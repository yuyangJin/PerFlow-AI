import torch
import torch.nn as nn
import time
import json
from collections import defaultdict
from pathlib import Path
import os
from typing import Optional, List, Dict, Any, Tuple


from .trace_nodes import TreeNode, AnalyzeTree, extract_info
from .ops_calculator import OpsCalculator
from .performance_calculator import PerformanceCalculator
from .device_controller import DeviceController

LOG_DIR = Path("hook_logs")
_events_buffer = []  # Global event buffer


class RooflineProfiler:
    """
    A comprehensive profiler for neural network models that tracks forward passes,
    builds call trees, calculates operations, and analyzes performance.
    Supports multi-process environments with distributed training ranks.
    """
    
    def __init__(self, log_dir: Optional[str] = None, 
                 filter_models: Optional[List[str]] = None,
                 rank: Optional[int] = None,
                 device_id: Optional[int] = None,
                 dp_rank: Optional[int] = None,
                 tp_rank: Optional[int] = None,
                 ep_rank: Optional[int] = None,
                 pp_rank: Optional[int] = None):
        """
        Initialize the RooflineProfiler.
        
        Args:
            log_dir: Directory to store profiling logs. If None, uses default "hook_logs"
            filter_models: List of model names to filter for profiling
            rank: Global rank in distributed training. If provided, only rank 0 will clean log directory
            device_id: Device ID in multi-device setup. If provided, only device 0 will clean log directory
            dp_rank: Data parallel rank. If provided, only rank 0 will clean log directory
            tp_rank: Tensor parallel rank. If provided, only rank 0 will clean log directory
            ep_rank: Expert parallel rank. If provided, only rank 0 will clean log directory
            pp_rank: Pipeline parallel rank. If provided, only rank 0 will clean log directory
        """
        self._handles = []
        self.call_trees = {}  # Stores all call trees
        self.analysis_trees = []  # Stores all analysis trees
        self.module_counters = defaultdict(int)  # Counts occurrences of each module
        self.active_calls = {}  # Tracks active calls {pid: [call_stack]}
        self.ops_calculator = OpsCalculator()
        self.performance_calculator = PerformanceCalculator()
        self.model_types = set()
        self.device_controller = DeviceController()

        # Set log directory
        if log_dir:
            global LOG_DIR
            LOG_DIR = Path(log_dir)
        
        self.filter = filter_models or ["Qwen2"]
        
        # Store rank information for multi-process handling
        self.rank = rank
        if self.rank is None:
            try:
                import torch.distributed as dist
                self.rank = dist.get_rank()
            except:
                pass
        self.device_id = device_id
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.ep_rank = ep_rank
        self.pp_rank = pp_rank
    
    def _should_clean_log_directory(self) -> bool:
        """
        Determine if this process should clean the log directory.
        
        Returns:
            True if this process should clean the directory, False otherwise
        """
        # If no rank information is provided, we're not in a multi-process environment
        if all(r is None for r in [self.rank, self.device_id, self.dp_rank, self.tp_rank, self.ep_rank, self.pp_rank]):
            return True
        
        # Check each rank type - only rank 0 should clean
        # If any rank is provided and not 0, don't clean
        if self.rank is not None and self.rank != 0:
            return False
        if self.device_id is not None and self.device_id != 0:
            return False
        if self.dp_rank is not None and self.dp_rank != 0:
            return False
        if self.tp_rank is not None and self.tp_rank != 0:
            return False
        if self.ep_rank is not None and self.ep_rank != 0:
            return False
        if self.pp_rank is not None and self.pp_rank != 0:
            return False
        
        # All provided ranks are 0 or None, so this process should clean
        return True
    
    def _clean_log_directory(self):
        """Clean up the log directory before starting a new profiling session."""
        if not self._should_clean_log_directory():
            print(f"Skipping log directory cleanup for process "
                  f"(rank={self.rank}, device_id={self.device_id}, "
                  f"dp_rank={self.dp_rank}, tp_rank={self.tp_rank}, ep_rank={self.ep_rank})")
            return
            
        if LOG_DIR.exists():
            for f in LOG_DIR.glob("*.json"):
                f.unlink()
            print(f"Cleaned log directory: {LOG_DIR}")
        LOG_DIR.mkdir(exist_ok=True)

    def _write_events_to_file(self, events: List[Dict]):
        """Write a batch of events to a new file with rank information in filename."""
        if not events:
            return
        
        # 构建文件名，包含rank信息
        filename_parts = []
        
        # 添加rank信息（如果存在）
        if self.rank is not None:
            filename_parts.append(f"Rank{self.rank}")
        if self.device_id is not None:
            filename_parts.append(f"Device{self.device_id}")
        if self.dp_rank is not None:
            filename_parts.append(f"DP{self.dp_rank}")
        if self.tp_rank is not None:
            filename_parts.append(f"TP{self.tp_rank}")
        if self.ep_rank is not None:
            filename_parts.append(f"EP{self.ep_rank}")
        if self.pp_rank is not None:
            filename_parts.append(f"PP{self.pp_rank}")
        
        # 添加进程ID和调用ID
        filename_parts.append(f"pid{events[0]['pid']}")
        filename_parts.append(f"{events[0]['call_id']}")
        
        # 组合文件名
        file_name = "_".join(filename_parts) + "_trace.json"
        file_path = LOG_DIR / file_name
        
        # 写入文件
        with open(file_path, "w") as f:
            for e in events:
                json.dump(e, f)
                f.write("\n")

    def _forward_pre_hook(self, module: nn.Module, inputs: Any):
        """Hook called before the forward pass of a module."""
        model_name = module.__class__.__name__
        input_info = [extract_info(i) for i in inputs]
            # 模块参数信息
        param_info = []
        for name, param in module.named_parameters(recurse=False):
            param_info.append({
                'name': name,
                'shape': list(param.shape),
                'dtype': str(param.dtype)
            })
        pid = os.getpid()
        
        # Create unique identifier using module name and occurrence count
        self.module_counters[model_name] += 1
        call_id = f"{model_name}_{self.module_counters[model_name]}"

        # Get current call stack
        if pid not in self.active_calls:
            self.active_calls[pid] = []
        call_stack = self.active_calls[pid]
        
        # Get parent call_id if exists
        parent_call_id = call_stack[-1] if call_stack else None
        start_time = time.time()
        
        record = {
            'phase': 'pre',
            'module': model_name,
            'timestamp': start_time,
            'input': input_info,
            'param': param_info,
            'pid': pid,
            'call_id': call_id,
            'parent_call_id': parent_call_id
        }
        
        _events_buffer.append(record)
        call_stack.append(call_id)  # Push current call to stack

    def _forward_hook(self, module: nn.Module, inputs: Any, output: Any):
        """Hook called after the forward pass of a module."""
        end_time = time.time()
        model_name = module.__class__.__name__
        output_info = extract_info(output)
        pid = os.getpid()
        
        # Get current call stack
        if pid not in self.active_calls or not self.active_calls[pid]:
            return  # No corresponding pre call
            
        call_stack = self.active_calls[pid]
        call_id = call_stack.pop() if call_stack else None
        
        if not call_id or not call_id.startswith(model_name):
            # Call mismatch, possibly an exception
            return
            
        # Get parent call_id
        parent_call_id = call_stack[-1] if call_stack else None

        record = {
            'phase': 'post',
            'module': model_name,
            'timestamp': end_time,
            'output': output_info,
            'pid': pid,
            'call_id': call_id,
            'parent_call_id': parent_call_id
        }
        
        _events_buffer.append(record)

        # If call stack is empty, write events to file if they match our filter
        if not call_stack:
            if self.filter is None:
                self._write_events_to_file(_events_buffer)
            else:
                for f in self.filter:
                    if f in _events_buffer[0]['module']:
                        self._write_events_to_file(_events_buffer)
                        break
            _events_buffer.clear()

    def start(self):
        """Start profiling by registering hooks."""
        self._clean_log_directory()
        h1 = torch.nn.modules.module.register_module_forward_pre_hook(self._forward_pre_hook)
        h2 = torch.nn.modules.module.register_module_forward_hook(self._forward_hook)
        self._handles.extend([h1, h2])

    def stop(self):
        """Stop profiling by removing all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.active_calls.clear()
    
    def build_tree(self, filepath: str):
        """

        """
        rank_info = self._extract_rank_info_from_filename(filepath.name)
        records = []
        rank_key = (
            f"Rank{rank_info['rank'] if rank_info['rank'] is not None else -1}_"
            f"Device{rank_info['device_id'] if rank_info['device_id'] is not None else -1}_"
            f"DP{rank_info['dp_rank'] if rank_info['dp_rank'] is not None else -1}_"
            f"TP{rank_info['tp_rank'] if rank_info['tp_rank'] is not None else -1}_"
            f"EP{rank_info['ep_rank'] if rank_info['ep_rank'] is not None else -1}_"
            f"PP{rank_info['pp_rank'] if rank_info['pp_rank'] is not None else -1}"
        )
        with open(filepath, "r") as fh:
            for line in fh:
                record = json.loads(line)
                records.append(record)

        records.sort(key=lambda x: x['timestamp'])
        nodes = {}
        for record in records:
            if record['phase'] == 'pre':
                node = TreeNode(record['module'], record['timestamp'], 
                               record['call_id'], record['pid'])
                node.input_info = record['input']
                node.param_info = record['param']
                node.parent_id = record.get('parent_call_id')
                nodes[record['call_id']] = node
            elif record['phase'] == 'post' and record['call_id'] in nodes:
                node = nodes[record['call_id']]
                node.end_time = record['timestamp']
                node.elapsed = node.end_time - node.start_time
                node.output_info = record['output']

        root = None
        for _, node in nodes.items():
            if node.parent_id is None:
                root = node
            else:
                parent_node = nodes[node.parent_id]
                parent_node.children.append(node)

        batchsize = root.input_info[0]['shape'][0]
        model_name = root.name

        if batchsize is None:
            batchsize = 0

        trees = self.call_trees.setdefault(rank_key, {})
        model_trees = trees.setdefault(model_name, {})
        batch_trees = model_trees.setdefault(str(batchsize), [])
        batch_trees.append(root)

    def build_trees(self):
        for f in LOG_DIR.glob("*_trace.json"):
            self.build_tree(f)

    def _extract_rank_info_from_filename(self, filename: str) -> Dict[str, Optional[int]]:
        """
        Extract rank information from trace file filename.
        
        Expected filename format: RankX_DeviceY_DPZ_TPA_EPB_PPC_pid12345_callid_trace.json
        """
        rank_info = {
            'rank': None,
            'device_id': None,
            'dp_rank': None,
            'tp_rank': None,
            'ep_rank': None,
            'pp_rank': None
        }
        
        parts = filename.split('_')
        for part in parts:
            if part.startswith('Rank') and part[4:].isdigit():
                rank_info['rank'] = int(part[4:])
            elif part.startswith('Device') and part[6:].isdigit():
                rank_info['device_id'] = int(part[6:])
            elif part.startswith('DP') and part[2:].isdigit():
                rank_info['dp_rank'] = int(part[2:])
            elif part.startswith('TP') and part[2:].isdigit():
                rank_info['tp_rank'] = int(part[2:])
            elif part.startswith('EP') and part[2:].isdigit():
                rank_info['ep_rank'] = int(part[2:])
            elif part.startswith('PP') and part[2:].isdigit():
                rank_info['pp_rank'] = int(part[2:])
        
        return rank_info
    
    def print_tree_summary(self):
        for rank_key, models in self.call_trees.items():
            print(f"Process {rank_key}:")
            for model_name, batches in models.items():
                print(f"  Model {model_name}:")
                for batchsize, trees in batches.items():
                    print(f"    Batch size={batchsize}: {len(trees)} instances")
    
    def print_tree(self, tree: Optional[Any] = None, indent: int = 0):
        """
        Print the call tree structure.
        
        Args:
            tree: Optional specific tree to print (TreeNode or dict)
            indent: Initial indentation level
        """
        if tree is None:
            for tree in self.call_trees:
                self._print_tree_node(tree, indent)
        elif isinstance(tree, TreeNode):
            self._print_tree_node(tree, indent)
        elif isinstance(tree, dict):
            self._print_tree_dict(tree, indent)
    
    def _print_tree_node(self, node: TreeNode, indent: int):
        """Print a tree node with proper formatting."""
        prefix = "  " * indent
        elapsed_str = f"{node.elapsed:.6f}s" if node.elapsed is not None else "N/A"
        print(f"{prefix}{node.name} [PID: {node.pid}, ID: {node.call_id}, "
              f"Elapsed: {elapsed_str}, FLOPs: {node.flops}, "
              f"Memory Access: {node.bytes}]")
        for child in node.children:
            self._print_tree_node(child, indent + 1)
    
    def _print_tree_dict(self, node: Dict, indent: int):
        """Print a dictionary tree node with proper formatting."""
        prefix = "  " * indent
        elapsed_str = f"{node['elapsed']:.6f}s" if node['elapsed'] is not None else "N/A"
        print(f"{prefix}{node['name']} [PID: {node['pid']}, ID: {node['call_id']}, "
              f"Elapsed: {elapsed_str}]")
        for child in node['children']:
            self._print_tree_dict(child, indent + 1)
    
    def save_tree_json(self, path: str):
        """Save call trees to a JSON file."""
        trees_dict = [tree.to_dict() for tree in self.call_trees]
        with open(path, "w") as f:
            json.dump(trees_dict, f, indent=4)
            
    def _collect_tree_nodes(self, node: TreeNode, nodes: List[TreeNode]):
        """Recursively collect all nodes from a tree."""
        nodes.append(node)
        for child in node.children:
            self._collect_tree_nodes(child, nodes)

    def _collect_leaf_nodes(self, node: TreeNode, nodes: List[TreeNode]):
        """Recursively collect all leaf nodes from a tree."""
        if not node.children:
            nodes.append(node)
        else:
            for child in node.children:
                self._collect_leaf_nodes(child, nodes)
    
    def group_and_merge_trees(self, analysis_trees: List[AnalyzeTree]) -> List[AnalyzeTree]:
        """
        Group AnalyzeTree objects by rank_key, model_name, batch_size and structural equality,
        then merge them by averaging leaf node durations.
        
        Args:
            analysis_trees: List of AnalyzeTree objects to merge
            
        Returns:
            List of merged AnalyzeTree objects
        """
        # Group trees by key and structural equality
        grouped_trees = self._group_trees_by_structure(analysis_trees)
        
        merged_trees = []
        for group_key, trees in grouped_trees.items():
            if len(trees) > 1:
                # Merge trees with the same structure
                merged_tree = self._merge_trees(trees)
                merged_trees.append(merged_tree)
            else:
                # Single tree, no need to merge
                merged_trees.append(trees[0])

        for tree in merged_trees:
            tree.tree_node.merged = True
        
        return merged_trees
    
    def _group_trees_by_structure(self, analysis_trees: List[AnalyzeTree]) -> Dict[Tuple, List[AnalyzeTree]]:
        """
        Group trees by (rank_key, model_name, batch_size) and structural equality.
        """
        groups = defaultdict(list)
        
        for tree in analysis_trees:
            # Create a key based on metadata
            metadata_key = (tree.rank_key, tree.model_name, tree.batch_size)
            
            # Check if this tree matches any existing group with the same structure
            found_group = False
            for existing_tree in groups.get(metadata_key, []):
                if tree.is_structurally_equal(existing_tree):
                    groups[metadata_key].append(tree)
                    found_group = True
                    break
            
            if not found_group:
                # Create new group
                groups[metadata_key] = [tree]
        
        return groups
    
    def _merge_trees(self, trees: List[AnalyzeTree]) -> AnalyzeTree:
        """
        Merge multiple trees with the same structure by averaging leaf node durations.
        """
        if not trees:
            raise ValueError("Cannot merge empty list of trees")
        
        # Use the first tree as template
        template_tree = trees[0]
        
        # Create a new merged tree
        merged_tree = AnalyzeTree(
            rank_key=template_tree.rank_key,
            model_name=template_tree.model_name,
            batch_size=template_tree.batch_size,
            tree_node=self._merge_tree_nodes([t.tree_node for t in trees]),
            rank_info=template_tree.rank_info.copy()
        )
        
        return merged_tree
    
    def _merge_tree_nodes(self, nodes: List[TreeNode]) -> TreeNode:
        """
        Recursively merge multiple tree nodes with the same structure.
        """
        if not nodes:
            raise ValueError("Cannot merge empty list of nodes")
        
        template_node = nodes[0]
        
        # Create new node with averaged timing information
        merged_node = TreeNode(
            name=template_node.name,
            start_time=None,
            call_id=template_node.call_id,
            pid=template_node.pid
        )
        
        # Copy structural information
        merged_node.input_info = template_node.input_info
        merged_node.output_info = template_node.output_info
        merged_node.param_info = template_node.param_info
        merged_node.parent_id = template_node.parent_id
        merged_node.flops = template_node.flops
        merged_node.bytes = template_node.bytes
        
        # Average elapsed time for leaf nodes, sum for non-leaf nodes
        if not template_node.children:
            # Leaf node: average the elapsed times
            elapsed_times = [node.elapsed for node in nodes if node.elapsed is not None]
            merged_node.elapsed = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
            
            actual_durs = [node.actual_dur for node in nodes if hasattr(node, 'actual_dur')]
            merged_node.actual_dur = sum(actual_durs) / len(actual_durs) if actual_durs else 0
        else:
            # Non-leaf node: merge children first, then calculate timing
            merged_children = []
            for i in range(len(template_node.children)):
                child_nodes = [node.children[i] for node in nodes]
                merged_child = self._merge_tree_nodes(child_nodes)
                merged_children.append(merged_child)
            
            merged_node.children = merged_children
            
            # Calculate elapsed as as sum of children's elapsed times
            merged_node.elapsed = sum(child.elapsed for child in merged_children)
            
            # Calculate actual_dur as sum of children's actual_dur
            merged_node.actual_dur = sum(child.actual_dur for child in merged_children)

            merged_node.overhead_ratio = (merged_node.elapsed - merged_node.actual_dur) / merged_node.elapsed * 100 if merged_node.elapsed > 0 else 0 
        
        return merged_node

    def interactive_show(self):
        """Interactive performance analysis with user guidance."""
        print("=" * 60)
        print("Roofline Performance Analyzer")
        print("=" * 60)
        
        # Step 1: Build call tree
        print("\nBuilding call tree...")
        print("=" * 60)
        self.build_trees()
        self.print_tree_summary()

        # Step 2: Rank selection
        print("\nRank selection:")
        print("=" * 60)
        available_ranks = list(self.call_trees.keys())
        for i, rank in enumerate(available_ranks, 1):
            print(f"{i}. {rank}")
        print(f"{len(available_ranks) + 1}. All ranks")
        
        rank_choice = input(f"Select ranks (comma-separated, 1-{len(available_ranks) + 1}): ").strip()
        selected_ranks = self._process_selection(rank_choice, available_ranks, "all")
        
        if selected_ranks == "all":
            selected_ranks = available_ranks
        
        # Step 3: Model selection
        print("\nModel selection:")
        print("=" * 60)
        # 收集所有可用的模型
        all_models = set()
        for rank in selected_ranks:
            all_models.update(self.call_trees[rank].keys())
        
        all_models = sorted(all_models)
        for i, model in enumerate(all_models, 1):
            print(f"{i}. {model}")
        print(f"{len(all_models) + 1}. All models")
        
        model_choice = input(f"Select models (comma-separated, 1-{len(all_models) + 1}): ").strip()
        selected_models = self._process_selection(model_choice, all_models, "all")
        
        if selected_models == "all":
            selected_models = all_models
        
        # Step 4: Batch size selection
        print("\nBatch size selection:")
        print("=" * 60)
        # 收集所有可用的batch size
        all_batch_sizes = set()
        for rank in selected_ranks:
            for model in selected_models:
                if model in self.call_trees[rank]:
                    all_batch_sizes.update(self.call_trees[rank][model].keys())
        
        all_batch_sizes = sorted(all_batch_sizes, key=lambda x: int(x) if x.isdigit() else -1)
        for i, batch_size in enumerate(all_batch_sizes, 1):
            print(f"{i}. {batch_size}")
        print(f"{len(all_batch_sizes) + 1}. All batch sizes")
        
        batch_choice = input(f"Select batch sizes (comma-separated, 1-{len(all_batch_sizes) + 1}): ").strip()
        selected_batch_sizes = self._process_selection(batch_choice, all_batch_sizes, "all")
        
        if selected_batch_sizes == "all":
            selected_batch_sizes = all_batch_sizes
        
        # 构建AnalysisTree对象列表
        self._build_analysis_trees(selected_ranks, selected_models, selected_batch_sizes)

        # Step 5: Calculate flops and memory access
        for tree in self.analysis_trees:
            self.ops_calculator.calculate_tree_ops(tree.tree_node)

        # Step 6: select device
        device_info = self.device_controller.interactive_device_selection()
        print(f"\nSelected device: {device_info}")

        # Step 7: Merge trees
        print("\nMerging trees...")
        print("=" * 60)
        print("Do you want to merge trees with the same information (y/n)?")
        merge_choice = input().strip().lower()
        if merge_choice == 'y':
            self.analysis_trees = self.group_and_merge_trees(self.analysis_trees)
            print(f"After merged, there are {len(self.analysis_trees)} trees.")
        else:
            print("Trees will not be merged.")

        # Step 8: Draw roofline
        print("\nDrawing roofline...")
        print("=" * 60)
        print("Please select a folder to save the roofline plots.")
        save_folder = input("Folder path: ").strip()
        os.makedirs(save_folder, exist_ok=True)
        for tree in self.analysis_trees:
            tree_nodes = []
            self._collect_tree_nodes(tree.tree_node, tree_nodes)
            save_path = f"{save_folder}/{tree.rank_key}_{tree.tree_node.call_id}_{tree.batch_size}.png"
            self.performance_calculator.plot_roofline(nodes=tree_nodes, device_info=device_info, save_path=save_path, show_plot=False)
        
        
        # TODO: add mfu and roofline and output to file
        # self._print_selection_summary()
        # self.save_selection_info("selection.json")
    
    def _build_analysis_trees(self, selected_ranks, selected_models, selected_batch_sizes):
        """Build AnalyzeTree objects based on selection."""
        self.analysis_trees = []
        
        for rank_key in selected_ranks:
            # 从rank_key解析rank信息
            rank_info = self._parse_rank_key(rank_key)
            
            for model_name in selected_models:
                if model_name in self.call_trees[rank_key]:
                    for batch_size in selected_batch_sizes:
                        if batch_size in self.call_trees[rank_key][model_name]:
                            for tree_node in self.call_trees[rank_key][model_name][batch_size]:
                                analysis_tree = AnalyzeTree(
                                    rank_key=rank_key,
                                    model_name=model_name,
                                    batch_size=batch_size,
                                    tree_node=tree_node,
                                    rank_info=rank_info
                                )
                                self.analysis_trees.append(analysis_tree)
    
    def _parse_rank_key(self, rank_key: str) -> Dict[str, Optional[int]]:
        """Parse rank key string back to rank info dictionary."""
        # 示例: "Rank0_Device0_DP0_TP0_EP0_PP0" -> rank_info字典
        rank_info = {
            'rank': None,
            'device_id': None,
            'dp_rank': None,
            'tp_rank': None,
            'ep_rank': None,
            'pp_rank': None
        }
        
        parts = rank_key.split('_')
        for part in parts:
            if part.startswith('Rank') and part[4:].isdigit():
                rank_info['rank'] = int(part[4:])
            elif part.startswith('Device') and part[6:].isdigit():
                rank_info['device_id'] = int(part[6:])
            elif part.startswith('DP') and part[2:].isdigit():
                rank_info['dp_rank'] = int(part[2:])
            elif part.startswith('TP') and part[2:].isdigit():
                rank_info['tp_rank'] = int(part[2:])
            elif part.startswith('EP') and part[2:].isdigit():
                rank_info['ep_rank'] = int(part[2:])
            elif part.startswith('PP') and part[2:].isdigit():
                rank_info['pp_rank'] = int(part[2:])
        
        return rank_info
    
    def _process_selection(self, choice_input, options, all_option="all"):
        """Process user selection input."""
        choices = choice_input.split(',')
        selected = []
        
        for choice in choices:
            choice = choice.strip()
            if not choice:
                continue
                
            try:
                choice_num = int(choice)
                if choice_num == len(options) + 1:
                    return all_option
                elif 1 <= choice_num <= len(options):
                    selected.append(options[choice_num - 1])
                else:
                    print(f"Warning: Invalid choice {choice_num}, skipping")
            except ValueError:
                # Check if it's a direct name match
                if choice in options:
                    selected.append(choice)
                elif choice == all_option:
                    return all_option
                else:
                    print(f"Warning: Invalid choice '{choice}', skipping")
        
        return selected if selected else all_option
    
    def _print_selection_summary(self):
        """Print selection summary."""
        print("\n" + "=" * 60)
        print("Selection Summary:")
        print("=" * 60)
        
        # 统计信息
        ranks = set(tree.rank_key for tree in self.analysis_trees)
        models = set(tree.model_name for tree in self.analysis_trees)
        batch_sizes = set(tree.batch_size for tree in self.analysis_trees)
        
        print(f"Selected ranks: {len(ranks)}")
        for rank in sorted(ranks):
            print(f"  - {rank}")
        
        print(f"\nSelected models: {len(models)}")
        for model in sorted(models):
            print(f"  - {model}")
        
        print(f"\nSelected batch sizes: {len(batch_sizes)}")
        for batch_size in sorted(batch_sizes, key=lambda x: int(x) if x.isdigit() else -1):
            print(f"  - {batch_size}")
        
        print(f"\nTotal trees selected: {len(self.analysis_trees)}")
        print("=" * 60)
    
    def get_trees_by_criteria(self, rank_key=None, model_name=None, batch_size=None):
        """Get trees filtered by specific criteria."""
        filtered_trees = self.analysis_trees
        
        if rank_key is not None:
            filtered_trees = [tree for tree in filtered_trees if tree.rank_key == rank_key]
        
        if model_name is not None:
            filtered_trees = [tree for tree in filtered_trees if tree.model_name == model_name]
        
        if batch_size is not None:
            filtered_trees = [tree for tree in filtered_trees if tree.batch_size == batch_size]
        
        return filtered_trees
    
    def save_selection_info(self, filepath: str):
        """Save selection information to JSON file."""
        selection_data = {
            'total_trees': len(self.analysis_trees),
            'trees': [tree.to_dict() for tree in self.analysis_trees]
        }
        
        with open(filepath, 'w') as f:
            json.dump(selection_data, f, indent=2)
        
        print(f"Selection info saved to {filepath}")