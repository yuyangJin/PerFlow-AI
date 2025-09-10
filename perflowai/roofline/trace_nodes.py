import json
from collections import defaultdict
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

class TreeNode:
    def __init__(self, name, start_time, call_id, pid):
        self.name = name
        self.merged = False
        self.start_time = start_time
        self.end_time = None
        self.elapsed = None
        self.actual_dur = None
        self.overhead_ratio = None
        self.children = []
        self.call_id = call_id
        self.pid = pid
        self.input_info = None
        self.param_info = None
        self.output_info = None
        self.parent_id = None

        self.flops = 0
        self.bytes = 0
        self.mfu = 0
        self.roofline = 0
    
    def to_dict(self):
        if self.merged:
            return {
                'name': self.name,
                'merged': self.merged,
                'elapsed': self.elapsed,
                'actual_dur': self.actual_dur,
                'overhead_ratio': self.overhead_ratio,
                'call_id': self.call_id,
                'pid': self.pid,
                'parent_id': self.parent_id,
                'input_info': self.input_info,
                'param_info': self.param_info,
                'output_info': self.output_info,
                'flops': self.flops,
                'bytes': self.bytes,
                'mfu': self.mfu,
                'roofline': self.roofline,
                'children': [child.to_dict() for child in self.children],
            }
        else:
            return {
                'name': self.name,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'elapsed': self.elapsed,
                'actual_dur': self.actual_dur,
                'overhead_ratio': self.overhead_ratio,
                'call_id': self.call_id,
                'pid': self.pid,
                'parent_id': self.parent_id,
                'input_info': self.input_info,
                'param_info': self.param_info,
                'output_info': self.output_info,
                'flops': self.flops,
                'bytes': self.bytes,
                'mfu': self.mfu,
                'roofline': self.roofline,
                'children': [child.to_dict() for child in self.children],
            }
    
    def is_structurally_equal(self, other: 'TreeNode') -> bool:
        """Check if two nodes have the same structure (name, input, output, params)."""
        if self.name != other.name:
            return False
        
        # Check input info
        if not self._compare_tensor_info(self.input_info, other.input_info):
            return False
        
        # Check output info
        if not self._compare_tensor_info(self.output_info, other.output_info):
            return False
        
        # Check param info
        if not self._compare_tensor_info(self.param_info, other.param_info):
            return False
        
        # Check children structure recursively
        if len(self.children) != len(other.children):
            return False
        
        for child1, child2 in zip(self.children, other.children):
            if not child1.is_structurally_equal(child2):
                return False
        
        return True
    
    def _compare_tensor_info(self, info1, info2) -> bool:
        """Compare tensor information for structural equality."""
        if info1 is None and info2 is None:
            return True
        if info1 is None or info2 is None:
            return False
        
        if isinstance(info1, list) and isinstance(info2, list):
            if len(info1) != len(info2):
                return False
            for item1, item2 in zip(info1, info2):
                if not self._compare_tensor_info(item1, item2):
                    return False
            return True
        elif isinstance(info1, dict) and isinstance(info2, dict):
            # Compare shape and dtype for tensors
            if info1.get('shape') != info2.get('shape'):
                return False
            if info1.get('dtype') != info2.get('dtype'):
                return False
            return True
        else:
            return info1 == info2
    
@dataclass
class AnalyzeTree:
    """Data structure to maintain analysis tree with full hierarchy information."""
    rank_key: str
    model_name: str
    batch_size: str
    tree_node: Any  # TreeNode对象
    rank_info: Dict[str, Optional[int]]  # 从文件名解析的rank详细信息
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'rank_key': self.rank_key,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'rank_info': self.rank_info,
            'tree_node': self.tree_node.to_dict() if hasattr(self.tree_node, 'to_dict') else str(self.tree_node)
        }
    
    def is_structurally_equal(self, other: 'AnalyzeTree') -> bool:
        """Check if two AnalyzeTree objects have the same structure."""
        return (self.rank_key == other.rank_key and
                self.model_name == other.model_name and
                self.batch_size == other.batch_size and
                self.tree_node.is_structurally_equal(other.tree_node))

def extract_info(x):
    if isinstance(x, torch.Tensor):
        return {'shape': tuple(x.shape), 'dtype': str(x.dtype)}
    elif isinstance(x, (list, tuple)):
        return [extract_info(i) for i in x]
    elif isinstance(x, dict):
        return {k: extract_info(v) for k, v in x.items()}
    else:
        return str(type(x))