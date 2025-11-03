'''
@module MST Mapper
Maps trace events from TorchProfiler to Model Structure Tree nodes.
'''

from typing import List, Dict, Optional
import json
from .mst import ModelStructureTree, MSTNode
from ..core.trace import Trace
from ..core.event import Event

class MSTMapper:
    '''
    Maps trace events to MST nodes based on call stacks and event names.
    '''
    
    @staticmethod
    def extract_callstack_from_torchprofiler(trace_json: Dict) -> Dict[int, List[str]]:
        '''
        Extract call stacks from TorchProfiler JSON trace.
        Uses Python parent id and Python id for better call stack reconstruction.
        
        Args:
            trace_json: Parsed TorchProfiler JSON
            
        Returns:
            Dictionary mapping event index to call stack
        '''
        callstacks = {}
        event_id_to_name = {}  # Map Python id to event name
        event_id_to_parent = {}  # Map Python id to parent id
        
        # First pass: build id mappings
        for i, event in enumerate(trace_json.get('traceEvents', [])):
            if 'args' in event:
                python_id = event['args'].get('Python id')
                python_parent_id = event['args'].get('Python parent id')
                name = event.get('name', 'unknown')
                
                if python_id is not None:
                    event_id_to_name[python_id] = name
                    if python_parent_id is not None:
                        event_id_to_parent[python_id] = python_parent_id
        
        # Second pass: reconstruct call stacks using parent relationships
        for i, event in enumerate(trace_json.get('traceEvents', [])):
            if 'args' in event:
                # Try to use Python id/parent id for hierarchical call stack
                python_id = event['args'].get('Python id')
                
                if python_id is not None and (python_id in event_id_to_name or python_id in event_id_to_parent):
                    # Reconstruct call stack by following parent chain
                    stack = []
                    current_id = python_id
                    visited = set()  # Prevent infinite loops
                    
                    while current_id is not None and current_id not in visited:
                        visited.add(current_id)
                        if current_id in event_id_to_name:
                            stack.append(event_id_to_name[current_id])
                        current_id = event_id_to_parent.get(current_id)
                    
                    # Reverse to get root-to-leaf order
                    stack.reverse()
                    if stack:
                        callstacks[i] = stack
                        continue
                
                # Fallback: try Python call stack string
                stack_info = event['args'].get('Python call stack', '')
                if stack_info:
                    # Parse the stack string into a list
                    stack = [line.strip() for line in stack_info.split('\n') if line.strip()]
                    callstacks[i] = stack
                    continue
                
                # Fallback: use event name
                name = event.get('name', '')
                callstacks[i] = [name] if name else ['unknown']
            else:
                # Use event name as minimal call stack
                callstacks[i] = [event.get('name', 'unknown')]
        
        return callstacks
    
    @staticmethod
    def build_mst_from_torchprofiler(trace_json: Dict) -> ModelStructureTree:
        '''
        Build an MST from TorchProfiler JSON trace.
        
        Args:
            trace_json: Parsed TorchProfiler JSON
            
        Returns:
            ModelStructureTree built from the trace
        '''
        mst = ModelStructureTree()
        
        # Extract unique operation names and build hierarchy
        operations = set()
        
        for event in trace_json.get('traceEvents', []):
            name = event.get('name', '')
            if name:
                operations.add(name)
        
        # Build a simple hierarchy based on naming patterns
        # Group by prefixes (e.g., 'forward_step', 'backward_step')
        groups = {}
        
        for op in operations:
            # Extract prefix (before first '_' or '-')
            if '_' in op:
                prefix = op.split('_')[0]
            elif '-' in op:
                prefix = op.split('-')[0]
            else:
                prefix = 'other'
            
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(op)
        
        # Create nodes for each group
        for group_name, ops in groups.items():
            group_node = mst.add_node(group_name, 'group', 0)
            
            for op in ops:
                op_node = mst.add_node(op, 'operation', group_node.node_id)
        
        return mst
    
    @staticmethod
    def map_trace_to_mst(trace: Trace, mst: ModelStructureTree, 
                        trace_json: Optional[Dict] = None) -> ModelStructureTree:
        '''
        Map trace events to MST nodes with improved name matching.
        
        Uses multiple strategies:
        1. Exact name matching
        2. Normalized name matching (handles aten:: prefix, case differences)
        3. Call stack-based matching
        4. Pattern-based matching for common PyTorch operations
        
        Args:
            trace: The trace with events
            mst: The Model Structure Tree to map to
            trace_json: Optional original TorchProfiler JSON for call stack info
            
        Returns:
            Updated MST with mapped events
        '''
        # Extract call stacks if trace_json is provided
        callstacks = {}
        if trace_json:
            callstacks = MSTMapper.extract_callstack_from_torchprofiler(trace_json)
        
        # Map each event to MST
        event_index = 0
        for dev_id in range(trace.get_ndevs()):
            events = trace.get_events(dev_id)
            
            for event in events:
                event_name = event.get_name()
                matched = False
                
                # Strategy 1: Exact name matching
                matching_nodes = mst.find_nodes_by_name(event_name)
                if matching_nodes:
                    matching_nodes[0].add_trace_event(event.get_id())
                    matched = True
                else:
                    # Strategy 2: Normalized name matching
                    matched_node = MSTMapper._find_node_by_normalized_name(mst, event_name)
                    if matched_node:
                        matched_node.add_trace_event(event.get_id())
                        matched = True
                    else:
                        # Strategy 3: Call stack-based matching
                        if event_index in callstacks:
                            callstack = callstacks[event_index]
                            matched_node = mst.map_trace_events_by_callstack(event.get_id(), callstack)
                            if matched_node:
                                matched = True
                        
                        # Strategy 4: If still not matched, create new node
                        if not matched:
                            new_node = mst.add_node(event_name, 'operation', 0)
                            new_node.add_trace_event(event.get_id())
                
                event_index += 1
        
        return mst
    
    @staticmethod
    def _find_node_by_normalized_name(mst: ModelStructureTree, event_name: str) -> Optional:
        '''
        Find MST node by normalized event name.
        
        Handles common naming differences between TorchProfiler and MST:
        - aten::embedding -> embedding
        - aten::linear -> linear
        - Case differences
        - Module path differences
        
        Args:
            mst: The Model Structure Tree
            event_name: The event name from trace
            
        Returns:
            Matching MSTNode or None
        '''
        # Normalize event name
        normalized_event = MSTMapper._normalize_operation_name(event_name)
        
        # Try to find by normalized name
        for node in mst.nodes.values():
            if node.node_type in ['root']:
                continue
            
            # Normalize MST node name
            normalized_node = MSTMapper._normalize_operation_name(node.name)
            
            # Check for match
            if normalized_event == normalized_node:
                return node
            
            # Check if event name ends with module name (e.g., "model.embedding" matches "embedding")
            if normalized_event.endswith(normalized_node) or normalized_node.endswith(normalized_event):
                return node
            
            # Check module path attribute
            if 'module_path' in node.attributes:
                module_path = node.attributes['module_path']
                if normalized_event in module_path or module_path in normalized_event:
                    return node
        
        return None
    
    @staticmethod
    def _normalize_operation_name(name: str) -> str:
        '''
        Normalize operation name for better matching.
        
        Handles:
        - aten:: prefix removal
        - torch.nn. prefix removal  
        - Case normalization
        - Path separator normalization
        
        Args:
            name: Original operation name
            
        Returns:
            Normalized name
        '''
        # Remove common prefixes
        prefixes_to_remove = ['aten::', 'torch.nn.', 'torch.', 'nn.']
        normalized = name
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        # Remove file path information (e.g., "file.py(123): func" -> "func")
        if '): ' in normalized:
            normalized = normalized.split('): ')[-1]
        
        # Convert to lowercase for comparison
        normalized = normalized.lower()
        
        # Replace path separators
        normalized = normalized.replace('/', '.').replace('\\', '.')
        
        # Remove trailing/leading whitespace
        normalized = normalized.strip()
        
        return normalized
    
    @staticmethod
    def build_and_map_from_file(trace_path: str, trace: Trace) -> ModelStructureTree:
        '''
        Build MST and map trace from a TorchProfiler JSON file.
        
        Args:
            trace_path: Path to TorchProfiler JSON file
            trace: The parsed trace
            
        Returns:
            MST with mapped events
        '''
        with open(trace_path, 'r') as f:
            trace_json = json.load(f)
        
        mst = MSTMapper.build_mst_from_torchprofiler(trace_json)
        mst = MSTMapper.map_trace_to_mst(trace, mst, trace_json)
        
        return mst
