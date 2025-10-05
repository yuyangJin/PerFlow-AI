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
        Map trace events to MST nodes.
        
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
                # Try to find matching node by name
                matching_nodes = mst.find_nodes_by_name(event.get_name())
                
                if matching_nodes:
                    # Map to first matching node
                    matching_nodes[0].add_trace_event(event.get_id())
                else:
                    # Try to find by call stack if available
                    if event_index in callstacks:
                        callstack = callstacks[event_index]
                        matched_node = mst.map_trace_events_by_callstack(event.get_id(), callstack)
                        if not matched_node:
                            # Create a new node if no match found
                            new_node = mst.add_node(event.get_name(), 'operation', 0)
                            new_node.add_trace_event(event.get_id())
                    else:
                        # Create a new node for unmapped event
                        new_node = mst.add_node(event.get_name(), 'operation', 0)
                        new_node.add_trace_event(event.get_id())
                
                event_index += 1
        
        return mst
    
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
