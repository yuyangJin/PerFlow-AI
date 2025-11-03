'''
@module Model Structure Tree (MST)
The MST represents the hierarchical structure of a model with call stacks.
'''

from typing import List, Dict, Optional, Any
import json

try:
    import torch
    import torch.fx as fx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

class MSTNode:
    '''
    A node in the Model Structure Tree representing an operation or module.
    '''
    def __init__(self, name: str, node_type: str, node_id: int = None):
        '''
        Args:
            name: Name of the operation/module
            node_type: Type of the node (e.g., 'module', 'operator', 'kernel')
            node_id: Unique identifier for the node
        '''
        self.name = name
        self.node_type = node_type
        self.node_id = node_id
        self.children: List[MSTNode] = []
        self.parent: Optional[MSTNode] = None
        self.attributes: Dict[str, Any] = {}
        self.call_stack: List[str] = []
        self.trace_events: List[int] = []  # IDs of events mapped to this node
        
    def add_child(self, child: 'MSTNode'):
        '''Add a child node to this node'''
        child.parent = self
        self.children.append(child)
        
    def add_attribute(self, key: str, value: Any):
        '''Add an attribute to this node'''
        self.attributes[key] = value
        
    def set_call_stack(self, call_stack: List[str]):
        '''Set the call stack for this node'''
        self.call_stack = call_stack
        
    def add_trace_event(self, event_id: int):
        '''Map a trace event ID to this node'''
        self.trace_events.append(event_id)
        
    def get_depth(self) -> int:
        '''Get the depth of this node in the tree'''
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth
        
    def to_dict(self) -> Dict:
        '''Convert node to dictionary representation'''
        return {
            'name': self.name,
            'node_type': self.node_type,
            'node_id': self.node_id,
            'attributes': self.attributes,
            'call_stack': self.call_stack,
            'trace_events': self.trace_events,
            'children': [child.to_dict() for child in self.children]
        }
        
    def __str__(self):
        return f"MSTNode({self.name}, type={self.node_type}, id={self.node_id})"
        
    def __repr__(self):
        return str(self)


class ModelStructureTree:
    '''
    Model Structure Tree (MST) represents the hierarchical structure of a model.
    It can be built from Torch.fx or mapped from trace events.
    '''
    def __init__(self):
        self.root = MSTNode('root', 'root', 0)
        self.nodes: Dict[int, MSTNode] = {0: self.root}
        self.next_id = 1
        
    def add_node(self, name: str, node_type: str, parent_id: int = 0) -> MSTNode:
        '''
        Add a node to the tree
        
        Args:
            name: Name of the node
            node_type: Type of the node
            parent_id: ID of the parent node (default: root)
            
        Returns:
            The created node
        '''
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node with id {parent_id} does not exist")
            
        node = MSTNode(name, node_type, self.next_id)
        self.nodes[self.next_id] = node
        self.next_id += 1
        
        parent = self.nodes[parent_id]
        parent.add_child(node)
        
        return node
        
    def get_node(self, node_id: int) -> Optional[MSTNode]:
        '''Get a node by its ID'''
        return self.nodes.get(node_id)
        
    def find_nodes_by_name(self, name: str) -> List[MSTNode]:
        '''Find all nodes with a given name'''
        return [node for node in self.nodes.values() if node.name == name]
        
    def find_nodes_by_type(self, node_type: str) -> List[MSTNode]:
        '''Find all nodes with a given type'''
        return [node for node in self.nodes.values() if node.node_type == node_type]
        
    def map_trace_event(self, event_id: int, node_id: int):
        '''Map a trace event to a node in the MST'''
        if node_id not in self.nodes:
            raise ValueError(f"Node with id {node_id} does not exist")
        self.nodes[node_id].add_trace_event(event_id)
        
    def map_trace_events_by_callstack(self, event_id: int, call_stack: List[str]) -> Optional[MSTNode]:
        '''
        Map a trace event to a node based on call stack matching
        
        Args:
            event_id: ID of the trace event
            call_stack: Call stack from the trace event
            
        Returns:
            The node the event was mapped to, or None if no match found
        '''
        # Find the best matching node based on call stack
        best_match = None
        best_match_depth = -1
        
        for node in self.nodes.values():
            if not node.call_stack:
                continue
                
            # Check if node's call stack is a prefix of event's call stack
            if len(node.call_stack) <= len(call_stack):
                match = all(node.call_stack[i] == call_stack[i] 
                           for i in range(len(node.call_stack)))
                if match and len(node.call_stack) > best_match_depth:
                    best_match = node
                    best_match_depth = len(node.call_stack)
                    
        if best_match:
            best_match.add_trace_event(event_id)
            
        return best_match
        
    def visualize(self, node: MSTNode = None, indent: int = 0) -> str:
        '''
        Generate a text visualization of the tree
        
        Args:
            node: Starting node (default: root)
            indent: Current indentation level
            
        Returns:
            String representation of the tree
        '''
        if node is None:
            node = self.root
            
        result = []
        prefix = "  " * indent
        
        # Show node info
        info = f"{prefix}{node.name} ({node.node_type})"
        if node.trace_events:
            info += f" [events: {len(node.trace_events)}]"
        if node.call_stack:
            info += f" [depth: {len(node.call_stack)}]"
        result.append(info)
        
        # Recursively show children
        for child in node.children:
            result.append(self.visualize(child, indent + 1))
            
        return "\n".join(result)
        
    def to_dict(self) -> Dict:
        '''Convert the entire tree to a dictionary'''
        return {
            'root': self.root.to_dict(),
            'node_count': len(self.nodes)
        }
        
    def to_json(self, filepath: str):
        '''Save the tree to a JSON file'''
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @staticmethod
    def from_dict(data: Dict) -> 'ModelStructureTree':
        '''Create a tree from a dictionary representation'''
        mst = ModelStructureTree()
        
        def build_node(node_data: Dict, parent_id: int = 0):
            node = mst.add_node(
                node_data['name'],
                node_data['node_type'],
                parent_id
            )
            node.attributes = node_data.get('attributes', {})
            node.call_stack = node_data.get('call_stack', [])
            node.trace_events = node_data.get('trace_events', [])
            
            for child_data in node_data.get('children', []):
                build_node(child_data, node.node_id)
                
        # Skip the root as it's already created
        for child_data in data['root'].get('children', []):
            build_node(child_data, 0)
            
        return mst
        
    @staticmethod
    def from_json(filepath: str) -> 'ModelStructureTree':
        '''Load a tree from a JSON file'''
        with open(filepath, 'r') as f:
            data = json.load(f)
        return ModelStructureTree.from_dict(data)
    
    @staticmethod
    def from_torch_fx_graph(graph, model_name: str = 'model', merge_similar_layers: bool = True) -> 'ModelStructureTree':
        '''
        Build hierarchical MST from a torch.fx graph.
        
        Creates a proper hierarchy based on module structure:
        - Module calls create parent nodes with their operations as children
        - Sequential operations are grouped under their respective modules
        - Input/output nodes are top-level
        - Similar layers can be merged into loop structures (optional)
        
        Args:
            graph: torch.fx.Graph object from symbolic_trace
            model_name: Name of the model (default: 'model')
            merge_similar_layers: Whether to merge similar sequential layers into loops (default: True)
            
        Returns:
            ModelStructureTree with hierarchical structure
        '''
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use this feature.")
        
        mst = ModelStructureTree()
        
        # Create a model node
        model_node = mst.add_node(model_name, 'model', 0)
        model_node.set_call_stack([model_name])
        
        # Track nodes by their fx node name and module paths
        fx_node_to_mst = {}
        module_nodes = {}  # Map module paths to MST nodes
        
        # First pass: Create module hierarchy and detect repeated patterns
        module_list = []
        for fx_node in graph.nodes:
            if fx_node.op == 'call_module' and fx_node.target:
                module_path = str(fx_node.target)
                module_list.append(module_path)
        
        # Detect repeated module patterns (e.g., layers.0, layers.1, layers.2)
        loop_groups = {}
        if merge_similar_layers:
            loop_groups = ModelStructureTree._detect_loop_patterns(module_list)
        
        # Build module hierarchy with loops
        # Strategy: Build normally, but when we encounter a module that should have a loop child,
        # create the loop node and make subsequent numeric children go under it
        for fx_node in graph.nodes:
            if fx_node.op == 'call_module' and fx_node.target:
                module_path = str(fx_node.target)
                parts = module_path.split('.')
                
                # Check if this entire path is a loop member
                is_loop_member = False
                loop_prefix_for_this_path = None
                for loop_prefix, loop_info in loop_groups.items():
                    if module_path in loop_info['members']:
                        is_loop_member = True
                        loop_prefix_for_this_path = loop_prefix
                        break
                
                # Build hierarchy level by level
                current_parent = model_node.node_id
                for i, part in enumerate(parts):
                    path_so_far = '.'.join(parts[:i+1])
                    
                    # Special handling: if next part is numeric and this path matches a loop prefix,
                    # we need to create a loop node here
                    if i + 1 < len(parts) and parts[i+1].isdigit():
                        # Check if there's a loop for path_so_far
                        for loop_prefix, loop_info in loop_groups.items():
                            if loop_prefix == path_so_far:
                                # This is the parent of a loop! Create loop node if not exists
                                loop_key = f"_loop_{loop_prefix}"
                                
                                # First ensure the current path exists
                                if path_so_far not in module_nodes:
                                    module_node = mst.add_node(part, 'module', current_parent)
                                    module_node.add_attribute('module_path', path_so_far)
                                    module_node.set_call_stack([model_name] + parts[:i+1])
                                    module_nodes[path_so_far] = module_node
                                
                                # Now create loop node under it
                                if loop_key not in module_nodes:
                                    loop_name = f"{loop_info['name']}_loop"
                                    loop_node = mst.add_node(loop_name, 'loop', module_nodes[path_so_far].node_id)
                                    loop_node.add_attribute('loop_count', loop_info['count'])
                                    loop_node.add_attribute('loop_pattern', loop_prefix)
                                    loop_node.add_attribute('start_index', loop_info.get('start_index', 0))
                                    loop_node.add_attribute('end_index', loop_info.get('end_index', loop_info['count']-1))
                                    
                                    call_stack_parts = parts[:i+1] + [loop_name]
                                    loop_node.set_call_stack([model_name] + call_stack_parts)
                                    module_nodes[loop_key] = loop_node
                                
                                # Set current parent to be under the loop for next iteration
                                current_parent = module_nodes[loop_key].node_id
                                break
                        else:
                            # No loop found, process normally
                            if path_so_far not in module_nodes:
                                module_node = mst.add_node(part, 'module', current_parent)
                                module_node.add_attribute('module_path', path_so_far)
                                module_node.set_call_stack([model_name] + parts[:i+1])
                                module_nodes[path_so_far] = module_node
                            current_parent = module_nodes[path_so_far].node_id
                    else:
                        # Normal processing
                        if path_so_far not in module_nodes:
                            module_node = mst.add_node(part, 'module', current_parent)
                            module_node.add_attribute('module_path', path_so_far)
                            module_node.set_call_stack([model_name] + parts[:i+1])
                            module_nodes[path_so_far] = module_node
                        current_parent = module_nodes[path_so_far].node_id
        
        # Second pass: Add operations under appropriate parents
        for fx_node in graph.nodes:
            node_name = fx_node.name
            node_op = fx_node.op
            
            # Determine node type based on operation
            if node_op == 'placeholder':
                node_type = 'input'
                parent_id = model_node.node_id
            elif node_op == 'get_attr':
                node_type = 'parameter'
                parent_id = model_node.node_id
            elif node_op == 'call_function':
                node_type = 'function'
                parent_id = model_node.node_id
            elif node_op == 'call_method':
                node_type = 'method'
                parent_id = model_node.node_id
            elif node_op == 'call_module':
                # Module call - add as operation under the module
                node_type = 'operation'
                module_path = str(fx_node.target)
                if module_path in module_nodes:
                    parent_id = module_nodes[module_path].node_id
                else:
                    parent_id = model_node.node_id
            elif node_op == 'output':
                node_type = 'output'
                parent_id = model_node.node_id
            else:
                node_type = 'operation'
                parent_id = model_node.node_id
            
            # Skip if this is just a module definition (already created)
            if node_op == 'call_module' and str(fx_node.target) in module_nodes:
                # Add operation node under the module
                mst_node = mst.add_node(f"{node_name}_call", node_type, parent_id)
            else:
                # Create the node
                mst_node = mst.add_node(node_name, node_type, parent_id)
            
            # Add attributes
            mst_node.add_attribute('op', node_op)
            if fx_node.target:
                mst_node.add_attribute('target', str(fx_node.target))
            
            # Set call stack based on parent hierarchy
            parent = mst.nodes[parent_id]
            if parent.call_stack:
                call_stack = parent.call_stack + [node_name]
            else:
                call_stack = [model_name, node_name]
            mst_node.set_call_stack(call_stack)
            
            # Store mapping
            fx_node_to_mst[node_name] = mst_node
        
        return mst
    
    @staticmethod
    def _detect_loop_patterns(module_list: List[str]) -> Dict[str, Dict]:
        '''
        Detect repeated module patterns that can be merged into loops.
        
        Detects patterns in two ways:
        1. Direct numeric indexing: layers.0, layers.1, layers.2 (same level)
        2. Nested indexing: layers.0.linear, layers.1.linear (with submodules)
        
        Args:
            module_list: List of module paths from torch.fx graph
            
        Returns:
            Dictionary mapping loop prefix to loop info
        '''
        import re
        
        # Strategy 1: Group by parent prefix and numeric child
        # This handles cases like: layers.0, layers.1, layers.2, layers.3
        parent_groups = {}
        
        for module_path in module_list:
            parts = module_path.split('.')
            
            # Check each level for numeric indices
            for i, part in enumerate(parts):
                if part.isdigit():
                    # Found a numeric index at position i
                    prefix_parts = parts[:i]
                    prefix = '.'.join(prefix_parts) if prefix_parts else ''
                    
                    # Group by (parent_prefix, depth, has_children)
                    # This ensures we only merge at the same structural level
                    has_children = i < len(parts) - 1
                    suffix_pattern = '.'.join(parts[i+1:]) if has_children else ''
                    
                    key = (prefix, i, suffix_pattern)
                    if key not in parent_groups:
                        parent_groups[key] = []
                    parent_groups[key].append((int(part), module_path))
                    
                    # Only process the first numeric level found in each path
                    break
        
        # Build loop groups from detected patterns
        loop_groups = {}
        for (prefix, depth, suffix_pattern), members in parent_groups.items():
            if len(members) < 2:
                continue
            
            # Sort by numeric index
            members.sort(key=lambda x: x[0])
            indices = [m[0] for m in members]
            paths = [m[1] for m in members]
            
            # Check if indices are consecutive starting from 0 or any number
            min_idx, max_idx = min(indices), max(indices)
            expected_indices = list(range(min_idx, max_idx + 1))
            
            if indices == expected_indices:
                # Determine the loop name
                if prefix:
                    loop_prefix = prefix
                    loop_name = prefix.split('.')[-1]
                else:
                    # No prefix, use the common part from first path
                    first_parts = paths[0].split('.')
                    # Find the part before the numeric index
                    for i, part in enumerate(first_parts):
                        if part == str(indices[0]):
                            if i > 0:
                                loop_name = first_parts[i-1]
                                loop_prefix = '.'.join(first_parts[:i])
                            else:
                                loop_name = 'layer'
                                loop_prefix = 'layers'
                            break
                    else:
                        loop_name = 'layer'
                        loop_prefix = 'layers'
                
                # Only create loop group if we have consecutive members
                if len(members) >= 2:
                    loop_groups[loop_prefix] = {
                        'name': loop_name,
                        'count': len(members),
                        'members': paths,
                        'indices': indices,
                        'start_index': min_idx,
                        'end_index': max_idx
                    }
        
        return loop_groups
    
    def visualize_graphviz(self, output_path: str = 'mst', format: str = 'pdf'):
        '''
        Visualize the MST using Graphviz and save as PDF or other formats.
        
        Args:
            output_path: Output file path (without extension)
            format: Output format ('pdf', 'png', 'svg', etc.)
            
        Returns:
            Path to the generated file
        '''
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("Graphviz is not available. Please install graphviz: pip install graphviz")
        
        # Create a new directed graph with improved aesthetics
        dot = graphviz.Digraph(comment='Model Structure Tree', format=format)
        dot.attr(rankdir='TB')  # Top to bottom layout
        dot.attr('node', shape='circle', style='filled', fixedsize='true', 
                 width='0.4', height='0.4', fontsize='9', labelloc='b')
        dot.attr('edge', color='#666666', penwidth='1.5')
        
        # Nature paper inspired color palette (elegant, professional)
        # Based on Nature journal figure guidelines - softer, more refined colors
        type_colors = {
            'root': '#E8E8E8',           # Light gray
            'model': '#4A90E2',          # Professional blue
            'module': '#7CB342',         # Nature green
            'loop': '#FFA726',           # Warm orange for loops
            'operation': '#F4A460',      # Soft orange
            'operator': '#E57373',       # Coral red
            'function': '#9C64A6',       # Muted purple
            'method': '#BA68C8',         # Light purple
            'layer': '#26A69A',          # Teal
            'group': '#8BC34A',          # Light green
            'input': '#FDD835',          # Warm yellow
            'output': '#EC407A',         # Pink
            'parameter': '#AB47BC',      # Deep purple
            'kernel': '#EF5350'          # Red
        }
        
        # Add nodes to the graph
        def add_nodes(node: MSTNode):
            node_id = str(node.node_id)
            color = type_colors.get(node.node_type, '#9E9E9E')
            
            # Create label with node name (text outside circle)
            label = node.name
            if node.node_type == 'loop' and 'loop_count' in node.attributes:
                label = f"{label}\n(i=0..{node.attributes['loop_count']-1})"
            if len(label) > 30:
                label = label[:27] + '...'
            
            # Add tooltip with more info
            tooltip = f"{node.name}\\nType: {node.node_type}"
            if node.node_type == 'loop' and 'loop_count' in node.attributes:
                tooltip += f"\\nIterations: {node.attributes['loop_count']}"
            if node.trace_events:
                tooltip += f"\\nEvents: {len(node.trace_events)}"
            
            # Empty label inside circle, actual label below (labelloc='b')
            dot.node(node_id, label=label, fillcolor=color, tooltip=tooltip,
                    fontcolor='#333333')
            
            # Add edges to children
            for child in node.children:
                # Use dashed edge for loop iterations
                if node.node_type == 'loop':
                    dot.edge(node_id, str(child.node_id), style='dashed')
                else:
                    dot.edge(node_id, str(child.node_id))
                add_nodes(child)
        
        add_nodes(self.root)
        
        # Render the graph
        output_file = dot.render(output_path, cleanup=True)
        return output_file
        
    def __str__(self):
        return f"ModelStructureTree(nodes={len(self.nodes)})"
        
    def __repr__(self):
        return str(self)
