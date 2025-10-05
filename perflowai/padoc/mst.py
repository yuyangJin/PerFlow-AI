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
    def from_torch_fx_graph(graph, model_name: str = 'model') -> 'ModelStructureTree':
        '''
        Build MST from a torch.fx graph.
        
        Args:
            graph: torch.fx.Graph object from symbolic_trace
            model_name: Name of the model (default: 'model')
            
        Returns:
            ModelStructureTree built from the graph
        '''
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use this feature.")
        
        mst = ModelStructureTree()
        
        # Create a model node
        model_node = mst.add_node(model_name, 'model', 0)
        model_node.set_call_stack([model_name])
        
        # Track nodes by their fx node name for building edges
        fx_node_to_mst = {}
        
        # Process each node in the graph
        for fx_node in graph.nodes:
            node_name = fx_node.name
            node_op = fx_node.op
            
            # Determine node type based on operation
            if node_op == 'placeholder':
                node_type = 'input'
            elif node_op == 'get_attr':
                node_type = 'parameter'
            elif node_op == 'call_function':
                node_type = 'function'
            elif node_op == 'call_method':
                node_type = 'method'
            elif node_op == 'call_module':
                node_type = 'module'
            elif node_op == 'output':
                node_type = 'output'
            else:
                node_type = 'operation'
            
            # Create MST node
            mst_node = mst.add_node(node_name, node_type, model_node.node_id)
            
            # Add attributes
            mst_node.add_attribute('op', node_op)
            if fx_node.target:
                mst_node.add_attribute('target', str(fx_node.target))
            
            # Set call stack
            call_stack = [model_name, node_name]
            if node_op == 'call_module' and fx_node.target:
                # Add module path to call stack
                module_path = str(fx_node.target).split('.')
                call_stack = [model_name] + module_path
            mst_node.set_call_stack(call_stack)
            
            # Store mapping
            fx_node_to_mst[node_name] = mst_node
        
        return mst
    
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
            if len(label) > 20:
                label = label[:17] + '...'
            
            # Add tooltip with more info
            tooltip = f"{node.name}\\nType: {node.node_type}"
            if node.trace_events:
                tooltip += f"\\nEvents: {len(node.trace_events)}"
            
            # Empty label inside circle, actual label below (labelloc='b')
            dot.node(node_id, label=label, fillcolor=color, tooltip=tooltip,
                    fontcolor='#333333')
            
            # Add edges to children
            for child in node.children:
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
