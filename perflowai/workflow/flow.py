'''
@module workflow
'''

'''
@class FlowNode
A FlowNode is a node in a flow graph.
'''

class FlowNode:
    def __init__(self, name, id, inputs, outputs):
        self.m_name = name
        self.m_id = id
        self.m_inputs = inputs
        self.m_outputs = outputs

    def __str__(self):
        return f"FlowNode({self.m_name})"

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

'''
@class FlowGraph
A FlowGraph is a diagram of tasks.
'''

class FlowGraph:
    def __init__(self):

        '''
        Dict<int, FlowNode> m_nodes
        The key is the node id, the value is the node.
        '''
        self.m_nodes = dict()
        
        '''
        Dict<int, List<int>>> m_edges
        The first int is the source node id, 
        the second List is the destination node ids.
        '''
        self.m_edges = dict()

    def add_node(self, node):
        self.m_nodes[node.m_id] = node
    
    def add_edge(self, src_node, dst_node):
        if src_node.m_id not in self.m_edges:
            self.m_edges[src_node.m_id] = []
        self.m_edges[src_node.m_id].append(dst_node.m_id)

    ''' 
    check the graph to ensure the output of 
    one node is the input of another node 
    '''
    def check(self):
        pass

    def run(self, *args, **kwargs):
        for node in self.nodes:
            node.run(*args, **kwargs)