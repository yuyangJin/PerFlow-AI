'''
@module workflow
'''

'''
@class FlowNode
A FlowNode is a node in a flow graph.
'''

from abc import ABC, abstractmethod

class FlowNode(ABC):
    def __init__(self, name, id, inputs, outputs):
        self.m_name = name
        self.m_id = id
        self.m_inputs = inputs
        self.m_outputs = outputs

    def __str__(self):
        return f"FlowNode({self.m_name})"

    # @abstractmethod
    def run(self):
        print('FlowNode runs virtially.')
        pass

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

    def get_node_by_id(self, id):
        return self.m_nodes[id]
    
    def get_next_nodelist_by_id(self, id):
        return self.m_edges[id]

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
        '''
        Not implemented yet
        '''
        pass

    def run(self, *args, **kwargs):
        # visited = set()

        # def traverse(node_id):
        #     if node_id in visited:
        #     return
        #     visited.add(node_id)
        #     node = self.get_node_by_id(node_id)
        #     print(f"Visiting node: {node}")
        #     for next_node_id in self.get_next_nodelist_by_id(node_id):
        #     traverse(next_node_id)

        # for node_id in self.m_nodes:
        #     traverse(node_id)

        for node in self.nodes:
            node.run(*args, **kwargs)