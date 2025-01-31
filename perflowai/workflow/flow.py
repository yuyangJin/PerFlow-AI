class FlowNode:
    def __init__(self, name, id, inputs, outputs):
        self.m_name = name
        self.m_id = id
        self.m_inputs = inputs
        self.m_outputs = outputs

    def __str__(self):
        return f"FlowNode({self.name})"

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class FlowGraph:
    def __init__(self):

        # 
        self.m_nodes = list()
        
        # 
        self.m_edges = dict()

    ''' 
    check the graph to ensure the output of 
    one node is the input of another node 
    '''
    def check(self):
        pass

    def run(self, *args, **kwargs):
        for node in self.nodes:
            node.run(*args, **kwargs)