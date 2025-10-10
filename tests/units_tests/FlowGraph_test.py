'''
test FlowGraph class
''' 

from perflowai.workflow import FlowGraph, FlowNode, FlowData

class ConcreteFlowNode(FlowNode):
    '''A concrete implementation of FlowNode for testing'''
    def __init__(self, name):
        super().__init__()
        self.name = name
        
    def run(self):
        # Add the name to outputs for testing
        self.m_outputs.add_data(f"output_{self.name}")

def test_FlowGraph():
    graph = FlowGraph()
    
    # Create nodes
    node0 = ConcreteFlowNode("test0")
    node1 = ConcreteFlowNode("test1")
    node2 = ConcreteFlowNode("test2")
    
    # Add nodes to graph
    graph.add_node(node0)
    graph.add_node(node1)
    graph.add_node(node2)
    
    # Add edges: node0 -> node1 -> node2
    graph.add_edge(node0, node1)
    graph.add_edge(node1, node2)
    
    # Test get_nodes
    assert len(graph.get_nodes()) == 3
    assert node0 in graph.get_nodes()
    assert node1 in graph.get_nodes()
    assert node2 in graph.get_nodes()
    
    # Test get_successors
    assert node1 in graph.get_successors(node0)
    assert node2 in graph.get_successors(node1)
    assert len(graph.get_successors(node2)) == 0
    
    # Test run - should execute nodes in topological order
    graph.run()
    
    # Verify data flow
    assert node0.m_outputs.size() == 1
    assert "output_test0" in node0.m_outputs.get_data()
    
    # node1 should have node0's output in its inputs
    assert "output_test0" in node1.m_inputs.get_data()
    assert "output_test1" in node1.m_outputs.get_data()
    
    # node2 should have node1's output in its inputs
    assert "output_test1" in node2.m_inputs.get_data()
    assert "output_test2" in node2.m_outputs.get_data()