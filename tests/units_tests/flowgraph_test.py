'''
test FlowGraph class
''' 

from perflowai.workflow import FlowGraph, FlowNode

def test_FlowGraph():
    graph = FlowGraph()
    graph.add_node(FlowNode("test0", 0, [], []))
    graph.add_node(FlowNode("test1", 1, [], []))
    graph.add_edge(graph.get_node_by_id(0), graph.get_node_by_id(1))
    print(graph.m_edges)
    assert graph.get_node_by_id(0).m_name == "test0"
    assert graph.get_node_by_id(1).m_name == "test1"
    assert graph.get_next_nodelist_by_id(0) == [1]
    assert str(graph.get_node_by_id(0)) == "FlowNode(test0)"
    assert str(graph.get_node_by_id(1)) == "FlowNode(test1)"
    assert graph.m_nodes[0].m_name == "test0"
    assert graph.m_nodes[1].m_name == "test1"
    assert graph.m_edges[0] == [1]
    assert graph.m_nodes[0].m_id == 0
    assert graph.m_nodes[1].m_id == 1
    assert graph.m_nodes[0].m_inputs == []
    assert graph.m_nodes[0].m_outputs == []
    assert graph.m_nodes[1].m_inputs == []
    assert graph.m_nodes[1].m_outputs == []