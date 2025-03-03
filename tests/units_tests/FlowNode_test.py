'''
test FlowNode class
'''

from perflowai.workflow import FlowNode

def test_FlowNode():
    node = FlowNode("test", 0, [], [])
    assert node.m_name == "test"
    assert node.m_id == 0
    assert node.m_inputs == []
    assert node.m_outputs == []
    assert str(node) == "FlowNode(test)"