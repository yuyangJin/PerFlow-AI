'''
test FlowNode class
'''

from perflowai.workflow import FlowNode

class ConcreteFlowNode(FlowNode):
    '''A concrete implementation of FlowNode for testing'''
    def run(self, *args, **kwargs):
        pass

def test_FlowNode():
    node = ConcreteFlowNode("test", 0, [], [])
    assert node.m_name == "test"
    assert node.m_id == 0
    assert node.m_inputs == []
    assert node.m_outputs == []
    assert str(node) == "FlowNode(test)"