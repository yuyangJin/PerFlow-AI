'''
test FlowNode class
'''

from perflowai.workflow import FlowNode, FlowData

class ConcreteFlowNode(FlowNode):
    '''A concrete implementation of FlowNode for testing'''
    def run(self):
        pass

def test_FlowNode():
    node = ConcreteFlowNode()
    assert isinstance(node.m_inputs, FlowData)
    assert isinstance(node.m_outputs, FlowData)
    assert node.m_inputs.size() == 0
    assert node.m_outputs.size() == 0
    
    # Test adding data
    node.m_inputs.add_data("test_data")
    assert node.m_inputs.size() == 1
    assert "test_data" in node.m_inputs.get_data()