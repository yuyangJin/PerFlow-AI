'''
@module operator simulator
'''


from ..simulator import Simulator

from ...core import ModelConfig, DeviceConfig, EventType

'''
@class OprtSimulator
'''

class OprtSimulator(Simulator):
    '''
    @method __init__
    Initialize the operator simulator.
    '''
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.m_compute_volume = 0
        self.m_memory_volume = 0
        self.m_model_config = model_config

    def memory(self):
        return self.m_memory_volume

    def compute(self):
        return self.m_compute_volume

    def model_config(self):
        return self.m_model_config
