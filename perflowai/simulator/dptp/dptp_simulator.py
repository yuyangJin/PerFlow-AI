'''
@module data/tensor parallel simulator
'''

from ..simulator import Simulator

'''
@class DPSimulator
A data parallel simulator.
'''
class DPSimulator(Simulator):
    def __init__(self):
        super().__init__('DPSimulator', 0)

    def simulate(self, *args, **kwargs):
        '''
        Simulate data parallelism.
        To be implemented.
        '''
        pass

'''
@class TPSimulator
A tensor parallel simulator.
'''
class TPSimulator(Simulator):
    def __init__(self):
        super().__init__('TPSimulator', 0)

    def simulate(self, *args, **kwargs):
        '''
        Simulate tensor parallelism.
        To be implemented.
        '''
        pass

'''
@class TPDPSimulator
A tensor pipeline data parallel parallel simulator.
'''
class TPDPSimulator(Simulator):
    def __init__(self):
        super().__init__('TPDPSimulator', 0)

    def simulate(self, *args, **kwargs):
        '''
        Simulate tensor, pipeline, and data parallelism.
        To be implemented.
        '''
        pass