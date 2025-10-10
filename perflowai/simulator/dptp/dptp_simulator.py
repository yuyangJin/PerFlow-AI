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
        super().__init__()

    def simulate(self):
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
        super().__init__()

    def simulate(self):
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
        super().__init__()

    def simulate(self):
        '''
        Simulate tensor, pipeline, and data parallelism.
        To be implemented.
        '''
        pass