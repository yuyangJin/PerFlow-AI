'''
@module simulator
'''

from ..workflow import FlowNode

'''
@class Simulator
A simulator.
'''
class Simulator(FlowNode):
    def __init__(self):
        super().__init__()

    def simulate(self):
        '''
        Simulate the workload.
        To be implemented by subclasses.
        '''
        pass

    def get_trace(self):
        '''
        Get the trace after simulation.
        '''
        pass

    def get_perf(self):
        '''
        Get the performance metrics.
        '''
        pass

    def get_config(self):
        '''
        Get the configuration.
        '''
        pass

    '''
    @method run
    Run the simulator.
    '''
    def run(self):
        '''
        Run the simulator by calling simulate().
        '''
        result = self.simulate()
        if result is not None:
            self.m_outputs.add_data(result)
        return result