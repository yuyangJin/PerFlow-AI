'''
@package perflowai.simulator.pp_simulator
'''

from enum import Enum

class PipeType(Enum):
    GPipe = 0
    PipeDream = 1
    Interleaved1F1B = 2
    ZeroBubble = 3
    Custom = 4

class PipePartitionType(Enum):
    Balanced = 0
    Custom = 1


'''
@class PipeConfig
A pipeline parallel configuration.
'''
class PipeConfig:
    def __init__(self):
        pass

'''
@class PipePartitionConfig
A pipeline partition configuration.
'''
class PipePartitionConfig:
    def __init__(self):
        pass

'''
@class PPSimulator
A pipeline simulator.
'''
class PPSimulator:
    def __init__(self):
        pass

    def simulate(self):
        pass

    def get_trace(self):
        pass

    def get_perf(self):
        pass

    def get_config(self):
        pass

    def get_partition_config(self):
        pass

    def set_config(self):
        pass

    def set_partition_config(self):
        pass

    def set_pipe_type(self):
        pass

    def set_pipe_partition_type(self):
        pass

    def get_pipe_type(self):
        pass

    def get_pipe_partition_type(self):
        pass 

    '''
    @method run
    Run the pipeline simulator.
    '''
    def run(self):
        pass