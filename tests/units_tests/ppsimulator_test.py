'''
test PPSimulator class
'''

from perflowai.simulator import PPSimulator, PipeType
from perflowai.parallel.pipeline_parallel import PipeCostConfig, GPipeGraph

def test_PPSimulator():

    g = GPipeGraph(4, 10, 1, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
    ))
    g.build_graph()
    
    sim = PPSimulator(PipeType.GPipe, g)