'''
test Visualize the simulated trace of Interleaved1F1B
'''

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.interleaved1f1b import Interleaved1F1BGraph
from perflowai.simulator.pp_simulator import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisiualizer

def test_Interleaved1F1B_Simulate_Visualize():
    g = Interleaved1F1BGraph(4, 10, 2, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
    ))
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()



# test_Interleaved1F1B_Simulate_Visualize()