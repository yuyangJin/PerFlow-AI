'''
test Visualize the simulated trace of PipeDream
'''

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.pipedream import PipeDreamGraph
from perflowai.simulator.pp_simulator import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisiualizer

def test_PipeDream_Simulate_Visualize():
    g = PipeDreamGraph(4, 10, 1, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
        ),
        eager = True
    )
    g.build_graph()

    sim = PPSimulator(PipeType.PipeDream, g)
    trace = sim.run()

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()



# test_PipeDream_Simulate_Visualize()