'''
test Generate imbalanced pipeline & Visualize the simulated trace
'''

from perflowai.parallel.pipeline_parallel import PipeCostConfig, GPipeGraph, PipeDreamGraph, Interleaved1F1BGraph, ZeroBubbleGraph, ScheduleType
from perflowai.simulator.pp_simulator import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisiualizer

def test_Imbalanced_GPipe_Simulate_Visualize():

    config = PipeCostConfig(fwd_time = [1314, 1518, 1419, 1217], 
                            bwd_time = [2439, 2218, 2983, 2427], 
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 1

    g = GPipeGraph(nstages, nmicrobatches, nchunks, cost_config = config)
    g.build_graph()

    sim = PPSimulator(PipeType.GPipe, g)
    trace = sim.run()

    TraceVisiualizer(trace).visualize()

def test_Imbalanced_PipeDream_Simulate_Visualize():

    config = PipeCostConfig(fwd_time = [1314, 1518, 1419, 1217], 
                            bwd_time = [2439, 2218, 2983, 2427], 
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 1

    g = PipeDreamGraph(nstages, nmicrobatches, nchunks, cost_config = config)
    g.build_graph()

    sim = PPSimulator(PipeType.PipeDream, g)
    trace = sim.run()

    TraceVisiualizer(trace).visualize()

def test_Imbalanced_Interleaved1F1B_Simulate_Visualize():

    config = PipeCostConfig(fwd_time = [1314, 1518, 1419, 1217], 
                            bwd_time = [2439, 2218, 2983, 2427], 
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g = Interleaved1F1BGraph(nstages, nmicrobatches, nchunks, cost_config = config)
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    TraceVisiualizer(trace).visualize()

def test_Imbalanced_ZeroBubble_Simulate_Visualize():

    config = PipeCostConfig(fwd_time = [1314, 1518, 1419, 1217], 
                            bwd_time = [2439, 2218, 2983, 2427], 
                            wgt_time = [2439, 2218, 2983, 2427]
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, cost_config = config)
    g.build_graph()

    sim = PPSimulator(PipeType.ZeroBubble, g)
    trace = sim.run()

    TraceVisiualizer(trace).visualize()

def test_Imbalanced_ZBV_Simulate_Visualize():

    config = PipeCostConfig(fwd_time = [1314, 1518, 1419, 1217], 
                            bwd_time = [2439, 2218, 2983, 2427], 
                            wgt_time = [2439, 2218, 2983, 2427]
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, cost_config = config, schedule_type = ScheduleType.ZBV)
    g.build_graph()

    sim = PPSimulator(PipeType.ZeroBubble, g)
    trace = sim.run()

    TraceVisiualizer(trace).visualize()