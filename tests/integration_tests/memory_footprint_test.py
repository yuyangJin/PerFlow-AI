'''
test Generate pipeline & Visualize the memory footprint of the simulated trace
'''

from perflowai.parallel.pipeline_parallel import PipeCostConfig, PipeOffloadConfig, GPipeGraph, PipeDreamGraph, Interleaved1F1BGraph, ZeroBubbleGraph, ScheduleType
from perflowai.simulator import PPSimulator, PipeType
from perflowai.visualizer import MemoryFootprintVisualizer
from perflowai.core import OffReLoadEvent, EventType, ResourceType

def test_GPipe_Memory_Footprint_Visualize():

    config = PipeCostConfig(fwd_time = 1314, 
                            bwd_time = 2439,
                            fwd_mem = 27.0,
                            bwd_mem = -27.0 
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 1

    g = GPipeGraph(nstages, nmicrobatches, nchunks, cost_config = config)
    g.build_graph()

    sim = PPSimulator(PipeType.GPipe, g)
    trace = sim.run()

    mfp = trace.get_memory_foorprint()

    MemoryFootprintVisualizer(mfp).visualize()

def test_PipeDream_Memory_Footprint_Visualize():

    config = PipeCostConfig(fwd_time = 1314, 
                            bwd_time = 2439,
                            fwd_mem = 27.0,
                            bwd_mem = -27.0 
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 1

    g = PipeDreamGraph(nstages, nmicrobatches, nchunks, cost_config = config)
    g.build_graph()

    sim = PPSimulator(PipeType.PipeDream, g)
    trace = sim.run()

    mfp = trace.get_memory_foorprint()

    MemoryFootprintVisualizer(mfp).visualize()

def test_Interleaved1F1B_Memory_Footprint_Visualize():

    config = PipeCostConfig(fwd_time = 1314, 
                            bwd_time = 2439,
                            fwd_mem = 27.0,
                            bwd_mem = -27.0 
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g = Interleaved1F1BGraph(nstages, nmicrobatches, nchunks, cost_config = config)
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    mfp = trace.get_memory_foorprint()

    MemoryFootprintVisualizer(mfp).visualize()


def test_ZeroBubble_Memory_Footprint_Visualize():

    config = PipeCostConfig(fwd_time = 1314, 
                            bwd_time = 1439,
                            wgt_time = 1533,
                            fwd_mem = 27.0,
                            bwd_mem = -13.5,
                            wgt_mem = -13.5 
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, cost_config = config, schedule_type = ScheduleType.ZB)
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    mfp = trace.get_memory_foorprint()

    MemoryFootprintVisualizer(mfp).visualize()


def test_ZBV_Memory_Footprint_Visualize():

    config = PipeCostConfig(fwd_time = 1314, 
                            bwd_time = 1439,
                            wgt_time = 1533,
                            fwd_mem = 27.0,
                            bwd_mem = -13.5,
                            wgt_mem = -13.5 
                            )
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, cost_config = config, schedule_type = ScheduleType.ZBV)
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    mfp = trace.get_memory_foorprint()

    MemoryFootprintVisualizer(mfp).visualize()


def test_offloading_Memory_Footprint_Visualize():

    config = PipeCostConfig(fwd_time = 1314, 
                            bwd_time = 2439,
                            fwd_mem = 27.0,
                            bwd_mem = -27.0 
                            )
    poc = PipeOffloadConfig(offload_ratio = 0.5)
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g = Interleaved1F1BGraph(nstages, nmicrobatches, nchunks, cost_config = config, offload_config = poc)
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    mfp = trace.get_memory_foorprint()

    MemoryFootprintVisualizer(mfp).visualize()
