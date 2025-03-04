'''
test Generate pipeline & Visualize the memory footprint of the simulated trace
'''

from perflowai.parallel.pipeline_parallel import PipeCostConfig, GPipeGraph, PipeDreamGraph, Interleaved1F1BGraph, ZeroBubbleGraph, ScheduleType
from perflowai.simulator import PPSimulator, PipeType
from perflowai.visualizer import MemoryFootprintVisualizer
from perflowai.core import OffReLoadEvent, EventType, LoadType

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
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g = Interleaved1F1BGraph(nstages, nmicrobatches, nchunks, cost_config = config)
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    load_ratio = 0
    for stage in range(trace.get_nstages()):
        load_events = []
        events = trace.get_events(stage)
        for event in events:  
            stage_id = event.get_stage_id()
            microbatch_id = event.get_microbatch_id()
            chunk_id = event.get_chunk_id()
            event_type = event.get_type()
            event_time = event.get_timestamp()
            event_duration = event.get_duration()
            end_time = event_time + event_duration

            if event_type == EventType.FWD:
                if (not stage_id == nstages - 1) or (not chunk_id == nchunks - 1):
                    load_events.append(OffReLoadEvent(id=0, type=EventType.OFFL, name="", timestamp=end_time, duration=1314, 
                                                    load_ratio=load_ratio, load_type=LoadType.GPU2CPU_PCIE,
                                                    stage_id=stage_id, microbatch_id=microbatch_id, chunk_id=chunk_id,
                                                    mem = -27.0 * load_ratio))
            else:
                if (not stage_id == nstages - 1) or (not chunk_id == 0):
                    load_events.append(OffReLoadEvent(id=0, type=EventType.REL, name="", timestamp=event_time-2439, duration=2439, 
                                                    load_ratio=load_ratio, load_type=LoadType.CPU2GPU_PCIE,
                                                    stage_id=stage_id, microbatch_id=microbatch_id, chunk_id=chunk_id,
                                                    mem = 27.0 * load_ratio))
        for event in load_events:
            trace.add_event(stage, event)


    mfp = trace.get_memory_foorprint()

    MemoryFootprintVisualizer(mfp).visualize()
