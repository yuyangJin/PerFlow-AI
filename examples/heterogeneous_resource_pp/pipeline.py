'''
Example: Gernerate heterogeneous graph & Visualize the simulated trace
'''

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.gpipe import GPipeGraph
from perflowai.parallel.pipeline_parallel.pipedream import PipeDreamGraph
from perflowai.parallel.pipeline_parallel.interleaved1f1b import Interleaved1F1BGraph
from perflowai.parallel.pipeline_parallel.zerobubble import ZeroBubbleGraph, ScheduleType
from perflowai.simulator.pp_simulator import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisiualizer
from perflowai.trace_op.filter import Filter
from perflowai.trace_op.merge import Merge
from perflowai.core.event import EventType

def Heterogeneous_GPipe_Simulate_Visualize():

    config0 = PipeCostConfig(fwd_time = 1000, bwd_time = 2000, wgt_time = 3000)
    config1 = PipeCostConfig(fwd_time = 2000, bwd_time = 1000, wgt_time = 3000)
    nstages = 4
    nmicrobatches = 10
    nchunks = 1

    g0 = GPipeGraph(nstages, nmicrobatches, nchunks, cost_config = config0)
    g1 = GPipeGraph(nstages, nmicrobatches, nchunks, cost_config = config1)
    g0.build_graph()
    g1.build_graph()

    fil = Filter()
    subg0 = fil.filter(g0, [0, 1])
    subg1 = fil.filter(g1, [2, 3])

    mer = Merge()
    res_edge_list = []

    for mb in range(nmicrobatches):
        for chk in range(nchunks):
            src1_id = g0.get_event_id(EventType.FWD, 1, mb, chk)
            dst1_id = g1.get_event_id(EventType.FWD, 2, mb, chk)
            res_edge_list.append([src1_id, dst1_id])

            src2_id = g0.get_event_id(EventType.BWD, 2, mb, chk)
            dst2_id = g1.get_event_id(EventType.BWD, 1, mb, chk)
            res_edge_list.append([src2_id, dst2_id])
    

    g = mer.merge([subg0,subg1], res_edge_list)

    sim = PPSimulator(PipeType.GPipe, g)
    trace = sim.run()

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()

def Heterogeneous_PipeDream_Simulate_Visualize():
    config0 = PipeCostConfig(fwd_time = 1000, bwd_time = 2000, wgt_time = 3000)
    config1 = PipeCostConfig(fwd_time = 2000, bwd_time = 1000, wgt_time = 3000)
    nstages = 4
    nmicrobatches = 10
    nchunks = 1

    g0 = PipeDreamGraph(nstages, nmicrobatches, nchunks, cost_config = config0)
    g1 = PipeDreamGraph(nstages, nmicrobatches, nchunks, cost_config = config1)
    g0.build_graph()
    g1.build_graph()

    fil = Filter()
    subg0 = fil.filter(g0, [0, 1])
    subg1 = fil.filter(g1, [2, 3])

    mer = Merge()
    res_edge_list = []

    for mb in range(nmicrobatches):
        for chk in range(nchunks):
            src1_id = g0.get_event_id(EventType.FWD, 1, mb, chk)
            dst1_id = g1.get_event_id(EventType.FWD, 2, mb, chk)
            res_edge_list.append([src1_id, dst1_id])

            src2_id = g0.get_event_id(EventType.BWD, 2, mb, chk)
            dst2_id = g1.get_event_id(EventType.BWD, 1, mb, chk)
            res_edge_list.append([src2_id, dst2_id])
    

    g = mer.merge([subg0,subg1], res_edge_list)

    sim = PPSimulator(PipeType.PipeDream, g)
    trace = sim.run()

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()

def Heterogeneous_Interleaved1F1B_Simulate_Visualize():
    config0 = PipeCostConfig(fwd_time = 1000, bwd_time = 2000, wgt_time = 3000)
    config1 = PipeCostConfig(fwd_time = 2000, bwd_time = 4000, wgt_time = 3000)
    nstages = 4
    nmicrobatches = 10
    nchunks = 4

    g0 = Interleaved1F1BGraph(nstages, nmicrobatches, nchunks, cost_config = config0)
    g1 = Interleaved1F1BGraph(nstages, nmicrobatches, nchunks, cost_config = config1)
    g0.build_graph()
    g1.build_graph()

    fil = Filter()
    subg0 = fil.filter(g0, [0, 1])
    subg1 = fil.filter(g1, [2, 3])

    mer = Merge()
    res_edge_list = []

    for mb in range(nmicrobatches):
        for chk in range(nchunks):
            src1_id = g0.get_event_id(EventType.FWD, 1, mb, chk)
            dst1_id = g1.get_event_id(EventType.FWD, 2, mb, chk)
            res_edge_list.append([src1_id, dst1_id])

            src2_id = g0.get_event_id(EventType.BWD, 2, mb, chk)
            dst2_id = g1.get_event_id(EventType.BWD, 1, mb, chk)
            res_edge_list.append([src2_id, dst2_id])
    

    g = mer.merge([subg0,subg1], res_edge_list)

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()

def Heterogeneous_ZeroBubble_Simulate_Visualize():
    config0 = PipeCostConfig(fwd_time = 1000, bwd_time = 2000, wgt_time = 3000)
    config1 = PipeCostConfig(fwd_time = 3000, bwd_time = 2000, wgt_time = 1000)
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g0 = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, cost_config = config0)
    g1 = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, cost_config = config1)
    g0.build_graph()
    g1.build_graph()

    fil = Filter()
    subg0 = fil.filter(g0, [0, 1])
    subg1 = fil.filter(g1, [2, 3])

    mer = Merge()
    res_edge_list = []

    for mb in range(nmicrobatches):
        for chk in range(nchunks):
            src1_id = g0.get_event_id(EventType.FWD, 1, mb, chk)
            dst1_id = g1.get_event_id(EventType.FWD, 2, mb, chk)
            res_edge_list.append([src1_id, dst1_id])

            src2_id = g0.get_event_id(EventType.BWD, 2, mb, chk)
            dst2_id = g1.get_event_id(EventType.BWD, 1, mb, chk)
            res_edge_list.append([src2_id, dst2_id])

            if chk != 0:
                src3_id = g1.get_event_id(EventType.FWD, 3, mb, chk - 1)
                dest3_id = g0.get_event_id(EventType.FWD, 0, mb, chk)
                res_edge_list.append([src3_id, dest3_id])

                src4_id = g0.get_event_id(EventType.BWD, 0, mb, chk)
                dest4_id = g1.get_event_id(EventType.BWD, 3, mb, chk - 1)
                res_edge_list.append([src4_id, dest4_id])
    

    g = mer.merge([subg0,subg1], res_edge_list)

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()


def Heterogeneous_ZBV_Simulate_Visualize():
    
    # Setup pipeline configurations on A100 and H100.
    config_h100 = PipeCostConfig(fwd_time = 1315, bwd_time = 2733, wgt_time = 2650)
    config_a100 = PipeCostConfig(fwd_time = 2743, bwd_time = 4563, wgt_time = 4324)
    
    nstages = 8
    nmicrobatches = 10
    nchunks = 2

    # Build tasks and dependence between tasks for pipelines on A100 and H100, respectively.
    g_h100 = ZeroBubbleGraph(nstages, 
                            nmicrobatches, 
                            nchunks, 
                            cost_config = config_h100, 
                            schedule_type = ScheduleType.ZBV)
    g_a100 = ZeroBubbleGraph(nstages, 
                            nmicrobatches, 
                            nchunks, 
                            cost_config = config_a100, 
                            schedule_type = ScheduleType.ZBV)
    g_h100.build_graph()
    g_a100.build_graph()

    # Get part stages from the A100 pipeline and H100 pipeline
    subg_h100 = Filter().filter(g_h100, [0, 1, 2, 3])
    subg_a100 = Filter().filter(g_a100, [4, 5, 6, 7])


    # Merge the two part stages and add inter-part data dependence (data dependence could be autoimatically generated, we are refining this part.)
    res_edge_list = []

    for mb in range(nmicrobatches):
        for chk in range(nchunks):
            if (chk & 1):
                src1_id = g_h100.get_event_id(EventType.FWD, 4, mb, chk)
                dst1_id = g_a100.get_event_id(EventType.FWD, 3, mb, chk)
                res_edge_list.append([src1_id, dst1_id])

                src2_id = g_h100.get_event_id(EventType.BWD, 3, mb, chk)
                dst2_id = g_a100.get_event_id(EventType.BWD, 4, mb, chk)
                res_edge_list.append([src2_id, dst2_id])
            else:
                src1_id = g_h100.get_event_id(EventType.FWD, 3, mb, chk)
                dst1_id = g_a100.get_event_id(EventType.FWD, 4, mb, chk)
                res_edge_list.append([src1_id, dst1_id])

                src2_id = g_h100.get_event_id(EventType.BWD, 4, mb, chk)
                dst2_id = g_a100.get_event_id(EventType.BWD, 3, mb, chk)
                res_edge_list.append([src2_id, dst2_id])
    

    g = Merge().merge([subg_h100, subg_a100], res_edge_list)

    # Simluation
    trace = PPSimulator(PipeType.Interleaved1F1B, g).run()

    # Visualization
    TraceVisiualizer(trace).visualize()


Heterogeneous_GPipe_Simulate_Visualize()
Heterogeneous_PipeDream_Simulate_Visualize()
Heterogeneous_ZeroBubble_Simulate_Visualize()
Heterogeneous_ZBV_Simulate_Visualize()