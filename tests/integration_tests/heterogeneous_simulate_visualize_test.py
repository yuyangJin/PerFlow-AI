'''
test Gernerate heterogeneous graph & Visualize the simulated trace
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

def test_Heterogeneous_GPipe_Simulate_Visualize():
    stage_events_str = ['[<Event 1:F:0-0>, <Event 2:F:1-0>, <Event 3:F:2-0>, <Event 4:F:3-0>, <Event 5:F:4-0>, <Event 6:F:5-0>, <Event 7:F:6-0>, <Event 8:F:7-0>, <Event 9:F:8-0>, <Event 10:F:9-0>, <Event 41:B:0-0>, <Event 42:B:1-0>, <Event 43:B:2-0>, <Event 44:B:3-0>, <Event 45:B:4-0>, <Event 46:B:5-0>, <Event 47:B:6-0>, <Event 48:B:7-0>, <Event 49:B:8-0>, <Event 50:B:9-0>]',
                    '[<Event 11:F:0-0>, <Event 12:F:1-0>, <Event 13:F:2-0>, <Event 14:F:3-0>, <Event 15:F:4-0>, <Event 16:F:5-0>, <Event 17:F:6-0>, <Event 18:F:7-0>, <Event 19:F:8-0>, <Event 20:F:9-0>, <Event 51:B:0-0>, <Event 52:B:1-0>, <Event 53:B:2-0>, <Event 54:B:3-0>, <Event 55:B:4-0>, <Event 56:B:5-0>, <Event 57:B:6-0>, <Event 58:B:7-0>, <Event 59:B:8-0>, <Event 60:B:9-0>]',
                    '[<Event 21:F:0-0>, <Event 22:F:1-0>, <Event 23:F:2-0>, <Event 24:F:3-0>, <Event 25:F:4-0>, <Event 26:F:5-0>, <Event 27:F:6-0>, <Event 28:F:7-0>, <Event 29:F:8-0>, <Event 30:F:9-0>, <Event 61:B:0-0>, <Event 62:B:1-0>, <Event 63:B:2-0>, <Event 64:B:3-0>, <Event 65:B:4-0>, <Event 66:B:5-0>, <Event 67:B:6-0>, <Event 68:B:7-0>, <Event 69:B:8-0>, <Event 70:B:9-0>]',
                    '[<Event 31:F:0-0>, <Event 32:F:1-0>, <Event 33:F:2-0>, <Event 34:F:3-0>, <Event 35:F:4-0>, <Event 36:F:5-0>, <Event 37:F:6-0>, <Event 38:F:7-0>, <Event 39:F:8-0>, <Event 40:F:9-0>, <Event 71:B:0-0>, <Event 72:B:1-0>, <Event 73:B:2-0>, <Event 74:B:3-0>, <Event 75:B:4-0>, <Event 76:B:5-0>, <Event 77:B:6-0>, <Event 78:B:7-0>, <Event 79:B:8-0>, <Event 80:B:9-0>]']
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

    assert trace.get_nstages() == 4
    assert trace.get_nmicrobatches() == 10

    for i in range(trace.get_nstages()):
        assert str(trace.get_events(i)) == stage_events_str[i]

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()

def test_Heterogeneous_PipeDream_Simulate_Visualize():
    stage_events_str = ['[<Event 1:F:0-0>, <Event 2:F:1-0>, <Event 3:F:2-0>, <Event 4:F:3-0>, <Event 41:B:0-0>, <Event 5:F:4-0>, <Event 42:B:1-0>, <Event 6:F:5-0>, <Event 43:B:2-0>, <Event 7:F:6-0>, <Event 44:B:3-0>, <Event 8:F:7-0>, <Event 45:B:4-0>, <Event 9:F:8-0>, <Event 46:B:5-0>, <Event 10:F:9-0>, <Event 47:B:6-0>, <Event 48:B:7-0>, <Event 49:B:8-0>, <Event 50:B:9-0>]',
                    '[<Event 11:F:0-0>, <Event 12:F:1-0>, <Event 13:F:2-0>, <Event 51:B:0-0>, <Event 14:F:3-0>, <Event 52:B:1-0>, <Event 15:F:4-0>, <Event 53:B:2-0>, <Event 16:F:5-0>, <Event 54:B:3-0>, <Event 17:F:6-0>, <Event 55:B:4-0>, <Event 18:F:7-0>, <Event 56:B:5-0>, <Event 19:F:8-0>, <Event 57:B:6-0>, <Event 20:F:9-0>, <Event 58:B:7-0>, <Event 59:B:8-0>, <Event 60:B:9-0>]',
                    '[<Event 21:F:0-0>, <Event 22:F:1-0>, <Event 61:B:0-0>, <Event 23:F:2-0>, <Event 62:B:1-0>, <Event 24:F:3-0>, <Event 63:B:2-0>, <Event 25:F:4-0>, <Event 64:B:3-0>, <Event 26:F:5-0>, <Event 65:B:4-0>, <Event 27:F:6-0>, <Event 66:B:5-0>, <Event 28:F:7-0>, <Event 67:B:6-0>, <Event 29:F:8-0>, <Event 68:B:7-0>, <Event 30:F:9-0>, <Event 69:B:8-0>, <Event 70:B:9-0>]',
                    '[<Event 31:F:0-0>, <Event 71:B:0-0>, <Event 32:F:1-0>, <Event 72:B:1-0>, <Event 33:F:2-0>, <Event 73:B:2-0>, <Event 34:F:3-0>, <Event 74:B:3-0>, <Event 35:F:4-0>, <Event 75:B:4-0>, <Event 36:F:5-0>, <Event 76:B:5-0>, <Event 37:F:6-0>, <Event 77:B:6-0>, <Event 38:F:7-0>, <Event 78:B:7-0>, <Event 39:F:8-0>, <Event 79:B:8-0>, <Event 40:F:9-0>, <Event 80:B:9-0>]']
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

    assert trace.get_nstages() == 4
    assert trace.get_nmicrobatches() == 10

    for i in range(trace.get_nstages()):
        assert str(trace.get_events(i)) == stage_events_str[i]

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()

def test_Heterogeneous_Interleaved1F1B_Simulate_Visualize():
    stage_events_str = ['[<Event 1:F:0-0>, <Event 3:F:1-0>, <Event 5:F:2-0>, <Event 7:F:3-0>, <Event 2:F:0-1>, <Event 4:F:1-1>, <Event 6:F:2-1>, <Event 8:F:3-1>, <Event 9:F:4-0>, <Event 11:F:5-0>, <Event 13:F:6-0>, <Event 98:B:0-1>, <Event 15:F:7-0>, <Event 100:B:1-1>, <Event 10:F:4-1>, <Event 102:B:2-1>, <Event 12:F:5-1>, <Event 104:B:3-1>, <Event 14:F:6-1>, <Event 97:B:0-0>, <Event 16:F:7-1>, <Event 99:B:1-0>, <Event 17:F:8-0>, <Event 101:B:2-0>, <Event 19:F:9-0>, <Event 103:B:3-0>, <Event 21:F:10-0>, <Event 106:B:4-1>, <Event 23:F:11-0>, <Event 108:B:5-1>, <Event 18:F:8-1>, <Event 110:B:6-1>, <Event 20:F:9-1>, <Event 112:B:7-1>, <Event 22:F:10-1>, <Event 105:B:4-0>, <Event 24:F:11-1>, <Event 107:B:5-0>, <Event 109:B:6-0>, <Event 111:B:7-0>, <Event 114:B:8-1>, <Event 116:B:9-1>, <Event 118:B:10-1>, <Event 120:B:11-1>, <Event 113:B:8-0>, <Event 115:B:9-0>, <Event 117:B:10-0>, <Event 119:B:11-0>]',
                    '[<Event 25:F:0-0>, <Event 27:F:1-0>, <Event 29:F:2-0>, <Event 31:F:3-0>, <Event 26:F:0-1>, <Event 28:F:1-1>, <Event 30:F:2-1>, <Event 32:F:3-1>, <Event 33:F:4-0>, <Event 122:B:0-1>, <Event 35:F:5-0>, <Event 124:B:1-1>, <Event 37:F:6-0>, <Event 126:B:2-1>, <Event 39:F:7-0>, <Event 128:B:3-1>, <Event 34:F:4-1>, <Event 121:B:0-0>, <Event 36:F:5-1>, <Event 123:B:1-0>, <Event 38:F:6-1>, <Event 125:B:2-0>, <Event 40:F:7-1>, <Event 127:B:3-0>, <Event 41:F:8-0>, <Event 130:B:4-1>, <Event 43:F:9-0>, <Event 132:B:5-1>, <Event 45:F:10-0>, <Event 134:B:6-1>, <Event 47:F:11-0>, <Event 136:B:7-1>, <Event 42:F:8-1>, <Event 129:B:4-0>, <Event 44:F:9-1>, <Event 131:B:5-0>, <Event 46:F:10-1>, <Event 133:B:6-0>, <Event 48:F:11-1>, <Event 135:B:7-0>, <Event 138:B:8-1>, <Event 140:B:9-1>, <Event 142:B:10-1>, <Event 144:B:11-1>, <Event 137:B:8-0>, <Event 139:B:9-0>, <Event 141:B:10-0>, <Event 143:B:11-0>]',
                    '[<Event 49:F:0-0>, <Event 51:F:1-0>, <Event 53:F:2-0>, <Event 55:F:3-0>, <Event 50:F:0-1>, <Event 52:F:1-1>, <Event 54:F:2-1>, <Event 146:B:0-1>, <Event 56:F:3-1>, <Event 148:B:1-1>, <Event 57:F:4-0>, <Event 150:B:2-1>, <Event 59:F:5-0>, <Event 152:B:3-1>, <Event 61:F:6-0>, <Event 145:B:0-0>, <Event 63:F:7-0>, <Event 147:B:1-0>, <Event 58:F:4-1>, <Event 149:B:2-0>, <Event 60:F:5-1>, <Event 151:B:3-0>, <Event 62:F:6-1>, <Event 154:B:4-1>, <Event 64:F:7-1>, <Event 156:B:5-1>, <Event 65:F:8-0>, <Event 158:B:6-1>, <Event 67:F:9-0>, <Event 160:B:7-1>, <Event 69:F:10-0>, <Event 153:B:4-0>, <Event 71:F:11-0>, <Event 155:B:5-0>, <Event 66:F:8-1>, <Event 157:B:6-0>, <Event 68:F:9-1>, <Event 159:B:7-0>, <Event 70:F:10-1>, <Event 162:B:8-1>, <Event 72:F:11-1>, <Event 164:B:9-1>, <Event 166:B:10-1>, <Event 168:B:11-1>, <Event 161:B:8-0>, <Event 163:B:9-0>, <Event 165:B:10-0>, <Event 167:B:11-0>]',
                    '[<Event 73:F:0-0>, <Event 75:F:1-0>, <Event 77:F:2-0>, <Event 79:F:3-0>, <Event 74:F:0-1>, <Event 170:B:0-1>, <Event 76:F:1-1>, <Event 172:B:1-1>, <Event 78:F:2-1>, <Event 174:B:2-1>, <Event 80:F:3-1>, <Event 176:B:3-1>, <Event 81:F:4-0>, <Event 169:B:0-0>, <Event 83:F:5-0>, <Event 171:B:1-0>, <Event 85:F:6-0>, <Event 173:B:2-0>, <Event 87:F:7-0>, <Event 175:B:3-0>, <Event 82:F:4-1>, <Event 178:B:4-1>, <Event 84:F:5-1>, <Event 180:B:5-1>, <Event 86:F:6-1>, <Event 182:B:6-1>, <Event 88:F:7-1>, <Event 184:B:7-1>, <Event 89:F:8-0>, <Event 177:B:4-0>, <Event 91:F:9-0>, <Event 179:B:5-0>, <Event 93:F:10-0>, <Event 181:B:6-0>, <Event 95:F:11-0>, <Event 183:B:7-0>, <Event 90:F:8-1>, <Event 186:B:8-1>, <Event 92:F:9-1>, <Event 188:B:9-1>, <Event 94:F:10-1>, <Event 190:B:10-1>, <Event 96:F:11-1>, <Event 192:B:11-1>, <Event 185:B:8-0>, <Event 187:B:9-0>, <Event 189:B:10-0>, <Event 191:B:11-0>]']
    config0 = PipeCostConfig(fwd_time = 1000, bwd_time = 2000, wgt_time = 3000)
    config1 = PipeCostConfig(fwd_time = 2000, bwd_time = 1000, wgt_time = 3000)
    nstages = 4
    nmicrobatches = 12
    nchunks = 2

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

    assert trace.get_nstages() == 4
    assert trace.get_nmicrobatches() == 12
    assert trace.get_nchunks() == 2

    for i in range(trace.get_nstages()):
        assert str(trace.get_events(i)) == stage_events_str[i]

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()

def test_Heterogeneous_ZeroBubble_Simulate_Visualize():
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

    assert trace.get_nstages() == 4
    assert trace.get_nmicrobatches() == 10
    assert trace.get_nchunks() == 2

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()

def test_Heterogeneous_ZBV_Simulate_Visualize():
    config0 = PipeCostConfig(fwd_time = 1000, bwd_time = 2000, wgt_time = 3000)
    config1 = PipeCostConfig(fwd_time = 3000, bwd_time = 2000, wgt_time = 1000)
    nstages = 4
    nmicrobatches = 10
    nchunks = 2

    g0 = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, cost_config = config0, schedule_type = ScheduleType.ZBV)
    g1 = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, cost_config = config1, schedule_type = ScheduleType.ZBV)
    g0.build_graph()
    g1.build_graph()

    fil = Filter()
    subg0 = fil.filter(g0, [0, 1])
    subg1 = fil.filter(g1, [2, 3])

    mer = Merge()
    res_edge_list = []

    for mb in range(nmicrobatches):
        for chk in range(nchunks):
            if (chk & 1):
                src1_id = g0.get_event_id(EventType.FWD, 2, mb, chk)
                dst1_id = g1.get_event_id(EventType.FWD, 1, mb, chk)
                res_edge_list.append([src1_id, dst1_id])

                src2_id = g0.get_event_id(EventType.BWD, 1, mb, chk)
                dst2_id = g1.get_event_id(EventType.BWD, 2, mb, chk)
                res_edge_list.append([src2_id, dst2_id])
            else:
                src1_id = g0.get_event_id(EventType.FWD, 1, mb, chk)
                dst1_id = g1.get_event_id(EventType.FWD, 2, mb, chk)
                res_edge_list.append([src1_id, dst1_id])

                src2_id = g0.get_event_id(EventType.BWD, 2, mb, chk)
                dst2_id = g1.get_event_id(EventType.BWD, 1, mb, chk)
                res_edge_list.append([src2_id, dst2_id])
    

    g = mer.merge([subg0,subg1], res_edge_list)

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    assert trace.get_nstages() == 4
    assert trace.get_nmicrobatches() == 10
    assert trace.get_nchunks() == 2

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()

