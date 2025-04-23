'''
test Visualize the simulated trace of Interleaved1F1B
'''

from perflowai.parallel.pipeline_parallel import PipeCostConfig, PipeOffloadConfig, PipeRecomputeConfig, Interleaved1F1BGraph 
from perflowai.simulator import PPSimulator, PipeType
from perflowai.visualizer import TraceVisiualizer, MemoryFootprintVisualizer

stage_events_str = ['[<Event 1:F:0-0>, <Event 3:F:1-0>, <Event 5:F:2-0>, <Event 7:F:3-0>, <Event 2:F:0-1>, <Event 4:F:1-1>, <Event 6:F:2-1>, <Event 8:F:3-1>, <Event 9:F:4-0>, <Event 11:F:5-0>, <Event 13:F:6-0>, <Event 98:B:0-1>, <Event 15:F:7-0>, <Event 100:B:1-1>, <Event 10:F:4-1>, <Event 102:B:2-1>, <Event 12:F:5-1>, <Event 104:B:3-1>, <Event 14:F:6-1>, <Event 97:B:0-0>, <Event 16:F:7-1>, <Event 99:B:1-0>, <Event 17:F:8-0>, <Event 101:B:2-0>, <Event 19:F:9-0>, <Event 103:B:3-0>, <Event 21:F:10-0>, <Event 106:B:4-1>, <Event 23:F:11-0>, <Event 108:B:5-1>, <Event 18:F:8-1>, <Event 110:B:6-1>, <Event 20:F:9-1>, <Event 112:B:7-1>, <Event 22:F:10-1>, <Event 105:B:4-0>, <Event 24:F:11-1>, <Event 107:B:5-0>, <Event 109:B:6-0>, <Event 111:B:7-0>, <Event 114:B:8-1>, <Event 116:B:9-1>, <Event 118:B:10-1>, <Event 120:B:11-1>, <Event 113:B:8-0>, <Event 115:B:9-0>, <Event 117:B:10-0>, <Event 119:B:11-0>]',
                    '[<Event 25:F:0-0>, <Event 27:F:1-0>, <Event 29:F:2-0>, <Event 31:F:3-0>, <Event 26:F:0-1>, <Event 28:F:1-1>, <Event 30:F:2-1>, <Event 32:F:3-1>, <Event 33:F:4-0>, <Event 122:B:0-1>, <Event 35:F:5-0>, <Event 124:B:1-1>, <Event 37:F:6-0>, <Event 126:B:2-1>, <Event 39:F:7-0>, <Event 128:B:3-1>, <Event 34:F:4-1>, <Event 121:B:0-0>, <Event 36:F:5-1>, <Event 123:B:1-0>, <Event 38:F:6-1>, <Event 125:B:2-0>, <Event 40:F:7-1>, <Event 127:B:3-0>, <Event 41:F:8-0>, <Event 130:B:4-1>, <Event 43:F:9-0>, <Event 132:B:5-1>, <Event 45:F:10-0>, <Event 134:B:6-1>, <Event 47:F:11-0>, <Event 136:B:7-1>, <Event 42:F:8-1>, <Event 129:B:4-0>, <Event 44:F:9-1>, <Event 131:B:5-0>, <Event 46:F:10-1>, <Event 133:B:6-0>, <Event 48:F:11-1>, <Event 135:B:7-0>, <Event 138:B:8-1>, <Event 140:B:9-1>, <Event 142:B:10-1>, <Event 144:B:11-1>, <Event 137:B:8-0>, <Event 139:B:9-0>, <Event 141:B:10-0>, <Event 143:B:11-0>]',
                    '[<Event 49:F:0-0>, <Event 51:F:1-0>, <Event 53:F:2-0>, <Event 55:F:3-0>, <Event 50:F:0-1>, <Event 52:F:1-1>, <Event 54:F:2-1>, <Event 146:B:0-1>, <Event 56:F:3-1>, <Event 148:B:1-1>, <Event 57:F:4-0>, <Event 150:B:2-1>, <Event 59:F:5-0>, <Event 152:B:3-1>, <Event 61:F:6-0>, <Event 145:B:0-0>, <Event 63:F:7-0>, <Event 147:B:1-0>, <Event 58:F:4-1>, <Event 149:B:2-0>, <Event 60:F:5-1>, <Event 151:B:3-0>, <Event 62:F:6-1>, <Event 154:B:4-1>, <Event 64:F:7-1>, <Event 156:B:5-1>, <Event 65:F:8-0>, <Event 158:B:6-1>, <Event 67:F:9-0>, <Event 160:B:7-1>, <Event 69:F:10-0>, <Event 153:B:4-0>, <Event 71:F:11-0>, <Event 155:B:5-0>, <Event 66:F:8-1>, <Event 157:B:6-0>, <Event 68:F:9-1>, <Event 159:B:7-0>, <Event 70:F:10-1>, <Event 162:B:8-1>, <Event 72:F:11-1>, <Event 164:B:9-1>, <Event 166:B:10-1>, <Event 168:B:11-1>, <Event 161:B:8-0>, <Event 163:B:9-0>, <Event 165:B:10-0>, <Event 167:B:11-0>]',
                    '[<Event 73:F:0-0>, <Event 75:F:1-0>, <Event 77:F:2-0>, <Event 79:F:3-0>, <Event 74:F:0-1>, <Event 170:B:0-1>, <Event 76:F:1-1>, <Event 172:B:1-1>, <Event 78:F:2-1>, <Event 174:B:2-1>, <Event 80:F:3-1>, <Event 176:B:3-1>, <Event 81:F:4-0>, <Event 169:B:0-0>, <Event 83:F:5-0>, <Event 171:B:1-0>, <Event 85:F:6-0>, <Event 173:B:2-0>, <Event 87:F:7-0>, <Event 175:B:3-0>, <Event 82:F:4-1>, <Event 178:B:4-1>, <Event 84:F:5-1>, <Event 180:B:5-1>, <Event 86:F:6-1>, <Event 182:B:6-1>, <Event 88:F:7-1>, <Event 184:B:7-1>, <Event 89:F:8-0>, <Event 177:B:4-0>, <Event 91:F:9-0>, <Event 179:B:5-0>, <Event 93:F:10-0>, <Event 181:B:6-0>, <Event 95:F:11-0>, <Event 183:B:7-0>, <Event 90:F:8-1>, <Event 186:B:8-1>, <Event 92:F:9-1>, <Event 188:B:9-1>, <Event 94:F:10-1>, <Event 190:B:10-1>, <Event 96:F:11-1>, <Event 192:B:11-1>, <Event 185:B:8-0>, <Event 187:B:9-0>, <Event 189:B:10-0>, <Event 191:B:11-0>]']

def test_Interleaved1F1B_Simulate_Visualize():
    g = Interleaved1F1BGraph(4, 12, 2, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
    ))
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    assert trace.get_nstages() == 4
    assert trace.get_nmicrobatches() == 12
    assert trace.get_nchunks() == 2

    for i in range(trace.get_nstages()):
        assert str(trace.get_events(i)) == stage_events_str[i]

    visualizer = TraceVisualizer(trace)
    visualizer.visualize()


def test_offload_Interleaved1F1B_Simulate_Visualize():
    g = Interleaved1F1BGraph(4, 8, 2, cost_config = PipeCostConfig(
        fwd_time = 1000,
        bwd_time = 2000,
        wgt_time = 0
    ), offload_config = PipeOffloadConfig(offload_ratio = 0.5))
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    #assert trace.get_nstages() == 4
    #assert trace.get_nmicrobatches() == 12
    #assert trace.get_nchunks() == 2

    #for i in range(trace.get_nstages()):
    #    assert str(trace.get_events(i)) == stage_events_str[i]

    visualizer = TraceVisualizer(trace)
    visualizer.visualize()

def test_recompoute_Interleaved1F1B_Simulate_Visualize():
    g = Interleaved1F1BGraph(4, 8, 2, cost_config = PipeCostConfig(
        fwd_time = 1000,
        bwd_time = 2000,
        wgt_time = 0
    ), recompute_config = PipeRecomputeConfig(recompute_mask = [0,1,0,1]))
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    mfp = trace.get_memory_foorprint()
    MemoryFootprintVisualizer(mfp).visualize()

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()




test_Interleaved1F1B_Simulate_Visualize()