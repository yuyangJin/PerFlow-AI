'''
test Visualize the simulated trace of zerobubble
'''

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.zerobubble import ZeroBubbleGraph
from perflowai.simulator.pp_simulator import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisiualizer

stage_events_str = ['[<Event 1:F:0-0>, <Event 3:F:1-0>, <Event 5:F:2-0>, <Event 7:F:3-0>, <Event 9:F:4-0>, <Event 2:F:0-1>, <Event 11:F:5-0>, <Event 4:F:1-1>, <Event 13:F:6-0>, <Event 6:F:2-1>, <Event 15:F:7-0>, <Event 8:F:3-1>, <Event 17:F:8-0>, <Event 10:F:4-1>, <Event 19:F:9-0>, <Event 12:F:5-1>, <Event 14:F:6-1>, <Event 82:B:0-1>, <Event 16:F:7-1>, <Event 162:W:0-1>, <Event 84:B:1-1>, <Event 18:F:8-1>, <Event 164:W:1-1>, <Event 20:F:9-1>, <Event 86:B:2-1>, <Event 166:W:2-1>, <Event 88:B:3-1>, <Event 81:B:0-0>, <Event 168:W:3-1>, <Event 90:B:4-1>, <Event 161:W:0-0>, <Event 83:B:1-0>, <Event 170:W:4-1>, <Event 92:B:5-1>, <Event 163:W:1-0>, <Event 85:B:2-0>, <Event 172:W:5-1>, <Event 94:B:6-1>, <Event 165:W:2-0>, <Event 87:B:3-0>, <Event 174:W:6-1>, <Event 96:B:7-1>, <Event 167:W:3-0>, <Event 89:B:4-0>, <Event 176:W:7-1>, <Event 98:B:8-1>, <Event 169:W:4-0>, <Event 91:B:5-0>, <Event 178:W:8-1>, <Event 100:B:9-1>, <Event 171:W:5-0>, <Event 93:B:6-0>, <Event 180:W:9-1>, <Event 173:W:6-0>, <Event 95:B:7-0>, <Event 175:W:7-0>, <Event 97:B:8-0>, <Event 177:W:8-0>, <Event 99:B:9-0>, <Event 179:W:9-0>]',
                    '[<Event 21:F:0-0>, <Event 23:F:1-0>, <Event 25:F:2-0>, <Event 27:F:3-0>, <Event 22:F:0-1>, <Event 29:F:4-0>, <Event 24:F:1-1>, <Event 31:F:5-0>, <Event 33:F:6-0>, <Event 26:F:2-1>, <Event 35:F:7-0>, <Event 28:F:3-1>, <Event 37:F:8-0>, <Event 30:F:4-1>, <Event 102:B:0-1>, <Event 39:F:9-0>, <Event 32:F:5-1>, <Event 182:W:0-1>, <Event 104:B:1-1>, <Event 34:F:6-1>, <Event 184:W:1-1>, <Event 106:B:2-1>, <Event 36:F:7-1>, <Event 186:W:2-1>, <Event 108:B:3-1>, <Event 38:F:8-1>, <Event 101:B:0-0>, <Event 188:W:3-1>, <Event 110:B:4-1>, <Event 40:F:9-1>, <Event 181:W:0-0>, <Event 103:B:1-0>, <Event 190:W:4-1>, <Event 112:B:5-1>, <Event 183:W:1-0>, <Event 105:B:2-0>, <Event 192:W:5-1>, <Event 114:B:6-1>, <Event 185:W:2-0>, <Event 107:B:3-0>, <Event 194:W:6-1>, <Event 116:B:7-1>, <Event 187:W:3-0>, <Event 109:B:4-0>, <Event 196:W:7-1>, <Event 118:B:8-1>, <Event 189:W:4-0>, <Event 111:B:5-0>, <Event 198:W:8-1>, <Event 120:B:9-1>, <Event 191:W:5-0>, <Event 113:B:6-0>, <Event 200:W:9-1>, <Event 193:W:6-0>, <Event 115:B:7-0>, <Event 195:W:7-0>, <Event 117:B:8-0>, <Event 197:W:8-0>, <Event 119:B:9-0>, <Event 199:W:9-0>]',
                    '[<Event 41:F:0-0>, <Event 43:F:1-0>, <Event 45:F:2-0>, <Event 47:F:3-0>, <Event 42:F:0-1>, <Event 49:F:4-0>, <Event 44:F:1-1>, <Event 51:F:5-0>, <Event 53:F:6-0>, <Event 46:F:2-1>, <Event 122:B:0-1>, <Event 55:F:7-0>, <Event 48:F:3-1>, <Event 202:W:0-1>, <Event 124:B:1-1>, <Event 57:F:8-0>, <Event 50:F:4-1>, <Event 204:W:1-1>, <Event 59:F:9-0>, <Event 126:B:2-1>, <Event 52:F:5-1>, <Event 206:W:2-1>, <Event 128:B:3-1>, <Event 54:F:6-1>, <Event 121:B:0-0>, <Event 208:W:3-1>, <Event 130:B:4-1>, <Event 56:F:7-1>, <Event 201:W:0-0>, <Event 123:B:1-0>, <Event 210:W:4-1>, <Event 132:B:5-1>, <Event 58:F:8-1>, <Event 203:W:1-0>, <Event 125:B:2-0>, <Event 212:W:5-1>, <Event 134:B:6-1>, <Event 60:F:9-1>, <Event 205:W:2-0>, <Event 127:B:3-0>, <Event 214:W:6-1>, <Event 136:B:7-1>, <Event 207:W:3-0>, <Event 129:B:4-0>, <Event 216:W:7-1>, <Event 138:B:8-1>, <Event 209:W:4-0>, <Event 131:B:5-0>, <Event 218:W:8-1>, <Event 140:B:9-1>, <Event 211:W:5-0>, <Event 133:B:6-0>, <Event 220:W:9-1>, <Event 213:W:6-0>, <Event 135:B:7-0>, <Event 215:W:7-0>, <Event 137:B:8-0>, <Event 217:W:8-0>, <Event 139:B:9-0>, <Event 219:W:9-0>]',
                    '[<Event 61:F:0-0>, <Event 63:F:1-0>, <Event 65:F:2-0>, <Event 67:F:3-0>, <Event 62:F:0-1>, <Event 69:F:4-0>, <Event 142:B:0-1>, <Event 64:F:1-1>, <Event 71:F:5-0>, <Event 222:W:0-1>, <Event 144:B:1-1>, <Event 73:F:6-0>, <Event 66:F:2-1>, <Event 224:W:1-1>, <Event 75:F:7-0>, <Event 146:B:2-1>, <Event 68:F:3-1>, <Event 77:F:8-0>, <Event 226:W:2-1>, <Event 148:B:3-1>, <Event 70:F:4-1>, <Event 141:B:0-0>, <Event 79:F:9-0>, <Event 228:W:3-1>, <Event 150:B:4-1>, <Event 72:F:5-1>, <Event 221:W:0-0>, <Event 143:B:1-0>, <Event 230:W:4-1>, <Event 152:B:5-1>, <Event 74:F:6-1>, <Event 223:W:1-0>, <Event 145:B:2-0>, <Event 232:W:5-1>, <Event 154:B:6-1>, <Event 76:F:7-1>, <Event 225:W:2-0>, <Event 147:B:3-0>, <Event 234:W:6-1>, <Event 156:B:7-1>, <Event 78:F:8-1>, <Event 227:W:3-0>, <Event 149:B:4-0>, <Event 236:W:7-1>, <Event 158:B:8-1>, <Event 80:F:9-1>, <Event 229:W:4-0>, <Event 151:B:5-0>, <Event 238:W:8-1>, <Event 160:B:9-1>, <Event 231:W:5-0>, <Event 153:B:6-0>, <Event 240:W:9-1>, <Event 233:W:6-0>, <Event 155:B:7-0>, <Event 235:W:7-0>, <Event 157:B:8-0>, <Event 237:W:8-0>, <Event 159:B:9-0>, <Event 239:W:9-0>]']
def test_ZeroBubble_Simulate_Visualize():
    g = ZeroBubbleGraph(4, 10, 2, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
    ))
    g.build_graph()

    sim = PPSimulator(PipeType.ZeroBubble, g)
    trace = sim.run()

    assert trace.get_nstages() == 4
    assert trace.get_nmicrobatches() == 10
    assert trace.get_nchunks() == 2

    def find_first_difference(str1, str2):
        for i in range(min(len(str1), len(str2))):
            if str1[i] != str2[i]:
                return i, str1[i-5:i+5], str2[i-5:i+5]
        return None
    for i in range(trace.get_nstages()):
        #assert str(trace.get_events(i)) == stage_events_str[i]
        print(str(trace.get_events(i)))
        print(stage_events_str[i])
        print(find_first_difference(str(trace.get_events(i)), stage_events_str[i]))

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()
    #assert false
# test_ZeroBubble_Simulate_Visualize()