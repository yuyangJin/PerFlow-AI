'''
test Visualize the simulated trace of zerobubble
'''

from perflowai.parallel.pipeline_parallel import PipeCostConfig, ZeroBubbleGraph, ScheduleType
from perflowai.simulator import PPSimulator, PipeType
from perflowai.visualizer import TraceVisiualizer

stage_events_str = ['[<Event 1:F:0-0>, <Event 3:F:1-0>, <Event 5:F:2-0>, <Event 7:F:3-0>, <Event 9:F:4-0>, <Event 11:F:5-0>, <Event 13:F:6-0>, <Event 2:F:0-1>, <Event 15:F:7-0>, <Event 122:B:0-1>, <Event 4:F:1-1>, <Event 17:F:8-0>, <Event 242:W:0-1>, <Event 124:B:1-1>, <Event 19:F:9-0>, <Event 6:F:2-1>, <Event 244:W:1-1>, <Event 21:F:10-0>, <Event 126:B:2-1>, <Event 8:F:3-1>, <Event 23:F:11-0>, <Event 246:W:2-1>, <Event 128:B:3-1>, <Event 10:F:4-1>, <Event 25:F:12-0>, <Event 248:W:3-1>, <Event 12:F:5-1>, <Event 130:B:4-1>, <Event 27:F:13-0>, <Event 14:F:6-1>, <Event 132:B:5-1>, <Event 250:W:4-1>, <Event 29:F:14-0>, <Event 16:F:7-1>, <Event 134:B:6-1>, <Event 252:W:5-1>, <Event 121:B:0-0>, <Event 18:F:8-1>, <Event 136:B:7-1>, <Event 254:W:6-1>, <Event 241:W:0-0>, <Event 123:B:1-0>, <Event 20:F:9-1>, <Event 138:B:8-1>, <Event 256:W:7-1>, <Event 243:W:1-0>, <Event 125:B:2-0>, <Event 22:F:10-1>, <Event 140:B:9-1>, <Event 258:W:8-1>, <Event 245:W:2-0>, <Event 127:B:3-0>, <Event 24:F:11-1>, <Event 142:B:10-1>, <Event 260:W:9-1>, <Event 247:W:3-0>, <Event 26:F:12-1>, <Event 129:B:4-0>, <Event 144:B:11-1>, <Event 262:W:10-1>, <Event 28:F:13-1>, <Event 249:W:4-0>, <Event 131:B:5-0>, <Event 146:B:12-1>, <Event 264:W:11-1>, <Event 30:F:14-1>, <Event 251:W:5-0>, <Event 133:B:6-0>, <Event 148:B:13-1>, <Event 266:W:12-1>, <Event 253:W:6-0>, <Event 150:B:14-1>, <Event 135:B:7-0>, <Event 268:W:13-1>, <Event 255:W:7-0>, <Event 137:B:8-0>, <Event 270:W:14-1>, <Event 257:W:8-0>, <Event 139:B:9-0>, <Event 259:W:9-0>, <Event 141:B:10-0>, <Event 261:W:10-0>, <Event 143:B:11-0>, <Event 263:W:11-0>, <Event 145:B:12-0>, <Event 147:B:13-0>, <Event 265:W:12-0>, <Event 149:B:14-0>, <Event 267:W:13-0>, <Event 269:W:14-0>]',
                    '[<Event 31:F:0-0>, <Event 33:F:1-0>, <Event 35:F:2-0>, <Event 37:F:3-0>, <Event 39:F:4-0>, <Event 32:F:0-1>, <Event 41:F:5-0>, <Event 34:F:1-1>, <Event 43:F:6-0>, <Event 45:F:7-0>, <Event 36:F:2-1>, <Event 152:B:0-1>, <Event 47:F:8-0>, <Event 38:F:3-1>, <Event 272:W:0-1>, <Event 154:B:1-1>, <Event 49:F:9-0>, <Event 40:F:4-1>, <Event 274:W:1-1>, <Event 51:F:10-0>, <Event 156:B:2-1>, <Event 42:F:5-1>, <Event 53:F:11-0>, <Event 276:W:2-1>, <Event 158:B:3-1>, <Event 44:F:6-1>, <Event 55:F:12-0>, <Event 278:W:3-1>, <Event 160:B:4-1>, <Event 46:F:7-1>, <Event 151:B:0-0>, <Event 57:F:13-0>, <Event 280:W:4-1>, <Event 162:B:5-1>, <Event 48:F:8-1>, <Event 271:W:0-0>, <Event 153:B:1-0>, <Event 59:F:14-0>, <Event 282:W:5-1>, <Event 164:B:6-1>, <Event 50:F:9-1>, <Event 273:W:1-0>, <Event 155:B:2-0>, <Event 284:W:6-1>, <Event 52:F:10-1>, <Event 166:B:7-1>, <Event 275:W:2-0>, <Event 157:B:3-0>, <Event 54:F:11-1>, <Event 286:W:7-1>, <Event 168:B:8-1>, <Event 277:W:3-0>, <Event 159:B:4-0>, <Event 56:F:12-1>, <Event 288:W:8-1>, <Event 170:B:9-1>, <Event 279:W:4-0>, <Event 161:B:5-0>, <Event 58:F:13-1>, <Event 172:B:10-1>, <Event 290:W:9-1>, <Event 281:W:5-0>, <Event 163:B:6-0>, <Event 60:F:14-1>, <Event 174:B:11-1>, <Event 292:W:10-1>, <Event 283:W:6-0>, <Event 165:B:7-0>, <Event 176:B:12-1>, <Event 294:W:11-1>, <Event 285:W:7-0>, <Event 167:B:8-0>, <Event 178:B:13-1>, <Event 296:W:12-1>, <Event 287:W:8-0>, <Event 180:B:14-1>, <Event 298:W:13-1>, <Event 169:B:9-0>, <Event 300:W:14-1>, <Event 289:W:9-0>, <Event 171:B:10-0>, <Event 291:W:10-0>, <Event 173:B:11-0>, <Event 293:W:11-0>, <Event 175:B:12-0>, <Event 177:B:13-0>, <Event 295:W:12-0>, <Event 179:B:14-0>, <Event 297:W:13-0>, <Event 299:W:14-0>]',
                    '[<Event 61:F:0-0>, <Event 63:F:1-0>, <Event 65:F:2-0>, <Event 62:F:0-1>, <Event 67:F:3-0>, <Event 64:F:1-1>, <Event 69:F:4-0>, <Event 66:F:2-1>, <Event 71:F:5-0>, <Event 68:F:3-1>, <Event 73:F:6-0>, <Event 75:F:7-0>, <Event 70:F:4-1>, <Event 182:B:0-1>, <Event 77:F:8-0>, <Event 72:F:5-1>, <Event 302:W:0-1>, <Event 184:B:1-1>, <Event 79:F:9-0>, <Event 74:F:6-1>, <Event 304:W:1-1>, <Event 186:B:2-1>, <Event 81:F:10-0>, <Event 76:F:7-1>, <Event 181:B:0-0>, <Event 306:W:2-1>, <Event 188:B:3-1>, <Event 83:F:11-0>, <Event 78:F:8-1>, <Event 301:W:0-0>, <Event 183:B:1-0>, <Event 308:W:3-1>, <Event 85:F:12-0>, <Event 190:B:4-1>, <Event 80:F:9-1>, <Event 303:W:1-0>, <Event 185:B:2-0>, <Event 87:F:13-0>, <Event 310:W:4-1>, <Event 192:B:5-1>, <Event 82:F:10-1>, <Event 305:W:2-0>, <Event 187:B:3-0>, <Event 89:F:14-0>, <Event 312:W:5-1>, <Event 84:F:11-1>, <Event 194:B:6-1>, <Event 307:W:3-0>, <Event 189:B:4-0>, <Event 86:F:12-1>, <Event 314:W:6-1>, <Event 196:B:7-1>, <Event 309:W:4-0>, <Event 191:B:5-0>, <Event 88:F:13-1>, <Event 316:W:7-1>, <Event 198:B:8-1>, <Event 311:W:5-0>, <Event 90:F:14-1>, <Event 193:B:6-0>, <Event 318:W:8-1>, <Event 200:B:9-1>, <Event 313:W:6-0>, <Event 195:B:7-0>, <Event 202:B:10-1>, <Event 320:W:9-1>, <Event 315:W:7-0>, <Event 197:B:8-0>, <Event 204:B:11-1>, <Event 322:W:10-1>, <Event 317:W:8-0>, <Event 206:B:12-1>, <Event 199:B:9-0>, <Event 324:W:11-1>, <Event 208:B:13-1>, <Event 319:W:9-0>, <Event 201:B:10-0>, <Event 326:W:12-1>, <Event 210:B:14-1>, <Event 321:W:10-0>, <Event 203:B:11-0>, <Event 328:W:13-1>, <Event 323:W:11-0>, <Event 205:B:12-0>, <Event 330:W:14-1>, <Event 207:B:13-0>, <Event 325:W:12-0>, <Event 209:B:14-0>, <Event 327:W:13-0>, <Event 329:W:14-0>]',
                    '[<Event 91:F:0-0>, <Event 92:F:0-1>, <Event 93:F:1-0>, <Event 94:F:1-1>, <Event 95:F:2-0>, <Event 96:F:2-1>, <Event 97:F:3-0>, <Event 98:F:3-1>, <Event 99:F:4-0>, <Event 100:F:4-1>, <Event 101:F:5-0>, <Event 102:F:5-1>, <Event 103:F:6-0>, <Event 105:F:7-0>, <Event 104:F:6-1>, <Event 212:B:0-1>, <Event 107:F:8-0>, <Event 106:F:7-1>, <Event 332:W:0-1>, <Event 211:B:0-0>, <Event 214:B:1-1>, <Event 109:F:9-0>, <Event 108:F:8-1>, <Event 331:W:0-0>, <Event 334:W:1-1>, <Event 213:B:1-0>, <Event 216:B:2-1>, <Event 111:F:10-0>, <Event 110:F:9-1>, <Event 333:W:1-0>, <Event 336:W:2-1>, <Event 215:B:2-0>, <Event 218:B:3-1>, <Event 113:F:11-0>, <Event 112:F:10-1>, <Event 335:W:2-0>, <Event 338:W:3-1>, <Event 217:B:3-0>, <Event 220:B:4-1>, <Event 115:F:12-0>, <Event 114:F:11-1>, <Event 337:W:3-0>, <Event 340:W:4-1>, <Event 219:B:4-0>, <Event 222:B:5-1>, <Event 117:F:13-0>, <Event 116:F:12-1>, <Event 339:W:4-0>, <Event 221:B:5-0>, <Event 342:W:5-1>, <Event 119:F:14-0>, <Event 118:F:13-1>, <Event 224:B:6-1>, <Event 341:W:5-0>, <Event 120:F:14-1>, <Event 344:W:6-1>, <Event 223:B:6-0>, <Event 226:B:7-1>, <Event 343:W:6-0>, <Event 225:B:7-0>, <Event 346:W:7-1>, <Event 228:B:8-1>, <Event 345:W:7-0>, <Event 348:W:8-1>, <Event 227:B:8-0>, <Event 230:B:9-1>, <Event 347:W:8-0>, <Event 232:B:10-1>, <Event 350:W:9-1>, <Event 229:B:9-0>, <Event 234:B:11-1>, <Event 352:W:10-1>, <Event 349:W:9-0>, <Event 231:B:10-0>, <Event 236:B:12-1>, <Event 354:W:11-1>, <Event 233:B:11-0>, <Event 351:W:10-0>, <Event 238:B:13-1>, <Event 356:W:12-1>, <Event 235:B:12-0>, <Event 353:W:11-0>, <Event 240:B:14-1>, <Event 358:W:13-1>, <Event 237:B:13-0>, <Event 355:W:12-0>, <Event 360:W:14-1>, <Event 239:B:14-0>, <Event 357:W:13-0>, <Event 359:W:14-0>]']

def test_ZBV_Simulate_Visualize():
    g = ZeroBubbleGraph(4, 15, 2, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
    ), schedule_type = ScheduleType.ZBV)
    g.build_graph()

    sim = PPSimulator(PipeType.ZeroBubble, g)
    trace = sim.run()

    assert trace.get_nstages() == 4
    assert trace.get_nmicrobatches() == 15
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
test_ZBV_Simulate_Visualize()