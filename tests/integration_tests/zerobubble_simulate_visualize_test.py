'''
test Visualize the simulated trace of zerobubble
'''

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.zerobubble import ZeroBubbleGraph
from perflowai.simulator.pp_simulator import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisiualizer

stage_events_str = ['[<Event 160:F:0-0>, <Event 161:F:1-0>, <Event 162:F:2-0>, <Event 163:F:3-0>, <Event 164:F:4-0>, <Event 200:F:0-1>, <Event 165:F:5-0>, <Event 201:F:1-1>, <Event 166:F:6-0>, <Event 202:F:2-1>, <Event 167:F:7-0>, <Event 203:F:3-1>, <Event 168:F:8-0>, <Event 204:F:4-1>, <Event 169:F:9-0>, <Event 205:F:5-1>, <Event 206:F:6-1>, <Event 280:B:0-1>, <Event 207:F:7-1>, <Event 360:W:0-1>, <Event 208:F:8-1>, <Event 281:B:1-1>, <Event 209:F:9-1>, <Event 361:W:1-1>, <Event 282:B:2-1>, <Event 362:W:2-1>, <Event 283:B:3-1>, <Event 240:B:0-0>, <Event 363:W:3-1>, <Event 284:B:4-1>, <Event 320:W:0-0>, <Event 241:B:1-0>, <Event 364:W:4-1>, <Event 285:B:5-1>, <Event 321:W:1-0>, <Event 242:B:2-0>, <Event 365:W:5-1>, <Event 286:B:6-1>, <Event 322:W:2-0>, <Event 243:B:3-0>, <Event 366:W:6-1>, <Event 287:B:7-1>, <Event 323:W:3-0>, <Event 244:B:4-0>, <Event 288:B:8-1>, <Event 367:W:7-1>, <Event 324:W:4-0>, <Event 245:B:5-0>, <Event 289:B:9-1>, <Event 368:W:8-1>, <Event 325:W:5-0>, <Event 246:B:6-0>, <Event 369:W:9-1>, <Event 326:W:6-0>, <Event 247:B:7-0>, <Event 327:W:7-0>, <Event 248:B:8-0>, <Event 328:W:8-0>, <Event 249:B:9-0>, <Event 329:W:9-0>]',
                    '[<Event 170:F:0-0>, <Event 171:F:1-0>, <Event 172:F:2-0>, <Event 173:F:3-0>, <Event 174:F:4-0>, <Event 210:F:0-1>, <Event 175:F:5-0>, <Event 211:F:1-1>, <Event 176:F:6-0>, <Event 212:F:2-1>, <Event 177:F:7-0>, <Event 213:F:3-1>, <Event 178:F:8-0>, <Event 214:F:4-1>, <Event 179:F:9-0>, <Event 290:B:0-1>, <Event 215:F:5-1>, <Event 370:W:0-1>, <Event 216:F:6-1>, <Event 291:B:1-1>, <Event 217:F:7-1>, <Event 371:W:1-1>, <Event 292:B:2-1>, <Event 218:F:8-1>, <Event 372:W:2-1>, <Event 293:B:3-1>, <Event 250:B:0-0>, <Event 219:F:9-1>, <Event 373:W:3-1>, <Event 294:B:4-1>, <Event 330:W:0-0>, <Event 251:B:1-0>, <Event 374:W:4-1>, <Event 295:B:5-1>, <Event 331:W:1-0>, <Event 252:B:2-0>, <Event 296:B:6-1>, <Event 375:W:5-1>, <Event 332:W:2-0>, <Event 253:B:3-0>, <Event 297:B:7-1>, <Event 376:W:6-1>, <Event 333:W:3-0>, <Event 254:B:4-0>, <Event 298:B:8-1>, <Event 377:W:7-1>, <Event 334:W:4-0>, <Event 299:B:9-1>, <Event 255:B:5-0>, <Event 378:W:8-1>, <Event 335:W:5-0>, <Event 256:B:6-0>, <Event 379:W:9-1>, <Event 336:W:6-0>, <Event 257:B:7-0>, <Event 337:W:7-0>, <Event 258:B:8-0>, <Event 338:W:8-0>, <Event 259:B:9-0>, <Event 339:W:9-0>]',
                    '[<Event 180:F:0-0>, <Event 181:F:1-0>, <Event 182:F:2-0>, <Event 183:F:3-0>, <Event 184:F:4-0>, <Event 220:F:0-1>, <Event 185:F:5-0>, <Event 221:F:1-1>, <Event 186:F:6-0>, <Event 222:F:2-1>, <Event 187:F:7-0>, <Event 300:B:0-1>, <Event 223:F:3-1>, <Event 188:F:8-0>, <Event 380:W:0-1>, <Event 224:F:4-1>, <Event 301:B:1-1>, <Event 189:F:9-0>, <Event 225:F:5-1>, <Event 381:W:1-1>, <Event 302:B:2-1>, <Event 226:F:6-1>, <Event 382:W:2-1>, <Event 303:B:3-1>, <Event 260:B:0-0>, <Event 227:F:7-1>, <Event 304:B:4-1>, <Event 383:W:3-1>, <Event 340:W:0-0>, <Event 228:F:8-1>, <Event 261:B:1-0>, <Event 305:B:5-1>, <Event 384:W:4-1>, <Event 229:F:9-1>, <Event 341:W:1-0>, <Event 262:B:2-0>, <Event 306:B:6-1>, <Event 385:W:5-1>, <Event 342:W:2-0>, <Event 307:B:7-1>, <Event 263:B:3-0>, <Event 386:W:6-1>, <Event 308:B:8-1>, <Event 343:W:3-0>, <Event 264:B:4-0>, <Event 387:W:7-1>, <Event 309:B:9-1>, <Event 344:W:4-0>, <Event 265:B:5-0>, <Event 388:W:8-1>, <Event 345:W:5-0>, <Event 266:B:6-0>, <Event 389:W:9-1>, <Event 346:W:6-0>, <Event 267:B:7-0>, <Event 347:W:7-0>, <Event 268:B:8-0>, <Event 348:W:8-0>, <Event 269:B:9-0>, <Event 349:W:9-0>]',
                    '[<Event 190:F:0-0>, <Event 191:F:1-0>, <Event 192:F:2-0>, <Event 193:F:3-0>, <Event 230:F:0-1>, <Event 194:F:4-0>, <Event 310:B:0-1>, <Event 195:F:5-0>, <Event 231:F:1-1>, <Event 390:W:0-1>, <Event 196:F:6-0>, <Event 232:F:2-1>, <Event 311:B:1-1>, <Event 197:F:7-0>, <Event 233:F:3-1>, <Event 312:B:2-1>, <Event 391:W:1-1>, <Event 198:F:8-0>, <Event 234:F:4-1>, <Event 313:B:3-1>, <Event 392:W:2-1>, <Event 270:B:0-0>, <Event 199:F:9-0>, <Event 235:F:5-1>, <Event 314:B:4-1>, <Event 393:W:3-1>, <Event 350:W:0-0>, <Event 236:F:6-1>, <Event 315:B:5-1>, <Event 271:B:1-0>, <Event 394:W:4-1>, <Event 237:F:7-1>, <Event 316:B:6-1>, <Event 351:W:1-0>, <Event 272:B:2-0>, <Event 395:W:5-1>, <Event 238:F:8-1>, <Event 317:B:7-1>, <Event 352:W:2-0>, <Event 273:B:3-0>, <Event 396:W:6-1>, <Event 239:F:9-1>, <Event 318:B:8-1>, <Event 353:W:3-0>, <Event 274:B:4-0>, <Event 397:W:7-1>, <Event 319:B:9-1>, <Event 354:W:4-0>, <Event 275:B:5-0>, <Event 398:W:8-1>, <Event 355:W:5-0>, <Event 276:B:6-0>, <Event 399:W:9-1>, <Event 356:W:6-0>, <Event 277:B:7-0>, <Event 357:W:7-0>, <Event 278:B:8-0>, <Event 358:W:8-0>, <Event 279:B:9-0>, <Event 359:W:9-0>]']
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

    for i in range(trace.get_nstages()):
        assert str(trace.get_events(i)) == stage_events_str[i]
        # print(trace.get_events(i))

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()

# test_ZeroBubble_Simulate_Visualize()