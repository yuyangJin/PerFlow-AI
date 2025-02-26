'''
test Visualize the simulated trace of Interleaved1F1B
'''

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.interleaved1f1b import Interleaved1F1BGraph
from perflowai.simulator.pp_simulator import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisiualizer

stage_events_str = ['[<Event 192:F:0-0>, <Event 193:F:1-0>, <Event 194:F:2-0>, <Event 195:F:3-0>, <Event 240:F:0-1>, <Event 241:F:1-1>, <Event 242:F:2-1>, <Event 243:F:3-1>, <Event 196:F:4-0>, <Event 197:F:5-0>, <Event 198:F:6-0>, <Event 336:B:0-1>, <Event 199:F:7-0>, <Event 337:B:1-1>, <Event 244:F:4-1>, <Event 338:B:2-1>, <Event 245:F:5-1>, <Event 339:B:3-1>, <Event 246:F:6-1>, <Event 288:B:0-0>, <Event 247:F:7-1>, <Event 289:B:1-0>, <Event 200:F:8-0>, <Event 290:B:2-0>, <Event 201:F:9-0>, <Event 291:B:3-0>, <Event 202:F:10-0>, <Event 340:B:4-1>, <Event 203:F:11-0>, <Event 341:B:5-1>, <Event 248:F:8-1>, <Event 342:B:6-1>, <Event 249:F:9-1>, <Event 343:B:7-1>, <Event 250:F:10-1>, <Event 292:B:4-0>, <Event 251:F:11-1>, <Event 293:B:5-0>, <Event 294:B:6-0>, <Event 295:B:7-0>, <Event 344:B:8-1>, <Event 345:B:9-1>, <Event 346:B:10-1>, <Event 347:B:11-1>, <Event 296:B:8-0>, <Event 297:B:9-0>, <Event 298:B:10-0>, <Event 299:B:11-0>]',
                    '[<Event 204:F:0-0>, <Event 205:F:1-0>, <Event 206:F:2-0>, <Event 207:F:3-0>, <Event 252:F:0-1>, <Event 253:F:1-1>, <Event 254:F:2-1>, <Event 255:F:3-1>, <Event 208:F:4-0>, <Event 348:B:0-1>, <Event 209:F:5-0>, <Event 349:B:1-1>, <Event 210:F:6-0>, <Event 350:B:2-1>, <Event 211:F:7-0>, <Event 351:B:3-1>, <Event 256:F:4-1>, <Event 300:B:0-0>, <Event 257:F:5-1>, <Event 301:B:1-0>, <Event 258:F:6-1>, <Event 302:B:2-0>, <Event 259:F:7-1>, <Event 303:B:3-0>, <Event 212:F:8-0>, <Event 352:B:4-1>, <Event 213:F:9-0>, <Event 353:B:5-1>, <Event 214:F:10-0>, <Event 354:B:6-1>, <Event 215:F:11-0>, <Event 355:B:7-1>, <Event 260:F:8-1>, <Event 304:B:4-0>, <Event 261:F:9-1>, <Event 305:B:5-0>, <Event 262:F:10-1>, <Event 306:B:6-0>, <Event 263:F:11-1>, <Event 307:B:7-0>, <Event 356:B:8-1>, <Event 357:B:9-1>, <Event 358:B:10-1>, <Event 359:B:11-1>, <Event 308:B:8-0>, <Event 309:B:9-0>, <Event 310:B:10-0>, <Event 311:B:11-0>]',
                    '[<Event 216:F:0-0>, <Event 217:F:1-0>, <Event 218:F:2-0>, <Event 219:F:3-0>, <Event 264:F:0-1>, <Event 265:F:1-1>, <Event 266:F:2-1>, <Event 360:B:0-1>, <Event 267:F:3-1>, <Event 361:B:1-1>, <Event 220:F:4-0>, <Event 362:B:2-1>, <Event 221:F:5-0>, <Event 363:B:3-1>, <Event 222:F:6-0>, <Event 312:B:0-0>, <Event 223:F:7-0>, <Event 313:B:1-0>, <Event 268:F:4-1>, <Event 314:B:2-0>, <Event 269:F:5-1>, <Event 315:B:3-0>, <Event 270:F:6-1>, <Event 364:B:4-1>, <Event 271:F:7-1>, <Event 365:B:5-1>, <Event 224:F:8-0>, <Event 366:B:6-1>, <Event 225:F:9-0>, <Event 367:B:7-1>, <Event 226:F:10-0>, <Event 316:B:4-0>, <Event 227:F:11-0>, <Event 317:B:5-0>, <Event 272:F:8-1>, <Event 318:B:6-0>, <Event 273:F:9-1>, <Event 319:B:7-0>, <Event 274:F:10-1>, <Event 368:B:8-1>, <Event 275:F:11-1>, <Event 369:B:9-1>, <Event 370:B:10-1>, <Event 371:B:11-1>, <Event 320:B:8-0>, <Event 321:B:9-0>, <Event 322:B:10-0>, <Event 323:B:11-0>]',
                    '[<Event 228:F:0-0>, <Event 229:F:1-0>, <Event 230:F:2-0>, <Event 231:F:3-0>, <Event 276:F:0-1>, <Event 372:B:0-1>, <Event 277:F:1-1>, <Event 373:B:1-1>, <Event 278:F:2-1>, <Event 374:B:2-1>, <Event 279:F:3-1>, <Event 375:B:3-1>, <Event 232:F:4-0>, <Event 324:B:0-0>, <Event 233:F:5-0>, <Event 325:B:1-0>, <Event 234:F:6-0>, <Event 326:B:2-0>, <Event 235:F:7-0>, <Event 327:B:3-0>, <Event 280:F:4-1>, <Event 376:B:4-1>, <Event 281:F:5-1>, <Event 377:B:5-1>, <Event 282:F:6-1>, <Event 378:B:6-1>, <Event 283:F:7-1>, <Event 379:B:7-1>, <Event 236:F:8-0>, <Event 328:B:4-0>, <Event 237:F:9-0>, <Event 329:B:5-0>, <Event 238:F:10-0>, <Event 330:B:6-0>, <Event 239:F:11-0>, <Event 331:B:7-0>, <Event 284:F:8-1>, <Event 380:B:8-1>, <Event 285:F:9-1>, <Event 381:B:9-1>, <Event 286:F:10-1>, <Event 382:B:10-1>, <Event 287:F:11-1>, <Event 383:B:11-1>, <Event 332:B:8-0>, <Event 333:B:9-0>, <Event 334:B:10-0>, <Event 335:B:11-0>]']

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

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()



test_Interleaved1F1B_Simulate_Visualize()