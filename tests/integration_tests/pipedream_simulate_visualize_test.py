'''
test Visualize the simulated trace of PipeDream
'''

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.pipedream import PipeDreamGraph
from perflowai.simulator.pp_simulator import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisiualizer

stage_events_str = ['[<Event 1:F:0-0>, <Event 2:F:1-0>, <Event 3:F:2-0>, <Event 4:F:3-0>, <Event 5:F:4-0>, <Event 6:F:5-0>, <Event 7:F:6-0>, <Event 41:B:0-0>, <Event 8:F:7-0>, <Event 42:B:1-0>, <Event 9:F:8-0>, <Event 43:B:2-0>, <Event 10:F:9-0>, <Event 44:B:3-0>, <Event 45:B:4-0>, <Event 46:B:5-0>, <Event 47:B:6-0>, <Event 48:B:7-0>, <Event 49:B:8-0>, <Event 50:B:9-0>]',
                    '[<Event 11:F:0-0>, <Event 12:F:1-0>, <Event 13:F:2-0>, <Event 14:F:3-0>, <Event 15:F:4-0>, <Event 51:B:0-0>, <Event 16:F:5-0>, <Event 52:B:1-0>, <Event 17:F:6-0>, <Event 53:B:2-0>, <Event 18:F:7-0>, <Event 54:B:3-0>, <Event 19:F:8-0>, <Event 55:B:4-0>, <Event 20:F:9-0>, <Event 56:B:5-0>, <Event 57:B:6-0>, <Event 58:B:7-0>, <Event 59:B:8-0>, <Event 60:B:9-0>]',
                    '[<Event 21:F:0-0>, <Event 22:F:1-0>, <Event 23:F:2-0>, <Event 61:B:0-0>, <Event 24:F:3-0>, <Event 62:B:1-0>, <Event 25:F:4-0>, <Event 63:B:2-0>, <Event 26:F:5-0>, <Event 64:B:3-0>, <Event 27:F:6-0>, <Event 65:B:4-0>, <Event 28:F:7-0>, <Event 66:B:5-0>, <Event 29:F:8-0>, <Event 67:B:6-0>, <Event 30:F:9-0>, <Event 68:B:7-0>, <Event 69:B:8-0>, <Event 70:B:9-0>]',
                    '[<Event 31:F:0-0>, <Event 71:B:0-0>, <Event 32:F:1-0>, <Event 72:B:1-0>, <Event 33:F:2-0>, <Event 73:B:2-0>, <Event 34:F:3-0>, <Event 74:B:3-0>, <Event 35:F:4-0>, <Event 75:B:4-0>, <Event 36:F:5-0>, <Event 76:B:5-0>, <Event 37:F:6-0>, <Event 77:B:6-0>, <Event 38:F:7-0>, <Event 78:B:7-0>, <Event 39:F:8-0>, <Event 79:B:8-0>, <Event 40:F:9-0>, <Event 80:B:9-0>]']


def test_PipeDream_Simulate_Visualize():
    g = PipeDreamGraph(4, 10, 1, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
        ),
        eager = True
    )
    g.build_graph()

    sim = PPSimulator(PipeType.PipeDream, g)
    trace = sim.run()

    assert trace.get_nstages() == 4
    assert trace.get_nmicrobatches() == 10

    for i in range(trace.get_nstages()):
        assert str(trace.get_events(i)) == stage_events_str[i]

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()



# test_PipeDream_Simulate_Visualize()