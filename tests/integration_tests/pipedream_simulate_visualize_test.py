'''
test Visualize the simulated trace of PipeDream
'''

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.pipedream import PipeDreamGraph
from perflowai.simulator.pp_simulator import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisiualizer

stage_events_str = ['[<Event 80:F:0-0>, <Event 81:F:1-0>, <Event 82:F:2-0>, <Event 83:F:3-0>, <Event 84:F:4-0>, <Event 85:F:5-0>, <Event 86:F:6-0>, <Event 120:B:0-0>, <Event 87:F:7-0>, <Event 121:B:1-0>, <Event 88:F:8-0>, <Event 122:B:2-0>, <Event 89:F:9-0>, <Event 123:B:3-0>, <Event 124:B:4-0>, <Event 125:B:5-0>, <Event 126:B:6-0>, <Event 127:B:7-0>, <Event 128:B:8-0>, <Event 129:B:9-0>]',
                    '[<Event 90:F:0-0>, <Event 91:F:1-0>, <Event 92:F:2-0>, <Event 93:F:3-0>, <Event 94:F:4-0>, <Event 130:B:0-0>, <Event 95:F:5-0>, <Event 131:B:1-0>, <Event 96:F:6-0>, <Event 132:B:2-0>, <Event 97:F:7-0>, <Event 133:B:3-0>, <Event 98:F:8-0>, <Event 134:B:4-0>, <Event 99:F:9-0>, <Event 135:B:5-0>, <Event 136:B:6-0>, <Event 137:B:7-0>, <Event 138:B:8-0>, <Event 139:B:9-0>]',
                    '[<Event 100:F:0-0>, <Event 101:F:1-0>, <Event 102:F:2-0>, <Event 140:B:0-0>, <Event 103:F:3-0>, <Event 141:B:1-0>, <Event 104:F:4-0>, <Event 142:B:2-0>, <Event 105:F:5-0>, <Event 143:B:3-0>, <Event 106:F:6-0>, <Event 144:B:4-0>, <Event 107:F:7-0>, <Event 145:B:5-0>, <Event 108:F:8-0>, <Event 146:B:6-0>, <Event 109:F:9-0>, <Event 147:B:7-0>, <Event 148:B:8-0>, <Event 149:B:9-0>]',
                    '[<Event 110:F:0-0>, <Event 150:B:0-0>, <Event 111:F:1-0>, <Event 151:B:1-0>, <Event 112:F:2-0>, <Event 152:B:2-0>, <Event 113:F:3-0>, <Event 153:B:3-0>, <Event 114:F:4-0>, <Event 154:B:4-0>, <Event 115:F:5-0>, <Event 155:B:5-0>, <Event 116:F:6-0>, <Event 156:B:6-0>, <Event 117:F:7-0>, <Event 157:B:7-0>, <Event 118:F:8-0>, <Event 158:B:8-0>, <Event 119:F:9-0>, <Event 159:B:9-0>]']


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