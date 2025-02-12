'''
test PPGraph class
'''

from perflowai.parallel.pipeline_parallel.ppgraph import PPGraph, PipeCostConfig
from perflowai.core.event import NoneTimestamp, EventType

std_nodes_str = 'dict_values([<Event 80:EventType.FWD-0-0>, <Event 81:EventType.FWD-1-0>, <Event 82:EventType.FWD-2-0>, <Event 83:EventType.FWD-3-0>, <Event 84:EventType.FWD-4-0>, <Event 85:EventType.FWD-5-0>, <Event 86:EventType.FWD-6-0>, <Event 87:EventType.FWD-7-0>, <Event 88:EventType.FWD-8-0>, <Event 89:EventType.FWD-9-0>, <Event 90:EventType.FWD-0-0>, <Event 91:EventType.FWD-1-0>, <Event 92:EventType.FWD-2-0>, <Event 93:EventType.FWD-3-0>, <Event 94:EventType.FWD-4-0>, <Event 95:EventType.FWD-5-0>, <Event 96:EventType.FWD-6-0>, <Event 97:EventType.FWD-7-0>, <Event 98:EventType.FWD-8-0>, <Event 99:EventType.FWD-9-0>, <Event 100:EventType.FWD-0-0>, <Event 101:EventType.FWD-1-0>, <Event 102:EventType.FWD-2-0>, <Event 103:EventType.FWD-3-0>, <Event 104:EventType.FWD-4-0>, <Event 105:EventType.FWD-5-0>, <Event 106:EventType.FWD-6-0>, <Event 107:EventType.FWD-7-0>, <Event 108:EventType.FWD-8-0>, <Event 109:EventType.FWD-9-0>, <Event 110:EventType.FWD-0-0>, <Event 111:EventType.FWD-1-0>, <Event 112:EventType.FWD-2-0>, <Event 113:EventType.FWD-3-0>, <Event 114:EventType.FWD-4-0>, <Event 115:EventType.FWD-5-0>, <Event 116:EventType.FWD-6-0>, <Event 117:EventType.FWD-7-0>, <Event 118:EventType.FWD-8-0>, <Event 119:EventType.FWD-9-0>, <Event 120:EventType.BWD-0-0>, <Event 121:EventType.BWD-1-0>, <Event 122:EventType.BWD-2-0>, <Event 123:EventType.BWD-3-0>, <Event 124:EventType.BWD-4-0>, <Event 125:EventType.BWD-5-0>, <Event 126:EventType.BWD-6-0>, <Event 127:EventType.BWD-7-0>, <Event 128:EventType.BWD-8-0>, <Event 129:EventType.BWD-9-0>, <Event 130:EventType.BWD-0-0>, <Event 131:EventType.BWD-1-0>, <Event 132:EventType.BWD-2-0>, <Event 133:EventType.BWD-3-0>, <Event 134:EventType.BWD-4-0>, <Event 135:EventType.BWD-5-0>, <Event 136:EventType.BWD-6-0>, <Event 137:EventType.BWD-7-0>, <Event 138:EventType.BWD-8-0>, <Event 139:EventType.BWD-9-0>, <Event 140:EventType.BWD-0-0>, <Event 141:EventType.BWD-1-0>, <Event 142:EventType.BWD-2-0>, <Event 143:EventType.BWD-3-0>, <Event 144:EventType.BWD-4-0>, <Event 145:EventType.BWD-5-0>, <Event 146:EventType.BWD-6-0>, <Event 147:EventType.BWD-7-0>, <Event 148:EventType.BWD-8-0>, <Event 149:EventType.BWD-9-0>, <Event 150:EventType.BWD-0-0>, <Event 151:EventType.BWD-1-0>, <Event 152:EventType.BWD-2-0>, <Event 153:EventType.BWD-3-0>, <Event 154:EventType.BWD-4-0>, <Event 155:EventType.BWD-5-0>, <Event 156:EventType.BWD-6-0>, <Event 157:EventType.BWD-7-0>, <Event 158:EventType.BWD-8-0>, <Event 159:EventType.BWD-9-0>])'

def test_PPGraph():

    g = PPGraph(4, 10, 1, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
    ))

    assert g.get_event_types() == [EventType.FWD, EventType.BWD]
    assert g.get_nstages() == 4
    assert g.get_nchunks() == 1
    assert g.get_nmicrobatches() == 10

    g.generate_nodes()

    g.check()

    assert str(g.get_nodes().values()) == std_nodes_str

    for event in g.get_nodes().values():
        assert event.get_timestamp() == NoneTimestamp

