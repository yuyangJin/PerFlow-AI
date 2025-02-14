'''
test ZeroBubbleGraph class
'''
import pytest

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.zerobubble import ZeroBubbleGraph
from perflowai.core.event import NoneTimestamp, EventType

std_nodes_str = "dict_values([<Event 80:EventType.FWD-0-0>, <Event 100:EventType.FWD-0-1>, <Event 81:EventType.FWD-1-0>, <Event 101:EventType.FWD-1-1>, <Event 82:EventType.FWD-2-0>, <Event 102:EventType.FWD-2-1>, <Event 83:EventType.FWD-3-0>, <Event 103:EventType.FWD-3-1>, <Event 84:EventType.FWD-4-0>, <Event 104:EventType.FWD-4-1>, <Event 85:EventType.FWD-0-0>, <Event 105:EventType.FWD-0-1>, <Event 86:EventType.FWD-1-0>, <Event 106:EventType.FWD-1-1>, <Event 87:EventType.FWD-2-0>, <Event 107:EventType.FWD-2-1>, <Event 88:EventType.FWD-3-0>, <Event 108:EventType.FWD-3-1>, <Event 89:EventType.FWD-4-0>, <Event 109:EventType.FWD-4-1>, <Event 90:EventType.FWD-0-0>, <Event 110:EventType.FWD-0-1>, <Event 91:EventType.FWD-1-0>, <Event 111:EventType.FWD-1-1>, <Event 92:EventType.FWD-2-0>, <Event 112:EventType.FWD-2-1>, <Event 93:EventType.FWD-3-0>, <Event 113:EventType.FWD-3-1>, <Event 94:EventType.FWD-4-0>, <Event 114:EventType.FWD-4-1>, <Event 95:EventType.FWD-0-0>, <Event 115:EventType.FWD-0-1>, <Event 96:EventType.FWD-1-0>, <Event 116:EventType.FWD-1-1>, <Event 97:EventType.FWD-2-0>, <Event 117:EventType.FWD-2-1>, <Event 98:EventType.FWD-3-0>, <Event 118:EventType.FWD-3-1>, <Event 99:EventType.FWD-4-0>, <Event 119:EventType.FWD-4-1>, <Event 120:EventType.BWD-0-0>, <Event 140:EventType.BWD-0-1>, <Event 121:EventType.BWD-1-0>, <Event 141:EventType.BWD-1-1>, <Event 122:EventType.BWD-2-0>, <Event 142:EventType.BWD-2-1>, <Event 123:EventType.BWD-3-0>, <Event 143:EventType.BWD-3-1>, <Event 124:EventType.BWD-4-0>, <Event 144:EventType.BWD-4-1>, <Event 125:EventType.BWD-0-0>, <Event 145:EventType.BWD-0-1>, <Event 126:EventType.BWD-1-0>, <Event 146:EventType.BWD-1-1>, <Event 127:EventType.BWD-2-0>, <Event 147:EventType.BWD-2-1>, <Event 128:EventType.BWD-3-0>, <Event 148:EventType.BWD-3-1>, <Event 129:EventType.BWD-4-0>, <Event 149:EventType.BWD-4-1>, <Event 130:EventType.BWD-0-0>, <Event 150:EventType.BWD-0-1>, <Event 131:EventType.BWD-1-0>, <Event 151:EventType.BWD-1-1>, <Event 132:EventType.BWD-2-0>, <Event 152:EventType.BWD-2-1>, <Event 133:EventType.BWD-3-0>, <Event 153:EventType.BWD-3-1>, <Event 134:EventType.BWD-4-0>, <Event 154:EventType.BWD-4-1>, <Event 135:EventType.BWD-0-0>, <Event 155:EventType.BWD-0-1>, <Event 136:EventType.BWD-1-0>, <Event 156:EventType.BWD-1-1>, <Event 137:EventType.BWD-2-0>, <Event 157:EventType.BWD-2-1>, <Event 138:EventType.BWD-3-0>, <Event 158:EventType.BWD-3-1>, <Event 139:EventType.BWD-4-0>, <Event 159:EventType.BWD-4-1>, <Event 160:EventType.WGT-0-0>, <Event 180:EventType.WGT-0-1>, <Event 161:EventType.WGT-1-0>, <Event 181:EventType.WGT-1-1>, <Event 162:EventType.WGT-2-0>, <Event 182:EventType.WGT-2-1>, <Event 163:EventType.WGT-3-0>, <Event 183:EventType.WGT-3-1>, <Event 164:EventType.WGT-4-0>, <Event 184:EventType.WGT-4-1>, <Event 165:EventType.WGT-0-0>, <Event 185:EventType.WGT-0-1>, <Event 166:EventType.WGT-1-0>, <Event 186:EventType.WGT-1-1>, <Event 167:EventType.WGT-2-0>, <Event 187:EventType.WGT-2-1>, <Event 168:EventType.WGT-3-0>, <Event 188:EventType.WGT-3-1>, <Event 169:EventType.WGT-4-0>, <Event 189:EventType.WGT-4-1>, <Event 170:EventType.WGT-0-0>, <Event 190:EventType.WGT-0-1>, <Event 171:EventType.WGT-1-0>, <Event 191:EventType.WGT-1-1>, <Event 172:EventType.WGT-2-0>, <Event 192:EventType.WGT-2-1>, <Event 173:EventType.WGT-3-0>, <Event 193:EventType.WGT-3-1>, <Event 174:EventType.WGT-4-0>, <Event 194:EventType.WGT-4-1>, <Event 175:EventType.WGT-0-0>, <Event 195:EventType.WGT-0-1>, <Event 176:EventType.WGT-1-0>, <Event 196:EventType.WGT-1-1>, <Event 177:EventType.WGT-2-0>, <Event 197:EventType.WGT-2-1>, <Event 178:EventType.WGT-3-0>, <Event 198:EventType.WGT-3-1>, <Event 179:EventType.WGT-4-0>, <Event 199:EventType.WGT-4-1>])"

def test_ZeroBubbleGraph():
    g = ZeroBubbleGraph(4, 5, 2, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
    ))

    g.build_graph()
    
    assert g.get_event_types() == [EventType.FWD, EventType.BWD, EventType.WGT]
    assert g.get_nstages() == 4
    assert g.get_nchunks() == 2
    assert g.get_nmicrobatches() == 5

    g.check()

    assert str(g.get_nodes().values()) == std_nodes_str

    for event in g.get_nodes().values():
        assert event.get_timestamp() == NoneTimestamp

