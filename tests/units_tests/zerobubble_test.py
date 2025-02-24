'''
test ZeroBubbleGraph class
'''
import pytest

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.zerobubble import ZeroBubbleGraph
from perflowai.core.event import NoneTimestamp, EventType

std_nodes_str = 'dict_values([<Event 80:F:0-0>, <Event 100:F:0-1>, <Event 81:F:1-0>, <Event 101:F:1-1>, <Event 82:F:2-0>, <Event 102:F:2-1>, <Event 83:F:3-0>, <Event 103:F:3-1>, <Event 84:F:4-0>, <Event 104:F:4-1>, <Event 85:F:0-0>, <Event 105:F:0-1>, <Event 86:F:1-0>, <Event 106:F:1-1>, <Event 87:F:2-0>, <Event 107:F:2-1>, <Event 88:F:3-0>, <Event 108:F:3-1>, <Event 89:F:4-0>, <Event 109:F:4-1>, <Event 90:F:0-0>, <Event 110:F:0-1>, <Event 91:F:1-0>, <Event 111:F:1-1>, <Event 92:F:2-0>, <Event 112:F:2-1>, <Event 93:F:3-0>, <Event 113:F:3-1>, <Event 94:F:4-0>, <Event 114:F:4-1>, <Event 95:F:0-0>, <Event 115:F:0-1>, <Event 96:F:1-0>, <Event 116:F:1-1>, <Event 97:F:2-0>, <Event 117:F:2-1>, <Event 98:F:3-0>, <Event 118:F:3-1>, <Event 99:F:4-0>, <Event 119:F:4-1>, <Event 120:B:0-0>, <Event 140:B:0-1>, <Event 121:B:1-0>, <Event 141:B:1-1>, <Event 122:B:2-0>, <Event 142:B:2-1>, <Event 123:B:3-0>, <Event 143:B:3-1>, <Event 124:B:4-0>, <Event 144:B:4-1>, <Event 125:B:0-0>, <Event 145:B:0-1>, <Event 126:B:1-0>, <Event 146:B:1-1>, <Event 127:B:2-0>, <Event 147:B:2-1>, <Event 128:B:3-0>, <Event 148:B:3-1>, <Event 129:B:4-0>, <Event 149:B:4-1>, <Event 130:B:0-0>, <Event 150:B:0-1>, <Event 131:B:1-0>, <Event 151:B:1-1>, <Event 132:B:2-0>, <Event 152:B:2-1>, <Event 133:B:3-0>, <Event 153:B:3-1>, <Event 134:B:4-0>, <Event 154:B:4-1>, <Event 135:B:0-0>, <Event 155:B:0-1>, <Event 136:B:1-0>, <Event 156:B:1-1>, <Event 137:B:2-0>, <Event 157:B:2-1>, <Event 138:B:3-0>, <Event 158:B:3-1>, <Event 139:B:4-0>, <Event 159:B:4-1>, <Event 160:W:0-0>, <Event 180:W:0-1>, <Event 161:W:1-0>, <Event 181:W:1-1>, <Event 162:W:2-0>, <Event 182:W:2-1>, <Event 163:W:3-0>, <Event 183:W:3-1>, <Event 164:W:4-0>, <Event 184:W:4-1>, <Event 165:W:0-0>, <Event 185:W:0-1>, <Event 166:W:1-0>, <Event 186:W:1-1>, <Event 167:W:2-0>, <Event 187:W:2-1>, <Event 168:W:3-0>, <Event 188:W:3-1>, <Event 169:W:4-0>, <Event 189:W:4-1>, <Event 170:W:0-0>, <Event 190:W:0-1>, <Event 171:W:1-0>, <Event 191:W:1-1>, <Event 172:W:2-0>, <Event 192:W:2-1>, <Event 173:W:3-0>, <Event 193:W:3-1>, <Event 174:W:4-0>, <Event 194:W:4-1>, <Event 175:W:0-0>, <Event 195:W:0-1>, <Event 176:W:1-0>, <Event 196:W:1-1>, <Event 177:W:2-0>, <Event 197:W:2-1>, <Event 178:W:3-0>, <Event 198:W:3-1>, <Event 179:W:4-0>, <Event 199:W:4-1>])'

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
