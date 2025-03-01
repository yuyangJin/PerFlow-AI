'''
test ZeroBubbleGraph class
'''
import pytest

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.zerobubble import ZeroBubbleGraph
from perflowai.core.event import NoneTimestamp, EventType

std_nodes_str = 'dict_values([<Event 1:F:0-0>, <Event 2:F:0-1>, <Event 3:F:1-0>, <Event 4:F:1-1>, <Event 5:F:2-0>, <Event 6:F:2-1>, <Event 7:F:3-0>, <Event 8:F:3-1>, <Event 9:F:4-0>, <Event 10:F:4-1>, <Event 11:F:0-0>, <Event 12:F:0-1>, <Event 13:F:1-0>, <Event 14:F:1-1>, <Event 15:F:2-0>, <Event 16:F:2-1>, <Event 17:F:3-0>, <Event 18:F:3-1>, <Event 19:F:4-0>, <Event 20:F:4-1>, <Event 21:F:0-0>, <Event 22:F:0-1>, <Event 23:F:1-0>, <Event 24:F:1-1>, <Event 25:F:2-0>, <Event 26:F:2-1>, <Event 27:F:3-0>, <Event 28:F:3-1>, <Event 29:F:4-0>, <Event 30:F:4-1>, <Event 31:F:0-0>, <Event 32:F:0-1>, <Event 33:F:1-0>, <Event 34:F:1-1>, <Event 35:F:2-0>, <Event 36:F:2-1>, <Event 37:F:3-0>, <Event 38:F:3-1>, <Event 39:F:4-0>, <Event 40:F:4-1>, <Event 41:B:0-0>, <Event 42:B:0-1>, <Event 43:B:1-0>, <Event 44:B:1-1>, <Event 45:B:2-0>, <Event 46:B:2-1>, <Event 47:B:3-0>, <Event 48:B:3-1>, <Event 49:B:4-0>, <Event 50:B:4-1>, <Event 51:B:0-0>, <Event 52:B:0-1>, <Event 53:B:1-0>, <Event 54:B:1-1>, <Event 55:B:2-0>, <Event 56:B:2-1>, <Event 57:B:3-0>, <Event 58:B:3-1>, <Event 59:B:4-0>, <Event 60:B:4-1>, <Event 61:B:0-0>, <Event 62:B:0-1>, <Event 63:B:1-0>, <Event 64:B:1-1>, <Event 65:B:2-0>, <Event 66:B:2-1>, <Event 67:B:3-0>, <Event 68:B:3-1>, <Event 69:B:4-0>, <Event 70:B:4-1>, <Event 71:B:0-0>, <Event 72:B:0-1>, <Event 73:B:1-0>, <Event 74:B:1-1>, <Event 75:B:2-0>, <Event 76:B:2-1>, <Event 77:B:3-0>, <Event 78:B:3-1>, <Event 79:B:4-0>, <Event 80:B:4-1>, <Event 81:W:0-0>, <Event 82:W:0-1>, <Event 83:W:1-0>, <Event 84:W:1-1>, <Event 85:W:2-0>, <Event 86:W:2-1>, <Event 87:W:3-0>, <Event 88:W:3-1>, <Event 89:W:4-0>, <Event 90:W:4-1>, <Event 91:W:0-0>, <Event 92:W:0-1>, <Event 93:W:1-0>, <Event 94:W:1-1>, <Event 95:W:2-0>, <Event 96:W:2-1>, <Event 97:W:3-0>, <Event 98:W:3-1>, <Event 99:W:4-0>, <Event 100:W:4-1>, <Event 101:W:0-0>, <Event 102:W:0-1>, <Event 103:W:1-0>, <Event 104:W:1-1>, <Event 105:W:2-0>, <Event 106:W:2-1>, <Event 107:W:3-0>, <Event 108:W:3-1>, <Event 109:W:4-0>, <Event 110:W:4-1>, <Event 111:W:0-0>, <Event 112:W:0-1>, <Event 113:W:1-0>, <Event 114:W:1-1>, <Event 115:W:2-0>, <Event 116:W:2-1>, <Event 117:W:3-0>, <Event 118:W:3-1>, <Event 119:W:4-0>, <Event 120:W:4-1>])'

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
