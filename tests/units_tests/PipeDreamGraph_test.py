'''
test PipeDreamGraph class
'''
import pytest

from perflowai.parallel.pipeline_parallel import PipeCostConfig, PipeDreamGraph
from perflowai.core import NoneTimestamp, EventType

def test_PipeDreamGraph():

    with pytest.raises(AssertionError):
        g = PipeDreamGraph(4, 10, 2, cost_config = PipeCostConfig(
            fwd_time = 5478,
            bwd_time = 5806,
            wgt_time = 3534
        ))

    g = PipeDreamGraph(4, 10, 1, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
    ))

    g.build_graph()
    
    assert g.get_event_types() == [EventType.FWD, EventType.BWD]
    assert g.get_nstages() == 4
    assert g.get_nchunks() == 1
    assert g.get_nmicrobatches() == 10

    g.check()

    for event in g.get_nodes().values():
        assert event.get_timestamp() == NoneTimestamp

