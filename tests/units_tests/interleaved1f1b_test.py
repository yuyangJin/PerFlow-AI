'''
test Interleaved1F1BGraph class
'''
import pytest

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.interleaved1f1b import Interleaved1F1BGraph
from perflowai.core.event import NoneTimestamp, EventType

def test_Interleaved1F1BGraphGraph():

    g = Interleaved1F1BGraph(4, 10, 2, cost_config = PipeCostConfig(
        fwd_time = 5478,
        bwd_time = 5806,
        wgt_time = 3534
    ))

    g.build_graph()
    
    assert g.get_event_types() == [EventType.FWD, EventType.BWD]
    assert g.get_nstages() == 4
    assert g.get_nchunks() == 2
    assert g.get_nmicrobatches() == 10

    g.check()

    for event in g.get_nodes().values():
        assert event.get_timestamp() == NoneTimestamp

