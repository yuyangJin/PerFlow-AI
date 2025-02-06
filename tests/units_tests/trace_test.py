'''
test Trace class
'''

import pytest
from perflowai.core.event import Event, FwdBwdEvent, EventType
from perflowai.core.trace import Trace

def test_Trace():
    event0 = Event(id=0, type=EventType.OPRT, name="Event0", timestamp=0, duration=10)
    event1 = FwdBwdEvent(id=1, type=EventType.FWD, name="Event1", timestamp=1, duration=20, stage_id=1, microbatch_id=3, chunk_id=5)
    event2 = FwdBwdEvent(id=2, type=EventType.BWD, name="Event2", timestamp=2, duration=30, stage_id=2, microbatch_id=4, chunk_id=6)
    event3 = Event(id=3, type=EventType.COMM, name="Event3", timestamp=3, duration=40)
    trace = Trace(ndevs = 2)
    assert trace.get_ndevs() == 2

    # 测试dev_id越界检查
    with pytest.raises(KeyError):
        trace.add_event(2, event0)
    with pytest.raises(KeyError):
        trace.get_events(-1)

    trace.add_event(0, event0)
    trace.add_event(1, event1)
    trace.add_event(1, event2)
    trace.add_event(1, event3)

    assert trace.get_events(0) == [event0]
    assert trace.get_events(1) == [event1, event2, event3]