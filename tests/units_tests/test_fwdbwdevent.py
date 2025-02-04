'''
test FwdBwdEvent class
'''

import pytest
from perflowai.core.event import FwdBwdEvent, EventType

def test_FwdBwdEvent():
    # 测试type参数检查
    with pytest.raises(ValueError):
        FwdBwdEvent(id=1, type=EventType.COMM, name="Event1", timestamp=1672531200, duration=3600, stage_id=1, microbatch_id=2, chunk_id=3)
    
    # 测试初始化方法
    event = FwdBwdEvent(id=1, type=EventType.FWD, name="Event1", timestamp=1672531200, duration=3600, stage_id=1, microbatch_id=2, chunk_id=3)
    assert event.get_id() == 1
    assert event.get_type() == EventType.FWD
    assert event.get_name() == "Event1"
    assert event.get_timestamp() == 1672531200
    assert event.get_duration() == 3600
    assert event.get_stage_id() == 1
    assert event.get_microbatch_id() == 2
    assert event.get_chunk_id() == 3

