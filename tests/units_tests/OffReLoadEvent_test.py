'''
test OffReLoadEvent class
'''

import pytest
from perflowai.core import OffReLoadEvent, EventType, ResourceType

def test_OffReLoadEvent():
    event = OffReLoadEvent(id=1, type=EventType.OFFL, name="Event1", timestamp=1672531200, duration=3600, load_ratio=0.5, resource_type=ResourceType.G2C_PCIE, stage_id=1, microbatch_id=2, chunk_id=3)
    assert event.get_id() == 1
    assert event.get_type() == EventType.OFFL
    assert event.get_name() == "Event1"
    assert event.get_timestamp() == 1672531200
    assert event.get_duration() == 3600
    assert event.get_load_ratio() == 0.5
    assert event.get_resource_type() == ResourceType.G2C_PCIE
    assert event.get_stage_id() == 1
    assert event.get_microbatch_id() == 2
    assert event.get_chunk_id() == 3