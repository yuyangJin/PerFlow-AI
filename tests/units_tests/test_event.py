'''
test Event class
'''

import pytest
from perflowai.core.event import Event, EventType

def test_Event():
    # 测试type参数检查
    with pytest.raises(ValueError):
        Event(id=1, type="InvalidType", name="Event1", timestamp=1672531200, duration=3600)

    # 测试初始化方法
    event = Event(id=1, type=EventType.OPRT, name="Event1", timestamp=1672531200, duration=3600)
    assert event.get_id() == 1
    assert event.get_type() == EventType.OPRT
    assert event.get_name() == "Event1"
    assert event.get_timestamp() == 1672531200
    assert event.get_duration() == 3600

    # 测试设置时长方法
    event.set_duration(7200)
    assert event.get_duration() == 7200

    # 测试设置时长方法时传入无效输入
    with pytest.raises(TypeError):
        event.set_duration("invalid_duration")