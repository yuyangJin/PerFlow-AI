from __future__ import annotations
from .utils import logger
from typing import List, Dict, Any
from abc import ABC, abstractmethod

class BaseEvent(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def is_merged(self) -> bool:
        pass

    @abstractmethod
    def to_dict(self):
        pass


class Event(BaseEvent):
    def __init__(self, raw: Dict[str, Any]):
        self.raw: Dict[str, Any] = raw

    def get_name(self) -> str:
        return self.raw.get("name", "unknown")
    
    def is_merged(self) -> bool:
        return False

    def to_dict(self):
        return self.raw


class MergeEvent(Event):
    def __init__(self, events: List[BaseEvent], merge_keys: List[str] = ["ts", "dur", "id", "args"]):
        # TODO: 这里默认输入的events都是Event，但可能还需要考虑是否需要支持MergeEvent
        assert len(events) > 1, "MergeEvent requires at least two event."
        self.merge_keys = merge_keys

        base = events[0].raw.copy()
        # TODO: 检查这些event是否可以合并，如果有问题，使用logger警告
        for key in merge_keys:
            base[key] = [e.raw.get(key) for e in events]

        # TODO: 可以自动识别哪些字段是连续型的（如ts、duration），后续动态生成 merge_keys，并使用logger信息
        self.raw = base

    def is_merged(self) -> bool:
        return True
    
    def add_events(self, events: List[BaseEvent]):
        # TODO: 这里默认输入的events都是Event，但可能还需要考虑是否需要支持MergeEvent
        # TODO: 检查这些event是否可以合并，如果有问题，使用logger警告
        for key in self.merge_keys:
            self.raw[key].extend([e.raw.get(key) for e in events])

    def to_dict(self):
        return self.raw
