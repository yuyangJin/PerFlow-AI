from __future__ import annotations
from .utils import logger
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import re

class BaseEvent(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_ts(self) -> float:
        pass

    @abstractmethod
    def get_dur(self) -> float:
        pass

    @abstractmethod
    def is_merged(self) -> bool:
        pass

    @abstractmethod
    def is_same_event(self, other: BaseEvent) -> bool:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class Event(BaseEvent):
    def __init__(self, raw: Dict[str, Any] = {}):
        self.raw: Dict[str, Any] = raw

    def get_name(self) -> str:
        return self.raw.get("name", "unknown")
    
    def get_ts(self) -> float:
        return self.raw.get("ts", 0.0)
    
    def get_dur(self) -> float:
        return self.raw.get("dur", 0.0)
    
    def is_merged(self) -> bool:
        return False
    
    def is_same_event(self, other: BaseEvent) -> bool:
        if not isinstance(other, BaseEvent):
            return False

        ignore_keys = {"ts", "dur", "id", "args"}

        for key, val in self.raw.items():
            if key in ignore_keys or key == "name":
                continue

            other_val = other.raw.get(key, None)
            if val != other_val:
                return False

        name1 = re.sub(r"\d+", "", self.get_name())
        name2 = re.sub(r"\d+", "", other.get_name())
        return name1 == name2

    def to_dict(self) -> Dict[str, Any]:
        return self.raw


class MergeEvent(Event):
    def __init__(self, events: List[BaseEvent] = [], merge_keys: List[str] = ["ts", "dur", "id", "args"]):
        self.merge_keys = merge_keys

        self.raw = {}

        if events:
            self.add_events(events)

    def is_merged(self) -> bool:
        return True
    
    def add_events(self, events: List[BaseEvent]):
        # TODO: 这里默认输入的events都是Event，但可能还需要考虑是否需要支持MergeEvent
        # TODO: 检查这些event是否可以合并，如果有问题，使用logger警告
        assert len(events) > 0, "events should not be empty"

        if not self.raw:
            self.raw = events[0].to_dict().copy()


        for key in self.merge_keys:
            self.raw[key] = [e.to_dict().get(key) for e in events]

    def to_dict(self) -> Dict[str, Any]:
        return self.raw
