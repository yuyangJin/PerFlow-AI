from __future__ import annotations
from .utils import logger
from typing import List, Dict, Any, Union
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
    
    def _ensure_list(self, v) -> List[Any]:
        if v is None:
            return [None]
        if isinstance(v, list):
            return v
        return [v]
    
    def _wrap_dict_leaves(self, d: Any) -> List[Any]:

        if not isinstance(d, dict):
            return self._ensure_list(d)
        
        out = {}
        for k, v in d.items():
            out[k] = self._wrap_dict_leaves(v)
        return out

    def _merge_values(self, acc, v):
        if acc is None:
            if isinstance(v, dict):
                return {k: self._merge_values(None, val) for k, val in v.items()}
            else:
                return self._ensure_list(v)

        if isinstance(acc, list):
            if isinstance(v, list):
                acc.extend(v)
            else:
                acc.append(v)
            return acc

        if isinstance(acc, dict):
            if not isinstance(v, dict):
                logger.error("MergeEvent: type mismatch (dict vs non-dict)")
                return acc
            all_keys = set(acc.keys()) | set(v.keys())
            for k in all_keys:
                acc[k] = self._merge_values(acc.get(k), v.get(k))
            return acc

        logger.error("MergeEvent: type mismatch (non-list/dict vs non-list/dict)")
        assert False, "unreachable"
    
    def _add_event(self, event: Union[Event, MergeEvent]):
        e_dict = event.to_dict()

        if not self.raw:
            self.raw = e_dict.copy()
            for key in self.merge_keys:
                val = self.raw.get(key, None)
                self.raw[key] = self._wrap_dict_leaves(val)

            return
        
        for key in self.merge_keys:
            old_val = self.raw.get(key, None)
            new_val = e_dict.get(key, None)
            self.raw[key] = self._merge_values(old_val, new_val)

    
    def add_events(self, events: List[Union[Event, MergeEvent]]):
        for event in events:
            self._add_event(event)

    def to_dict(self) -> Dict[str, Any]:
        return self.raw
