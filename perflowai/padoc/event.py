"""
Core event data structures for PADOC trace representation.

This module defines the base event abstraction (`BaseEvent`), the standard
event implementation (`Event`), and the merged-event representation
(`MergeEvent`) used for template-based compression. These classes provide
a unified interface for accessing event metadata, comparing event patterns,
serializing/deserializing event dictionaries, and grouping structurally
similar events into mergeable templates.
"""

from __future__ import annotations
from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod
import re

class BaseEvent(ABC):
    """
    Abstract base class for all trace events.

    This class defines the minimal interface that all event types must
    implement.
    """

    def __init__(self, raw: Dict[str, Any] | None = None):
        self.raw: Dict[str, Any] = raw or {}

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the event."""
        return "unknown"

    def get_ts(self) -> float:
        """Get the timestamp of the event."""
        return self.raw.get("ts", 0.0)

    def get_dur(self) -> float:
        """Get the duration of the event."""
        return self.raw.get("dur", 0.0)

    def is_same_structure(self, a: Any, b: Any) -> bool:
        """Check if two values are the same structure."""
        if not isinstance(a, type(b)):
            return False
        if isinstance(a, (list, tuple)):
            if len(a)!= len(b):
                return False
            return all(self.is_same_structure(x, y) for x, y in zip(a, b))
        elif isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self.is_same_structure(a[k], b[k]) for k in a)

        return True

    def is_same_event(self, other: BaseEvent) -> bool:
        """Check if the event is the same as another event."""
        if not isinstance(other, BaseEvent):
            return False

        name1 = re.sub(r"\d+", "", self.get_name())
        name2 = re.sub(r"\d+", "", other.get_name())
        if name1!= name2:
            return False

        ignore_keys = {"ts", "dur", "id", "args"}

        for key, val in self.raw.items():
            if key in ignore_keys or key == "name":
                continue

            other_val = other.to_dict().get(key, None)
            if val != other_val:
                return False

        args1 = self.raw.get("args", {})
        args2 = other.to_dict().get("args", {})

        # if not self.is_same_structure(args1, args2):
        if not set(args1.keys()) == set(args2.keys()):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event to a dictionary."""
        return self.raw

    @abstractmethod
    def is_merged(self) -> bool:
        """Check if the event is a merged event."""
        return True

    @classmethod
    @abstractmethod
    def from_dict(cls, raw: Dict[str, Any]) -> BaseEvent:
        """Deserialize the event from a dictionary."""
        return cls()


class Event(BaseEvent):
    """
    Standard event implementation.
    """

    def get_name(self) -> str:
        return self.raw.get("name", "unknown")

    def is_merged(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return self.raw

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> Event:
        return cls(raw)


class MergeEvent(BaseEvent):
    """
    Merged event representation.

    This class represents a collection of events that are structurally
    similar and can be merged into a single event. The events are grouped
    based on a set of keys (`merge_keys`) that are used to compare and merge
    the events. The resulting merged event is represented as a dictionary
    with the merged values for each key.
    """

    def __init__(self, events: List[BaseEvent] | None = None, merge_keys: List[str] | None = None):
        super().__init__()
        self.merge_keys = merge_keys or ["ts", "dur", "id", "args", "name"]

        if events:
            self.add_events(events)

    def get_name(self) -> str:
        return self.raw.get("name_pattern", "unknown")

    def is_merged(self) -> bool:
        return True

    def get_event_by_index(self, index: int) -> Event:
        """Get the event at the specified index."""

        def _extract_indexed(value):

            if self._is_basic_value(value):
                return value

            if isinstance(value, list):
                if isinstance(value[0], list):
                    return [_extract_indexed(v) for v in value]
                else:
                    return value[index]
            elif isinstance(value, dict):
                return {k: _extract_indexed(v) for k, v in value.items()}
            else:
                return value

        content = {}
        name_pattern = ""
        for k, v in self.raw.items():
            if k == "name_pattern":
                name_pattern = v
            elif k in self.merge_keys:
                content[k] = _extract_indexed(v)
            else:
                content[k] = v

        content["name"] = self._format_name(name_pattern, content.get("name", []))

        return Event(content)

    def _parse_name(self, name: str):
        nums = [int(x) for x in re.findall(r"\d+", name)]
        pattern = re.sub(r"\d+", "0", name)
        return pattern, nums

    def _format_name(self, pattern: str, nums: List[int]):
        it = iter(nums)
        return re.sub(r"0", lambda _: str(next(it)), pattern)

    def _is_basic_value(self, val: Any) -> bool:
        if isinstance(val, list):
            if not val:
                return True
        elif isinstance(val, dict):
            if not val:
                return True
        elif isinstance(val, tuple):
            if not val:
                return True
        else:
            return True

        return False

    def _wrap_to_list(self, val: Any) -> Any:
        if self._is_basic_value(val):
            return [val]

        if isinstance(val, list):
            if len(val) == 0:
                return [[]]
            return [self._wrap_to_list(v) for v in val]

        if isinstance(val, dict):
            return {k: self._wrap_to_list(v) for k, v in val.items()}

        return [val]

    def _wrap_add_list(self, val1: Any, val2: Any) -> Any:
        if isinstance(val1, list) and self._is_basic_value(val2):
            val1.append(val2)
            return val1
        if isinstance(val1, list) and isinstance(val2, list):
            return [self._wrap_add_list(v1, v2) for v1, v2 in zip(val1, val2)]
        if isinstance(val1, dict) and isinstance(val2, dict):
            return {k: self._wrap_add_list(v1, v2) for k, v1, v2 \
                    in zip(val1.keys(), val1.values(), val2.values())}

        val1.append(val2)
        return val1

    def _wrap_extend_list(self, val1: Any, val2: Any) -> Any:
        if isinstance(val1, list) and isinstance(val2, list):
            if self._is_basic_value(val1[0]) and self._is_basic_value(val2[0]):
                return val1 + val2

            if isinstance(val1[0], (dict, list)):
                return [self._wrap_extend_list(v1, v2) for v1, v2 in zip(val1, val2)]

            return val1 + val2
        if isinstance(val1, dict) and isinstance(val2, dict):
            return {k: self._wrap_extend_list(v1, v2) for k, v1, v2 \
                    in zip(val1.keys(), val1.values(), val2.values())}

        assert False, "unreachable"

    def _add_event(self, event: Union[Event, MergeEvent]):
        e_dict = event.to_dict().copy()
        is_mergeevent = isinstance(event, MergeEvent)
        pattern = ""

        if not is_mergeevent:
            name = e_dict.get("name", None)
            assert name is not None, "name should not be None"
            pattern, nums = self._parse_name(name)
            e_dict["name"] = nums

        if not self.raw:
            self.raw = e_dict.copy()
            if "name_pattern" not in self.raw:
                self.raw["name_pattern"] = pattern

            if not is_mergeevent:
                for key in self.merge_keys:
                    if key in self.raw:
                        val = self.raw.get(key, None)
                        self.raw[key] = self._wrap_to_list(val)

            return

        for key in self.merge_keys:
            if key in self.raw:
                old_val = self.raw.get(key, None)
                new_val = e_dict.get(key, None)
                if is_mergeevent:
                    self.raw[key] = self._wrap_extend_list(old_val, new_val)
                else:
                    self.raw[key] = self._wrap_add_list(old_val, new_val)


    def add_events(self, events: List[Union[Event, MergeEvent]]):
        """Add a list of events to the merged event."""
        for event in events:
            self._add_event(event)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> MergeEvent:
        assert "name_pattern" in raw, "name_pattern should be in raw"

        obj = cls.__new__(cls)
        obj.raw = raw.copy()
        obj.merge_keys = ["ts", "dur", "id", "args", "name"]

        return obj
