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
from .slp import SegmentedLinearPredictorCompressor as SLP
from .utils import logger

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

    def get_ts(self) -> int:
        """Get the timestamp of the event."""
        return self.raw.get("ts", 0)

    def get_dur(self) -> int:
        """Get the duration of the event."""
        return self.raw.get("dur", 0)

    def __getitem__(self, key: str) -> Any:
        """
        Allows dictionary-like access to the internal raw data (self.raw).
        
        Example: event_instance['pid']
        """
        return self.raw[key]

    def is_same_event(self, other: BaseEvent, debug: bool = False) -> bool:
        """Check if the event is the same as another event."""
        if not isinstance(other, BaseEvent):
            return False

        name1 = re.sub(r"\d+", "", self.get_name())
        name2 = re.sub(r"\d+", "", other.get_name())
        if name1!= name2:
            if debug:
                logger.debug(f"Name mismatch: {name1} vs {name2}")
            return False

        ignore_keys = {"ts", "dur", "id", "args", "name_pattern"}

        if self.raw.keys() - {"name_pattern"} != other.to_dict().keys() - {"name_pattern"}:
            if debug:
                logger.debug(f"Keys mismatch: {self.raw.keys()} vs {other.to_dict().keys()}")
            return False

        for key, val in self.raw.items():
            if key in ignore_keys or key == "name":
                continue

            other_val = other.to_dict().get(key, None)
            if val != other_val:
                if debug:
                    logger.debug(f"Value mismatch: {key} = {val} vs {key} = {other_val}")
                return False

        args1 = self.raw.get("args", {})
        args2 = other.to_dict().get("args", {})

        # if not self.is_same_structure(args1, args2):
        if not set(args1.keys()) == set(args2.keys()):
            if debug:
                logger.debug(f"Args mismatch: {set(args1.keys())} vs {set(args2.keys())}")
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

    def get_len(self) -> int:
        """Get the number of events in the merged event."""
        return len(self.raw.get("ts", []))

    def get_event_by_index(self, index: int) -> Event:
        """Get the event at the specified index."""

        content = {}
        name_pattern = ""
        for k, v in self.raw.items():
            if k == "name_pattern":
                continue
            elif k in self.merge_keys:
                if k == "args":
                    content[k] = {}
                    for arg_key, arg_val in v.items():
                        content[k][arg_key] = arg_val[index]
                elif k == "name":
                    content[k] = \
                            SLP.decompress_names(v, self.raw["name_pattern"], index)
                else:
                    content[k] = v[index]
            else:
                content[k] = v

        # content["name"] = self._format_name(name_pattern, content.get("name", []))

        return Event(content)

    def _parse_name(self, name: str):
        nums = [x for x in re.findall(r"\d+", name)]
        pattern = re.sub(r"\d+", "0", name)
        return pattern, nums

    def _format_name(self, pattern: str, nums: List[str]):
        it = iter(nums)
        return re.sub(r"0", lambda _: str(next(it)), pattern)

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
                        if key == "args":
                            args = e_dict.get("args", {})
                            new_args = {}
                            for k, v in args.items():
                                new_args[k] = [v]
                            self.raw[key] = new_args
                        else:
                            self.raw[key] = [e_dict.get(key, None)]

            return

        for key in self.merge_keys:
            if key in self.raw:
                if is_mergeevent:
                    if key == "args":
                        new_args = e_dict.get("args", {})
                        for k in self.raw[key].keys():
                            self.raw[key][k].extend(new_args.get(k, []))
                    else:
                        self.raw[key].extend(e_dict.get(key, []))
                else:
                    if key == "args":
                        args = e_dict.get("args", {})
                        for k in self.raw[key].keys():
                            self.raw[key][k].append(args.get(k, None))
                    else:
                        self.raw[key].append(e_dict.get(key, None))


    def add_events(self, events: List[Union[Event, MergeEvent]]):
        """Add a list of events to the merged event."""
        for event in events:
            self._add_event(event)

    def segmented_linear_predictor_compress(self):
        """Compress the merged event using segmented linear predictor."""
        # for key in ["ts", "dur", "id"]:
        #     if key in self.raw:
        #         val = self.raw.get(key, None)
        #         self.raw[key] = SegmentedLinearPredictorCompressor.compress(val)

        # if "name" in self.raw:
        #    val = self.raw.get("name", None)
        #    self.raw["name"] = [SegmentedLinearPredictorCompressor.compress(v) for v in val]

        # for key in ["args", "ts", "dur", "name", "id"]:
        #     if key in self.raw:
        #         val = self.raw.get(key, None)
        #         self.raw[key] = None
        #         if key == "ts":
        #             self.raw[key] = len(val)
        # names = self.raw.get("name", [])
        # result = SegmentedLinearPredictorCompressor.compress_names(names)
        # if result is None:
        #     if len(names[0]) > 0:
        #         nums = []
        #         for name in names:
        #             nums.append(name[0])
        #         self.raw["name"] = self._format_name(self.raw["name_pattern"], nums)
        #     else:
        #         self.raw["name"] = self.raw["name_pattern"]
        # else:
        #     self.raw["name"] = result

        self.raw["name"], self.raw["name_pattern"] = \
            SLP.compress_names(self.raw["name"], self.raw["name_pattern"])

        # self.raw["args"] = {k: [] for k in self.raw.get("args", {})}
        # self.raw["args"] = {}
        # self.raw["ts"] = [len(self.raw.get("ts", []))]
        # self.raw["dur"] = []

        # args = self.raw.get("args", {})
        # for key, val in args.items():
        #     if isinstance(val, list):
        #         logger.info(f"Compressing args {key}")
        #         args[key] = SegmentedLinearPredictorCompressor.compress_ids(val)

        # self.raw["args"] = args
        return

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> MergeEvent:
        assert "name_pattern" in raw, "name_pattern should be in raw"

        obj = cls.__new__(cls)
        obj.raw = raw.copy()
        obj.merge_keys = ["ts", "dur", "id", "args", "name"]

        return obj
