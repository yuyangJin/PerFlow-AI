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
import numpy as np
from .slp import SegmentedLinearPredictorCompressor as SLP
from .utils import logger, to_json_safe
from ._compat import asizeof

def is_same_event(e1: Union[Event, MergeEvent], e2: Union[Event, MergeEvent], debug: bool = False) -> bool:
    """
    Compare two events for structural similarity.
    """

    if isinstance(e1, (KernelEvent, MergeKernelEvent)) and isinstance(e2, (KernelEvent, MergeKernelEvent)):
        name1 = re.sub(r"\d+", "", e1.get_name())
        name2 = re.sub(r"\d+", "", e2.get_name())
        if name1 != name2:
            if debug:
                logger.debug("Name mismatch: %s vs %s", name1, name2)
            return False

        if e1.cat != e2.cat:
            if debug:
                logger.debug("Category mismatch: %s vs %s", e1.cat, e2.cat)
            return False

        if e1.args is not None or e2.args is not None:
            if not (e1.args is not None and e2.args is not None):
                if debug:
                    logger.debug("One event has args while the other doesn't")
                return False

            if not set(e1.args.keys()) == set(e2.args.keys()):
                if debug:
                    logger.debug("Argument keys mismatch: %s vs %s", e1.args.keys(), e2.args.keys())
                return False

        return True

    if isinstance(e1, (Event, MergeEvent)) and isinstance(e2, (Event, MergeEvent)):
        name1 = re.sub(r"\d+", "", e1.get_name())
        name2 = re.sub(r"\d+", "", e2.get_name())
        if name1 != name2:
            if debug:
                logger.debug("Name mismatch: %s vs %s", name1, name2)
            return False

        if e1.cat != e2.cat:
            if debug:
                logger.debug("Category mismatch: %s vs %s", e1.cat, e2.cat)
            return False

        if e1.bp != e2.bp:
            if debug:
                logger.debug("Branchpoint mismatch: %s vs %s", e1.bp, e2.bp)
            return False

        if e1.s != e2.s:
            if debug:
                logger.debug("Stack depth mismatch: %s vs %s", e1.s, e2.s)
            return False

        if e1.args is not None or e2.args is not None:
            if not (e1.args is not None and e2.args is not None):
                if debug:
                    logger.debug("One event has args while the other doesn't")
                return False

            if not set(e1.args.keys()) == set(e2.args.keys()):
                if debug:
                    logger.debug("Argument keys mismatch: %s vs %s", e1.args.keys(), e2.args.keys())
                return False

        return True
    
    return False


class Event:
    """
    Standard event implementation.
    """
    __slots__ = ("name", "ts", "cat", "args", "dur", "id", "bp", "s")

    def __init__(self, raw: Dict[str, Any] | None = None):
        # 必定存在的字段
        self.name = raw.get("name", "unknown") if raw else "unknown"
        self.ts = raw.get("ts", 0) if raw else 0
        self.cat = raw.get("cat", None) if raw else None

        # 可选字段
        self.args = raw.get("args", None) if raw else None
        self.dur  = raw.get("dur", None) if raw else None
        self.id   = raw.get("id", None) if raw else None
        self.bp   = raw.get("bp", None) if raw else None
        self.s    = raw.get("s", None) if raw else None

    def get_name(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        res = {}
        res["name"] = self.name
        res["ts"] = self.ts
        if self.cat is not None:
            res["cat"] = self.cat
        if self.args is not None:
            res["args"] = self.args
        if self.dur is not None:
            res["dur"] = self.dur
        if self.id is not None:
            res["id"] = self.id
        if self.bp is not None:
            res["bp"] = self.bp
        if self.s is not None:
            res["s"] = self.s
        return res

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> Event:
        return cls(raw)

class MergeEvent:
    """
    Merged event representation.
    """
    # __slots__ = ("ts", "cat", "args", "dur", "id", "bp", "s",
    #              "name_pattern", "name_nums")

    def __init__(self, events):

        self.ts = []
        self.cat: str = None
        self.args: Dict[str, List[Any]] = None
        self.dur = []
        self.id = []
        self.bp: str = None
        self.s: str = None

        self.name_pattern: str = None
        self.name_nums: List[str] = []

        if events:
            self.add_events(events)

    def get_name(self) -> str:
        return self.name_pattern

    def get_len(self) -> int:
        """Get the number of events in the merged event."""
        return len(self.ts)

    def get_event_by_index(self, index: int) -> Event:
        """Get the event at the specified index."""

        event = Event()
        event.ts = SLP.decompress_linear_segment(self.ts, index)
        event.name = SLP.decompress_names(self.name_nums, self.name_pattern, index)
        event.cat = self.cat
        event.args = SLP.decompress_same_args(self.args, index)
        event.dur = SLP.decompress_linear_segment(self.dur, index)
        event.id = SLP.decompress_ids(self.id, index)
        event.bp = self.bp
        event.s = self.s

        return event

    def _parse_name(self, name: str):
        """
        Parse name into pattern and nums based on observed logic:
        - Each leading '0' in a digit sequence is extracted as integer 0.
        - The remaining non-zero part of the sequence is extracted as a single integer.
        - In the pattern, '0' is used as a placeholder for every number extracted.
        """
        nums = []
        pattern_parts = []
        last_idx = 0

        # 查找所有连续的数字片段
        for match in re.finditer(r"\d+", name):
            start, end = match.start(), match.end()
            #保留非数字部分的文本
            pattern_parts.append(name[last_idx:start])
            
            digit_seq = match.group()
            i = 0
            # 处理该数字片段
            while i < len(digit_seq):
                if digit_seq[i] == '0':
                    # 如果是0，当作独立的数字0处理
                    nums.append(0)
                    pattern_parts.append('0') # 占位符
                    i += 1
                else:
                    # 如果是非0数字，剩下的部分作为一个整体整数处理
                    rest = digit_seq[i:]
                    nums.append(int(rest))
                    pattern_parts.append('0') # 占位符
                    break # 这一段数字处理完毕

            last_idx = end

        pattern_parts.append(name[last_idx:])
        pattern = ''.join(pattern_parts)
        return pattern, nums

    def _format_name(self, pattern: str, nums: List[str]):
        it = iter(nums)
        return re.sub(r"0", lambda _: str(next(it)), pattern)
    
    def _add_single_args(self, dst: dict, src: dict):
        for k, v in src.items():
            if isinstance(v, dict):
                if k not in dst:
                    dst[k] = {}
                self._add_single_args(dst[k], v)
            elif isinstance(v, list):
                if k not in dst:
                    dst[k] = [[] for _ in v]  # 每个元素都独立收集
                for i, val in enumerate(v):
                    dst[k][i].append(val)
            else:
                if k not in dst:
                    dst[k] = []
                dst[k].append(v)

    def _add_single_event(self, e: Event):
        pat, nums = self._parse_name(e.name)

        if self.name_pattern is None:
            self.name_pattern = pat

        self.name_nums.append(nums)

        self.ts.append(e.ts)
        self.cat = e.cat
        if e.dur is not None:
            self.dur.append(e.dur)
        if e.id is not None:
            self.id.append(e.id)
        self.bp = e.bp
        self.s = e.s

        if e.args is not None:
            if self.args is None:
                self.args = {}
                self._add_single_args(self.args, e.args)
            else:
                self._add_single_args(self.args, e.args)

    def _add_merged_event(self, e: MergeEvent):

        self.ts.extend(e.ts)
        self.cat = e.cat
        self.dur.extend(e.dur)
        self.id.extend(e.id)
        self.bp = e.bp
        self.s = e.s

        self.name_nums.extend(e.name_nums)
        self.name_pattern = e.name_pattern

        if e.args is not None:
            if not self.args:
                self.args = e.args.copy()
                return

            for k in self.args.keys():
                self.args[k].extend(e.args[k])


    def add_event(self, event: Union[Event, MergeEvent]):
        """Add an event to the merged event."""
        if isinstance(event, Event):
            self._add_single_event(event)
        else:
            print(type(event))
            self._add_merged_event(event)


    def add_events(self, events: List[Union[Event, MergeEvent]]):
        """Add a list of events to the merged event."""
        for event in events:
            self.add_event(event)

    def compress_values(self):
        """Compress the merged event."""

        self.name_nums, self.name_pattern = \
            SLP.compress_names(self.name_nums, self.name_pattern)

        if self.args is not None:
            SLP.compress_same_args(self.args)

        self.ts = SLP.segment_linear_compress(self.ts)

        if len(self.dur) > 0:
            self.dur = SLP.segment_linear_compress(self.dur)

        if len(self.id) > 0:
            self.id = SLP.compress_ids(self.id)

        return


    def to_dict(self) -> Dict[str, Any]:
        res = {}
        res["name_pattern"] = self.name_pattern
        res["name"] = to_json_safe(self.name_nums)
        res["ts"] = to_json_safe(self.ts)

        if len(self.dur) > 0:
            res["dur"] = to_json_safe(self.dur)
        if self.cat is not None:
            res["cat"] = self.cat
        if self.args:
            res["args"] = to_json_safe(self.args)
        if len(self.id) > 0:
            res["id"] = to_json_safe(self.id)
        if self.bp is not None:
            res["bp"] = self.bp
        if self.s is not None:
            res["s"] = self.s
        return res

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> MergeEvent:
        assert "name_pattern" in raw, "name_pattern should be in raw"

        obj = cls.__new__(cls)
        obj.name_pattern = raw["name_pattern"]
        obj.name_nums = raw.get("name", [])
        for i, v in enumerate(obj.name_nums):
            if isinstance(v, list):
                obj.name_nums[i] = np.asarray(v, dtype=np.int64)
        obj.ts = np.asarray(raw.get("ts", []), dtype=np.int64)
        obj.dur = np.asarray(raw.get("dur", []), dtype=np.int64)
        obj.cat = raw.get("cat", None)
        obj.args = raw.get("args", None)
        raw_id = raw.get("id", [])
        if isinstance(raw_id, dict):
            values = raw_id.get("values", [])
            if isinstance(values, list):
                raw_id["values"] = np.asarray(values, dtype=np.int64)
            obj.id = raw_id
        elif isinstance(raw_id, list) and len(raw_id) == 2 and isinstance(raw_id[0], str):
            values = raw_id[1]
            if isinstance(values, list):
                values = np.asarray(values, dtype=np.int64)
            obj.id = [raw_id[0], values]
        else:
            obj.id = np.asarray(raw_id, dtype=np.int64)
        obj.bp = raw.get("bp", None)
        obj.s = raw.get("s", None)

        return obj
    
class KernelEvent:
    """
    Standard event implementation.
    """
    __slots__ = ("name", "ts", "cat", "pid", "tid", "ph", "args", "dur")

    def __init__(self, e: Event, pid: int, tid: int, ph: str):
        # 必定存在的字段
        self.name = e.name
        self.ts = e.ts
        self.cat = e.cat

        self.pid = pid
        self.tid = tid
        self.ph = ph

        # 可选字段
        self.args = e.args
        self.dur  = e.dur
        assert e.id is None, "KernelEvent should not have id"
        assert e.bp is None, "KernelEvent should not have bp"
        assert e.s is None, "KernelEvent should not have s"

    def get_name(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        res = {}
        res["name"] = self.name
        res["ts"] = self.ts
        if self.cat is not None:
            res["cat"] = self.cat
        if self.args is not None:
            res["args"] = self.args
        if self.dur is not None:
            res["dur"] = self.dur

        res["pid"] = self.pid
        res["tid"] = self.tid
        res["ph"] = self.ph

        return res

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> Event:
        return cls(raw)
    
class MergeKernelEvent:
    """
    Merged event representation.
    """
    __slots__ = ("ts", "cat", "pid", "tid", "ph", "args", "dur",
                 "name_pattern", "name_nums")

    def __init__(self, events: List[KernelEvent]):

        self.ts = []
        self.cat: str = None
        self.pid: List[str] = []
        self.tid: List[str] = []
        self.ph: List[str] = []
        self.args: Dict[str, List[Any]] = None
        self.dur = []

        self.name_pattern: str = None
        self.name_nums: List[str] = []

        if events:
            self.add_events(events)

    def get_name(self) -> str:
        return self.name_pattern

    def get_len(self) -> int:
        """Get the number of events in the merged event."""
        return len(self.ts)

    def get_event_by_index(self, index: int) -> KernelEvent:
        """Get the event at the specified index."""

        event = KernelEvent(Event(), self.pid[index], self.tid[index], self.ph[index])
        event.ts = SLP.decompress_linear_segment(self.ts, index)
        event.name = SLP.decompress_names(self.name_nums, self.name_pattern, index)
        event.cat = self.cat
        event.args = SLP.decompress_same_args(self.args, index)
        event.dur = SLP.decompress_linear_segment(self.dur, index)

        return event

    def _parse_name(self, name: str):
        """
        Parse name into pattern and nums based on observed logic:
        - Each leading '0' in a digit sequence is extracted as integer 0.
        - The remaining non-zero part of the sequence is extracted as a single integer.
        - In the pattern, '0' is used as a placeholder for every number extracted.
        """
        nums = []
        pattern_parts = []
        last_idx = 0

        # 查找所有连续的数字片段
        for match in re.finditer(r"\d+", name):
            start, end = match.start(), match.end()
            #保留非数字部分的文本
            pattern_parts.append(name[last_idx:start])
            
            digit_seq = match.group()
            i = 0
            # 处理该数字片段
            while i < len(digit_seq):
                if digit_seq[i] == '0':
                    # 如果是0，当作独立的数字0处理
                    nums.append(0)
                    pattern_parts.append('0') # 占位符
                    i += 1
                else:
                    # 如果是非0数字，剩下的部分作为一个整体整数处理
                    rest = digit_seq[i:]
                    nums.append(int(rest))
                    pattern_parts.append('0') # 占位符
                    break # 这一段数字处理完毕

            last_idx = end

        pattern_parts.append(name[last_idx:])
        pattern = ''.join(pattern_parts)
        return pattern, nums

    def _format_name(self, pattern: str, nums: List[str]):
        it = iter(nums)
        return re.sub(r"0", lambda _: str(next(it)), pattern)
    
    def _add_single_args(self, dst: dict, src: dict):
        for k, v in src.items():
            if isinstance(v, dict):
                if k not in dst:
                    dst[k] = {}
                self._add_single_args(dst[k], v)
            elif isinstance(v, list):
                if k not in dst:
                    dst[k] = [[] for _ in v]  # 每个元素都独立收集
                for i, val in enumerate(v):
                    dst[k][i].append(val)
            else:
                if k not in dst:
                    dst[k] = []
                dst[k].append(v)

    def _add_single_event(self, e: Event):
        pat, nums = self._parse_name(e.name)

        if self.name_pattern is None:
            self.name_pattern = pat

        self.name_nums.append(nums)

        self.ts.append(e.ts)
        self.pid.append(e.pid)
        self.tid.append(e.tid)
        self.ph.append(e.ph)
        self.cat = e.cat
        if e.dur is not None:
            self.dur.append(e.dur)

        if e.args is not None:
            if self.args is None:
                self.args = {}
                self._add_single_args(self.args, e.args)
            else:
                self._add_single_args(self.args, e.args)

    def _add_merged_event(self, e: MergeEvent):

        self.ts.extend(e.ts)
        self.cat = e.cat
        self.dur.extend(e.dur)

        self.name_nums.extend(e.name_nums)
        self.name_pattern = e.name_pattern

        if e.args is not None:
            if not self.args:
                self.args = e.args.copy()
                return

            for k in self.args.keys():
                self.args[k].extend(e.args[k])


    def add_event(self, event: Union[Event, MergeEvent]):
        """Add an event to the merged event."""
        if isinstance(event, KernelEvent):
            self._add_single_event(event)
        else:
            self._add_merged_event(event)


    def add_events(self, events: List[Union[Event, MergeEvent]]):
        """Add a list of events to the merged event."""
        for event in events:
            self.add_event(event)

    def compress_values(self):
        """Compress the merged event."""

        self.name_nums, self.name_pattern = \
            SLP.compress_names(self.name_nums, self.name_pattern)

        if self.args is not None:
            SLP.compress_same_args(self.args)

        self.ts = SLP.segment_linear_compress(self.ts)

        if len(self.dur) > 0:
            self.dur = SLP.segment_linear_compress(self.dur)

        return


    def to_dict(self) -> Dict[str, Any]:
        res = {}
        res["name_pattern"] = self.name_pattern
        res["name"] = to_json_safe(self.name_nums)
        res["ts"] = to_json_safe(self.ts)
        res["pid"] = self.pid
        res["tid"] = self.tid
        res["ph"] = self.ph

        if len(self.dur) > 0:
            res["dur"] = to_json_safe(self.dur)
        if self.cat is not None:
            res["cat"] = self.cat
        if self.args:
            res["args"] = to_json_safe(self.args)

        return res

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> MergeKernelEvent:
        assert "name_pattern" in raw, "name_pattern should be in raw"

        obj = cls.__new__(cls)
        obj.name_pattern = raw["name_pattern"]
        obj.name_nums = raw.get("name", [])
        for i, v in enumerate(obj.name_nums):
            if isinstance(v, list):
                obj.name_nums[i] = np.asarray(v, dtype=np.int64)
        obj.ts = np.asarray(raw.get("ts", []), dtype=np.int64)
        obj.dur = np.asarray(raw.get("dur", []), dtype=np.int64)
        obj.cat = raw.get("cat", None)
        obj.args = raw.get("args", None)
        obj.pid = raw.get("pid", [])
        obj.tid = raw.get("tid", [])
        obj.ph = raw.get("ph", [])

        return obj




def memory_breakdown_templates(templates: List[MergeEvent]):
    """
    返回 templates 中各个 key 占用的总内存
    """
    keys = ["name_pattern", "name_nums", "ts", "dur", "id", "cat", "args", "bp", "s", "pid", "tid", "ph"]
    sizes = {k: 0 for k in keys}

    for tmpl in templates:
        if isinstance(tmpl, MergeEvent):
            sizes["name_pattern"] += asizeof.asizeof(tmpl.name_pattern)
            sizes["name_nums"] += asizeof.asizeof(tmpl.name_nums)
            sizes["ts"] += asizeof.asizeof(tmpl.ts)
            sizes["dur"] += asizeof.asizeof(tmpl.dur)
            sizes["id"] += asizeof.asizeof(tmpl.id)
            sizes["cat"] += asizeof.asizeof(tmpl.cat)
            sizes["args"] += asizeof.asizeof(tmpl.args)
            sizes["bp"] += asizeof.asizeof(tmpl.bp)
            sizes["s"] += asizeof.asizeof(tmpl.s)
        elif isinstance(tmpl, MergeKernelEvent):
            sizes["name_pattern"] += asizeof.asizeof(tmpl.name_pattern)
            sizes["name_nums"] += asizeof.asizeof(tmpl.name_nums)
            sizes["ts"] += asizeof.asizeof(tmpl.ts)
            sizes["dur"] += asizeof.asizeof(tmpl.dur)
            sizes["cat"] += asizeof.asizeof(tmpl.cat)
            sizes["args"] += asizeof.asizeof(tmpl.args)
            sizes["pid"] += asizeof.asizeof(tmpl.pid)
            sizes["tid"] += asizeof.asizeof(tmpl.tid)
            sizes["ph"] += asizeof.asizeof(tmpl.ph)

    total = sum(sizes.values())
    sizes["total"] = total
    return sizes
