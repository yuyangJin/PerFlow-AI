"""
Node structures for trace trees used in compression and reconstruction.

This module defines three node types:
- Node: a concrete tree node with real events and children.
- TemplateNode: a merged representation of multiple identical nodes, storing
  merged events and merged subtrees.
- RefNode: a lightweight reference into a TemplateNode at a given index.

All node types implement the BaseNode interface, which standardizes event
access, traversal, comparison, and (de)serialization. This abstraction allows
the compressor and decompressor to operate on both raw and compressed trees
using a unified API.
"""


from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Optional, Generator
import numpy as np
from .event import Event, MergeEvent, KernelEvent, is_same_event
from .utils import logger
from collections import defaultdict


class BaseNode(ABC):
    """Abstract base class for all node types in the trace tree.

    This interface defines a unified API for concrete node implementations
    (Node, TemplateNode, RefNode). Any subclass must implement event access,
    structural access, serialization, and comparison.

    Subclasses:
        - Node: a concrete event node in the trace tree.
        - TemplateNode: a merged/compressed representation of multiple nodes.
        - RefNode: a reference to a TemplateNode at a specific index.
    """

    @abstractmethod
    def get_events(self):
        """Get a list of events in this node."""
        return []

    @abstractmethod
    def get_all_events(self):
        """Get a list of all events in this node and its children."""
        return []

    @abstractmethod
    def get_children(self) -> List[BaseNode]:
        """Get a list of children of this node."""
        return []

    @abstractmethod
    def set_children(self, children: List[BaseNode]):
        """Set the list of children of this node."""
        return

    @abstractmethod
    def add_child(self, child: BaseNode):
        """Add a child to this node."""
        return

class Node(BaseNode):
    """A concrete tree node that stores actual Event objects.

    A Node contains:
        - A list of Event objects.
        - A list of child nodes (Node or RefNode).

    It represents the uncompressed, regular trace tree structure.
    """

    def __init__(self, events: List[Event] | None = None):
        self.events: List[Event] = events or []
        self.children: List[Union[Node, RefNode, GroupRefNode]] = []

    def get_events(self) -> List[Event]:
        return self.events

    def get_all_events(self) -> List[Event]:
        all_events = self.events.copy()
        for child in self.children:
            all_events.extend(child.get_all_events())
        return all_events

    def get_node_count(self) -> int:
        return 1

    def add_events(self, events: List[Event]):
        """Add a list of events to this node."""
        self.events.extend(events)

    def get_children(self) -> List[Union[Node, RefNode, GroupRefNode]]:
        return self.children

    def set_children(self, children: List[BaseNode]):
        self.children = children

    def add_child(self, child: BaseNode):
        self.children.append(child)

    def get_first_event_name(self):
        if len(self.events) > 0:
            return self.events[0].get_name()
        if len(self.children) > 0:
            return self.children[0].get_first_event_name()

        return "none"

    def show(self, indent: int = 0):
        prefix = " " * indent

        if self.events:
            names = [e.get_name() for e in self.events]
            print(f"{prefix}Node: {names}")
        else:
            print(f"{prefix}Node: <empty>")

        for child in self.children:
            child.show(indent + 2)

class CPUNode:

    __slots__ = ["template_index", "instance_index", "children", "slots"]

    def __init__(self, template_index: int, instance_index: int):
        self.template_index = template_index
        self.instance_index = instance_index
        self.children = None
        self.slots = None

    def add_child(self, child):
        if self.children is None:
            self.children = []
        self.children.append(child)

    def get_children(self):
        if self.children is None:
            return []
        return self.children

    def is_kernel_node(self):
        return False

    def _chain_visitors(self, visitors):
        for it in visitors:
            yield from it


    def event_visitor(self, event_templates: List[MergeEvent], include_kernels: bool = False):
        if self.template_index != -1:
            yield event_templates[self.template_index].get_event_by_index(self.instance_index)

        child_iter = None
        slot_iter = None

        if self.children:
            child_iter = self._chain_visitors(
                c.event_visitor(event_templates, include_kernels) for c in self.children
            )

        if self.slots:
            slot_iter = self._chain_visitors(
                s.event_visitor(event_templates, include_kernels) for s in self.slots
            )

        child_ev = next(child_iter, None) if child_iter else None
        slot_ev = next(slot_iter, None) if slot_iter else None

        while child_ev is not None or slot_ev is not None:
            if slot_ev is None or (
                child_ev is not None and child_ev.ts <= slot_ev.ts
            ):
                yield child_ev
                child_ev = next(child_iter, None)
            else:
                yield slot_ev
                slot_ev = next(slot_iter, None)

    @classmethod
    def from_dict(cls, d: dict):
        node = cls(
            template_index=d["t"],
            instance_index=d["i"],
        )

        if "c" in d:
            node.children = [node_from_dict(c) for c in d["c"]]

        if "s" in d:
            node.slots = [node_from_dict(s) for s in d["s"]]

        return node

    def to_dict(self):
        d = {
            "t": self.template_index,
            "i": self.instance_index,
        }

        if self.children:
            d["c"] = [c.to_dict() for c in self.children]

        if self.slots:
            d["s"] = [s.to_dict() for s in self.slots]

        return d

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}CPUNode(template_index={self.template_index}, \
              instance_index={self.instance_index})")
        if self.children:
            for child in self.children:
                child.show(indent + 1)


class SameCPUNode:

    __slots__ = ["template_index", "instance_index", "children", "slots"]

    def __init__(self, template_index: int, instance_index: list):
        self.template_index = template_index
        self.instance_index = np.asarray(instance_index, dtype=np.int32)
        self.children = None
        self.slots = None

    def add_child(self, child):
        if self.children is None:
            self.children = []
        self.children.append(child)

    def is_kernel_node(self):
        return False

    def _chain_visitors(self, visitors):
        for it in visitors:
            yield from it

    def event_visitor_index(self, event_templates: List[MergeEvent], index: int = 0, include_kernels: bool = False):

        yield event_templates[self.template_index].get_event_by_index(self.instance_index[index])

        child_iter = None
        slot_iter = None

        if self.children:
            child_iter = self._chain_visitors(
                c.event_visitor_index(event_templates, index, include_kernels) for c in self.children
            )

        if self.slots:
            slot_iter = self._chain_visitors(
                s.event_visitor(event_templates, include_kernels) for s in self.slots[index]
            )

        child_ev = next(child_iter, None) if child_iter else None
        slot_ev = next(slot_iter, None) if slot_iter else None

        while child_ev is not None or slot_ev is not None:
            if slot_ev is None or (
                child_ev is not None and child_ev.ts <= slot_ev.ts
            ):
                yield child_ev
                child_ev = next(child_iter, None)
            else:
                yield slot_ev
                slot_ev = next(slot_iter, None)

    def event_visitor(self, event_templates: List[MergeEvent], include_kernels: bool = False):
        for index in range(len(self.instance_index)):
            yield from self.event_visitor_index(event_templates, index, include_kernels)

    @classmethod
    def from_dict(cls, d: dict):
        node = cls(
            template_index=d["t"],
            instance_index=d["i"],
        )

        if "c" in d:
            node.children = [node_from_dict(c) for c in d["c"]]

        if "s" in d:
            node.slots = [
                [node_from_dict(s) for s in slot]
                for slot in d["s"]
            ]

        return node

    def to_dict(self):
        d = {
            "t": self.template_index,
            "i": self.instance_index.tolist(),
        }

        if self.children:
            d["c"] = [c.to_dict() for c in self.children]

        if self.slots:
            d["s"] = [
                [s.to_dict() for s in slot]
                for slot in self.slots
            ]

        return d

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}SameCPUNode(template_index={self.template_index}, \
              instance_index_len={len(self.instance_index)})")
        if self.children:
            for child in self.children:
                child.show(indent + 1)


class KernelNode:

    __slots__ = ["template_index", "instance_index"]

    def __init__(self, template_index: int, instance_index: int):
        self.template_index = template_index
        self.instance_index = instance_index

    def get_children(self):
        return []

    def is_kernel_node(self):
        return True

    def event_visitor(self, event_templates: List[MergeEvent], include_kernels: bool = False):
        if include_kernels:
            yield event_templates[self.template_index].get_event_by_index(self.instance_index)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            template_index=d["k"],
            instance_index=d["i"],
        )

    def to_dict(self):
        return {
            "k": self.template_index,
            "i": self.instance_index,
        }

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}KernelNode(template_index={self.template_index}, \
              instance_index={self.instance_index})")


class SameKernelNode:
    __slots__ = ["template_index", "instance_index"]

    def __init__(self, template_index: int, instance_index: list):
        self.template_index = template_index
        self.instance_index = np.asarray(instance_index, dtype=np.int32)

    def get_children(self):
        return []

    def is_kernel_node(self):
        return True

    def event_visitor_index(self, event_templates: List[MergeEvent], index: int = 0, include_kernels: bool = False):
        if include_kernels:
            yield event_templates[self.template_index].get_event_by_index(self.instance_index[index])

    def event_visitor(self, event_templates: List[MergeEvent], include_kernels: bool = False):
        if include_kernels:
            for index in range(len(self.instance_index)):
                yield from self.event_visitor_index(event_templates, index)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            template_index=d["k"],
            instance_index=d["i"],
        )

    def to_dict(self):
        return {
            "k": self.template_index,
            "i": self.instance_index.tolist(),
        }

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}GroupKernelNode(template_index={self.template_index}, \
              instance_index_len={len(self.instance_index)})")

class GPUNode:

    __slots__ = ["template_index", "instance_index", "event_start"]

    def __init__(self):
        self.template_index = []
        self.instance_index = []
        self.event_start = None

    def add_event(self, template_index, instance_index):
        self.template_index.append(template_index)
        self.instance_index.append(instance_index)

    def is_kernel_node(self):
        return False

    def set_start_event(self, event_start):
        self.event_start = event_start

    def event_visitor(self, event_templates: List[MergeEvent], include_kernels: bool = False):
        if False:
            yield None


    @classmethod
    def from_dict(cls, d: dict):
        node = cls()
        node.template_index = list(d["t"])
        node.instance_index = list(d["i"])
        node.event_start = d.get("e", None)
        return node

    def to_dict(self):
        return {
            "t": self.template_index,
            "i": self.instance_index,
            "e": None
        }

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}GPUNode(template_index_len={len(self.template_index)}, instance_index_len={len(self.instance_index)})")

def node_from_dict(d: dict):
    """
    根据 dict 的结构判断 Node 类型，并调用对应 from_dict
    """

    if "e" in d:
        return GPUNode.from_dict(d)


    if "k" in d:
        if isinstance(d.get("i"), list):
            return SameKernelNode.from_dict(d)
        else:
            return KernelNode.from_dict(d)

    if isinstance(d.get("i"), list):
        return SameCPUNode.from_dict(d)

    if isinstance(d.get("i"), int):
        return CPUNode.from_dict(d)

    raise ValueError(f"Unknown node dict format: {d}")

def count_nodes(node, counter=None):
    if counter is None:
        counter = defaultdict(int)

    # ===== 统计当前节点类型 =====
    counter[type(node).__name__] += 1

    # ===== CPUNode / SameCPUNode =====
    if hasattr(node, "children") and node.children:
        for c in node.children:
            count_nodes(c, counter)

    if hasattr(node, "slots") and node.slots:
        # CPUNode: slots = List[Node]
        if isinstance(node.slots, list) and node.slots and not isinstance(node.slots[0], list):
            for s in node.slots:
                count_nodes(s, counter)
        # SameCPUNode: slots = List[List[Node]]
        else:
            for slot in node.slots:
                for s in slot:
                    count_nodes(s, counter)

    return counter

def count_trace_nodes(compressed_trace):
    counter = defaultdict(int)

    for rank in compressed_trace.get_ranks():
        for _, _, _, _, node in compressed_trace.iter_nodes(rank):
            count_nodes(node, counter)

    print_node_stats(counter, "Trace Node Statistics")

def print_node_stats(counter, title="Node Statistics"):
    print(f"\n=== {title} ===")
    total = sum(counter.values())
    for k, v in sorted(counter.items()):
        print(f"{k:15s}: {v}")
    print(f"{'-'*20}")
    print(f"{'TOTAL':15s}: {total}")
