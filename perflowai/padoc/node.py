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
from typing import List, Dict, Union, Any, Optional, Generator, Tuple
import numpy as np
import sys
import struct
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

    __slots__ = ["template_index", "instance_index", "children", "slots", "launch_range"]

    def __init__(self, template_index: int, instance_index: int):
        self.template_index = template_index
        self.instance_index = instance_index
        self.children = None
        self.slots = None
        self.launch_range = None

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

    __slots__ = ["template_index", "instance_index", "children", "slots", "launch_range"]

    def __init__(self, template_index: int, instance_index: list):
        self.template_index = template_index
        self.instance_index = np.asarray(instance_index, dtype=np.int32)
        self.children = None
        self.slots = None
        self.launch_range = None

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


class KernelLaunchNode:
    __slots__ = ["template_index", "instance_index", "gpu_template_index", "gpu_instance_index"]

    def __init__(
        self,
        template_index: int,
        instance_index: int,
        gpu_template_index: int,
        gpu_instance_index: int,
    ):
        # template/instance describe CPU launch event; gpu_* indexes point to related GPU event.
        self.template_index = template_index
        self.instance_index = instance_index
        self.gpu_template_index = gpu_template_index
        self.gpu_instance_index = gpu_instance_index

    def get_children(self):
        return []

    def is_kernel_node(self):
        return True

    def event_visitor(self, event_templates: List[MergeEvent], include_kernels: bool = False):
        if include_kernels:
            yield event_templates[self.gpu_template_index].get_event_by_index(self.gpu_instance_index)

    @classmethod
    def from_dict(cls, d: dict):
        gpu_template_index = d["gt"] if "gt" in d else d["k"]
        gpu_instance_index = d["gi"] if "gi" in d else d["i"]
        return cls(
            template_index=d["k"],
            instance_index=d["i"],
            gpu_template_index=gpu_template_index,
            gpu_instance_index=gpu_instance_index,
        )

    def to_dict(self):
        return {
            "k": self.template_index,
            "i": self.instance_index,
            "gt": self.gpu_template_index,
            "gi": self.gpu_instance_index,
        }

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}KernelLaunchNode(template_index={self.template_index}, \
              instance_index={self.instance_index}, gpu_template_index={self.gpu_template_index}, \
              gpu_instance_index={self.gpu_instance_index})")


class KernelsLaunchNode:
    __slots__ = ["template_index", "instance_index", "gpu_template_index", "gpu_instance_index"]

    def __init__(
        self,
        template_index: int,
        instance_index: list,
        gpu_template_index: list,
        gpu_instance_index: list,
    ):
        self.template_index = template_index
        self.instance_index = np.asarray(instance_index, dtype=np.int32)
        self.gpu_template_index = np.asarray(gpu_template_index, dtype=np.int32)
        self.gpu_instance_index = np.asarray(gpu_instance_index, dtype=np.int32)

    def get_children(self):
        return []

    def is_kernel_node(self):
        return True

    def event_visitor_index(self, event_templates: List[MergeEvent], index: int = 0, include_kernels: bool = False):
        if include_kernels:
            yield event_templates[self.gpu_template_index[index]].get_event_by_index(self.gpu_instance_index[index])

    def event_visitor(self, event_templates: List[MergeEvent], include_kernels: bool = False):
        if include_kernels:
            for index in range(len(self.instance_index)):
                yield from self.event_visitor_index(event_templates, index, include_kernels)

    @classmethod
    def from_dict(cls, d: dict):
        gpu_template_index = d["gt"] if "gt" in d else d["k"]
        gpu_instance_index = d["gi"] if "gi" in d else d["i"]
        return cls(
            template_index=d["k"],
            instance_index=d["i"],
            gpu_template_index=gpu_template_index,
            gpu_instance_index=gpu_instance_index,
        )

    def to_dict(self):
        return {
            "k": self.template_index,
            "i": self.instance_index.tolist(),
            "gt": self.gpu_template_index.tolist(),
            "gi": self.gpu_instance_index.tolist(),
        }

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}KernelsLaunchNode(template_index={self.template_index}, \
              instance_index_len={len(self.instance_index)})")

class GPUNode:

    __slots__ = ["template_index", "instance_index"]

    def __init__(self):
        self.template_index = np.empty(0, dtype=np.int32)
        self.instance_index = np.empty(0, dtype=np.int32)

    def set_events(self, template_index, instance_index):
        self.template_index = np.asarray(template_index, dtype=np.int32)
        self.instance_index = np.asarray(instance_index, dtype=np.int32)

    def add_event(self, template_index, instance_index):
        # Keep backward compatibility for call sites that still append one-by-one.
        self.template_index = np.append(self.template_index, np.int32(template_index))
        self.instance_index = np.append(self.instance_index, np.int32(instance_index))

    def is_kernel_node(self):
        return False

    def event_visitor(self, event_templates: List[MergeEvent], include_kernels: bool = False):
        for index in range(len(self.template_index)):
            yield event_templates[int(self.template_index[index])].get_event_by_index(
                int(self.instance_index[index])
            )

    @classmethod
    def from_dict(cls, d: dict):
        node = cls()
        node.template_index = np.atleast_1d(np.asarray(d["t"], dtype=np.int32))
        node.instance_index = np.atleast_1d(np.asarray(d["i"], dtype=np.int32))
        return node

    def to_dict(self):
        return {
            "g": 1,
            "t": self.template_index.tolist(),
            "i": self.instance_index.tolist(),
        }

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}GPUNode(template_index_len={len(self.template_index)}, instance_index_len={len(self.instance_index)})")

def node_from_dict(d: dict):
    """
    根据 dict 的结构判断 Node 类型，并调用对应 from_dict
    """

    if d.get("g", 0) == 1 or "e" in d:
        return GPUNode.from_dict(d)


    if "k" in d:
        if isinstance(d.get("i"), list):
            return KernelsLaunchNode.from_dict(d)
        return KernelLaunchNode.from_dict(d)

    if isinstance(d.get("i"), list):
        return SameCPUNode.from_dict(d)

    if isinstance(d.get("i"), int):
        return CPUNode.from_dict(d)

    raise ValueError(f"Unknown node dict format: {d}")


class LaunchSubtreeIndex:
    """Index launch nodes by subtree range for fast repeated lookups.

    Build once from a root node, then query any node under that root in O(1)+O(K),
    where K is the number of launch nodes in the queried subtree.
    """

    __slots__ = ["root", "_launch_nodes"]

    def __init__(self, root):
        self.root = root
        self._launch_nodes: List[Union[KernelLaunchNode, KernelsLaunchNode]] = []
        self._build(root)

    @staticmethod
    def _is_launch_node(node: Any) -> bool:
        return isinstance(node, (KernelLaunchNode, KernelsLaunchNode))

    def _children_iter(self, node: Any):
        children = getattr(node, "children", None)
        if children:
            for child in children:
                yield child

        slots = getattr(node, "slots", None)
        if not slots:
            return

        first = slots[0]
        if isinstance(first, list):
            for slot in slots:
                for child in slot:
                    yield child
            return

        for child in slots:
            yield child

    def _build(self, node: Any) -> None:
        if self._is_launch_node(node):
            self._launch_nodes.append(node)
            return

        children = list(self._children_iter(node))
        if not children:
            if hasattr(node, "launch_range"):
                node.launch_range = None
            return

        start = len(self._launch_nodes)
        for child in children:
            self._build(child)

        if hasattr(node, "launch_range"):
            node.launch_range = (start, len(self._launch_nodes))

    def has_node(self, node: Any) -> bool:
        if self._is_launch_node(node):
            return True
        return hasattr(node, "launch_range")

    def get_range(self, node: Any = None) -> Tuple[int, int]:
        target = self.root if node is None else node
        launch_range = getattr(target, "launch_range", None)
        if launch_range is None:
            return (0, 0)
        return launch_range

    def get_launch_nodes(self, node: Any = None) -> List[Union[KernelLaunchNode, KernelsLaunchNode]]:
        target = self.root if node is None else node
        if self._is_launch_node(target):
            return [target]
        start, end = self.get_range(node)
        return self._launch_nodes[start:end]

    def iter_launch_nodes(self, node: Any = None):
        target = self.root if node is None else node
        if self._is_launch_node(target):
            yield target
            return
        start, end = self.get_range(node)
        for i in range(start, end):
            yield self._launch_nodes[i]


def build_launch_subtree_index(root) -> LaunchSubtreeIndex:
    return LaunchSubtreeIndex(root)

_EMPTY_LIST_SIZE = sys.getsizeof([])
_PTR_SIZE = struct.calcsize("P")


def _list_ref_memory(lst) -> int:
    if lst is None:
        return 0
    return _EMPTY_LIST_SIZE + len(lst) * _PTR_SIZE


def _node_shallow_memory(node) -> int:
    return sum(_node_shallow_memory_breakdown(node).values())


def _node_shallow_memory_breakdown(node) -> Dict[str, int]:
    breakdown: Dict[str, int] = defaultdict(int)

    breakdown["object"] += sys.getsizeof(node)

    if hasattr(node, "__slots__"):
        for slot in node.__slots__:
            if slot in ("children", "slots"):
                continue
            try:
                value = getattr(node, slot)
            except AttributeError:
                continue
            breakdown[slot] += sys.getsizeof(value)

    children = getattr(node, "children", None)
    if children is not None:
        breakdown["children_refs"] += _list_ref_memory(children)

    slots = getattr(node, "slots", None)
    if slots is not None:
        breakdown["slots_refs"] += _list_ref_memory(slots)
        if isinstance(slots, list):
            for inner in slots:
                if isinstance(inner, list):
                    breakdown["slots_inner_refs"] += _list_ref_memory(inner)

    return breakdown


def count_nodes(node, counter=None, type_memory=None, seen=None, type_field_memory=None):
    if counter is None:
        counter = defaultdict(int)
    if type_memory is None:
        type_memory = defaultdict(int)
    if seen is None:
        seen = set()
    if type_field_memory is None:
        type_field_memory = defaultdict(lambda: defaultdict(int))

    oid = id(node)
    if oid in seen:
        return counter, type_memory
    seen.add(oid)

    # ===== 统计当前节点类型 =====
    cls_name = type(node).__name__
    counter[cls_name] += 1
    field_breakdown = _node_shallow_memory_breakdown(node)
    type_memory[cls_name] += sum(field_breakdown.values())
    for field, sz in field_breakdown.items():
        type_field_memory[cls_name][field] += sz

    # ===== CPUNode / SameCPUNode =====
    if hasattr(node, "children") and node.children:
        for c in node.children:
            count_nodes(c, counter, type_memory, seen, type_field_memory)

    if hasattr(node, "slots") and node.slots:
        # CPUNode: slots = List[Node]
        if isinstance(node.slots, list) and node.slots and not isinstance(node.slots[0], list):
            for s in node.slots:
                count_nodes(s, counter, type_memory, seen, type_field_memory)
        # SameCPUNode: slots = List[List[Node]]
        else:
            for slot in node.slots:
                for s in slot:
                    count_nodes(s, counter, type_memory, seen, type_field_memory)

    return counter, type_memory, type_field_memory


def count_trace_nodes(compressed_trace):
    counter, type_memory, type_field_memory = collect_trace_node_stats(compressed_trace)
    print_node_stats(counter, "Trace Node Statistics", type_memory, type_field_memory)
    return counter, type_memory, type_field_memory


def collect_trace_node_stats(compressed_trace):
    """Collect node statistics for a compressed trace without printing."""
    counter = defaultdict(int)
    type_memory = defaultdict(int)
    type_field_memory = defaultdict(lambda: defaultdict(int))
    seen = set()

    for rank in compressed_trace.get_ranks():
        for _, _, _, _, node in compressed_trace.iter_nodes(rank):
            count_nodes(node, counter, type_memory, seen, type_field_memory)

    return counter, type_memory, type_field_memory


def print_node_stats(counter, title="Node Statistics", type_memory=None, type_field_memory=None):
    print(f"\n=== {title} ===")
    total = sum(counter.values())
    total_mem = 0 if type_memory is None else sum(type_memory.values())
    for k, v in sorted(counter.items()):
        if type_memory is None:
            print(f"{k:15s}: {v}")
        else:
            mem_mb = type_memory[k] / 1024 / 1024
            avg_kb = (type_memory[k] / v / 1024) if v > 0 else 0.0
            print(f"{k:15s}: count={v:8d}, mem={mem_mb:8.2f} MB, avg={avg_kb:8.2f} KB")
    print(f"{'-'*20}")
    if type_memory is None:
        print(f"{'TOTAL':15s}: {total}")
    else:
        print(f"{'TOTAL':15s}: count={total:8d}, mem={total_mem / 1024 / 1024:8.2f} MB")

    if type_field_memory is not None and "SameCPUNode" in type_field_memory:
        same_count = counter.get("SameCPUNode", 0)
        if same_count > 0:
            print("\n=== SameCPUNode Memory Breakdown (Shallow + Refs) ===")
            fields = type_field_memory["SameCPUNode"]
            total_same = sum(fields.values())
            for field, sz in sorted(fields.items(), key=lambda x: x[1], reverse=True):
                pct = (sz / total_same * 100.0) if total_same > 0 else 0.0
                avg_kb = sz / same_count / 1024.0
                print(
                    f"{field:16s}: {sz / 1024 / 1024:8.2f} MB "
                    f"({pct:5.1f}%), avg={avg_kb:8.2f} KB/node"
                )
