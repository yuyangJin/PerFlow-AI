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
from .event import Event, MergeEvent, is_same_event
from .utils import logger


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

    @abstractmethod
    def get_first_event_name(self):
        """Get the name of the first event in this node."""
        return

    @abstractmethod
    def events_visitor(self, index: int = 0) -> Generator[Event, None, None]:
        """A generator that yields all events (must be Event) in this node and its children."""
        return

    @abstractmethod
    def is_same_node(self, other: BaseNode, debug: bool = False, indent = 0) -> bool:
        """Check if this node is the same as another node."""
        return False

    @abstractmethod
    def is_similar_node(self, other: BaseNode, debug: bool = False, indent = 0) -> bool:
        """Check if the first event of this node is similar to another node."""
        return False

    @abstractmethod
    def get_node_count(self) -> int:
        """Get the total number of nodes in this subtree."""
        return 1

    @abstractmethod
    def show(self, indent=0):
        return

    @abstractmethod
    def to_dict(self) -> Dict:
        """Serialize this node to a dictionary."""
        return {}

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[str, Any] | None = None,
                  templates: Dict[str, TemplateNode] | None = None):
        """Deserialize a dictionary to a node."""
        return None

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

    def events_visitor(self, index: int = 0) -> Generator[Event, None, None]:
        for event in self.events:
            yield event

        for child in self.children:
            yield from child.events_visitor(index)

    def is_same_node(self, other: Union[Node, TemplateNode, RefNode],
                     debug: bool = False, indent = 0) -> bool:
        if isinstance(other, RefNode):
            return other.is_same_node(self)

        ind = " " * indent
        if len(self.events) != len(other.get_events()):
            if debug:
                logger.debug("%s Node is_same_node: Different event count(%s): %d vs %d",
                             ind, self.events[0].get_name(),
                             len(self.events), len(other.get_events()))
                for e in other.get_events():
                    logger.debug("%s Node is_same_node: %s", ind, e.get_name())
            return False

        for i, event in enumerate(self.events):
            if not is_same_event(event, other.get_events()[i], debug):
                if debug:
                    logger.debug("%s Node is_same_node: Different event: %s vs %s",
                                 ind, event, other.get_events()[i])
                return False

        if len(self.children) != len(other.get_children()):
            if debug:
                logger.debug("%s Node is_same_node: Different child count: %d vs %d",
                             ind, len(self.children), len(other.get_children()))
                for i in range(len(other.get_children())):
                    if i < len(self.children):
                        logger.debug("%s Node is_same_node: %s %s %s",
                                     ind, \
                                     self.children[i].get_first_event_name() == \
                                        other.get_children()[i].get_first_event_name(), \
                                             self.children[i].get_first_event_name(), \
                                                 other.get_children()[i].get_first_event_name())
                    else:
                        logger.debug("%s Node is_same_node: %s",
                                     ind, other.get_children()[i].get_first_event_name())
            return False

        for i, child in enumerate(self.children):
            if not child.is_same_node(other.get_children()[i], debug=debug, indent=indent+2):
                if debug:
                    logger.debug("%s Node is_same_node: Different child: %s vs %s",
                                 ind, child.get_first_event_name(), \
                                 other.get_children()[i].get_first_event_name())
                return False

        return True

    def is_similar_node(self, other, debug = False, indent=0):
        if isinstance(other, RefNode):
            return other.is_same_node(self)

        ind = " " * indent
        if len(self.events) != len(other.get_events()):
            if debug:
                logger.debug("%s Node is_same_node: Different event count(%s): %d vs %d",
                             ind, self.events[0].get_name(),
                             len(self.events), len(other.get_events()))
                for e in other.get_events():
                    logger.debug("%s Node is_same_node: %s", ind, e.get_name())
            return False

        for i, event in enumerate(self.events):
            if not is_same_event(event, other.get_events()[i], debug):
                if debug:
                    logger.debug("%s Node is_same_node: Different event: %s vs %s",
                                 ind, event, other.get_events()[i])
                return False

        return True

    def sort_events(self):
        """Sort events in this node by their timestamps."""
        assert len(self.children) == 0, "Cannot sort events in a non-leaf node."

        self.events.sort(key=lambda e: e.get_ts())

    def show(self, indent: int = 0):
        prefix = " " * indent

        if self.events:
            names = [e.get_name() for e in self.events]
            print(f"{prefix}Node: {names}")
        else:
            print(f"{prefix}Node: <empty>")

        for child in self.children:
            child.show(indent + 2)

    def to_dict(self) -> Dict:
        return {
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children]
        }

    @classmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[str, Any] | None = None,
                  templates: Dict[str, TemplateNode] | None = None):
        assert "events" in data, f"Node must have events. {data.keys()}"
        assert "children" in data, "Node must have children."

        obj = cls.__new__(cls)
        obj.events = [Event.from_dict(e) for e in data["events"]]
        obj.children = []
        for c in data["children"]:
            if "ref_node_id" in c:
                obj.children.append(RefNode.from_dict(c, templates_dict, templates))
            elif "ref_node_ids" in c:
                obj.children.append(GroupRefNode.from_dict(c, templates_dict, templates))
            else:
                obj.children.append(Node.from_dict(c, templates_dict, templates))
        return obj

class TemplateNode(BaseNode):
    """A merged/compressed node representing multiple structurally-identical nodes.

    TemplateNode performs hierarchical merging:
        - MergeEvent stores merged attributes across multiple events.
        - Subtrees are recursively merged into TemplateNode children.
        - RefNode denotes repeated occurrences of this template.

    This class is central to structural compression (template extraction).
    """

    def __init__(self, nodes: List[Union[Node, TemplateNode]]):
        self.events : List[MergeEvent] = []
        self.children : List[Union[TemplateNode, RefNode]] = []
        self.slots = []
        self.node_count = sum(n.get_node_count() for n in nodes)

        self.ev_idxs = None
        self.external_ids = None
        self.sequence_ids = None
        self.correlations = None

        if len(nodes) == 0:
            return

        event_count = len(nodes[0].events)
        for n in nodes[1:]:
            assert len(n.events) == event_count, \
                "All nodes must have the same number of events to merge."

        for i in range(event_count):
            events_to_merge = [n.events[i] for n in nodes]
            merged_event = MergeEvent(events_to_merge)
            self.events.append(merged_event)

        children_count = len(nodes[0].children)
        for n in nodes[1:]:
            assert len(n.children) == children_count, \
                "All nodes must have the same number of children to merge."

        for i in range(children_count):
            child_nodes_to_merge = [n.children[i] for n in nodes]
            merged_child_node = TemplateNode(child_nodes_to_merge)
            self.children.append(merged_child_node)

    def add_events(self, events: List[Event]):
        """Add a list of events to this node."""
        merged_event = MergeEvent(events)
        self.events.append(merged_event)

    def add_nodes(self, nodes: List[Union[Node, TemplateNode]]):
        """Add a list of nodes to this node."""
        self.node_count += sum(n.get_node_count() for n in nodes)
        event_count = len(self.events)
        for n in nodes:
            assert len(n.events) == event_count, \
                "All nodes must have the same number of events as original node to merge."

        for i in range(event_count):
            events_to_merge = [n.events[i] for n in nodes]
            self.events[i].add_events(events_to_merge)

        child_count = len(self.children)
        for n in nodes:
            assert len(n.children) == child_count, \
                "All nodes must have the same number of children as original node to merge."

        for i in range(child_count):
            child_nodes_to_merge = [n.children[i] for n in nodes]
            self.children[i].add_nodes(child_nodes_to_merge)

    def get_events(self) -> List[MergeEvent]:
        return self.events

    def get_all_events(self) -> List[Any]:
        all_events = self.events.copy()
        for child in self.children:
            all_events.extend(child.get_all_events())
        return all_events

    def get_events_by_index(self, index: int) -> List[Event]:
        """Get a list of events at a specific index in this node and its children."""
        all_events = [e.get_event_by_index(index) for e in self.events]
        for child in self.children:
            if isinstance(child, TemplateNode):
                all_events.extend(child.get_events_by_index(index))
            else:
                all_events.extend(child.get_all_events(index))
        return all_events

    def get_node_count(self) -> int:
        return self.node_count

    def get_children(self) -> List[BaseNode]:
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

    def events_visitor(self, index: int = 0) -> Generator[Event, None, None]:
        for e in self.events:
            yield e.get_event_by_index(index)

        for child in self.children:
            yield from child.events_visitor(index)

    def is_same_node(self, other: BaseNode, debug: bool = False, indent = 0) -> bool:
        if isinstance(other, RefNode):
            return other.is_same_node(self)

        ind = " " * indent

        if len(self.events) != len(other.get_events()):
            if debug:
                logger.debug("%s TemplateNode is_same_node: Different event count: %d vs %d",
                             ind, len(self.events), len(other.get_events()))
            return False

        for i, event in enumerate(self.events):
            if not is_same_event(event, other.get_events()[i], debug):
                if debug:
                    logger.debug("TemplateNode is_same_node: Different event: %s vs %s",
                                 event, other.get_events()[i])
                return False

        if len(self.children) != len(other.get_children()):
            if debug:
                logger.debug("TemplateNode is_same_node: Different child count: %d vs %d",
                             len(self.children), len(other.get_children()))
            return False

        for i, event in enumerate(self.children):
            if not event.is_same_node(other.get_children()[i], debug=debug):
                if debug:
                    logger.debug("TemplateNode is_same_node: Different child: %s vs %s",
                                 event, other.get_children()[i])
                return False

        return True

    def is_similar_node(self, other: BaseNode, debug: bool = False, indent = 0) -> bool:
        if isinstance(other, RefNode):
            return other.is_same_node(self)

        ind = " " * indent

        if len(self.events) != len(other.get_events()):
            if debug:
                logger.debug("%s TemplateNode is_same_node: Different event count: %d vs %d",
                             ind, len(self.events), len(other.get_events()))
            return False

        for i, event in enumerate(self.events):
            if not is_same_event(event, other.get_events()[i], debug):
                if debug:
                    logger.debug("TemplateNode is_same_node: Different event: %s vs %s",
                                 event, other.get_events()[i])
                return False

        return True

    def compress_event_values(self):
        """Compress event values."""
        for e in self.events:
            e.compress_values()
        for c in self.children:
            c.compress_event_values()

    def check_args_id_same_detla(self,
                             external_ids: Optional[List[int]] = None,
                             ev_idxs: Optional[List[int]] = None
        ) -> bool:
        """Check if args_id has the same delta."""

        for e in self.events:
            args = e.get_args()
            if "External id" in args:
                e_external_ids = args["External id"]
                if external_ids is None:
                    self.external_ids = e_external_ids
                    external_ids = e_external_ids
                else:
                    if not self._all_same_detla(external_ids, e_external_ids):
                        return False
            if "Ev Idx" in args:
                e_ev_idxs = args["Ev Idx"]
                if external_ids is None:
                    self.ev_idxs = e_ev_idxs
                    ev_idxs = e_ev_idxs
                else:
                    if not self._all_same_detla(ev_idxs, e_ev_idxs):
                        return False

        for c in self.children:
            if not c.check_args_id_same_detla(external_ids, ev_idxs):
                return False

        return True

    def try_compress_args_id(self,
                             external_ids: Optional[List[int]] = None,
                             ev_idxs: Optional[List[int]] = None,
                             sequence_ids: Optional[List[int]] = None,
                             correlations: Optional[List[int]] = None
        ) -> None:
        """Try to compress args_id."""

        for e in self.events:
            e_args = e.args
            if "External id" in e_args:
                e_external_ids = e_args["External id"]
                if external_ids is None:
                    self.external_ids = e_external_ids
                    external_ids = e_external_ids
                    e_args["External id"] = 0
                else:
                    if self._all_same_detla(external_ids, e_external_ids):
                        e_args["External id"] = e_external_ids[0] - external_ids[0]

            if "Ev Idx" in e_args:
                e_ev_idxs = e_args["Ev Idx"]
                if ev_idxs is None:
                    self.ev_idxs = e_ev_idxs
                    ev_idxs = e_ev_idxs
                    e_args["Ev Idx"] = 0
                else:
                    if self._all_same_detla(ev_idxs, e_ev_idxs):
                        e_args["Ev Idx"] = e_ev_idxs[0] - ev_idxs[0]

            if "Sequence number" in e_args:
                e_sequence_ids = e_args["Sequence number"]
                if sequence_ids is None:
                    self.sequence_ids = e_sequence_ids
                    sequence_ids = e_sequence_ids
                    e_args["Sequence number"] = 0
                else:
                    if self._all_same_detla(sequence_ids, e_sequence_ids):
                        e_args["Sequence number"] = e_sequence_ids[0] - sequence_ids[0]

            if "correlation" in e_args:
                e_correlations = e_args["correlation"]
                if correlations is None:
                    self.correlations = e_correlations
                    correlations = e_correlations
                    e_args["correlation"] = 0
                else:
                    if self._all_same_detla(correlations, e_correlations):
                        e_args["correlation"] = e_correlations[0] - correlations[0]

        for c in self.children:
            if isinstance(c, TemplateNode):
                c.try_compress_args_id(external_ids, ev_idxs, sequence_ids)


    def _all_same_detla(self, a: List[int], b: List[int]) -> bool:
        if len(a) != len(b):
            return False

        delta = a[0] - b[0]
        for i in range(1, len(a)):
            if a[i] - b[i] != delta:
                return False

        return True
    
    def show(self, indent: int = 0):
        prefix = " " * indent

        if self.events:
            names = [e.get_name() + " " + str(e.get_len()) + " " + str(e.ts[0]) for e in self.events]
            print(f"{prefix}TemplateNode x{self.node_count} {len(self.slots)}: {names}")
        else:
            print(f"{prefix}TemplateNode x{self.node_count} {len(self.slots)}: <empty>")

        for child in self.children:
            child.show(indent + 2)

        for si, slot in enumerate(self.slots):
            if len(slot) == 0:
                continue
            print(f"{prefix}  slot[{si}]")
            for oi, opt in enumerate(slot):
                print(f"{prefix}    option[{oi}]")
                opt.show(indent + 4)

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children]
        }

        if self.external_ids is not None:
            res["external_ids"] = self.external_ids

        if self.ev_idxs is not None:
            res["ev_idxs"] = self.ev_idxs

        if self.sequence_ids is not None:
            res["sequence_ids"] = self.sequence_ids

        if self.correlations is not None:
            res["correlations"] = self.correlations

        return res

    @classmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[int, Any] | None = None,
                  templates: Dict[int, TemplateNode] | None = None):
        assert "events" in data, "TemplateNode must have events."
        assert "children" in data, "TemplateNode must have children."
        obj = cls.__new__(cls)
        obj.events = [MergeEvent.from_dict(e) for e in data["events"]]
        if "external_ids" in data:
            obj.external_ids = data["external_ids"]
        if "ev_idxs" in data:
            obj.ev_idxs = data["ev_idxs"]
        if "sequence_ids" in data:
            obj.sequence_ids = data["sequence_ids"]
        if "correlations" in data:
            obj.correlations = data["correlations"]
        obj.children = []
        for c in data["children"]:
            if "ref_node_id" in c:
                obj.children.append(RefNode.from_dict(c, templates_dict, templates))
            elif "ref_node_ids" in c:
                obj.children.append(GroupRefNode.from_dict(c, templates_dict, templates))
            else:
                obj.children.append(TemplateNode.from_dict(c, templates_dict, templates))
        obj.node_count = obj.events[0].get_len()
        return obj


class RefNode(BaseNode):
    """A lightweight reference to a TemplateNode at a given index.

    Instead of storing events or children, RefNode resolves data
    dynamically from the template via index lookup.

    This enables repeated structure references without duplication.
    """

    def __init__(self, ref: TemplateNode, index: int):
        self.ref: TemplateNode = ref
        self.index: int = index

    def update_template(self, template: TemplateNode, index: int):
        """Update the reference to a new template and index."""
        self.ref = template
        self.index += index

    def is_same_node(self, other: BaseNode, debug: bool = False, indent = 0) -> bool:
        return False

    def is_similar_node(self, other: BaseNode, debug: bool = False, indent = 0) -> bool:
        return False

    def get_events(self) -> List[Event]:
        return self.ref.get_events_by_index(self.index)

    def get_all_events(self, index: Optional[int] = None) -> List[Event]:
        if index is not None:
            return self.ref.get_events_by_index(self.index + index)
        return self.ref.get_events_by_index(self.index)

    def get_children(self) -> List[BaseNode]:
        pass

    def set_children(self, children: List[BaseNode]):
        pass

    def add_child(self, child: BaseNode):
        pass

    def get_first_event_name(self):
        return self.ref.get_first_event_name()

    def events_visitor(self, index: int = 0) -> Generator[Event, None, None]:
        return self.ref.events_visitor(self.index + index)

    def get_node_count(self) -> int:
        return self.ref.get_node_count()

    def compress_event_values(self):
        """Compress event values."""
        return


    def show(self, indent: int = 0):
        prefix = " " * indent
        ref_id = getattr(self.ref, "id", "unknown")
        print(f"{prefix}RefNode -> Template {ref_id} [index={self.index}]")

    def to_dict(self) -> Dict:
        assert hasattr(self.ref, "id"), "RefNode must have a ref_node_id."
        return {
            "ref_node_id": self.ref.id,
            "index": self.index
        }

    @classmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[str, Any] | None = None,
                  templates: Dict[str, TemplateNode] | None = None):
        assert "ref_node_id" in data, "RefNode must have a ref_node_id."
        assert "index" in data, "RefNode must have an index."

        ref_id = data["ref_node_id"]
        assert ref_id in templates_dict, (
            f"RefNode ref_node_id {ref_id} not found in "
            f"templates_dict {templates_dict.keys()}."
        )

        if int(ref_id) not in templates:
            templates[ref_id] = TemplateNode.from_dict(
                templates_dict[ref_id],
                templates_dict,
                templates
            )

        return RefNode(templates[ref_id], int(data["index"]))


class GroupRefNode(BaseNode):
    """A lightweight reference to a list of TemplateNodes at given indexs.
    """

    def __init__(self, ref_list: List[TemplateNode], index: list[int]):
        self.ref = ref_list
        self.index = index

    def add_ref(self, ref: TemplateNode, index: int):
        """Add a reference to a new template and index."""
        self.ref.append(ref)
        self.index.append(index)

    def is_same_node(self, other: BaseNode, debug: bool = False, indent = 0) -> bool:
        return False

    def is_similar_node(self, other: BaseNode, debug: bool = False, indent = 0) -> bool:
        return False

    def get_events(self) -> List[Event]:
        results = []
        for r in self.ref:
            results.extend(r.get_events_by_index(self.index))
        return results

    def get_all_events(self, index: Optional[int] = None) -> List[Event]:
        results = []
        if index is not None:
            for r, i in zip(self.ref, self.index):
                results.extend(r.get_events_by_index(i + index))
        else:
            for r, i in zip(self.ref, self.index):
                results.extend(r.get_events_by_index(i))
        return results

    def get_children(self) -> List[BaseNode]:
        pass

    def set_children(self, children: List[BaseNode]):
        pass

    def add_child(self, child: BaseNode):
        pass

    def get_first_event_name(self):
        return self.ref[0].get_first_event_name()

    def events_visitor(self, index: int = 0) -> Generator[Event, None, None]:
        for r, i in zip(self.ref, self.index):
            yield from r.events_visitor(i + index)

    def get_node_count(self) -> int:
        return sum(r.get_node_count() for r in self.ref)

    def compress_event_values(self):
        """Compress event values."""
        return

    def show(self, indent: int = 0):
        prefix = " " * indent
        ref_ids = [getattr(r, "id", "unknown") for r in self.ref]
        print(f"{prefix}GroupRefNode -> Templates {ref_ids} [index={self.index}]")

    def to_dict(self) -> Dict:
        # assert hasattr(self.ref, "id"), "RefNode must have a ref_node_id."
        return {
            "ref_node_ids": [r.id for r in self.ref],
            "index": self.index
        }

    @classmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[str, Any] | None = None,
                  templates: Dict[str, TemplateNode] | None = None):
        assert "ref_node_ids" in data, "RefNode must have a ref_node_id."
        assert "index" in data, "RefNode must have an index."

        ref_ids = data["ref_node_ids"]
        assert all(ref_id in templates_dict for ref_id in ref_ids), (
            f"RefNode ref_node_ids {ref_ids} not found in "
            f"templates_dict {templates_dict.keys()}."
        )

        for ref_id in ref_ids:
            if int(ref_id) not in templates:
                templates[ref_id] = TemplateNode.from_dict(
                    templates_dict[ref_id],
                    templates_dict,
                    templates
                )

        return GroupRefNode([templates[ref_id] for ref_id in ref_ids], data["index"])

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
    
    def to_dict(self):
        d = {
            "template_index": self.template_index,
            "instance_index": self.instance_index,
        }

        if self.children:
            d["children"] = [c.to_dict() for c in self.children]

        if self.slots:
            d["slots"] = [s.to_dict() for s in self.slots]

        return d

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}CPUNode(template_index={self.template_index}, instance_index={self.instance_index})")
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

    def to_dict(self):
        d = {
            "template_index": self.template_index,
            "instance_index": self.instance_index.tolist(),
        }

        if self.children:
            d["children"] = [c.to_dict() for c in self.children]

        if self.slots:
            d["slots"] = [
                [s.to_dict() for s in slot]
                for slot in self.slots
            ]

        return d

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}SameCPUNode(template_index={self.template_index}, instance_index_len={len(self.instance_index)})")
        if self.children:
            for child in self.children:
                child.show(indent + 1)

class GroupCPUNode:

    __slots__ = ["template_index", "instance_index", "children", "slots"]

    def __init__(self, template_index, instance_index):
        # template_index: list[int] | np.ndarray
        # instance_index: list[int] | np.ndarray
        self.template_index = np.asarray(template_index, dtype=np.int32)
        self.instance_index = np.asarray(instance_index, dtype=np.int32)
        self.children = None
        self.slots = None

    def add_child(self, child):
        if self.children is None:
            self.children = []
        self.children.append(child)

    def to_dict(self):
        d = {
            "template_index": self.template_index.tolist(),
            "instance_index": self.instance_index.tolist(),
        }

        if self.children:
            d["children"] = [c.to_dict() for c in self.children]

        if self.slots:
            d["slots"] = [
                [s.to_dict() for s in slot]
                for slot in self.slots
            ]

        return d

    def show(self, indent=0):
        prefix = "  " * indent
        print(
            f"{prefix}GroupCPUNode("
            f"template_index_len={len(self.template_index)}, "
            f"instance_index_len={len(self.instance_index)})"
        )
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

    def to_dict(self):
        return {
            "template_index": self.template_index,
            "instance_index": self.instance_index,
        }

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}KernelNode(template_index={self.template_index}, instance_index={self.instance_index})")

class GPUNode:

    __slots__ = ["template_index", "instance_index", "event_start"]

    def __init__(self):
        self.template_index = []
        self.instance_index = []
        self.event_start = None

    def add_event(self, template_index, instance_index):
        self.template_index.append(template_index)
        self.instance_index.append(instance_index)

    def set_start_event(self, event_start):
        self.event_start = event_start

    def to_dict(self):
        return {
            "template_index": self.template_index,
            "instance_index": self.instance_index,
            "event_start": None
        }

    # ===== show 函数 =====
    def show(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}GPUNode(template_index_len={len(self.template_index)}, instance_index_len={len(self.instance_index)})")
