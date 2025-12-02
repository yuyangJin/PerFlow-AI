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
from .event import BaseEvent, Event, MergeEvent
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
    def get_all_events(self) -> List[BaseEvent]:
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
    def get_node_count(self) -> int:
        """Get the total number of nodes in this subtree."""
        return 1

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
        self.children: List[Union[Node, RefNode]] = []

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

    def get_children(self) -> List[Union[Node, RefNode]]:
        return self.children

    def set_children(self, children: List[Union[Node, RefNode]]):
        self.children = children

    def add_child(self, child: Union[Node, RefNode]):
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
            if not event.is_same_event(other.get_events()[i], debug):
                if debug:
                    logger.debug("%s Node is_same_node: Different event: %s vs %s",
                                 ind, event, other.get_events()[i])
                return False

        if len(self.children) != len(other.get_children()):
            if debug:
                logger.debug("%s Node is_same_node: Different child count: %d vs %d",
                             ind, len(self.children), len(other.get_children()))
            return False

        for i, child in enumerate(self.children):
            if not child.is_same_node(other.get_children()[i], debug=debug, indent=indent+2):
                if debug:
                    logger.debug("%s Node is_same_node: Different child: %s vs %s",
                                 ind, child, other.get_children()[i])
                return False

        return True

    def to_dict(self) -> Dict:
        return {
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children]
        }

    @classmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[str, Any] | None = None,
                  templates: Dict[str, TemplateNode] | None = None):
        assert "events" in data, "Node must have events."
        assert "children" in data, "Node must have children."

        obj = cls.__new__(cls)
        obj.events = [Event.from_dict(e) for e in data["events"]]
        obj.children = []
        for c in data["children"]:
            if "ref_node_id" in c:
                obj.children.append(RefNode.from_dict(c, templates_dict, templates))
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
        self.node_count = sum(n.get_node_count() for n in nodes)

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

    def get_all_events(self) -> List[BaseEvent]:
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
        # TODO: 这里需要做一些限制，比如不能直接设置到TemplateNode，只能通过merge_nodes
        pass

    def add_child(self, child: BaseNode):
        # TODO: 这里需要做一些限制，比如不能直接添加到TemplateNode，只能通过merge_nodes
        pass

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
            if not event.is_same_event(other.get_events()[i], debug):
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

    def segmented_linear_predictor_compress(self):
        """Compress this node using segmented linear predictor."""
        for e in self.events:
            e.segmented_linear_predictor_compress()
        for c in self.children:
            c.segmented_linear_predictor_compress()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children]
        }

    @classmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[int, Any] | None = None,
                  templates: Dict[int, TemplateNode] | None = None):
        assert "events" in data, "TemplateNode must have events."
        assert "children" in data, "TemplateNode must have children."
        obj = cls.__new__(cls)
        obj.events = [MergeEvent.from_dict(e) for e in data["events"]]
        obj.children = []
        for c in data["children"]:
            if "ref_node_id" in c:
                obj.children.append(RefNode.from_dict(c, templates_dict, templates))
            else:
                obj.children.append(TemplateNode.from_dict(c, templates_dict, templates))
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

    def is_same_node(self, other: BaseNode, debug: bool = False) -> bool:
        # TODO:
        pass

    def get_events(self) -> List[BaseEvent]:
        return self.ref.get_events_by_index(self.index)

    def get_all_events(self, index: Optional[int] = None) -> List[BaseEvent]:
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

    def segmented_linear_predictor_compress(self):
        """Compress this node using segmented linear predictor."""
        return

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
