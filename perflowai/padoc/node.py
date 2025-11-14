from __future__ import annotations
from .event import BaseEvent, Event, MergeEvent
from .utils import logger
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Optional


class BaseNode(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_events(self):
        pass

    @abstractmethod
    def get_all_events(self) -> List[BaseEvent]:
        pass

    @abstractmethod
    def add_events(self, events: List[BaseEvent]):
        pass

    @abstractmethod
    def get_children(self) -> List[BaseNode]:
        pass

    @abstractmethod
    def set_children(self, children: List[BaseNode]):
        pass

    @abstractmethod
    def add_child(self, child: BaseNode):
        pass

    @abstractmethod
    def is_same_node(self, other: BaseNode) -> bool:
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[str, Any] = {}, templates: Dict[str, TemplateNode] = {}):
        pass

    # TODO: 这里需要一个access接口
    
class Node(BaseNode):
    def __init__(self, events: List[Event] = []):
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
        self.events.extend(events)

    def get_children(self) -> List[Union[Node, RefNode]]:
        return self.children

    def set_children(self, children: List[Union[Node, RefNode]]):
        self.children = children

    def add_child(self, child: Union[Node, RefNode]):
        self.children.append(child)

    def is_same_node(self, other: Union[Node, TemplateNode]) -> bool:
        if isinstance(other, RefNode):
            return other.is_same_node(self)

        if len(self.events) != len(other.get_events()):
            return False
        
        for i in range(len(self.events)):
            if not self.events[i].is_same_event(other.get_events()[i]):
                return False
        
        if len(self.children) != len(other.get_children()):
            return False
        
        for i in range(len(self.children)):
            if not self.children[i].is_same_node(other.get_children()[i]):
                return False
        
        return True


    def to_dict(self) -> Dict:
        return {
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[str, Any] = {}, templates: Dict[str, TemplateNode] = {}):
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
    def __init__(self, nodes: List[Union[Node, TemplateNode]]):
        self.events : List[MergeEvent] = []
        self.children : List[Union[TemplateNode, RefNode]] = []
        self.node_count = sum([n.get_node_count() for n in nodes])
        
        logger.debug("TemplateNode __init__: Merging %d nodes", len(nodes))
        event_count = len(nodes[0].events)
        for n in nodes[1:]:
            assert len(n.events) == event_count, "All nodes must have the same number of events to merge."

        for i in range(event_count):
            events_to_merge = [n.events[i] for n in nodes]
            merged_event = MergeEvent(events_to_merge)
            self.events.append(merged_event)

        children_count = len(nodes[0].children)
        for n in nodes[1:]:
            assert len(n.children) == children_count, "All nodes must have the same number of children to merge."

        for i in range(children_count):
            child_nodes_to_merge = [n.children[i] for n in nodes]
            merged_child_node = TemplateNode(child_nodes_to_merge)
            self.children.append(merged_child_node)

    def add_nodes(self, nodes: List[Union[Node, TemplateNode]]):
        logger.debug(f"TemplateNode add_nodes: {[type(n) for n in nodes]}")
        self.node_count += sum([n.get_node_count() for n in nodes])
        event_count = len(self.events)
        for n in nodes:
            assert len(n.events) == event_count, "All nodes must have the same number of events as original node to merge."

        for i in range(event_count):
            events_to_merge = [n.events[i] for n in nodes]
            self.events[i].add_events(events_to_merge)

        child_count = len(self.children)
        for n in nodes:
            assert len(n.children) == child_count, "All nodes must have the same number of children as original node to merge."

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
        all_events = [e.get_event_by_index(index) for e in self.events]
        for child in self.children:
            if isinstance(child, TemplateNode):
                all_events.extend(child.get_events_by_index(index))
            else:
                all_events.extend(child.get_all_events(index))
        return all_events
    
    def get_node_count(self) -> int:
        return self.node_count

    def add_events(self, events: List[BaseEvent]):
        # TODO: 这里需要做一些限制，比如不能直接添加到TemplateNode，只能通过merge_nodes
        pass

    def get_children(self) -> List[BaseNode]:
        return self.children

    def set_children(self, children: List[BaseNode]):
        # TODO: 这里需要做一些限制，比如不能直接设置到TemplateNode，只能通过merge_nodes
        pass

    def add_child(self, child: BaseNode):
        # TODO: 这里需要做一些限制，比如不能直接添加到TemplateNode，只能通过merge_nodes
        pass

    def is_same_node(self, other: BaseNode) -> bool:
        if isinstance(other, RefNode):
            return other.is_same_node(self)

        if len(self.events) != len(other.get_events()):
            return False
        
        for i in range(len(self.events)):
            if not self.events[i].is_same_event(other.get_events()[i]):
                return False
        
        if len(self.children) != len(other.get_children()):
            return False
        
        for i in range(len(self.children)):
            if not self.children[i].is_same_node(other.get_children()[i]):
                return False
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[int, Any] = {}, templates: Dict[int, TemplateNode] = {}):
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
    def __init__(self, ref: TemplateNode, index: int):
        self.ref: TemplateNode = ref
        self.index: int = index
    
    def is_same_node(self, other: BaseNode) -> bool:
        # TODO:
        pass

    def get_events(self) -> List[BaseEvent]:
        return self.ref.get_events_by_index(self.index)
    
    def get_all_events(self, index: Optional[int] = None) -> List[BaseEvent]:
        if index is not None:
            return self.ref.get_events_by_index(self.index + index)
        return self.ref.get_events_by_index(self.index)

    def add_events(self, events: List[BaseEvent]):
        pass

    def get_children(self) -> List[BaseNode]:
        pass

    def set_children(self, children: List[BaseNode]):
        pass

    def add_child(self, child: BaseNode):
        pass

    def to_dict(self) -> Dict:
        assert hasattr(self.ref, "id"), "RefNode must have a ref_node_id."
        return {
            "ref_node_id": self.ref.id,
            "index": self.index
        }
    
    @classmethod
    def from_dict(cls, data: Dict, templates_dict: Dict[str, Any] = {}, templates: Dict[str, TemplateNode] = {}):
        assert "ref_node_id" in data, "RefNode must have a ref_node_id."
        assert "index" in data, "RefNode must have an index."
        assert data["ref_node_id"] in templates_dict, f"RefNode ref_node_id {data['ref_node_id']} not found in templates_dict {templates_dict.keys()}."
        if int(data["ref_node_id"]) not in templates:
            templates[data["ref_node_id"]] = TemplateNode.from_dict(templates_dict[data["ref_node_id"]], templates_dict, templates)
        return RefNode(templates[data["ref_node_id"]], int(data["index"]))
