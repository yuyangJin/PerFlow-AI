from __future__ import annotations
from .event import BaseEvent, Event, MergeEvent
from .utils import logger
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any


class BaseNode(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_events(self) -> List[BaseEvent]:
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
    def from_dict(cls, data: Dict):
        pass

    # TODO: 这里需要一个access接口
    
class Node(BaseNode):
    def __init__(self, events: List[Event] = None):
        self.events: List[Event] = events or []
        self.children: List[BaseNode] = []

    def get_events(self) -> List[BaseEvent]:
        return self.events

    def add_events(self, events: List[BaseEvent]):
        self.events.extend(events)

    def get_children(self) -> List[BaseNode]:
        return self.children

    def set_children(self, children: List[BaseNode]):
        self.children = children

    def add_child(self, child: BaseNode):
        self.children.append(child)

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


    def to_dict(self) -> Dict:
        return {
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children]
        }
    
    def from_dict(cls, data: Dict):
        # TODO:
        pass

class TemplateNode(BaseNode):
    def __init__(self, nodes: List[Union[Node, TemplateNode]]):
        self.events : List[MergeEvent] = []
        self.children : List[TemplateNode] = []
        self.node_count = len(nodes)
        
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
        self.node_count += len(nodes)
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
    
    def from_dict(cls, data: Dict):
        # TODO:
        pass


class RefNode(BaseNode):
    def __init__(self, ref: TemplateNode, index: int):
        self.ref: TemplateNode = ref
        self.index: int = index
    
    def is_same_node(self, other: BaseNode) -> bool:
        # TODO:
        pass

    def get_events(self) -> List[BaseEvent]:
        pass

    def add_events(self, events: List[BaseEvent]):
        pass

    def get_children(self) -> List[BaseNode]:
        pass

    def set_children(self, children: List[BaseNode]):
        pass

    def add_child(self, child: BaseNode):
        pass

    def to_dict(self) -> Dict:
        return {
            "ref_node_id": 0, # TODO: 这里需要一个ref_node_id
            "index": self.index
        }
    
    def from_dict(cls, data: Dict):
        # TODO:
        pass
