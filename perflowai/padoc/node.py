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
        # TODO: type字段是不是必要的，可不可以省略
        return {
            "type": "Node",
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children]
        }
    
    def from_dict(cls, data: Dict):
        # TODO:
        pass

class TemplateNode(BaseNode):
    def __init__(self, nodes: List[Union[Node, TemplateNode]]):
        # TODO: 这里默认nodes都是Node类型，可能需要考虑支持混合类型
        # TODO: 这里其实还需要维护一个全部唯一标识符之类的，可以使用BaseNode的_global_id，但是并不是所有的Node需要维护，如果没有被Ref，就不需要id
        self.events : List[MergeEvent] = [MergeEvent()] * len(nodes[0].events)
        self.children : List[BaseNode] = []
        self.node_count = len(nodes)

        assert len(nodes) > 1, "TemplateNode requires at least two nodes."
        if len(nodes) < 2:
            logger.warning("TemplateNode __init__: Only one node provided. Please provide at least two nodes to merge.")
            self.events = nodes[0].events
            self.children = nodes[0].children
            return
        
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
        # TODO: 这里默认nodes都是Node类型，可能需要考虑支持混合类型
        logger.debug("TemplateNode merge_nodes: Merging %d nodes", len(nodes))
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
            self.children[i].merge_nodes(child_nodes_to_merge)

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
        # TODO: type字段是不是必要的，可不可以省略
        return {
            "type": "TemplateNode",
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
        # TODO: type字段是不是必要的，可不可以省略
        # TODO: 这里如何写入其实是一个问题，因为程序结构中存储的是引用，这里需要想一个办法以最低的存储开销维护这个ref关系
        return {
            "type": "RefNode",
            "ref_node_id": 0,
            "index": self.index
        }
    
    def from_dict(cls, data: Dict):
        # TODO:
        pass
