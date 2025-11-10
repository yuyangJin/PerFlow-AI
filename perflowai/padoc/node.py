from __future__ import annotations
from .event import Event, MergeEvent
from .utils import logger
from abc import ABC, abstractmethod
from typing import List, Dict, Union


class BaseNode(ABC):
    _global_id = 0

    def __init__(self):
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

    def add_child(self, child: BaseNode):
        self.children.append(child)

    def add_event(self, event: Event):
        self.events.append(event)

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

class MergeNode(BaseNode):
    def __init__(self, nodes: List[Union["Node", "MergeNode"]]):
        # TODO: 这里默认nodes都是Node类型，可能需要考虑支持混合类型
        # TODO: 这里其实还需要维护一个全部唯一标识符之类的，可以使用BaseNode的_global_id，但是并不是所有的Node需要维护，如果没有被Ref，就不需要id
        # TODO: 因为这里的merge有两种情况，一种是他们同为子节点，在同一个父节点下(init时)，另一种是他们是不同父节点下的子节点(merge_nodes时，会被Ref)，需要考虑两种情况的合并，需要区分这两种，不然树结构就损坏了
        self.events : List[MergeEvent] = []
        self.children : List[MergeNode] = []

        assert len(nodes) > 1, "MergeNode requires at least two nodes."
        if len(nodes) < 2:
            logger.warning("MergeNode __init__: Only one node provided. Please provide at least two nodes to merge.")
            self.events = nodes[0].events
            self.children = nodes[0].children
            return
        
        logger.debug("MergeNode __init__: Merging %d nodes", len(nodes))
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
            merged_child_node = MergeNode(child_nodes_to_merge)
            self.children.append(merged_child_node)

    def merge_nodes(self, nodes: List[Union["Node", "MergeNode"]]):
        # TODO: 这里默认nodes都是Node类型，可能需要考虑支持混合类型
        logger.debug("MergeNode merge_nodes: Merging %d nodes", len(nodes))
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

    def to_dict(self) -> Dict:
        # TODO: type字段是不是必要的，可不可以省略
        return {
            "type": "MergeNode",
            "events": [e.to_dict() for e in self.events]
        }
    
    def from_dict(cls, data: Dict):
        # TODO:
        pass


class RefNode(BaseNode):
    def __init__(self, ref: MergeNode, index: int):
        self.ref: MergeNode = ref
        self.index: int = index

    def to_dict(self) -> Dict:
        # TODO: type字段是不是必要的，可不可以省略
        # TODO: 这里如何写入其实是一个问题，因为程序结构中存储的是引用，这里需要想一个办法以最低的存储开销维护这个ref关系
        return {
            "type": "RefNode",
            "ref_name": self.ref,
            "index": self.index
        }
    
    def from_dict(cls, data: Dict):
        # TODO:
        pass
