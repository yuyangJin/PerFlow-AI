from .trace import BaseTrace, Trace, CompressedTrace
from .node import BaseNode, Node, MergeNode, RefNode
from .event import Event, MergeEvent
from .utils import logger
from typing import List, Dict, Union
from abc import ABC, abstractmethod


class Compressor(ABC):
    
    @abstractmethod
    def intra_compress(self, trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError
    
    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError

    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError
    
    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError


class TemplateCompressor(Compressor):

    def intra_compress(self, trace: BaseTrace, rank: int = 0) -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        # rank -> pid -> tid -> node
        compressed_ranks: Dict[int, Dict[int, Dict[int, BaseNode]]] = {}
        
        # TODO: compress when building call tree
        # 1. build call tree
        # 1.1 get each pid and tid node
        # 1.2 build call tree for each pid and tid node
        for r, pid, tid, node in trace.iter_nodes(rank):

            compressed_ranks.setdefault(r, {}).setdefault(pid, {})

            root = self.build_call_tree(node)
            compressed_ranks[r][pid][tid] = root

            # TODO: compress call tree
        

        # TODO:
        # 2. compress call tree
        # 2.1 combine nodes if there are multiple same nodes
        # 2.2 find template nodes and compress them
        return CompressedTrace(compressed_ranks, trace.get_metadata())

    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        # TODO
        pass

    def build_call_tree(self, node: Node) -> Node:
        events = sorted(node.events, key=lambda e: e.get_ts())
        root = Node(node.get_events()[0:1])
        stack: List[Node] = [root]

        for e in events:
            while len(stack) > 1:
                top: Event = stack[-1].events[0]
                if top.get_ts() + top.get_dur() <= e.get_ts():
                    stack.pop()
                else:
                    break

            child: Node = Node([e])

            stack[-1].add_child(child)
            stack.append(child)

        root = self._flatten_tree(root)
        return root
    
    def _flatten_tree(self, node: Node):
        if (len(node.get_children())) == 0:
            return node
        
        while (len(node.get_children())) == 1:
            logger.debug(f"flattening {node.get_children()[0].get_events()[0].get_name()} to {node.get_events()[-1].get_name()}")
            node.add_events(node.get_children()[0].get_events())
            new_children = node.get_children()[0].get_children()
            node.children = new_children

        new_children = []
        for child in node.get_children():
            new_children.append(self._flatten_tree(child))

        node.children = new_children

        return node

    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        assert isinstance(compressed_trace, Trace), "Compressed trace must be of type Trace"
        # TODO
        pass
    
    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        assert isinstance(compressed_trace, Trace), "Compressed trace must be of type Trace"
        # TODO
        pass