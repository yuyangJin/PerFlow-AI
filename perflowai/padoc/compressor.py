from __future__ import annotations
from .trace import BaseTrace, Trace, CompressedTrace
from .node import BaseNode, Node, TemplateNode, RefNode
from .event import Event, MergeEvent
from .utils import logger
from typing import List, Dict, Union, Optional, Tuple
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

        self.templates: Dict[int, BaseNode] = {}
        self.next_template_id = 0

        # rank -> pid -> tid -> node
        compressed_ranks: Dict[int, Dict[int, Dict[int, BaseNode]]] = {}
        
        for r, pid, tid, node in trace.iter_nodes(rank):

            compressed_ranks.setdefault(r, {}).setdefault(pid, {})

            root = self._build_call_tree(node)
            root = self._find_template_node(root)
            compressed_ranks[r][pid][tid] = root

        logger.info(f"compressed {len(compressed_ranks)} ranks")
        logger.info(f"compressed {len(self.templates)} templates")

        return CompressedTrace(self.templates, compressed_ranks, trace.get_metadata())

    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        # TODO
        pass

    def _build_call_tree(self, node: Node) -> Node:
        events = sorted(node.events, key=lambda e: e.get_ts())
        
        roots: List[Node] = []
        stack: List[Node] = []

        for e in events:
            while stack:
                top_event = stack[-1].events[0]
                if top_event.get_ts() + top_event.get_dur() <= e.get_ts():
                    stack.pop()
                else:
                    break

            new_node = Node(events=[e])

            if stack:
                stack[-1].add_child(new_node)
            else:
                roots.append(new_node)

            stack.append(new_node)

        logger.info(f"build call tree: {len(roots)} roots")
        if len(roots) == 1:
            root = roots[0]
        else:
            root = Node()
            for r in roots:
                root.add_child(r)

        root = self._flatten_tree(root)

        return root
    
    def _group_same_children(self, node: BaseNode) -> Tuple[List[List[BaseNode]], List[BaseNode]]:
        children = node.get_children()
        n = len(children)
        used = [False] * n
        groups = []

        for i in range(n):
            if used[i]:
                continue

            current_group = [children[i]]

            for j in range(i+1, n):
                if used[j]:
                    continue
                
                if children[i].is_same_node(children[j]):
                    logger.debug(f"found same node: {children[i].get_events()[0].get_name()} and {children[j].get_events()[0].get_name()}")
                    current_group.append(children[j])
                    used[j] = True

            if len(current_group) > 1:
                used[i] = True
                groups.append(current_group)

        unused = [children[i] for i in range(n) if not used[i]]

        return groups, unused
    
    def _find_node_in_templates(self, node: Node) -> Optional[TemplateNode]:
        for template in self.templates.values():
            if template.is_same_node(node):
                return template
        return None
    
    def _find_template_node(self, node: Node) -> Node:
        # there is no ref node
        
        if not hasattr(node, "id"): # only root template node has id
            tem = self._find_node_in_templates(node)
            if tem is not None:
                index = tem.get_node_count()
                tem.add_nodes([node])

                return RefNode(tem, index)
        
        groups, _ = self._group_same_children(node)
        node_to_ref_node: Dict[Node, RefNode] = {}
        for group in groups:
            tem = self._find_node_in_templates(group[0])
            if tem is not None:
                index = tem.get_node_count()
                tem.add_nodes(group)
                for n in group:
                    node_to_ref_node[n] = RefNode(tem, index)
                    index += 1
            else:
                tem = TemplateNode(group)
                tem.id = self.next_template_id
                self.templates[self.next_template_id] = tem
                self.next_template_id += 1
                index = 0
                for n in group:
                    node_to_ref_node[n] = RefNode(tem, index)
                    index += 1

                self._find_template_node(tem)

        new_children = []
        for child in node.get_children():
            if child in node_to_ref_node:
                new_children.append(node_to_ref_node[child])
            else:
                new_children.append(self._find_template_node(child))

        node.children = new_children

        return node
    
    def _compress_templates(self):
        # TODO:
        pass
        

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

class SegmentDeltaCompressor(Compressor):
    def intra_compress(self, trace: BaseTrace) -> BaseTrace:
        pass
    
    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        pass

    def _compress_timestamps(self, timestamps: List[int]):
        """
        TODO: 改为英文
        分段压缩时间戳
        """
        # TODO: 
        # 1. 确定返回什么数据
        # 2. 实现分段压缩算法
        # 3. 实现分段解压算法，因为这里我不知道你如何设计数据结构，所以解压函数的声明没写
        pass


    
    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        pass
    
    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        pass