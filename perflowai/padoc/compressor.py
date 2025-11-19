"""Compressor implementations for trace compression.

This module defines abstract and concrete compressor classes used to
perform intra-rank and inter-rank compression on trace structures.
Each compressor follows a unified interface and provides both compression
and decompression methods.
"""

from __future__ import annotations
from typing import List, Dict, Union, Optional, Tuple
from abc import ABC, abstractmethod
from .trace import BaseTrace, Trace, CompressedTrace
from .node import BaseNode, Node, TemplateNode, RefNode
from .utils import logger


class Compressor(ABC):
    """Abstract base class for all compressors.

    Subclasses must implement methods for intra-rank and inter-rank
    compression and decompression.
    """

    @abstractmethod
    def intra_compress(self, trace: BaseTrace) -> BaseTrace:
        """Perform intra-rank compression."""
        raise NotImplementedError

    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        """Perform inter-rank compression."""
        raise NotImplementedError

    @abstractmethod
    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        """Perform intra-rank decompression."""
        raise NotImplementedError

    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        """Perform inter-rank decompression."""
        raise NotImplementedError


class TemplateCompressor(Compressor):
    """Template-based compressor using subtree deduplication.

    This compressor identifies repeated subtrees in the execution trace
    (common in transformer layers and decoding loops) and replaces them
    with references to shared ``TemplateNode`` structures.
    """

    def __init__(self):
        super().__init__()
        self.templates: Dict[str, Union[TemplateNode, RefNode]] = {}
        self.next_template_id = 0

    def intra_compress(self, trace: BaseTrace, rank: str = "0") -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"

        self.templates = {}
        self.next_template_id = 0

        # rank -> pid -> tid -> node
        compressed_ranks: Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]] = {}

        logger.info("Intra compressing rank %s", rank)

        for r, pid, tid, node in trace.iter_nodes(rank):
            logger.info("Compressing pid %s tid %s", pid, tid)

            compressed_ranks.setdefault(str(r), {}).setdefault(str(pid), {})

            root = self._build_call_tree(node)
            root = self._find_template_node(root)
            compressed_ranks[str(r)][str(pid)][str(tid)] = root

        logger.info("There are %d templates", len(self.templates))

        return CompressedTrace(self.templates, compressed_ranks, trace.get_metadata())

    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        raise NotImplementedError

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
                    next_index = index + n.get_node_count()
                    node_to_ref_node[n] = RefNode(tem, index)
                    index = next_index
            else:
                tem = TemplateNode(group)
                tem.id = str(self.next_template_id)
                self.templates[str(self.next_template_id)] = tem
                self.next_template_id += 1
                index = 0
                for n in group:
                    next_index = index + n.get_node_count()
                    node_to_ref_node[n] = RefNode(tem, index)
                    index = next_index

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
        raise NotImplementedError


    def _flatten_tree(self, node: Node):
        if (len(node.get_children())) == 0:
            return node

        while (len(node.get_children())) == 1:
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
        raise NotImplementedError

    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        assert isinstance(compressed_trace, Trace), "Compressed trace must be of type Trace"
        raise NotImplementedError

class SegmentDeltaCompressor(Compressor):
    """Segmented delta compression.

    Applies segmented delta compression to timestamps using timestamp differences.
    Additionally, leverages the periodic patterns in the IDs embedded in names to
    achieve further compression.
    """

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
        raise NotImplementedError


    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        pass

    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        pass
