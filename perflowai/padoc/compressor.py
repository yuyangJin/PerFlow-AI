"""Compressor implementations for trace compression.

This module defines abstract and concrete compressor classes used to
perform intra-rank and inter-rank compression on trace structures.
Each compressor follows a unified interface and provides both compression
and decompression methods.
"""

from __future__ import annotations
from typing import List, Dict, Union, Optional, Tuple
from abc import ABC, abstractmethod
import time
import re
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
        self.templates: Dict[str, TemplateNode] = {}
        self.next_template_id = 0
        self.name2id: Dict[str, List[str]] = {}

        self.build_tree_time = 0
        self.compress_tree_time = 0
        self.find_template_time = 0
        self.check_same_node_time = 0

    def intra_compress(self, trace: BaseTrace, rank: str = "0") -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        self.build_tree_time = 0
        self.compress_tree_time = 0
        self.find_template_time = 0
        self.check_same_node_time = 0

        strat_time = time.time()

        self.templates = {}
        self.next_template_id = 0
        self.name2id = {}

        # rank -> pid -> tid -> node
        compressed_ranks: Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]] = {}

        logger.info("Intra compressing rank %s", rank)

        for r, pid, tid, ph, node in trace.iter_nodes(rank):
            logger.info("Compressing pid %s tid %s ph %s", pid, tid, ph)

            compressed_ranks.setdefault(
                str(r), {}).setdefault(str(pid), {}).setdefault(str(tid), {})

            if ph == "X":
                build_start_time = time.time()
                root = self._build_call_tree(node)
                self.build_tree_time += time.time() - build_start_time
            else:
                root = self._divide_events(node)
            conpress_start_time = time.time()
            root = self._compress_node(root)
            self.compress_tree_time += time.time() - conpress_start_time
            compressed_ranks[str(r)][str(pid)][str(tid)][ph] = root

        compress_time = time.time() - strat_time
        logger.info("There are %d templates, cost %.3f s", len(self.templates), compress_time)
        logger.info("Build tree cost %.3f s, find template cost %.3f s, " \
            "compress tree cost %.3f s, check same node cost %.3f s", \
            self.build_tree_time, self.find_template_time, \
            self.compress_tree_time, self.check_same_node_time)

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
                if top_event.get_ts() + top_event.get_dur() <= e.get_ts() and e.get_dur() > 0:
                    stack.pop()
                else:
                    break

            new_node = Node(events=[e])

            if stack:
                stack[-1].add_child(new_node)
            else:
                roots.append(new_node)

            stack.append(new_node)

        logger.info("Building %d call trees", len(roots))

        if len(roots) == 1:
            root = roots[0]
        else:
            root = Node()
            for r in roots:
                root.add_child(r)

        root = self._flatten_tree(root)

        return root

    def _divide_events(self, node: Node) -> Node:
        new_node = Node()
        for e in node.get_events():
            new_node.add_child(Node(events=[e]))
        return new_node

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

                debug_flag = False
                check_start_time = time.time()
                is_same = children[i].is_same_node(children[j], debug_flag)
                self.check_same_node_time += time.time() - check_start_time
                if is_same:
                    current_group.append(children[j])
                    used[j] = True

            if len(current_group) > 1:
                used[i] = True
                groups.append(current_group)

        unused = [children[i] for i in range(n) if not used[i]]

        return groups, unused

    def _find_node_in_templates(self, node: Node, ids: List[str]) -> Optional[TemplateNode]:
        for k in ids:
            if k in self.templates:
                template = self.templates[k]
                debug_flag = False
                check_start_time = time.time()
                is_same = template.is_same_node(node, debug_flag)
                self.check_same_node_time += time.time() - check_start_time
                if is_same:
                    return template

        return None

    def _compress_node(self, node: Node) -> Node:
        # there is no ref node

        if not hasattr(node, "id"): # only root template node has id
            find_start_time = time.time()
            name = node.get_first_event_name()
            pattern = re.sub(r"\d+", "0", name)
            ids = self.name2id.get(pattern, [])
            tem = self._find_node_in_templates(node, ids)
            self.find_template_time += time.time() - find_start_time
            if tem is not None:
                index = tem.get_node_count()
                tem.add_nodes([node])

                return RefNode(tem, index)

        groups, _ = self._group_same_children(node)
        node_to_ref_node: Dict[Node, RefNode] = {}
        for group in groups:
            find_start_time = time.time()
            name = group[0].get_first_event_name()
            pattern = re.sub(r"\d+", "0", name)
            ids = self.name2id.get(pattern, [])
            tem = self._find_node_in_templates(group[0], ids)
            self.find_template_time += time.time() - find_start_time
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
                name = tem.get_first_event_name()
                self.name2id.setdefault(name, []).append(tem.id)
                self.templates[str(self.next_template_id)] = tem
                self.next_template_id += 1
                index = 0
                for n in group:
                    next_index = index + n.get_node_count()
                    node_to_ref_node[n] = RefNode(tem, index)
                    index = next_index

                self._compress_node(tem)

        new_children = []
        for child in node.get_children():
            if child in node_to_ref_node:
                new_children.append(node_to_ref_node[child])
            else:
                new_children.append(self._compress_node(child))

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
