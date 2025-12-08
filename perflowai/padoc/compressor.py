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
from .node import BaseNode, Node, TemplateNode, RefNode, GroupRefNode
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
        self.templates_refs: Dict[str, List[RefNode]] = {}
        self.name2id: Dict[str, List[str]] = {}

        self.build_tree_time = 0
        self.compress_tree_time = 0
        self.find_template_time = 0
        self.check_same_node_time = 0

    def _compress_rank(self, trace: BaseTrace, rank: str) -> \
        Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        self.build_tree_time = 0
        self.compress_tree_time = 0
        self.find_template_time = 0
        self.check_same_node_time = 0

        strat_time = time.time()

        self.templates = {}
        self.name2id = {}
        self.templates_refs = {}

        # pid -> tid -> ph -> node
        compressed_ranks: Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]] = {}

        logger.info("Intra compressing rank %s", rank)

        for _, pid, tid, ph, node in trace.iter_nodes(rank):

            compressed_ranks.setdefault(
                str(pid), {}).setdefault(str(tid), {})

            if ph == "X":
                build_start_time = time.time()
                root, abnoraml_root = self._build_call_tree(node)
                if abnoraml_root is not None:
                    logger.warning("Abnormal root found, pid %s tid %s", pid, tid)
                    new_root = self._compress_node(abnoraml_root, self.templates, self.name2id)
                    compressed_ranks[str(pid)][str(tid) + "-abnormal"] = {}
                    compressed_ranks[str(pid)][str(tid) + "-abnormal"][ph] = new_root
                self.build_tree_time += time.time() - build_start_time
            else:
                root = self._divide_events(node)
            conpress_start_time = time.time()
            root = self._compress_node(root, self.templates, self.name2id)
            self.compress_tree_time += time.time() - conpress_start_time
            compressed_ranks[str(pid)][str(tid)][ph] = root

        compress_time = time.time() - strat_time
        logger.info("There are %d templates, cost %.3f s", len(self.templates), compress_time)
        logger.info("Build tree cost %.3f s, find template cost %.3f s, " \
            "compress tree cost %.3f s, check same node cost %.3f s", \
            self.build_tree_time, self.find_template_time, \
            self.compress_tree_time, self.check_same_node_time)

        return compressed_ranks

    def intra_compress(self, trace: BaseTrace, rank: str = "") -> BaseTrace:

        if rank == "":
            rank = trace.get_ranks()[0]
        compressed_rank = self._compress_rank(trace, rank)
        ranks = {rank: compressed_rank}

        logger.info("Compressing templates")
        for tem_node in list(self.templates.values()):
            self._compress_node(tem_node, self.templates, self.name2id)
        logger.info("After compressing, have %d templates", len(self.templates))

        for k, v in self.templates.items():
            self.templates[k] = self._merge_ref(v)

        for _, compressed_ranks in ranks.items():
            for _, tids in compressed_ranks.items():
                for _, phs in tids.items():
                    for ph, node in phs.items():
                        phs[ph] = self._merge_ref(node)

        return CompressedTrace(self.templates, ranks, trace.get_metadata())

    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"

        self.templates = {}
        final_templates: Dict[str, TemplateNode] = {}
        all_name2id: Dict[str, List[str]] = {}
        all_compressed_ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]] = {}

        for rank in trace.get_ranks():
            compressed_ranks = self._compress_rank(trace, rank)
            all_compressed_ranks[rank] = compressed_ranks
            final_templates, all_name2id = self._merge_templates(final_templates, all_name2id)

        logger.info("After merging, have %d templates, start compressing templates", \
                    len(final_templates))

        for tem_node in list(final_templates.values()):
            self._compress_node(tem_node, final_templates, all_name2id)

        for rank, compressed_ranks in all_compressed_ranks.items():
            for pid, tids in compressed_ranks.items():
                for tid, phs in tids.items():
                    for ph, node in phs.items():
                        phs[ph] = self._merge_ref(node)

        for k, v in final_templates.items():
            final_templates[k] = self._merge_ref(v)

        return CompressedTrace(final_templates, all_compressed_ranks, trace.get_metadata())

    def _merge_templates(self, all_templates: Dict[str, TemplateNode],
                        all_name2id: Dict[str, List[str]]):
        logger.info("Merging %d templates, before merged have %d templates",
                    len(self.templates), len(all_templates))

        if all_templates == {}:

            all_templates = self.templates

            all_name2id = self.name2id

            return all_templates, all_name2id

        for k, v in self.templates.items():
            name = v.get_first_event_name()
            ids = all_name2id.get(name, [])
            found = False
            for i in ids:
                if all_templates[i].is_same_node(v):
                    index = all_templates[i].get_node_count()
                    for ref_node in self.templates_refs[k]:
                        ref_node.update_template(all_templates[i], index)
                    all_templates[i].add_nodes([v])
                    found = True
                    break

            if not found:
                print(f"{name} not found, ids: {ids} {k in ids}")
                next_id = str(len(all_templates))
                all_templates[next_id] = v
                all_templates[next_id].id = next_id
                all_name2id.setdefault(name, []).append(next_id)

        logger.info("After merged have %d templates", len(all_templates))

        return all_templates, all_name2id

    def _merge_ref(self, node: BaseNode) -> BaseNode:
        if isinstance(node, RefNode) or isinstance(node, GroupRefNode):
            return node

        new_children: List[BaseNode] = []
        current_ref_list: List[TemplateNode] = []
        current_index_list: List[int] = []

        children = node.get_children()

        for child in children:
            merged_child = self._merge_ref(child)

            if isinstance(merged_child, RefNode):
                current_ref_list.append(merged_child.ref)
                current_index_list.append(merged_child.index)

            elif isinstance(merged_child, GroupRefNode):
                current_ref_list.extend(merged_child.ref)
                current_index_list.extend(merged_child.index)

            else:

                if current_ref_list:
                    group_node = GroupRefNode(current_ref_list, current_index_list)
                    print(f"merge {len(current_index_list)} refs")
                    new_children.append(group_node)

                    current_ref_list = []
                    current_index_list = []

                new_children.append(merged_child)

        if current_ref_list:
            group_node = GroupRefNode(current_ref_list, current_index_list)
            new_children.append(group_node)

        node.set_children(new_children)
        return node

    def _build_call_tree(self, node: Node) -> Tuple[Node, Optional[Node]]:
        events = sorted(node.events, key=lambda e: e.get_ts())

        roots: List[Node] = []
        stack: List[Node] = []

        abnormal_root = Node()

        for e in events:
            while stack:
                top_event = stack[-1].events[0]
                if top_event.get_ts() + top_event.get_dur() <= e.get_ts() and e.get_dur() > 0:
                    stack.pop()
                else:
                    break

            new_node = Node(events=[e])

            if stack:
                top_event = stack[-1].events[0]
                if top_event.get_ts() + top_event.get_dur() < e.get_ts() + e.get_dur():
                    logger.warning("Abnormal node found, last event %s %d\n current event %s %d", \
                                   top_event.get_name(), top_event.get_ts(), \
                                   e.get_name(), e.get_ts())
                    abnormal_root.add_child(new_node)
                    continue
                else:
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
                if len(roots) < 10:
                    print(r.get_first_event_name())
                root.add_child(r)

        if len(abnormal_root.get_children()) == 0:
            abnormal_root = None

        return root, abnormal_root

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

    def _find_node_in_templates(self,
                                node: Node,
                                templates: Dict[str, TemplateNode],
                                ids: List[str],
                                debug_flag: bool = False
        ) -> Optional[TemplateNode]:
        for k in ids:
            if k in templates:
                template = templates[k]
                check_start_time = time.time()
                is_same = template.is_same_node(node, debug_flag)
                self.check_same_node_time += time.time() - check_start_time
                if is_same:
                    return template
            else:
                logger.error("Template %s not found", k)

        return None

    def _compress_node(self,
                       node: Union[Node, TemplateNode],
                       templates: Dict[str, TemplateNode],
                       name2id: Dict[str, List[str]]
        ) -> Node:
        # there is no ref node

        if not hasattr(node, "id"): # only root template node has id
            find_start_time = time.time()
            name = node.get_first_event_name()
            pattern = re.sub(r"\d+", "0", name)
            ids = name2id.get(pattern, [])
            tem = self._find_node_in_templates(node, templates, ids)
            self.find_template_time += time.time() - find_start_time
            if tem is not None:
                index = tem.get_node_count()
                tem.add_nodes([node])

                ref_node = RefNode(tem, index)
                self.templates_refs.setdefault(tem.id, []).append(ref_node)

                return ref_node

        groups, _ = self._group_same_children(node)
        node_to_ref_node: Dict[BaseNode, RefNode] = {}
        for group in groups:
            find_start_time = time.time()
            name = group[0].get_first_event_name()
            pattern = re.sub(r"\d+", "0", name)
            debug_flag = False
            ids = name2id.get(pattern, [])
            tem = self._find_node_in_templates(group[0], templates, ids, debug_flag)
            self.find_template_time += time.time() - find_start_time
            if tem is not None:
                index = tem.get_node_count()
                tem.add_nodes(group)
                for n in group:
                    next_index = index + n.get_node_count()
                    ref_node = RefNode(tem, index)
                    self.templates_refs.setdefault(tem.id, []).append(ref_node)
                    node_to_ref_node[n] = ref_node
                    index = next_index
            else:
                tem = TemplateNode(group)
                tem.id = str(len(templates))
                name = tem.get_first_event_name()
                name2id.setdefault(name, []).append(tem.id)
                templates[tem.id] = tem
                index = 0
                for n in group:
                    next_index = index + n.get_node_count()
                    ref_node = RefNode(tem, index)
                    self.templates_refs.setdefault(tem.id, []).append(ref_node)
                    node_to_ref_node[n] = ref_node
                    index = next_index

                # self._compress_node(tem)

        new_children = []
        for child in node.get_children():
            if child in node_to_ref_node:
                new_children.append(node_to_ref_node[child])
            else:
                new_children.append(self._compress_node(child, templates, name2id))

        node.children = new_children

        return node


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

    def _decompress_rank(self, compressed_trace: BaseTrace, rank: str):
        assert isinstance(compressed_trace, CompressedTrace), \
            "Compressed trace must be of type CompressedTrace"

        new_rank: Dict[str, Dict[str, Dict[str, Node]]] = {}
        for _, pid, tid, ph, node in compressed_trace.iter_nodes(rank):
            new_node = Node()
            events = node.get_all_events()
            new_node.add_events(events)
            new_rank.setdefault(pid, {}).setdefault(tid, {})[ph] = new_node

        return new_rank

    def intra_decompress(self, compressed_trace: BaseTrace, rank: str = "") -> BaseTrace:
        assert isinstance(compressed_trace, CompressedTrace), \
            "Compressed trace must be of type CompressedTrace"

        if rank == "":
            rank = compressed_trace.get_ranks()[0]
        new_rank = self._decompress_rank(compressed_trace, rank)
        ranks = {rank: new_rank}
        trace = Trace(metadata=compressed_trace.get_metadata())
        trace.set_ranks(ranks)
        return trace

    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        assert isinstance(compressed_trace, CompressedTrace), \
            "Compressed trace must be of type CompressedTrace"

        all_ranks = {}
        for rank in compressed_trace.get_ranks():
            logger.info("decompressing rank %s", rank)
            new_rank = self._decompress_rank(compressed_trace, rank)
            all_ranks[rank] = new_rank

        trace = Trace(metadata=compressed_trace.get_metadata())
        trace.set_ranks(all_ranks)
        return trace
