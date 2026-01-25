"""Compressor implementations for trace compression.

This module defines abstract and concrete compressor classes used to
perform intra-rank and inter-rank compression on trace structures.
Each compressor follows a unified interface and provides both compression
and decompression methods.
"""

from __future__ import annotations
from typing import List, Dict, Union, Optional, Tuple, Any
from abc import ABC, abstractmethod
import time
import re
from collections import defaultdict
from .trace import BaseTrace, Trace, CompressedTrace
from .node import BaseNode, Node, TemplateNode, RefNode, GroupRefNode, CPUNode, GPUNode, SameCPUNode
from .event import Event, MergeEvent, is_same_event, memory_breakdown_templates
from .utils import logger, log_memory_breakdown, log_memory_diff

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
        self.event_templates: List[MergeEvent] = []
        self.name2indexes: Dict[str, List[int]] = defaultdict(list)
        self.gpu_events: Dict[int, Event] = {}
        self.gpu_visited = set()
        self.templates: Dict[str, TemplateNode] = {}
        self.templates_refs: Dict[str, List[RefNode]] = {}
        self.name2id: Dict[str, List[str]] = {}

        self.corr2node: Dict[int, CPUNode] = {}
        self.corr_info: Dict[Any, int] = {}
        self.corr_info_set = set()

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

        for rank, pid, tid, ph, events in trace.iter_events(rank):
            if not "stream" in tid:
                continue

            key = (rank, pid, tid, ph)
            for event in events:
                args: Dict[str, Any] = event.args
                if args is None:
                    continue
                corr = args.get("correlation", None)
                if corr is None:
                    continue
                if key not in self.corr_info:
                    self.corr_info[key] = corr
                    self.corr_info_set.add(corr)
                assert corr not in self.gpu_events, \
                    f"GPU events should not have same correlation {corr}"
                self.gpu_events[corr] = event

        logger.info("There are %d GPU events", len(self.gpu_events))

        for _, pid, tid, ph, events in trace.iter_events(rank):

            compressed_ranks.setdefault(
                str(pid), {}).setdefault(str(tid), {})

            is_gpu = "stream" in tid

            if is_gpu:
                continue

            if ph == "X":
                build_start_time = time.time()
                root, abnoraml_root = self._build_call_tree(events, is_gpu)
                if abnoraml_root is not None:
                    logger.warning("Abnormal root found, pid %s tid %s", pid, tid)
                    # new_root = self._compress_node_new(abnoraml_root, self.templates, self.name2id)
                    compressed_ranks[str(pid)][str(tid) + "-abnormal"] = {}
                    compressed_ranks[str(pid)][str(tid) + "-abnormal"][ph] = abnoraml_root
                self.build_tree_time += time.time() - build_start_time
            else:
                root = self._divide_events(events, is_gpu)
                root = self._compress_node_new(root)
            conpress_start_time = time.time()
            # root = self._compress_node_new(root, self.templates, self.name2id)
            self.compress_tree_time += time.time() - conpress_start_time
            compressed_ranks[str(pid)][str(tid)][ph] = root

        for rank, pid, tid, ph, events in trace.iter_events(rank):

            compressed_ranks.setdefault(
                str(pid), {}).setdefault(str(tid), {})

            is_gpu = "stream" in tid

            if not is_gpu:
                continue

            if ph == "X":
                build_start_time = time.time()
                root, abnoraml_root = self._build_call_tree(events, is_gpu)
                if abnoraml_root is not None:
                    logger.warning("Abnormal root found, pid %s tid %s", pid, tid)
                    # new_root = self._compress_node_new(abnoraml_root, self.templates, self.name2id)
                    compressed_ranks[str(pid)][str(tid) + "-abnormal"] = {}
                    compressed_ranks[str(pid)][str(tid) + "-abnormal"][ph] = abnoraml_root
                self.build_tree_time += time.time() - build_start_time
            else:
                root = self._divide_events(events, is_gpu)
                root = self._compress_node_new(root)
            conpress_start_time = time.time()
            # root = self._compress_node_new(root, self.templates, self.name2id)
            self.compress_tree_time += time.time() - conpress_start_time
            root.set_start_event(self.corr2node[self.corr_info[(rank, pid, tid, ph)]])
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
        # for tem_node in list(self.templates.values()):
        #     self._compress_node(tem_node, self.templates, self.name2id)
        for id, tem in self.templates.items():
            print(f"template {id}:")
            tem.show()
        logger.info("After compressing, have %d templates", len(self.templates))

        # for k, v in self.templates.items():
        #     self.templates[k] = self._merge_ref(v)

        # for _, compressed_ranks in ranks.items():
        #     for _, tids in compressed_ranks.items():
        #         for _, phs in tids.items():
        #             for ph, node in phs.items():
        #                 phs[ph] = self._merge_ref(node)

        before = memory_breakdown_templates(self.event_templates)

        for m in self.event_templates:
            m.compress_values()

        after = memory_breakdown_templates(self.event_templates)

        logger.info("=== Template memory AFTER value compression ===")
        log_memory_breakdown(logger, after)

        log_memory_diff(logger, before, after)


        return CompressedTrace(self.event_templates, ranks, trace.get_metadata())

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

        for _, compressed_ranks in all_compressed_ranks.items():
            for _, tids in compressed_ranks.items():
                for _, phs in tids.items():
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
                pass

            elif isinstance(merged_child, GroupRefNode):
                current_ref_list.extend(merged_child.ref)
                current_index_list.extend(merged_child.index)
                pass

            else:

                if current_ref_list:
                    group_node = GroupRefNode(current_ref_list, current_index_list)
                    new_children.append(group_node)

                    current_ref_list = []
                    current_index_list = []

                new_children.append(merged_child)

        if current_ref_list:
            group_node = GroupRefNode(current_ref_list, current_index_list)
            new_children.append(group_node)

        node.set_children(new_children)
        return node

    def _normalize_name(self, name: str) -> str:
        return re.sub(r"\d+", "0", name)

    def _find_event_template(self, e: Event) -> int | None:
        key = self._normalize_name(e.get_name())
        for tid in self.name2indexes.get(key, []):
            tmpl = self.event_templates[tid]
            if is_same_event(tmpl, e):
                return tid
        return None

    def _create_event_template(self, e: Event) -> int:
        index = len(self.event_templates)
        tmpl = MergeEvent([e])
        self.event_templates.append(tmpl)
        self.name2indexes[self._normalize_name(e.get_name())].append(index)
        return index

    def _add_event(self, e: Event) -> Tuple[int, int]:
        temp_index = self._find_event_template(e)
        if temp_index is None:
            temp_index = self._create_event_template(e)
            inst_id = 0
        else:
            self.event_templates[temp_index].add_event(e)
            inst_id = self.event_templates[temp_index].get_len() - 1

        return temp_index, inst_id

    def _convert_node_to_cpunode(self, node: Node) -> CPUNode:
        assert len(node.events) == 1, "Node should have only one event"

        e = node.events[0]

        template_index, instance_index = self._add_event(e)

        cpu_node = CPUNode(template_index, instance_index)

        for child in node.get_children():
            cpu_child = self._convert_node_to_cpunode(child)
            cpu_node.add_child(cpu_child)

        return cpu_node

    def _build_call_tree(
        self,
        events: List[Event],
        is_gpu: bool
    ) -> Tuple[Node, Optional[Node]]:

        t_build_start = time.perf_counter()

        # ========================
        # sort events
        # ========================
        t_sort_start = time.perf_counter()
        events = sorted(
            events,
            key=lambda e: (e.ts, -e.dur if e.dur is not None else 0)
        )
        t_sort = time.perf_counter() - t_sort_start

        # ========================
        # 统计 _add_event 耗时
        # ========================
        add_event_time = 0.0
        add_event_calls = 0

        def timed_add_event(e: Event):
            nonlocal add_event_time, add_event_calls
            t0 = time.perf_counter()
            res = self._add_event(e)
            add_event_time += time.perf_counter() - t0
            add_event_calls += 1
            return res

        # ========================
        # GPU branch
        # ========================
        if is_gpu:
            t_gpu_start = time.perf_counter()

            gpu_node = GPUNode()

            skipped_corr = 0

            for e in events:
                args = e.args
                if args is None:
                    temp_index, inst_id = timed_add_event(e)
                    gpu_node.add_event(temp_index, inst_id)
                    continue

                corr = args.get("correlation", None)
                if corr is not None and corr in self.gpu_visited:
                    skipped_corr += 1
                    continue

                temp_index, inst_id = timed_add_event(e)
                gpu_node.add_event(temp_index, inst_id)

            t_gpu = time.perf_counter() - t_gpu_start
            t_total = time.perf_counter() - t_build_start

            logger.info(
                "[BUILD GPU TREE] events=%d | total=%.3f ms | sort=%.3f ms | "
                "_add_event=%.3f ms (%d calls) | skipped_corr=%d",
                len(events),
                t_total * 1e3,
                t_sort * 1e3,
                add_event_time * 1e3,
                add_event_calls,
                skipped_corr,
            )

            return gpu_node, None

        # ========================
        # CPU branch
        # ========================
        t_cpu_start = time.perf_counter()

        roots: List[Node] = []
        ref_roots: List[CPUNode] = []
        stack: List[Node] = []
        ref_stack: List[CPUNode] = []

        abnormal_root = Node()
        abnormal_ref_root = CPUNode(-1, -1)

        for e in events:
            while stack:
                top_event = stack[-1].events[0]
                if (
                    (top_event.ts + top_event.dur <= e.ts and e.dur > 0)
                    or (top_event.ts + top_event.dur < e.ts)
                ):
                    stack.pop()
                    ref_stack.pop()
                else:
                    break

            new_node = Node(events=[e])

            temp_index, inst_id = timed_add_event(e)
            new_ref_node = CPUNode(temp_index, inst_id)

            e_args = e.args
            if e_args is not None and "correlation" in e_args:
                corr = e_args["correlation"]
                if corr in self.gpu_events:
                    self.gpu_visited.add(corr)
                    # new_ref_node.add_child(Node(events=[self.gpu_events[corr]]))
                    if corr in self.corr_info_set:
                        self.corr2node[corr] = new_ref_node

            if stack:
                top_event = stack[-1].events[0]
                if top_event.ts + top_event.dur < e.ts + e.dur:
                    abnormal_root.add_child(new_node)
                    abnormal_ref_root.add_child(new_ref_node)
                    continue
                else:
                    stack[-1].add_child(new_node)
                    ref_stack[-1].add_child(new_ref_node)
            else:
                roots.append(new_node)
                ref_roots.append(new_ref_node)

            stack.append(new_node)
            ref_stack.append(new_ref_node)

        # ========================
        # build root
        # ========================
        if len(ref_roots) == 1:
            root = ref_roots[0]
        else:
            root = CPUNode(-1, -1)
            for r in ref_roots:
                root.add_child(r)

        if len(abnormal_ref_root.get_children()) == 0:
            abnormal_ref_root = None

        t_cpu = time.perf_counter() - t_cpu_start
        t_total = time.perf_counter() - t_build_start

        logger.info(
            "[BUILD CPU TREE] events=%d | total=%.3f ms | sort=%.3f ms | "
            "_add_event=%.3f ms (%d calls) | cpu_build=%.3f ms | abnormal=%s",
            len(events),
            t_total * 1e3,
            t_sort * 1e3,
            add_event_time * 1e3,
            add_event_calls,
            t_cpu * 1e3,
            abnormal_ref_root is not None,
        )

        root = self._compress_node_new(root)

        return root, abnormal_ref_root

    def _divide_events(self, events: List[Event], is_gpu: bool) -> Node:
        new_node = CPUNode(-1, -1)
        for e in events:
            temp_index, inst_id = self._add_event(e)
            temp = CPUNode(temp_index, inst_id)
            new_node.add_child(temp)
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

                debug_flag = "SCH-forward_step" in children[i].get_first_event_name() and "SCH-forward_step" in children[j].get_first_event_name()
                if debug_flag:
                    print(f"{debug_flag} {children[i].get_first_event_name()} {children[j].get_first_event_name()}")
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

    def _compress_node_new(self, node):

        if len(node.get_children()) == 0:
            return node

        groups, unused = self._group_similar_nodes(node.get_children())

        temps = []
        for group in groups:
            temp = self._build_template(group)
            temps.append(temp)


        node.children = temps
        for child in unused:
            if node.slots is None:
                node.slots = []
            node.slots.append(self._compress_node_new(child))

        return node

    def _compress_nodes(self, nodes):

        groups, unused = self._group_similar_nodes(nodes)

        results = []
        for group in groups:
            temp = self._build_template(group)
            results.append(temp)


        for child in unused:
            results.append(child)

        return results

    def _build_template(self, group):
        tnode = SameCPUNode(group[0].template_index, [n.instance_index for n in group])

        group_children = [n.get_children() for n in group]


        child_sizes = [len(c) for c in group_children]

        if all(size == 0 for size in child_sizes):
            return tnode


        if any(size == 0 for size in child_sizes):
            # 不做 LCS，全部 children 进 slot
            tnode.slots = group_children
            return tnode

        matched, unmatched = self._extract_anchors(group_children)

        for i in range(len(matched[0])):
            sub_group = [matched[j][i] for j in range(len(matched))]
            sub_tem = self._build_template(sub_group)
            tnode.add_child(sub_tem)

        if unmatched and any(unmatched):
            tnode.slots = [self._compress_nodes(n) for n in unmatched]

        return tnode

    def _group_similar_nodes(self, nodes):
        n = len(nodes)
        used = [False] * n
        groups = []

        for i in range(n):
            if used[i]:
                continue

            current_group = [nodes[i]]

            for j in range(i+1, n):
                if used[j]:
                    continue

                is_similar = nodes[i].template_index == nodes[j].template_index
                if is_similar:
                    current_group.append(nodes[j])
                    used[j] = True

            if len(current_group) > 1:
                used[i] = True
                groups.append(current_group)

        unused = [nodes[i] for i in range(n) if not used[i]]

        return groups, unused


    def _extract_anchors(
        self,
        sequences: List[List[Node]]
    ) -> Tuple[List[List[Node]], List[List[Node]]]:
        """
        返回:
        matched   : 每个序列对应的 anchor 子序列
        unmatched : 每个序列剩余的 slot 内容
        """

        if not sequences:
            return [], []

        # 如果只有一个序列，没必要建模板
        if len(sequences) == 1:
            return [list(sequences[0])], [[]]

        # 1. 选最短序列作为 reference（更稳）
        ref_idx = min(range(len(sequences)), key=lambda i: len(sequences[i]))
        ref = sequences[ref_idx]

        # 2. 为每个序列维护扫描指针
        cursors = [0] * len(sequences)

        anchors = []  # List[List[Node]]，每一列

        for ref_node in ref:
            matched_nodes = [None] * len(sequences)
            ok = True

            for i, seq in enumerate(sequences):
                found = False
                for j in range(cursors[i], len(seq)):
                    if ref_node.template_index == seq[j].template_index:
                        matched_nodes[i] = seq[j]
                        cursors[i] = j + 1
                        found = True
                        break
                if not found:
                    ok = False
                    break

            if ok:
                anchors.append(matched_nodes)

        # 3. 如果 anchor 为空，整体失败 → 全进 slot
        if not anchors:
            return (
                [[] for _ in sequences],
                [list(seq) for seq in sequences]
            )

        # 4. 构造 matched / unmatched
        matched = [[] for _ in sequences]
        used_indices = [set() for _ in sequences]

        for col in anchors:
            for i, node in enumerate(col):
                matched[i].append(node)
                used_indices[i].add(node)

        unmatched = []
        for i, seq in enumerate(sequences):
            unmatched.append([n for n in seq if n not in used_indices[i]])

        return matched, unmatched

    def _flatten_slots(self, slots: List[List[Node]]) -> List[Node]:
        flat = []
        for s in slots:
            flat.extend(s)
        return flat

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
