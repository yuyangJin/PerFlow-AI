"""Compressor implementations for trace compression.

This module defines abstract and concrete compressor classes used to
perform intra-rank and inter-rank compression on trace structures.
Each compressor follows a unified interface and provides both compression
and decompression methods.
"""

from __future__ import annotations
import copy
from typing import List, Dict, Union, Optional, Tuple, Any
from abc import ABC, abstractmethod
import time
import re
from collections import defaultdict
from .trace import BaseTrace, Trace, CompressedTrace, TraceFileData, TraceLoadStats
from .node import Node, CPUNode, GPUNode, SameCPUNode, KernelLaunchNode, KernelsLaunchNode
from .event import Event, MergeEvent, KernelEvent, MergeKernelEvent, is_same_event, memory_breakdown_templates
from .utils import logger, log_memory_diff

class Compressor(ABC):
    """Abstract base class for all compressors.

    Subclasses must implement methods for intra-rank and inter-rank
    compression and decompression.
    """

    @abstractmethod
    def intra_compress(self, trace: BaseTrace) -> BaseTrace:
        """Perform intra-rank compression."""
        raise NotImplementedError

    def inter_compress(self, trace: BaseTrace, merge_ranks: bool = False) -> BaseTrace:
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
        self.gpu_events: Dict[int, KernelEvent] = {}
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
        self.last_memory_before: Dict[str, int] = {}
        self.last_memory_after: Dict[str, int] = {}
        self.last_gpu_event_count = 0

    def _reset_template_state(self) -> None:
        self.event_templates = []
        self.name2indexes = defaultdict(list)

    def _compress_rank(self, trace: BaseTrace, rank: str) -> \
        Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        self.build_tree_time = 0
        self.compress_tree_time = 0
        self.find_template_time = 0
        self.check_same_node_time = 0
        self.gpu_events = {}
        self.gpu_visited = set()
        self.corr2node = {}
        self.corr_info = {}
        self.corr_info_set = set()

        strat_time = time.time()

        self.templates = {}
        self.name2id = {}
        self.templates_refs = {}

        # pid -> tid -> ph -> node
        compressed_ranks: Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]] = {}

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
                self.gpu_events[corr] = KernelEvent(event, pid, tid.split(" ")[1], ph)

        self.last_gpu_event_count = len(self.gpu_events)

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

        compress_time = time.time() - strat_time

        return compressed_ranks

    def intra_compress(
        self,
        trace: BaseTrace,
        rank: str = "",
        emit_summary: bool = True,
    ) -> BaseTrace:

        if rank == "":
            rank = trace.get_ranks()[0]
        compressed_rank = self._compress_rank(trace, rank)
        ranks = {rank: compressed_rank}

        before = memory_breakdown_templates(self.event_templates)

        for m in self.event_templates:
            m.compress_values()

        after = memory_breakdown_templates(self.event_templates)
        self.last_memory_before = before
        self.last_memory_after = after

        logger.info(
            "Compressed rank %s | templates=%d | gpu_events=%d",
            rank,
            len(self.event_templates),
            self.last_gpu_event_count,
        )

        if emit_summary:
            log_memory_diff(logger, before, after)


        return CompressedTrace(self.event_templates, ranks, trace.get_metadata(), trace.get_start_time())

    def _finalize_template_values(self, emit_summary: bool = True) -> None:
        before = memory_breakdown_templates(self.event_templates)
        for merged_event in self.event_templates:
            merged_event.compress_values()
        after = memory_breakdown_templates(self.event_templates)
        self.last_memory_before = before
        self.last_memory_after = after
        if emit_summary:
            log_memory_diff(logger, before, after)

    def _inter_compress_shared(self, trace: Trace) -> CompressedTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"

        self._reset_template_state()
        all_compressed_ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]] = {}

        for rank in trace.get_ranks():
            compressed_ranks = self._compress_rank(trace, rank)
            all_compressed_ranks[rank] = compressed_ranks

        logger.info("After compressing all ranks, have %d templates", len(self.event_templates))

        self._finalize_template_values()

        return CompressedTrace(
            self.event_templates,
            all_compressed_ranks,
            trace.get_metadata(),
            trace.get_start_time(),
        )

    @staticmethod
    def _copy_rank_trace(trace: Trace, rank: str) -> Trace:
        rank_trace = Trace(
            metadata={rank: trace.get_metadata()[rank]},
            start_timestamp={rank: trace.get_start_time()[rank]},
        )
        rank_trace.set_ranks({rank: trace.ranks[rank]})
        return rank_trace

    @staticmethod
    def _trace_from_file_data(file_data: TraceFileData) -> Trace:
        return Trace._build_from_loaded_files([file_data])

    @staticmethod
    def _canonicalize_rank_trace(trace: Trace, rank: str) -> Trace:
        """Rebuild a decompressed rank trace into the same normalized form as file loading."""
        events: List[Dict[str, Any]] = []
        start_timestamp = trace.get_start_time()[rank]

        for _, pid, tid, ph, rank_events in trace.iter_events(rank):
            for event in rank_events:
                event_dict = event.to_dict().copy()
                if "pid" not in event_dict:
                    event_dict["pid"] = pid
                normalized_tid = tid
                if normalized_tid.startswith("stream "):
                    normalized_tid = normalized_tid.split(" ", maxsplit=1)[1]
                if normalized_tid.endswith("-abnormal"):
                    normalized_tid = normalized_tid.split("-", maxsplit=1)[0]
                if "tid" not in event_dict:
                    event_dict["tid"] = normalized_tid
                if "ph" not in event_dict:
                    event_dict["ph"] = ph
                event_dict["ts"] += start_timestamp
                events.append(event_dict)

        metadata = {rank: trace.get_metadata()[rank]}
        return Trace(events={rank: events}, metadata=metadata)

    @classmethod
    def _compress_file_data(
        cls,
        file_data: TraceFileData,
        emit_summary: bool = False,
    ) -> Tuple[CompressedTrace, TraceLoadStats]:
        stats = TraceLoadStats()
        stats.add_file(file_data)
        trace = cls._trace_from_file_data(file_data)
        compressor = cls()
        compressed_trace = compressor.intra_compress(
            trace,
            file_data.rank,
            emit_summary=emit_summary,
        )
        compressed_trace._memory_before = compressor.last_memory_before
        compressed_trace._memory_after = compressor.last_memory_after
        return compressed_trace, stats

    @staticmethod
    def _offset_node_templates(node: Any, template_offset: int) -> Any:
        shifted = copy.deepcopy(node)

        def visit(current: Any) -> None:
            if hasattr(current, "template_index"):
                template_index = getattr(current, "template_index")
                if isinstance(template_index, int):
                    if template_index >= 0:
                        current.template_index += template_offset
                else:
                    current.template_index = template_index + template_offset

            if hasattr(current, "gpu_template_index"):
                gpu_template_index = getattr(current, "gpu_template_index")
                if isinstance(gpu_template_index, int):
                    if gpu_template_index >= 0:
                        current.gpu_template_index += template_offset
                else:
                    current.gpu_template_index = gpu_template_index + template_offset

            children = getattr(current, "children", None)
            if children:
                for child in children:
                    visit(child)

            slots = getattr(current, "slots", None)
            if not slots:
                return
            if isinstance(slots[0], list):
                for slot in slots:
                    for child in slot:
                        visit(child)
                return
            for child in slots:
                visit(child)

        visit(shifted)
        return shifted

    def _merge_independent_compressed_traces(
        self,
        compressed_traces: List[CompressedTrace],
    ) -> CompressedTrace:
        event_templates: List[MergeEvent] = []
        merged_ranks: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        metadata: Dict[str, Any] = {}
        start_timestamp: Dict[str, int] = {}

        for compressed_trace in compressed_traces:
            offset = len(event_templates)
            event_templates.extend(copy.deepcopy(compressed_trace.event_templates))
            metadata.update(compressed_trace.get_metadata())
            start_timestamp.update(compressed_trace.get_start_time())

            for rank, processes in compressed_trace.ranks.items():
                merged_ranks.setdefault(rank, {})
                for pid, threads in processes.items():
                    merged_ranks[rank].setdefault(pid, {})
                    for tid, phases in threads.items():
                        merged_ranks[rank][pid].setdefault(tid, {})
                        for ph, node in phases.items():
                            merged_ranks[rank][pid][tid][ph] = self._offset_node_templates(
                                node,
                                offset,
                            )

        return CompressedTrace(event_templates, merged_ranks, metadata, start_timestamp)

    def _append_independent_rank(
        self,
        compressed_trace: CompressedTrace,
        independent_rank_trace: CompressedTrace,
    ) -> CompressedTrace:
        if not compressed_trace.event_templates and not compressed_trace.ranks:
            return independent_rank_trace
        return self._merge_independent_compressed_traces([compressed_trace, independent_rank_trace])

    def _compress_ranks_independently(self, trace: Trace) -> CompressedTrace:
        compressed_traces: List[CompressedTrace] = []
        for rank in trace.get_ranks():
            rank_compressor = TemplateCompressor()
            rank_trace = self._copy_rank_trace(trace, rank)
            compressed_traces.append(rank_compressor.intra_compress(rank_trace, rank))
        return self._merge_independent_compressed_traces(compressed_traces)

    def _merge_independent_ranks(self, compressed_trace: CompressedTrace) -> CompressedTrace:
        merge_compressor = TemplateCompressor()
        merge_compressor._reset_template_state()
        merged_ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]] = {}

        for rank in compressed_trace.get_ranks():
            rank_only = CompressedTrace(
                compressed_trace.event_templates,
                {rank: compressed_trace.ranks[rank]},
                {rank: compressed_trace.get_metadata()[rank]},
                {rank: compressed_trace.get_start_time()[rank]},
            )
            raw_rank_trace = TemplateCompressor().intra_decompress(rank_only, rank)
            canonical_rank_trace = self._canonicalize_rank_trace(raw_rank_trace, rank)
            merged_ranks[rank] = merge_compressor._compress_rank(canonical_rank_trace, rank)

        merge_compressor._finalize_template_values()
        return CompressedTrace(
            merge_compressor.event_templates,
            merged_ranks,
            compressed_trace.get_metadata(),
            compressed_trace.get_start_time(),
        )

    def inter_compress(self, trace: BaseTrace, merge_ranks: bool = False) -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        independently_compressed = self._compress_ranks_independently(trace)
        if not merge_ranks:
            return independently_compressed
        return self._merge_independent_ranks(independently_compressed)

    def compress_file(
        self,
        path: str,
        emit_summary: bool = True,
    ) -> Tuple[CompressedTrace, TraceLoadStats]:
        """Compress one trace file using the same pipeline as directory mode."""
        file_data = Trace.load_file_data(path)
        return self._compress_file_data(file_data, emit_summary=emit_summary)

    def merge_compressed_files(
        self,
        paths: List[str],
    ) -> CompressedTrace:
        """Merge independently compressed rank files into a shared compressed trace."""
        merge_compressor = TemplateCompressor()
        merge_compressor._reset_template_state()
        started_at = time.perf_counter()
        total_paths = len(paths)
        merged_ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]] = {}
        merged_metadata: Dict[str, Any] = {}
        merged_start_timestamp: Dict[str, int] = {}

        for index, path in enumerate(sorted(paths), start=1):
            compressed_trace = CompressedTrace.from_file(path)
            rank = compressed_trace.get_ranks()[0]
            raw_rank_trace = TemplateCompressor().intra_decompress(compressed_trace, rank)
            canonical_rank_trace = self._canonicalize_rank_trace(raw_rank_trace, rank)
            merged_ranks[rank] = merge_compressor._compress_rank(canonical_rank_trace, rank)
            logger.info(
                "Merged rank %s | templates=%d",
                rank,
                len(merge_compressor.event_templates),
            )
            merged_metadata.update(compressed_trace.get_metadata())
            merged_start_timestamp.update(compressed_trace.get_start_time())
            elapsed = time.perf_counter() - started_at
            average = elapsed / index if index else 0.0
            eta = average * max(total_paths - index, 0)
            logger.info(
                "merge progress %d/%d | elapsed=%.1fs | eta=%.1fs",
                index,
                total_paths,
                elapsed,
                eta,
            )

        merge_compressor._finalize_template_values(emit_summary=True)
        self.last_memory_before = merge_compressor.last_memory_before
        self.last_memory_after = merge_compressor.last_memory_after
        return CompressedTrace(
            merge_compressor.event_templates,
            merged_ranks,
            merged_metadata,
            merged_start_timestamp,
        )

    def inter_compress_dir(
        self,
        path: str,
        max_workers: Optional[int] = None,
        merge_ranks: bool = False,
    ) -> Tuple[CompressedTrace, TraceLoadStats]:
        """Compress a multi-rank trace directory with threaded file workers."""
        load_stats = TraceLoadStats()
        independent_result = CompressedTrace([], {}, {}, {})
        merge_compressor = TemplateCompressor() if merge_ranks else None
        merged_ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]] = {}
        merged_metadata: Dict[str, Any] = {}
        merged_start_timestamp: Dict[str, int] = {}
        aggregated_before: Dict[str, int] = defaultdict(int)
        aggregated_after: Dict[str, int] = defaultdict(int)
        for file_data in Trace.iter_dir_data(path, max_workers=max_workers):
            independent_rank_trace, file_stats = self._compress_file_data(
                file_data,
                emit_summary=False,
            )
            load_stats.file_count += file_stats.file_count
            load_stats.event_count += file_stats.event_count
            load_stats.source_size_bytes += file_stats.source_size_bytes
            load_stats.loaded_memory_bytes += file_stats.loaded_memory_bytes

            independent_result = self._append_independent_rank(
                independent_result,
                independent_rank_trace,
            )
            for key, value in getattr(independent_rank_trace, "_memory_before", {}).items():
                aggregated_before[key] += value
            for key, value in getattr(independent_rank_trace, "_memory_after", {}).items():
                aggregated_after[key] += value

            if merge_ranks and merge_compressor is not None:
                rank = independent_rank_trace.get_ranks()[0]
                raw_rank_trace = TemplateCompressor().intra_decompress(
                    independent_rank_trace,
                    rank,
                )
                canonical_rank_trace = self._canonicalize_rank_trace(raw_rank_trace, rank)
                rank_ranks = merge_compressor._compress_rank(canonical_rank_trace, rank)
                if rank not in merged_ranks:
                    merged_ranks[rank] = rank_ranks
                else:
                    for pid, threads in rank_ranks.items():
                        merged_ranks[rank].setdefault(pid, {}).update(threads)
                merged_metadata.update(independent_rank_trace.get_metadata())
                merged_start_timestamp.update(independent_rank_trace.get_start_time())
                del raw_rank_trace

        if not merge_ranks or merge_compressor is None:
            self.last_memory_before = dict(aggregated_before)
            self.last_memory_after = dict(aggregated_after)
            if self.last_memory_before or self.last_memory_after:
                log_memory_diff(logger, self.last_memory_before, self.last_memory_after)
            return independent_result, load_stats

        merge_compressor._finalize_template_values(emit_summary=True)
        self.last_memory_before = merge_compressor.last_memory_before
        self.last_memory_after = merge_compressor.last_memory_after
        return (
            CompressedTrace(
                merge_compressor.event_templates,
                merged_ranks,
                merged_metadata,
                merged_start_timestamp,
            ),
            load_stats,
        )

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
        tmpl = None
        if isinstance(e, Event):
            tmpl = MergeEvent([e])
        elif isinstance(e, KernelEvent):
            tmpl = MergeKernelEvent([e])
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
            gpu_template_indexes = []
            gpu_instance_indexes = []

            skipped_corr = 0

            for e in events:
                args = e.args
                if args is None:
                    temp_index, inst_id = timed_add_event(e)
                    gpu_template_indexes.append(temp_index)
                    gpu_instance_indexes.append(inst_id)
                    continue

                corr = args.get("correlation", None)
                if corr is not None and corr in self.gpu_visited:
                    skipped_corr += 1
                    continue

                temp_index, inst_id = timed_add_event(e)
                gpu_template_indexes.append(temp_index)
                gpu_instance_indexes.append(inst_id)

            gpu_node.set_events(gpu_template_indexes, gpu_instance_indexes)

            t_gpu = time.perf_counter() - t_gpu_start
            t_total = time.perf_counter() - t_build_start

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

            cpu_template_index, cpu_instance_index = timed_add_event(e)
            new_ref_node = CPUNode(cpu_template_index, cpu_instance_index)

            e_args = e.args
            if e_args is not None and "correlation" in e_args:
                corr = e_args["correlation"]
                if corr in self.gpu_events:
                    self.gpu_visited.add(corr)
                    gpu_template_index, gpu_instance_index = timed_add_event(self.gpu_events[corr])
                    new_ref_node.add_child(
                        KernelLaunchNode(
                            template_index=cpu_template_index,
                            instance_index=cpu_instance_index,
                            gpu_template_index=gpu_template_index,
                            gpu_instance_index=gpu_instance_index,
                        )
                    )
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

        root = self._compress_node_new(root)

        return root, abnormal_ref_root

    def _divide_events(self, events: List[Event], is_gpu: bool) -> Node:
        new_node = CPUNode(-1, -1)
        for e in events:
            temp_index, inst_id = self._add_event(e)
            temp = CPUNode(temp_index, inst_id)
            new_node.add_child(temp)
        return new_node

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
        if group[0].is_kernel_node():
            return KernelsLaunchNode(
                group[0].template_index,
                [n.instance_index for n in group],
                [n.gpu_template_index for n in group],
                [n.gpu_instance_index for n in group],
            )

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

    def _decompress_rank(self, compressed_trace: BaseTrace, rank: str):
        assert isinstance(compressed_trace, CompressedTrace), \
            "Compressed trace must be of type CompressedTrace"

        new_rank: Dict[str, Dict[str, Dict[str, Node]]] = {}
        for _, pid, tid, ph, node in compressed_trace.iter_nodes(rank):
            events = []
            for e in node.event_visitor(compressed_trace.event_templates, True):
                if is_same_event(e, KernelEvent):
                    print("bbbbbb")
                events.append(e)
            new_rank.setdefault(pid, {}).setdefault(tid, {})[ph] = events

        return new_rank

    def intra_decompress(self, compressed_trace: BaseTrace, rank: str = "") -> BaseTrace:
        assert isinstance(compressed_trace, CompressedTrace), \
            "Compressed trace must be of type CompressedTrace"

        if rank == "":
            rank = compressed_trace.get_ranks()[0]
        new_rank = self._decompress_rank(compressed_trace, rank)
        ranks = {rank: new_rank}
        trace = Trace(metadata=compressed_trace.get_metadata(), \
                      start_timestamp=compressed_trace.get_start_time())
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

        trace = Trace(
            metadata=compressed_trace.get_metadata(),
            start_timestamp=compressed_trace.get_start_time(),
        )
        trace.set_ranks(all_ranks)
        return trace
