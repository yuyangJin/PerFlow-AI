"""Analysis-task wrappers for the bench harness.

Each task is a thin adapter around :class:`perflowai.padoc.analysis.TraceAnalysis`
that:

1. Accepts an already-loaded :class:`Trace` (raw trace ready for analysis).
2. Returns a small ``dict`` summarizing the result (DataFrames are *not*
   returned; we collect aggregate stats so the harness can pickle them).
3. Optionally exposes ``run_in_situ(compressor, blob)`` which lets PADOC
   skip the decompress step.

The set of tasks here mirrors the paper's *analysis tasks* axis:
operator hotspots (GPU kernel breakdown), comm/compute overlap, temporal
breakdown.  Stubs for *operator-balance* and *parallel-group* are
included with ``NotImplementedError`` so the runner can still produce
the matrix layout while implementations are added.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..analysis import TraceAnalysis
from ..baselines import BaselineCompressor
from ..baselines.padoc_adapter import PADOCCompressor
from ..trace import BaseTrace, Trace


# ----------------------------------------------------------------------
# Base interface
# ----------------------------------------------------------------------


class AnalysisTask(ABC):
    """Common interface for every analysis task in the bench."""

    name: str = "base"

    @abstractmethod
    def run_on_raw(self, trace: Trace) -> Dict[str, Any]:
        """Run the analysis on a raw decompressed :class:`Trace`."""

    # ------------------------------------------------------------------
    # In-situ hooks (opt-in per task)
    # ------------------------------------------------------------------

    def has_in_situ_for(self, compressor: BaselineCompressor) -> bool:
        """Whether this task can run on the compressor's compressed format.

        Default ``False`` -- subclasses override and provide a
        :meth:`run_in_situ` implementation when they support it.
        """
        return False

    def run_in_situ(
        self,
        compressor: BaselineCompressor,
        blob: bytes,
    ) -> Dict[str, Any]:
        """Run the analysis directly on the compressed blob (no full decode).

        Default raises ``NotImplementedError``.  Subclasses overriding this
        method **must** also override :meth:`has_in_situ_for` to declare
        which compressors they support.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not have an in-situ implementation."
        )


# ----------------------------------------------------------------------
# Concrete tasks
# ----------------------------------------------------------------------


@dataclass
class _DataFrameSummary:
    rows: int
    columns: int
    head_rows: List[Dict[str, Any]]


def _df_summary(df: Any, head: int = 5) -> Dict[str, Any]:
    if df is None or not hasattr(df, "shape"):
        return {"rows": 0, "columns": 0, "head_rows": []}
    head_rows = df.head(head).to_dict(orient="records") if hasattr(df, "head") else []
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]) if len(df.shape) > 1 else 1,
        "head_rows": head_rows,
    }


class GpuKernelBreakdownTask(AnalysisTask):
    """Operator hotspot via :class:`TraceAnalysis.get_gpu_kernel_breakdown`."""

    name = "gpu_kernel_breakdown"

    def run_on_raw(self, trace: Trace) -> Dict[str, Any]:
        analysis = TraceAnalysis(trace)
        result = analysis.get_gpu_kernel_breakdown(visualize=False)
        if isinstance(result, tuple) and len(result) == 2:
            kernel_type_df, kernel_df = result
        else:
            kernel_type_df, kernel_df = result, None
        return {
            "kernel_type": _df_summary(kernel_type_df),
            "kernel": _df_summary(kernel_df),
        }


class CommCompOverlapTask(AnalysisTask):
    """Communication / computation overlap percentage per rank."""

    name = "comm_comp_overlap"

    def run_on_raw(self, trace: Trace) -> Dict[str, Any]:
        analysis = TraceAnalysis(trace)
        df = analysis.get_comm_comp_overlap(visualize=False)
        return _df_summary(df)


class TemporalBreakdownTask(AnalysisTask):
    """Idle vs compute vs non-compute breakdown per rank."""

    name = "temporal_breakdown"

    def run_on_raw(self, trace: Trace) -> Dict[str, Any]:
        analysis = TraceAnalysis(trace)
        df = analysis.get_temporal_breakdown(visualize=False)
        return _df_summary(df)


# ----------------------------------------------------------------------
# Pure-python supplementary tasks (no HTA dependency)
# ----------------------------------------------------------------------


class OperatorHotspotTask(AnalysisTask):
    """Top-N CPU + GPU operators by total duration.

    The raw implementation iterates every event once.  For PADOC we ship
    an in-situ implementation that walks the compressed structure
    directly -- it never materializes an :class:`Event` object and keeps
    the duration sum at the *template* level using numpy.
    """

    name = "operator_hotspot"

    def __init__(self, top_n: int = 20, bucket_by_pattern_for_padoc: bool = True) -> None:
        self.top_n = top_n
        self.bucket_by_pattern_for_padoc = bucket_by_pattern_for_padoc

    def run_on_raw(self, trace: Trace) -> Dict[str, Any]:
        cpu_totals: Dict[str, int] = defaultdict(int)
        gpu_totals: Dict[str, int] = defaultdict(int)
        for _rank, _pid, tid, _ph, events in trace.iter_events():
            is_gpu_stream = isinstance(tid, str) and tid.startswith("stream ")
            target = gpu_totals if is_gpu_stream else cpu_totals
            for event in events:
                if event.dur is None:
                    continue
                target[str(event.name)] += int(event.dur)
        return {
            "top_cpu": _top_n(cpu_totals, self.top_n),
            "top_gpu": _top_n(gpu_totals, self.top_n),
            "unique_cpu_ops": len(cpu_totals),
            "unique_gpu_kernels": len(gpu_totals),
        }

    def has_in_situ_for(self, compressor: BaselineCompressor) -> bool:
        return isinstance(compressor, PADOCCompressor)

    def run_in_situ(
        self,
        compressor: BaselineCompressor,
        blob: bytes,
    ) -> Dict[str, Any]:
        from ..event import MergeEvent, MergeKernelEvent
        import numpy as np

        assert isinstance(compressor, PADOCCompressor)
        ct = compressor.load_compressed_trace(blob)

        cpu_totals: Dict[str, int] = defaultdict(int)
        gpu_totals: Dict[str, int] = defaultdict(int)

        for tmpl in ct.event_templates:
            durs = np.asarray(tmpl.dur)
            if durs.size == 0:
                continue
            total_dur = int(durs.sum())
            name_key = (
                tmpl.name_pattern
                if self.bucket_by_pattern_for_padoc
                else tmpl.name_pattern
            )
            target = gpu_totals if isinstance(tmpl, MergeKernelEvent) else cpu_totals
            target[name_key] += total_dur

        return {
            "top_cpu": _top_n(cpu_totals, self.top_n),
            "top_gpu": _top_n(gpu_totals, self.top_n),
            "unique_cpu_ops": len(cpu_totals),
            "unique_gpu_kernels": len(gpu_totals),
            "in_situ_implementation": True,
        }


def _top_n(mapping: Dict[str, int], n: int) -> List[Dict[str, Any]]:
    return [
        {"name": name, "duration_us": int(duration)}
        for name, duration in sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)[:n]
    ]


class StreamLoadBalanceTask(AnalysisTask):
    """Per-stream busy-time imbalance (used as the 'operator balance' proxy)."""

    name = "stream_load_balance"

    def run_on_raw(self, trace: Trace) -> Dict[str, Any]:
        stream_totals: Dict[str, int] = defaultdict(int)
        for rank, _pid, tid, _ph, events in trace.iter_events():
            key = f"rank{rank}/{tid}"
            for event in events:
                if event.dur is not None:
                    stream_totals[key] += int(event.dur)
        return self._summarize(stream_totals)

    @staticmethod
    def _summarize(stream_totals: Dict[str, int]) -> Dict[str, Any]:
        if not stream_totals:
            return {
                "max_us": 0,
                "min_us": 0,
                "mean_us": 0.0,
                "imbalance": 0.0,
                "streams": 0,
            }
        values = list(stream_totals.values())
        max_v = max(values)
        min_v = min(values)
        mean_v = sum(values) / len(values)
        imbalance = (max_v - min_v) / mean_v if mean_v > 0 else 0.0
        return {
            "max_us": int(max_v),
            "min_us": int(min_v),
            "mean_us": float(mean_v),
            "imbalance": float(imbalance),
            "streams": len(stream_totals),
        }

    def has_in_situ_for(self, compressor: BaselineCompressor) -> bool:
        return isinstance(compressor, PADOCCompressor)

    def run_in_situ(
        self,
        compressor: BaselineCompressor,
        blob: bytes,
    ) -> Dict[str, Any]:
        """Walk the compressed structure per (rank, tid) and sum dur arrays.

        Subtle point: in PADOC's compressed form a GPU kernel event is
        referenced by the :class:`KernelLaunchNode` in the **CPU**
        stream's tree (rather than from a GPUNode), but logically its
        home is the GPU stream from which it was originally emitted.
        Sum-by-stream therefore needs to route the GPU kernel ``dur``
        back to ``rank<R>/<gpu_tid>`` -- we do that by reading the
        kernel template's per-instance ``tid`` field.
        """
        import numpy as np
        from ..node import (
            CPUNode,
            GPUNode,
            KernelLaunchNode,
            KernelsLaunchNode,
            SameCPUNode,
        )

        assert isinstance(compressor, PADOCCompressor)
        ct = compressor.load_compressed_trace(blob)
        templates = ct.event_templates

        stream_totals: Dict[str, int] = defaultdict(int)

        for rank, processes in ct.ranks.items():
            for _pid, threads in processes.items():
                for tid, phases in threads.items():
                    cpu_key = f"rank{rank}/{tid}"
                    for _ph, root in phases.items():
                        if root is None:
                            continue
                        _accumulate_dur(
                            root,
                            templates,
                            cpu_key=cpu_key,
                            rank=str(rank),
                            stream_totals=stream_totals,
                            CPUNode=CPUNode,
                            SameCPUNode=SameCPUNode,
                            GPUNode=GPUNode,
                            KernelLaunchNode=KernelLaunchNode,
                            KernelsLaunchNode=KernelsLaunchNode,
                            np=np,
                        )

        summary = self._summarize(dict(stream_totals))
        summary["in_situ_implementation"] = True
        return summary


def _accumulate_dur(
    node,
    templates,
    *,
    cpu_key: str,
    rank: str,
    stream_totals: Dict[str, int],
    CPUNode,
    SameCPUNode,
    GPUNode,
    KernelLaunchNode,
    KernelsLaunchNode,
    np,
) -> None:
    """Walk ``node`` and add every event's ``dur`` to the right stream key.

    ``cpu_key`` is the stream key associated with the surrounding tree
    (``rank<R>/<tid>``).  GPU-kernel events referenced via
    KernelLaunchNode / KernelsLaunchNode are routed to
    ``rank<R>/<their_kernel_template.tid>`` so the per-stream totals
    match what the raw walk would produce.
    """
    if isinstance(node, GPUNode):
        tids = np.asarray(node.template_index).tolist()
        iids = np.asarray(node.instance_index).tolist()
        for tid_idx, inst_idx in zip(tids, iids):
            tmpl = templates[int(tid_idx)]
            durs = np.asarray(tmpl.dur)
            if durs.size > int(inst_idx):
                key = _stream_key_from_kernel_template(tmpl, int(inst_idx), rank, cpu_key)
                stream_totals[key] += int(durs[int(inst_idx)])
        return

    if isinstance(node, KernelLaunchNode):
        # CPU launch lives on the surrounding CPU stream.
        tmpl = templates[node.template_index]
        durs = np.asarray(tmpl.dur)
        if durs.size > node.instance_index:
            stream_totals[cpu_key] += int(durs[node.instance_index])
        # GPU kernel goes to its own stream.
        kernel_tmpl = templates[node.gpu_template_index]
        kernel_durs = np.asarray(kernel_tmpl.dur)
        if kernel_durs.size > node.gpu_instance_index:
            key = _stream_key_from_kernel_template(
                kernel_tmpl, node.gpu_instance_index, rank, cpu_key
            )
            stream_totals[key] += int(kernel_durs[node.gpu_instance_index])
        return

    if isinstance(node, KernelsLaunchNode):
        tmpl = templates[node.template_index]
        durs = np.asarray(tmpl.dur)
        cpu_inst = np.asarray(node.instance_index).tolist()
        for inst_idx in cpu_inst:
            if durs.size > int(inst_idx):
                stream_totals[cpu_key] += int(durs[int(inst_idx)])
        gpu_tids = np.asarray(node.gpu_template_index).tolist()
        gpu_iids = np.asarray(node.gpu_instance_index).tolist()
        for tid_idx, inst_idx in zip(gpu_tids, gpu_iids):
            kernel_tmpl = templates[int(tid_idx)]
            kernel_durs = np.asarray(kernel_tmpl.dur)
            if kernel_durs.size > int(inst_idx):
                key = _stream_key_from_kernel_template(
                    kernel_tmpl, int(inst_idx), rank, cpu_key
                )
                stream_totals[key] += int(kernel_durs[int(inst_idx)])
        return

    if isinstance(node, SameCPUNode):
        tmpl = templates[node.template_index]
        durs = np.asarray(tmpl.dur)
        instance_idx = np.asarray(node.instance_index)
        if durs.size > 0 and instance_idx.size > 0:
            valid = instance_idx[instance_idx < durs.size]
            stream_totals[cpu_key] += int(durs[valid].sum())
        for child in (node.children or []):
            _accumulate_dur(
                child, templates,
                cpu_key=cpu_key, rank=rank, stream_totals=stream_totals,
                CPUNode=CPUNode, SameCPUNode=SameCPUNode, GPUNode=GPUNode,
                KernelLaunchNode=KernelLaunchNode, KernelsLaunchNode=KernelsLaunchNode,
                np=np,
            )
        if node.slots:
            for slot in node.slots:
                if isinstance(slot, list):
                    for entry in slot:
                        _accumulate_dur(
                            entry, templates,
                            cpu_key=cpu_key, rank=rank, stream_totals=stream_totals,
                            CPUNode=CPUNode, SameCPUNode=SameCPUNode, GPUNode=GPUNode,
                            KernelLaunchNode=KernelLaunchNode, KernelsLaunchNode=KernelsLaunchNode,
                            np=np,
                        )
                else:
                    _accumulate_dur(
                        slot, templates,
                        cpu_key=cpu_key, rank=rank, stream_totals=stream_totals,
                        CPUNode=CPUNode, SameCPUNode=SameCPUNode, GPUNode=GPUNode,
                        KernelLaunchNode=KernelLaunchNode, KernelsLaunchNode=KernelsLaunchNode,
                        np=np,
                    )
        return

    if isinstance(node, CPUNode):
        if node.template_index >= 0:
            tmpl = templates[node.template_index]
            durs = np.asarray(tmpl.dur)
            if durs.size > node.instance_index:
                stream_totals[cpu_key] += int(durs[node.instance_index])
        for child in (node.children or []):
            _accumulate_dur(
                child, templates,
                cpu_key=cpu_key, rank=rank, stream_totals=stream_totals,
                CPUNode=CPUNode, SameCPUNode=SameCPUNode, GPUNode=GPUNode,
                KernelLaunchNode=KernelLaunchNode, KernelsLaunchNode=KernelsLaunchNode,
                np=np,
            )


def _stream_key_from_kernel_template(template, instance_index: int, rank: str, fallback_cpu_key: str) -> str:
    """Produce ``rank<R>/<gpu_tid>`` from a kernel template + instance index."""
    tid_value = None
    tid_attr = getattr(template, "tid", None)
    if isinstance(tid_attr, list):
        if 0 <= instance_index < len(tid_attr):
            tid_value = tid_attr[instance_index]
    elif tid_attr is not None:
        tid_value = tid_attr
    if tid_value is None:
        # Fall back to the surrounding CPU stream key (matches the legacy
        # in-situ behaviour) if the kernel template has no tid metadata.
        return fallback_cpu_key
    return f"rank{rank}/{tid_value}"


# ----------------------------------------------------------------------
# Layer-level operator balance
# ----------------------------------------------------------------------


class LayerOperatorBalanceTask(AnalysisTask):
    """Operator-level imbalance across transformer layers.

    A *layer* is identified by an annotation event whose name matches a
    regex (default looks for ``layer_<N>`` / ``Layer_<N>`` /
    ``decoder.layers.<N>`` style markers).  For every layer instance we
    accumulate the total busy time on each stream; the imbalance metric
    is ``(max - min) / mean`` across layers, plus a
    ``coefficient_of_variation`` for the same population.
    """

    name = "layer_operator_balance"

    _DEFAULT_LAYER_REGEX = (
        r"(?:^|[._/])layer[._]?(\d+)|"
        r"(?:^|[._/])layers[._/](\d+)|"
        r"transformer\..*?(\d+)\.attention"
    )

    def __init__(self, layer_regex: Optional[str] = None) -> None:
        import re

        self.layer_regex = re.compile(layer_regex or self._DEFAULT_LAYER_REGEX, re.IGNORECASE)

    def _extract_layer_id(self, name: str) -> Optional[int]:
        if not isinstance(name, str):
            return None
        match = self.layer_regex.search(name)
        if not match:
            return None
        for group in match.groups():
            if group is not None:
                try:
                    return int(group)
                except ValueError:
                    continue
        return None

    def run_on_raw(self, trace: Trace) -> Dict[str, Any]:
        per_layer_dur: Dict[int, int] = defaultdict(int)
        per_layer_event_count: Dict[int, int] = defaultdict(int)
        per_layer_op_dur: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for _rank, _pid, _tid, _ph, events in trace.iter_events():
            for event in events:
                layer_id = self._extract_layer_id(event.name)
                if layer_id is None:
                    continue
                if event.dur is None:
                    continue
                dur = int(event.dur)
                per_layer_dur[layer_id] += dur
                per_layer_event_count[layer_id] += 1
                per_layer_op_dur[layer_id][str(event.name)] += dur

        if not per_layer_dur:
            return {
                "layers_detected": 0,
                "imbalance": 0.0,
                "coefficient_of_variation": 0.0,
                "per_layer_total_us": [],
            }

        values = list(per_layer_dur.values())
        max_v = max(values)
        min_v = min(values)
        mean_v = sum(values) / len(values)
        imbalance = (max_v - min_v) / mean_v if mean_v > 0 else 0.0

        if mean_v > 0 and len(values) > 1:
            variance = sum((v - mean_v) ** 2 for v in values) / len(values)
            coeff_var = (variance ** 0.5) / mean_v
        else:
            coeff_var = 0.0

        return {
            "layers_detected": len(values),
            "imbalance": float(imbalance),
            "coefficient_of_variation": float(coeff_var),
            "max_us": int(max_v),
            "min_us": int(min_v),
            "mean_us": float(mean_v),
            "per_layer_total_us": [
                {"layer": layer_id, "total_us": per_layer_dur[layer_id], "event_count": per_layer_event_count[layer_id]}
                for layer_id in sorted(per_layer_dur)
            ],
        }


# ----------------------------------------------------------------------
# Parallel-group identification
# ----------------------------------------------------------------------


class ParallelGroupTask(AnalysisTask):
    """Detect TP / DP / PP / EP parallel groups from collective traffic.

    Heuristics (paper-friendly approximation):

    * ranks that issue collectives with the same group name + size at the
      same logical iteration are grouped together;
    * we coarsely tag a group as TP if its members differ only in the
      lowest-order rank bits (``rank % world_size_tp``), as DP if they
      differ in the highest-order bits, and so on.  When we cannot match
      a heuristic we fall back to ``unknown_parallel_group``.

    The output is intentionally compact -- the harness only needs a
    *count* per kind plus a sample membership list to populate the paper
    table.
    """

    name = "parallel_group"

    _COLLECTIVE_NAME_HINTS = (
        "all_reduce", "allreduce",
        "all_gather", "allgather",
        "reduce_scatter", "reducescatter",
        "broadcast",
        "send", "recv",
        "all_to_all", "alltoall",
    )

    def _is_collective(self, name: str) -> bool:
        if not isinstance(name, str):
            return False
        n = name.lower()
        return any(hint in n for hint in self._COLLECTIVE_NAME_HINTS)

    def run_on_raw(self, trace: Trace) -> Dict[str, Any]:
        ranks = sorted(trace.get_ranks())
        if not ranks:
            return {
                "ranks": 0,
                "collective_count": 0,
                "groups": [],
            }

        # group key -> {rank: count}
        group_to_ranks: Dict[Tuple[Any, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for rank, _pid, _tid, _ph, events in trace.iter_events():
            for event in events:
                if not self._is_collective(event.name):
                    continue
                args = event.args or {}
                key = (
                    str(event.name),
                    args.get("group_size") or args.get("Group size") or args.get("nranks"),
                    args.get("group_name") or args.get("Group name"),
                )
                group_to_ranks[key][str(rank)] += 1

        groups_summary: List[Dict[str, Any]] = []
        for (name, group_size, group_name), rank_counts in group_to_ranks.items():
            ranks_in_group = sorted(rank_counts)
            kind = self._classify_group_kind(ranks_in_group, group_size)
            groups_summary.append(
                {
                    "name": name,
                    "kind": kind,
                    "group_size": group_size,
                    "group_name": group_name,
                    "rank_count": len(ranks_in_group),
                    "sample_ranks": ranks_in_group[:8],
                    "collective_invocations": sum(rank_counts.values()),
                }
            )

        groups_summary.sort(key=lambda g: -g["collective_invocations"])
        kinds: Dict[str, int] = defaultdict(int)
        for entry in groups_summary:
            kinds[entry["kind"]] += 1

        return {
            "ranks": len(ranks),
            "collective_count": sum(g["collective_invocations"] for g in groups_summary),
            "kinds": dict(kinds),
            "groups": groups_summary[:32],
        }

    def _classify_group_kind(
        self,
        ranks_in_group: List[str],
        group_size: Any,
    ) -> str:
        """Best-effort TP/DP/PP/EP labelling from rank-ID arithmetic."""
        try:
            int_ranks = sorted(int(r) for r in ranks_in_group)
        except (TypeError, ValueError):
            return "unknown_parallel_group"

        if len(int_ranks) <= 1:
            return "singleton"

        diffs = [int_ranks[i + 1] - int_ranks[i] for i in range(len(int_ranks) - 1)]
        if not diffs:
            return "unknown_parallel_group"

        # Stride-1 contiguous ranks usually indicate TP groups.
        if all(d == 1 for d in diffs):
            return "tp_or_ep"
        # Constant stride > 1 typically indicates PP / DP.
        if all(d == diffs[0] for d in diffs):
            stride = diffs[0]
            if stride >= 8:
                return "pp"
            return "dp"
        return "unknown_parallel_group"


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


_TASKS: Dict[str, Callable[[], AnalysisTask]] = {
    "gpu_kernel_breakdown": GpuKernelBreakdownTask,
    "comm_comp_overlap": CommCompOverlapTask,
    "temporal_breakdown": TemporalBreakdownTask,
    "operator_hotspot": OperatorHotspotTask,
    "stream_load_balance": StreamLoadBalanceTask,
    "layer_operator_balance": LayerOperatorBalanceTask,
    "parallel_group": ParallelGroupTask,
}


def builtin_tasks() -> List[str]:
    return sorted(_TASKS)


def get_task(name: str, **kwargs: Any) -> AnalysisTask:
    if name not in _TASKS:
        raise KeyError(
            f"Unknown task {name!r}. Available: {', '.join(builtin_tasks())}"
        )
    return _TASKS[name](**kwargs)
