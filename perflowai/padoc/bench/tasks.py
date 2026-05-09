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
from typing import Any, Callable, Dict, List, Optional

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


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


_TASKS: Dict[str, Callable[[], AnalysisTask]] = {
    "gpu_kernel_breakdown": GpuKernelBreakdownTask,
    "comm_comp_overlap": CommCompOverlapTask,
    "temporal_breakdown": TemporalBreakdownTask,
    "operator_hotspot": OperatorHotspotTask,
    "stream_load_balance": StreamLoadBalanceTask,
}


def builtin_tasks() -> List[str]:
    return sorted(_TASKS)


def get_task(name: str, **kwargs: Any) -> AnalysisTask:
    if name not in _TASKS:
        raise KeyError(
            f"Unknown task {name!r}. Available: {', '.join(builtin_tasks())}"
        )
    return _TASKS[name](**kwargs)
