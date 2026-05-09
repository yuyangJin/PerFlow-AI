"""Central matrix-running engine.

The bench harness exposes three top-level sweeps:

* :func:`run_compression_matrix` -- ``(trace, compressor)`` -> sizes / times.
* :func:`run_analysis_matrix` -- ``(trace, compressor, task)`` -> analysis times.
* :func:`run_scalability` -- vary one scale knob (gpus, layers, iterations,
  threads/processes) and rerun the inner sweeps.

Every sweep returns a typed result object with a ``records`` list ready
for :mod:`perflowai.padoc.bench.report`.
"""

from __future__ import annotations

import gc
import os
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..baselines import BaselineCompressor, available_compressors, get_compressor
from ..baselines.padoc_adapter import PADOCCompressor
from ..trace import Trace, TraceLoadStats
from .datasets import TraceDataset
from .metrics import AnalysisRecord, CompressionRecord, measure_compression
from .tasks import AnalysisTask, builtin_tasks, get_task


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_compressors(
    names: Sequence[str],
    options: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[BaselineCompressor]:
    options = options or {}
    out: List[BaselineCompressor] = []
    for name in names:
        kwargs = options.get(name, {})
        out.append(get_compressor(name, **kwargs))
    return out


def _resolve_tasks(names: Sequence[str]) -> List[AnalysisTask]:
    return [get_task(name) for name in names]


def _on_disk_size(path: str) -> int:
    if os.path.isdir(path):
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total
    return os.path.getsize(path)


def _load_trace(dataset: TraceDataset) -> Tuple[Trace, TraceLoadStats]:
    if dataset.is_directory:
        result = Trace.from_dir_with_stats(dataset.path)
    else:
        result = Trace.from_file_with_stats(dataset.path)
    return result.trace, result.stats


def _safe_event_signatures(trace: Trace) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Population fingerprint per ``(rank, name)`` -- ``(count, dur_sum)``."""
    out: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for rank, _pid, _tid, _ph, events in trace.iter_events():
        for event in events:
            entry = out[(str(rank), str(event.name))]
            entry[0] += 1
            if event.dur is not None:
                entry[1] += int(event.dur)
    return {key: (count, dur) for key, (count, dur) in out.items()}


# ----------------------------------------------------------------------
# Compression matrix
# ----------------------------------------------------------------------


@dataclass
class CompressionMatrixResult:
    records: List[CompressionRecord] = field(default_factory=list)

    def by_compressor(self) -> Dict[str, List[CompressionRecord]]:
        out: Dict[str, List[CompressionRecord]] = defaultdict(list)
        for record in self.records:
            out[record.compressor].append(record)
        return dict(out)


def run_compression_matrix(
    datasets: Sequence[TraceDataset],
    compressors: Sequence[str] = ("raw_msgpack", "gzip_msgpack", "tracezip", "scalatrace", "padoc"),
    *,
    compressor_options: Optional[Dict[str, Dict[str, Any]]] = None,
    verify: bool = True,
    track_memory: bool = True,
    progress_cb: Optional[Callable[[CompressionRecord], None]] = None,
) -> CompressionMatrixResult:
    """Sweep ``(dataset, compressor)`` and collect compression metrics."""
    compressor_options = compressor_options or {}
    result = CompressionMatrixResult()

    for dataset in datasets:
        trace, stats = _load_trace(dataset)
        source_signatures = _safe_event_signatures(trace) if verify else None
        original_disk_size = _on_disk_size(dataset.path)

        for compressor in _resolve_compressors(compressors, compressor_options):
            record = CompressionRecord(
                trace_name=dataset.name,
                trace_path=dataset.path,
                compressor=compressor.name,
                file_count=getattr(stats, "file_count", 1),
                event_count=getattr(stats, "event_count", 0),
                source_size_bytes=original_disk_size,
                source_loaded_memory_bytes=getattr(stats, "loaded_memory_bytes", 0),
            )

            artifact = measure_compression(
                compressor=compressor,
                trace=trace,
                track_memory=track_memory,
            )
            record.compressed_size_bytes = artifact.size_bytes
            record.compress_seconds = artifact.compress_seconds
            record.compress_peak_memory_bytes = int(
                artifact.metadata.get("compress_peak_memory_bytes", 0)
            )
            record.metadata = dict(artifact.metadata)

            if verify:
                start = time.perf_counter()
                decoded = compressor.decompress_to_trace(artifact.blob)
                record.decompress_seconds = time.perf_counter() - start
                decoded_signatures = _safe_event_signatures(decoded)
                if decoded_signatures == source_signatures:
                    record.verify_passed = True
                else:
                    record.verify_passed = False
                    record.verify_message = _diff_signatures_summary(
                        source_signatures, decoded_signatures
                    )

            result.records.append(record)
            if progress_cb is not None:
                progress_cb(record)

        # Free the trace object before moving on to the next dataset.
        del trace
        gc.collect()

    return result


def _diff_signatures_summary(
    expected: Optional[Dict[Tuple[str, str], Tuple[int, int]]],
    actual: Dict[Tuple[str, str], Tuple[int, int]],
) -> str:
    if expected is None:
        return ""
    missing = sum(1 for k in expected if k not in actual)
    extra = sum(1 for k in actual if k not in expected)
    diffs = sum(
        1
        for k, v in expected.items()
        if k in actual and actual[k] != v
    )
    return f"missing_keys={missing} extra_keys={extra} value_diffs={diffs}"


# ----------------------------------------------------------------------
# Analysis matrix
# ----------------------------------------------------------------------


@dataclass
class AnalysisMatrixResult:
    records: List[AnalysisRecord] = field(default_factory=list)


def run_analysis_matrix(
    datasets: Sequence[TraceDataset],
    compressors: Sequence[str] = ("raw_msgpack", "gzip_msgpack", "tracezip", "scalatrace", "padoc"),
    tasks: Sequence[str] = ("operator_hotspot", "stream_load_balance"),
    *,
    compressor_options: Optional[Dict[str, Dict[str, Any]]] = None,
    progress_cb: Optional[Callable[[AnalysisRecord], None]] = None,
) -> AnalysisMatrixResult:
    """Sweep ``(dataset, compressor, task)`` and collect analysis times.

    For each ``(dataset, compressor)`` we compress *once*, then run every
    task against the resulting blob.  In-situ compressors (PADOC) skip
    the decompression step where the task supports it.
    """
    compressor_options = compressor_options or {}
    result = AnalysisMatrixResult()
    task_objs = _resolve_tasks(tasks)

    for dataset in datasets:
        trace, _ = _load_trace(dataset)
        for compressor in _resolve_compressors(compressors, compressor_options):
            artifact = compressor.compress_trace(trace)
            blob = artifact.blob

            for task in task_objs:
                record = AnalysisRecord(
                    trace_name=dataset.name,
                    trace_path=dataset.path,
                    compressor=compressor.name,
                    task=task.name,
                    in_situ=False,
                )

                gc.collect()
                tracemalloc.start()
                try:
                    can_in_situ = (
                        compressor.supports_in_situ_analysis
                        and task.has_in_situ_for(compressor)
                    )
                    if can_in_situ:
                        record.in_situ = True
                        start = time.perf_counter()
                        summary = task.run_in_situ(compressor, blob)
                        record.end_to_end_seconds = time.perf_counter() - start
                        record.analysis_seconds = record.end_to_end_seconds
                    else:
                        decompress_start = time.perf_counter()
                        decoded = compressor.decompress_to_trace(blob)
                        record.decompress_seconds = time.perf_counter() - decompress_start
                        analysis_start = time.perf_counter()
                        summary = task.run_on_raw(decoded)
                        record.analysis_seconds = time.perf_counter() - analysis_start
                        record.end_to_end_seconds = (
                            record.decompress_seconds + record.analysis_seconds
                        )
                    record.result_summary = summary
                except Exception as exc:  # pragma: no cover - bench captures errors
                    record.success = False
                    record.error_message = f"{type(exc).__name__}: {exc}"
                finally:
                    _current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    record.peak_memory_bytes = int(peak)

                result.records.append(record)
                if progress_cb is not None:
                    progress_cb(record)
        del trace
        gc.collect()

    return result


# ----------------------------------------------------------------------
# Scalability sweep wrappers
# ----------------------------------------------------------------------


def run_scalability_compression(
    datasets: Sequence[TraceDataset],
    compressor: str = "padoc",
    *,
    compressor_options: Optional[Dict[str, Any]] = None,
    track_memory: bool = True,
) -> CompressionMatrixResult:
    """Same as :func:`run_compression_matrix` but on a single compressor.

    Use this with a list of datasets that vary in scale (different gpus,
    layers, iterations) to populate the scalability tables.
    """
    options = {compressor: compressor_options or {}}
    return run_compression_matrix(
        datasets=datasets,
        compressors=(compressor,),
        compressor_options=options,
        verify=False,
        track_memory=track_memory,
    )


def run_thread_scalability(
    dataset: TraceDataset,
    thread_counts: Sequence[int],
    compressor: str = "padoc",
    *,
    compressor_options: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, CompressionRecord]]:
    """Re-run compression with different thread counts.

    PADOC honours the ``OMP_NUM_THREADS`` style env var via its MPI worker
    helpers.  For non-MPI compressors this is a no-op; the returned record
    will simply repeat the same single-thread number.

    Returns a list ``[(thread_count, record), ...]``.
    """
    base_options = compressor_options or {}
    out: List[Tuple[int, CompressionRecord]] = []
    for threads in thread_counts:
        options = {compressor: dict(base_options)}
        os.environ["PADOC_NUM_THREADS"] = str(threads)
        result = run_compression_matrix(
            datasets=[dataset],
            compressors=(compressor,),
            compressor_options=options,
            verify=False,
        )
        out.append((threads, result.records[0]))
    os.environ.pop("PADOC_NUM_THREADS", None)
    return out
