"""Measurement helpers for the bench harness.

Everything here is intentionally cheap and side-effect free so the
runner can call it repeatedly inside a sweep.
"""

from __future__ import annotations

import gc
import os
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from ..baselines import BaselineCompressor
from ..baselines.base import CompressArtifact
from ..trace import Trace, TraceLoadStats


@dataclass
class CompressionRecord:
    """One ``(trace, compressor)`` row of the compression matrix."""

    trace_name: str
    trace_path: str
    compressor: str

    file_count: int = 0
    event_count: int = 0
    source_size_bytes: int = 0
    source_loaded_memory_bytes: int = 0

    compressed_size_bytes: int = 0
    compress_seconds: float = 0.0
    compress_peak_memory_bytes: int = 0
    decompress_seconds: float = 0.0

    verify_passed: bool = False
    verify_message: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["compression_ratio"] = self.compression_ratio
        d["compression_pct"] = self.compression_pct
        return d

    @property
    def compression_ratio(self) -> float:
        """Original / compressed (TraceZip-paper definition; higher = better)."""
        if self.compressed_size_bytes <= 0:
            return 0.0
        return self.source_size_bytes / self.compressed_size_bytes

    @property
    def compression_pct(self) -> float:
        """Compressed / original (lower = better)."""
        if self.source_size_bytes <= 0:
            return 0.0
        return self.compressed_size_bytes / self.source_size_bytes


@dataclass
class AnalysisRecord:
    """One ``(trace, compressor, task)`` row of the analysis matrix."""

    trace_name: str
    trace_path: str
    compressor: str
    task: str

    in_situ: bool = False

    decompress_seconds: float = 0.0
    analysis_seconds: float = 0.0
    end_to_end_seconds: float = 0.0
    peak_memory_bytes: int = 0

    success: bool = True
    error_message: str = ""

    result_summary: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Wrappers
# ----------------------------------------------------------------------


def measure_compression(
    compressor: BaselineCompressor,
    trace: Trace,
    track_memory: bool = True,
) -> CompressArtifact:
    """Run :meth:`compress_trace` while tracking wall time + peak memory."""
    if track_memory:
        gc.collect()
        tracemalloc.start()
    start = time.perf_counter()
    artifact = compressor.compress_trace(trace)
    elapsed = time.perf_counter() - start
    if track_memory:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        artifact.metadata.setdefault("compress_peak_memory_bytes", peak)
    artifact.compress_seconds = elapsed
    return artifact


def measure_decompression(
    compressor: BaselineCompressor,
    blob: bytes,
    track_memory: bool = False,
) -> tuple[Trace, float, Optional[int]]:
    """Run :meth:`decompress_to_trace` while tracking wall time + peak memory."""
    peak: Optional[int] = None
    if track_memory:
        gc.collect()
        tracemalloc.start()
    start = time.perf_counter()
    trace = compressor.decompress_to_trace(blob)
    elapsed = time.perf_counter() - start
    if track_memory:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return trace, elapsed, peak
