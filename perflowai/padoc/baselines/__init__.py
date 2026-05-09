"""Baseline compressors used for paper-level comparison.

Every baseline implements :class:`BaselineCompressor` so that the
:mod:`perflowai.padoc.bench` harness can drive compression-ratio,
roundtrip-correctness, and analysis-time experiments over an arbitrary
mix of compressors and traces.

Currently shipped baselines:

- :class:`RawJsonCompressor` / :class:`RawMsgpackCompressor` -- "no compression" reference points.
- :class:`GzipJsonCompressor` -- raw chrome trace JSON piped through gzip.
- :class:`TracezipCompressor` -- AI-trace adaptation of TraceZip (ISSTA'25).
- :class:`ScalaTraceCompressor` -- RSD/PRSD style sequence compression.
- :class:`PADOCCompressor` -- adapter around :class:`TemplateCompressor`.
"""

from __future__ import annotations

from .base import (
    BaselineCompressor,
    CompressArtifact,
    available_compressors,
    get_compressor,
    register_compressor,
)
from .gzip_baseline import GzipJsonCompressor, GzipMsgpackCompressor
from .padoc_adapter import PADOCCompressor
from .raw import RawJsonCompressor, RawMsgpackCompressor
from .scalatrace import ScalaTraceCompressor
from .tracezip import TracezipCompressor

__all__ = [
    "BaselineCompressor",
    "CompressArtifact",
    "available_compressors",
    "get_compressor",
    "register_compressor",
    "RawJsonCompressor",
    "RawMsgpackCompressor",
    "GzipJsonCompressor",
    "GzipMsgpackCompressor",
    "TracezipCompressor",
    "ScalaTraceCompressor",
    "PADOCCompressor",
]
