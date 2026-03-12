from __future__ import annotations

from .event import Event, MergeEvent

from .trace import BaseTrace, Trace, CompressedTrace, TraceLoadResult, TraceLoadStats
from .compressor import Compressor, TemplateCompressor
from .utils import logger
from .slp import SegmentedLinearPredictorCompressor
from .analysis import TraceAnalysis
from .verify import (
    compare_trace_directories,
    compare_trace_directories_with_report,
    compare_trace_files,
    compare_trace_files_with_report,
)

__all__ = [
    "Event",
    "MergeEvent",

    "BaseNode",
    "Node",
    "TemplateNode",
    "RefNode",

    "BaseTrace",
    "Trace",
    "CompressedTrace",
    "TraceLoadResult",
    "TraceLoadStats",

    "Compressor",
    "TemplateCompressor",

    "SegmentedLinearPredictorCompressor",

    "TraceAnalysis",
    "compare_trace_directories",
    "compare_trace_directories_with_report",
    "compare_trace_files",
    "compare_trace_files_with_report",
]


__version__ = "0.1.0"

logger.info("padoc package initialized (v%s)", __version__)
