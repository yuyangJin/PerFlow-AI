from __future__ import annotations

from .event import Event, MergeEvent
from .node import BaseNode, Node, TemplateNode, RefNode
from .trace import BaseTrace, Trace, CompressedTrace
from .compressor import Compressor, TemplateCompressor
from .utils import logger
from .slp import SegmentedLinearPredictorCompressor
from .analysis import TraceAnalysis

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

    "Compressor",
    "TemplateCompressor",

    "SegmentedLinearPredictorCompressor",

    "TraceAnalysis",
]


__version__ = "0.1.0"

logger.info("padoc package initialized (v%s)", __version__)
