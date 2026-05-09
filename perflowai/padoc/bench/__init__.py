"""End-to-end benchmark harness for PADOC paper experiments.

Top-level layout::

    bench/
      metrics.py    -- size / time / memory measurements
      tasks.py      -- analysis-task wrappers
      datasets.py   -- trace dataset descriptors
      runner.py     -- main matrix runners (compression / analysis / scaling / ablation)
      report.py     -- markdown / csv / json output formatters
      __main__.py   -- ``python -m perflowai.padoc.bench`` CLI

The harness drives every experiment through the same
:class:`perflowai.padoc.baselines.BaselineCompressor` abstraction so that
adding a new baseline or a new analysis task is one file each.
"""

from __future__ import annotations

from .datasets import TraceDataset, builtin_datasets, load_dataset
from .metrics import (
    CompressionRecord,
    AnalysisRecord,
    measure_compression,
    measure_decompression,
)
from .runner import (
    AnalysisMatrixResult,
    CompressionMatrixResult,
    run_analysis_matrix,
    run_compression_matrix,
)
from .tasks import AnalysisTask, builtin_tasks, get_task

__all__ = [
    "TraceDataset",
    "builtin_datasets",
    "load_dataset",
    "CompressionRecord",
    "AnalysisRecord",
    "measure_compression",
    "measure_decompression",
    "CompressionMatrixResult",
    "AnalysisMatrixResult",
    "run_compression_matrix",
    "run_analysis_matrix",
    "AnalysisTask",
    "builtin_tasks",
    "get_task",
]
