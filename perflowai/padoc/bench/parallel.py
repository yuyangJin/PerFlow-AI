"""Thread / process scalability benchmark.

This sweep measures wall-clock compression and analysis time as we
increase the number of worker processes used to handle a multi-rank
trace directory.

The unit of parallelism is **one rank file**.  We delegate each file to
:class:`PADOCCompressor` (or any other :class:`BaselineCompressor`
exposing ``compress_trace``) and aggregate the per-file artifacts.

Two execution backends are exposed:

* ``"process"`` -- :class:`concurrent.futures.ProcessPoolExecutor`,
  which is the realistic scaling target for compute-bound work and is
  the one the paper uses;
* ``"thread"`` -- :class:`concurrent.futures.ThreadPoolExecutor`, kept
  around for sanity-check runs because it removes IPC noise.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..baselines import get_compressor
from ..trace import Trace
from ..utils import logger


@dataclass
class ParallelRunResult:
    backend: str
    workers: int
    files: int
    total_input_bytes: int
    total_output_bytes: int
    wall_seconds: float
    speedup_vs_serial: float = 0.0
    serial_baseline_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compress_single_file(args):
    file_path, compressor_name, compressor_options = args
    compressor = get_compressor(compressor_name, **(compressor_options or {}))
    trace = Trace.from_file(file_path)
    artifact = compressor.compress_trace(trace)
    file_size = os.path.getsize(file_path)
    return file_size, len(artifact.blob), artifact.compress_seconds


def run_parallel_compression(
    trace_dir: str,
    *,
    compressor_name: str = "padoc",
    compressor_options: Optional[Dict[str, Any]] = None,
    workers_grid: Sequence[int] = (1, 2, 4, 8),
    backend: str = "process",
) -> List[ParallelRunResult]:
    """Compress every file in ``trace_dir`` for each worker count.

    Returns one :class:`ParallelRunResult` per ``workers_grid`` entry.
    Speedup is computed against the ``workers=1`` row.
    """
    trace_files = _list_trace_files(trace_dir)
    if not trace_files:
        logger.warning("No trace files in %s", trace_dir)
        return []

    serial_seconds: Optional[float] = None
    out: List[ParallelRunResult] = []
    for workers in workers_grid:
        result = _run_one_worker_count(
            trace_files,
            compressor_name=compressor_name,
            compressor_options=compressor_options,
            workers=int(workers),
            backend=backend,
        )
        if workers == 1:
            serial_seconds = result.wall_seconds
        if serial_seconds is not None and result.wall_seconds > 0:
            result.serial_baseline_seconds = serial_seconds
            result.speedup_vs_serial = serial_seconds / result.wall_seconds
        out.append(result)
    return out


def _run_one_worker_count(
    trace_files: Sequence[str],
    *,
    compressor_name: str,
    compressor_options: Optional[Dict[str, Any]],
    workers: int,
    backend: str,
) -> ParallelRunResult:
    if backend not in {"process", "thread"}:
        raise ValueError(f"Unknown backend: {backend!r}")

    pool_factory = (
        ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    )

    args_list = [
        (file_path, compressor_name, compressor_options or {})
        for file_path in trace_files
    ]
    start = time.perf_counter()
    total_input = 0
    total_output = 0
    if workers <= 1:
        for entry in args_list:
            in_size, out_size, _ = _compress_single_file(entry)
            total_input += in_size
            total_output += out_size
    else:
        with pool_factory(max_workers=workers) as pool:
            for in_size, out_size, _ in pool.map(_compress_single_file, args_list):
                total_input += in_size
                total_output += out_size
    wall = time.perf_counter() - start

    return ParallelRunResult(
        backend=backend,
        workers=int(workers),
        files=len(args_list),
        total_input_bytes=int(total_input),
        total_output_bytes=int(total_output),
        wall_seconds=float(wall),
    )


def _list_trace_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    out: List[str] = []
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        if not os.path.isfile(full):
            continue
        if not (entry.endswith(".json") or entry.endswith(".json.gz")):
            continue
        out.append(full)
    return out


def render_parallel_markdown(rows: Sequence[ParallelRunResult]) -> str:
    if not rows:
        return "_no parallel runs_\n"
    headers = ["workers", "wall_s", "speedup", "input_bytes", "output_bytes"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                [
                    str(row.workers),
                    f"{row.wall_seconds:.3f}",
                    f"{row.speedup_vs_serial:.2f}x" if row.speedup_vs_serial else "-",
                    str(row.total_input_bytes),
                    str(row.total_output_bytes),
                ]
            ) + " |"
        )
    return "\n".join(lines) + "\n"
