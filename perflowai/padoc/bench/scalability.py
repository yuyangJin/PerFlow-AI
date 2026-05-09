"""Scalability sweeps for the bench harness.

Three sweeps are exposed:

* :func:`run_gpu_sweep` -- vary the number of GPUs (synthetic trace).
* :func:`run_layer_sweep` -- vary the number of transformer layers.
* :func:`run_iteration_sweep` -- vary the number of training iterations.

Each sweep writes a temp synthetic trace, runs the requested compressors
on it, and returns a typed result list with one row per sweep point.

All three sweeps reuse :func:`run_compression_matrix`, so the recorded
metrics are exactly those used by the main compression-matrix table.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..synthetic import SyntheticTraceSpec, write_trace
from .datasets import TraceDataset
from .metrics import CompressionRecord
from .runner import CompressionMatrixResult, run_compression_matrix


@dataclass
class ScalabilityPoint:
    """One sweep point + its corresponding compression records."""

    axis: str
    axis_value: int
    spec: SyntheticTraceSpec
    records: List[CompressionRecord] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "axis_value": self.axis_value,
            "spec": self.spec.__dict__,
            "records": [r.as_dict() for r in self.records],
        }


def _default_compressors() -> Sequence[str]:
    return ("raw_msgpack", "gzip_msgpack", "tracezip", "scalatrace", "padoc")


def _run_one_point(
    spec: SyntheticTraceSpec,
    axis: str,
    axis_value: int,
    compressors: Sequence[str],
    workdir: str,
) -> ScalabilityPoint:
    """Generate one synthetic trace and run the compression matrix on it."""
    trace_dir = os.path.join(workdir, f"{axis}_{axis_value}")
    spec.output_dir = True
    write_trace(spec, trace_dir)

    dataset = TraceDataset(
        name=f"synth_{axis}_{axis_value}",
        path=trace_dir,
        is_directory=True,
        gpus=spec.gpus,
        layers=spec.layers,
        iterations=spec.iterations,
        extras={"sweep_axis": axis, "sweep_value": axis_value},
    )
    result = run_compression_matrix(
        datasets=[dataset],
        compressors=tuple(compressors),
        verify=False,
        track_memory=False,
    )
    point = ScalabilityPoint(axis=axis, axis_value=axis_value, spec=spec)
    point.records = result.records
    return point


# ----------------------------------------------------------------------
# Sweep helpers
# ----------------------------------------------------------------------


def run_gpu_sweep(
    gpu_counts: Sequence[int] = (4, 8, 16, 32),
    *,
    base_spec: Optional[SyntheticTraceSpec] = None,
    compressors: Sequence[str] = _default_compressors(),
    workdir: Optional[str] = None,
    cleanup: bool = True,
) -> List[ScalabilityPoint]:
    """Vary the GPU count, holding everything else fixed."""
    return _sweep_axis(
        axis="gpus",
        values=gpu_counts,
        param_setter=lambda spec, v: _set(spec, gpus=v, dp=v),
        base_spec=base_spec,
        compressors=compressors,
        workdir=workdir,
        cleanup=cleanup,
    )


def run_layer_sweep(
    layer_counts: Sequence[int] = (4, 8, 16, 32),
    *,
    base_spec: Optional[SyntheticTraceSpec] = None,
    compressors: Sequence[str] = _default_compressors(),
    workdir: Optional[str] = None,
    cleanup: bool = True,
) -> List[ScalabilityPoint]:
    """Vary transformer layer count."""
    return _sweep_axis(
        axis="layers",
        values=layer_counts,
        param_setter=lambda spec, v: _set(spec, layers=v),
        base_spec=base_spec,
        compressors=compressors,
        workdir=workdir,
        cleanup=cleanup,
    )


def run_iteration_sweep(
    iteration_counts: Sequence[int] = (1, 2, 4, 8, 16),
    *,
    base_spec: Optional[SyntheticTraceSpec] = None,
    compressors: Sequence[str] = _default_compressors(),
    workdir: Optional[str] = None,
    cleanup: bool = True,
) -> List[ScalabilityPoint]:
    """Vary the iteration count."""
    return _sweep_axis(
        axis="iterations",
        values=iteration_counts,
        param_setter=lambda spec, v: _set(spec, iterations=v),
        base_spec=base_spec,
        compressors=compressors,
        workdir=workdir,
        cleanup=cleanup,
    )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _set(spec: SyntheticTraceSpec, **kwargs: Any) -> SyntheticTraceSpec:
    for key, value in kwargs.items():
        setattr(spec, key, value)
    spec.__post_init__()
    return spec


def _sweep_axis(
    axis: str,
    values: Sequence[int],
    param_setter,
    base_spec: Optional[SyntheticTraceSpec],
    compressors: Sequence[str],
    workdir: Optional[str],
    cleanup: bool,
) -> List[ScalabilityPoint]:
    if base_spec is None:
        base_spec = SyntheticTraceSpec(
            gpus=4, dp=4, layers=4, iterations=4, micro_batches=2,
        )

    own_workdir = workdir is None
    if own_workdir:
        workdir = tempfile.mkdtemp(prefix=f"padoc_bench_{axis}_")
    else:
        os.makedirs(workdir, exist_ok=True)

    out: List[ScalabilityPoint] = []
    try:
        for value in values:
            spec = SyntheticTraceSpec(**base_spec.__dict__)
            spec = param_setter(spec, int(value))
            point = _run_one_point(
                spec=spec,
                axis=axis,
                axis_value=int(value),
                compressors=compressors,
                workdir=workdir,
            )
            out.append(point)
    finally:
        if own_workdir and cleanup:
            shutil.rmtree(workdir, ignore_errors=True)
    return out


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------


def render_scalability_markdown(points: Sequence[ScalabilityPoint]) -> str:
    """Render a sweep as ``axis_value x compressor`` markdown table."""
    if not points:
        return "_no scalability points_\n"
    compressors = []
    for record in points[0].records:
        if record.compressor not in compressors:
            compressors.append(record.compressor)
    headers = [points[0].axis] + compressors
    rows: List[List[str]] = [headers]
    for point in points:
        row = [str(point.axis_value)]
        size_by_compressor = {r.compressor: r.compressed_size_bytes for r in point.records}
        for c in compressors:
            row.append(str(size_by_compressor.get(c, "-")))
        rows.append(row)
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(rows[0])) + " |",
        sep,
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines) + "\n"
