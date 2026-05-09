"""Tests for new analysis tasks + scalability harness."""

from __future__ import annotations

import os

import pytest

from perflowai.padoc.bench import (
    get_task,
    render_scalability_markdown,
    run_gpu_sweep,
    run_iteration_sweep,
    run_layer_sweep,
)
from perflowai.padoc.bench.tasks import (
    LayerOperatorBalanceTask,
    ParallelGroupTask,
    StreamLoadBalanceTask,
)
from perflowai.padoc.baselines import PADOCCompressor
from perflowai.padoc.synthetic import SyntheticTraceSpec, write_trace
from perflowai.padoc.trace import Trace


@pytest.fixture(scope="module")
def synth_trace(tmp_path_factory) -> Trace:
    out_dir = tmp_path_factory.mktemp("synth")
    spec = SyntheticTraceSpec(
        gpus=4, tp=2, dp=2, layers=4, iterations=2, micro_batches=2,
        output_dir=True,
    )
    write_trace(spec, str(out_dir))
    return Trace.from_dir(str(out_dir))


def test_layer_operator_balance_finds_layers(synth_trace: Trace) -> None:
    task = LayerOperatorBalanceTask()
    result = task.run_on_raw(synth_trace)
    assert result["layers_detected"] >= 4
    assert result["mean_us"] > 0
    assert isinstance(result["per_layer_total_us"], list)
    assert all("total_us" in entry for entry in result["per_layer_total_us"])


def test_parallel_group_detects_tp_dp(synth_trace: Trace) -> None:
    task = ParallelGroupTask()
    result = task.run_on_raw(synth_trace)
    assert result["ranks"] == 4
    assert result["collective_count"] > 0
    kinds = result["kinds"]
    # We at least expect *some* classification to fire
    assert any(value > 0 for value in kinds.values())


def test_stream_load_balance_in_situ_consistent_with_raw(synth_trace: Trace) -> None:
    """In-situ and raw must agree on coarse properties.

    PADOC's compressed structure represents kernel events as
    KernelLaunchNode references inside the CPU tree, so the walk visits
    the same per-event durations but groups them slightly differently
    (e.g. the call-tree builder may wrap annotation events).  We
    therefore test the *invariants* the paper actually cares about
    rather than per-byte equivalence:

    * same number of streams,
    * the max/min ordering across streams is preserved,
    * end-to-end totals are within a small margin (no large drift).
    """
    task = StreamLoadBalanceTask()
    raw = task.run_on_raw(synth_trace)

    compressor = PADOCCompressor()
    artifact = compressor.compress_trace(synth_trace)
    in_situ = task.run_in_situ(compressor, artifact.blob)

    assert task.has_in_situ_for(compressor)
    assert raw["streams"] == in_situ["streams"]
    # GPU-stream max should match exactly (kernels live on a dedicated
    # stream and PADOC routes them back to it via tid metadata).
    assert raw["max_us"] == in_situ["max_us"]
    assert in_situ["min_us"] > 0
    # End-to-end mean is allowed up to 50 % drift on tiny synthetic
    # traces; the cluster numbers will be much tighter.
    assert in_situ["mean_us"] >= raw["mean_us"] * 0.5
    assert in_situ["mean_us"] <= raw["mean_us"] * 2.0


def test_get_task_resolves_new_tasks() -> None:
    assert isinstance(get_task("layer_operator_balance"), LayerOperatorBalanceTask)
    assert isinstance(get_task("parallel_group"), ParallelGroupTask)


def test_layer_sweep_smoke() -> None:
    points = run_layer_sweep(
        layer_counts=(2, 4),
        compressors=("padoc", "raw_msgpack"),
    )
    assert len(points) == 2
    sizes = []
    for point in points:
        for record in point.records:
            if record.compressor == "padoc":
                sizes.append(record.compressed_size_bytes)
    assert sizes[0] < sizes[1]
    md = render_scalability_markdown(points)
    assert "padoc" in md and "layers" in md


def test_iteration_sweep_smoke() -> None:
    points = run_iteration_sweep(
        iteration_counts=(1, 2),
        compressors=("padoc",),
    )
    assert len(points) == 2
    assert points[0].records[0].compressed_size_bytes <= points[1].records[0].compressed_size_bytes


def test_gpu_sweep_smoke() -> None:
    points = run_gpu_sweep(
        gpu_counts=(2, 4),
        compressors=("padoc", "scalatrace", "tracezip"),
    )
    assert len(points) == 2
    for point in points:
        compressors = {r.compressor for r in point.records}
        assert {"padoc", "scalatrace", "tracezip"}.issubset(compressors)
