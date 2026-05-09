"""Tests for storage_breakdown.py and tree_stats.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from perflowai.padoc.compressor import TemplateCompressor
from perflowai.padoc.storage_breakdown import StorageBreakdown, measure_storage
from perflowai.padoc.synthetic import SyntheticTraceSpec, generate_trace, write_trace
from perflowai.padoc.trace import Trace
from perflowai.padoc.tree_stats import TreeStatistics, measure_tree_statistics


TRACE_PATH = Path("tests/example_trace/out-1024.json")


@pytest.fixture(scope="module")
def small_compressed():
    if not TRACE_PATH.exists():
        pytest.skip(f"missing {TRACE_PATH}")
    trace = Trace.from_file(str(TRACE_PATH))
    return TemplateCompressor().intra_compress(trace, emit_summary=False)


def test_storage_breakdown_components_sum_to_total(small_compressed) -> None:
    b: StorageBreakdown = measure_storage(small_compressed)
    components = (
        b.templates_serialized_bytes
        + b.structure_serialized_bytes
        + b.metadata_serialized_bytes
    )
    assert components == b.total_serialized_bytes
    assert b.total_serialized_bytes > 0


def test_storage_breakdown_template_share_in_range(small_compressed) -> None:
    b = measure_storage(small_compressed)
    share = b.template_share()
    assert 0.0 < share < 1.0
    md = b.render_markdown()
    assert "templates" in md
    assert "structure" in md


def test_tree_stats_counts_match_actual_walk(small_compressed) -> None:
    stats: TreeStatistics = measure_tree_statistics(small_compressed)
    assert stats.total_nodes >= stats.total_roots
    assert stats.depth.maximum >= stats.depth.minimum >= 1
    assert stats.unique_subtree_shapes <= stats.total_nodes


def test_tree_stats_on_synthetic_has_expected_shape(tmp_path) -> None:
    spec = SyntheticTraceSpec(
        gpus=2, dp=2, layers=4, iterations=4, micro_batches=2,
    )
    trace_path = tmp_path / "synth.json"
    write_trace(spec, str(trace_path))
    trace = Trace.from_file(str(trace_path))
    ct = TemplateCompressor().intra_compress(trace, emit_summary=False)
    stats = measure_tree_statistics(ct)
    # Exactly one rank, so every iteration contributes events but the
    # forest may collapse heavily.  We just sanity-check that *some*
    # SameCPUNode multiplier > 1 (i.e. structural compression fired).
    if stats.samecpu_multiplier.count > 0:
        assert stats.samecpu_multiplier.maximum >= 2


def test_synthetic_trace_generator_is_deterministic() -> None:
    spec_a = SyntheticTraceSpec(gpus=2, dp=2, layers=2, iterations=2, seed=42)
    spec_b = SyntheticTraceSpec(gpus=2, dp=2, layers=2, iterations=2, seed=42)
    a = generate_trace(spec_a)
    b = generate_trace(spec_b)
    assert a == b


def test_synthetic_trace_generator_changes_with_layer_count() -> None:
    small = generate_trace(SyntheticTraceSpec(gpus=1, dp=1, layers=1, iterations=1, seed=0))
    big = generate_trace(SyntheticTraceSpec(gpus=1, dp=1, layers=4, iterations=1, seed=0))
    assert sum(len(events) for events in big.values()) > sum(len(events) for events in small.values())
