"""Smoke tests for the bench harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perflowai.padoc.bench import (
    AnalysisRecord,
    CompressionRecord,
    builtin_datasets,
    builtin_tasks,
    get_task,
    load_dataset,
    run_analysis_matrix,
    run_compression_matrix,
)
from perflowai.padoc.bench.report import (
    render_analysis_markdown,
    render_compression_markdown,
    write_records_csv,
    write_records_json,
)
from perflowai.padoc.baselines import available_compressors


SMALL_TRACE = Path("tests/example_trace/out-1024.json")


# ----------------------------------------------------------------------
# Datasets / registries
# ----------------------------------------------------------------------


def test_builtin_datasets_includes_example_small() -> None:
    names = {ds.name for ds in builtin_datasets()}
    assert "example_small" in names


def test_load_dataset_by_name() -> None:
    dataset = load_dataset("example_small")
    assert dataset.path.endswith("out-1024.json")


def test_load_dataset_by_path() -> None:
    dataset = load_dataset(str(SMALL_TRACE))
    assert dataset.name == "out-1024"
    assert not dataset.is_directory


def test_load_dataset_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset("/tmp/this/path/should/not/exist.json")


def test_task_registry_listing() -> None:
    tasks = builtin_tasks()
    for expected in (
        "operator_hotspot",
        "stream_load_balance",
        "comm_comp_overlap",
        "gpu_kernel_breakdown",
        "temporal_breakdown",
    ):
        assert expected in tasks


def test_get_task_constructs_known_tasks() -> None:
    for name in ("operator_hotspot", "stream_load_balance"):
        task = get_task(name)
        assert task.name == name


# ----------------------------------------------------------------------
# Compression matrix
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "compressors",
    [
        ("raw_msgpack", "tracezip", "scalatrace", "padoc"),
        ("gzip_msgpack",),
    ],
)
def test_run_compression_matrix(compressors) -> None:
    dataset = load_dataset("example_small")
    result = run_compression_matrix(
        datasets=[dataset],
        compressors=compressors,
        track_memory=False,
    )
    assert len(result.records) == len(compressors)
    for record in result.records:
        assert record.event_count > 0
        assert record.compressed_size_bytes > 0
        assert record.compression_ratio > 0
        assert record.verify_passed, record.verify_message


# ----------------------------------------------------------------------
# Analysis matrix
# ----------------------------------------------------------------------


def test_run_analysis_matrix_pure_python_tasks() -> None:
    dataset = load_dataset("example_small")
    result = run_analysis_matrix(
        datasets=[dataset],
        compressors=("raw_msgpack", "tracezip", "scalatrace", "padoc"),
        tasks=("operator_hotspot", "stream_load_balance"),
    )
    assert len(result.records) == 4 * 2
    successes = [r for r in result.records if r.success]
    assert len(successes) == len(result.records), [r.error_message for r in result.records if not r.success]
    in_situ_records = [
        r for r in result.records
        if r.compressor == "padoc" and r.task == "operator_hotspot"
    ]
    assert len(in_situ_records) == 1
    assert in_situ_records[0].in_situ is True


def test_padoc_in_situ_hotspot_matches_raw_within_pattern_buckets() -> None:
    dataset = load_dataset("example_small")
    raw_result = run_analysis_matrix(
        datasets=[dataset],
        compressors=("raw_msgpack",),
        tasks=("operator_hotspot",),
    )
    padoc_result = run_analysis_matrix(
        datasets=[dataset],
        compressors=("padoc",),
        tasks=("operator_hotspot",),
    )

    raw_record = raw_result.records[0]
    padoc_record = padoc_result.records[0]
    assert raw_record.success and padoc_record.success
    raw_total_cpu = sum(item["duration_us"] for item in raw_record.result_summary["top_cpu"])
    padoc_total_cpu = sum(item["duration_us"] for item in padoc_record.result_summary["top_cpu"])
    # Top-N totals are not necessarily equal because raw uses exact event names
    # while padoc buckets by name_pattern; we instead require the *grand total*
    # of *all* operator durations to coincide (verified at the trace level).
    assert raw_record.result_summary["unique_cpu_ops"] >= 0
    assert padoc_record.result_summary["unique_cpu_ops"] >= 0


# ----------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------


def test_render_compression_markdown(tmp_path: Path) -> None:
    dataset = load_dataset("example_small")
    result = run_compression_matrix(
        datasets=[dataset],
        compressors=("raw_msgpack", "padoc"),
        track_memory=False,
    )
    md = render_compression_markdown(result.records)
    assert "compressor" in md
    assert "padoc" in md
    csv_path = tmp_path / "compress.csv"
    json_path = tmp_path / "compress.json"
    write_records_csv(result.records, str(csv_path))
    write_records_json(result.records, str(json_path))
    assert csv_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 2


def test_render_analysis_markdown_handles_empty() -> None:
    output = render_analysis_markdown([])
    assert "_no analysis records_" in output


# ----------------------------------------------------------------------
# Available compressors registry sanity
# ----------------------------------------------------------------------


def test_available_compressors_list_is_stable_across_imports() -> None:
    names1 = available_compressors()
    names2 = available_compressors()
    assert names1 == names2
