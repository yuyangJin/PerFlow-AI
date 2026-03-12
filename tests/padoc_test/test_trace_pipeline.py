from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perflowai.padoc import (
    TemplateCompressor,
    Trace,
    compare_trace_directories_with_report,
    compare_trace_files,
    compare_trace_files_with_report,
)
from perflowai.padoc.slp import SegmentedLinearPredictorCompressor as SLP


TRACE_PATH = Path("tests/example_trace/out-1024.json")
MERGE_SMALL_DIR = Path("tests/example_trace/merge_small")


def write_rank_trace(source: Path, destination: Path, rank: int) -> None:
    """Write a source trace with a modified rank."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.setdefault("distributedInfo", {})["rank"] = rank
    destination.write_text(json.dumps(payload), encoding="utf-8")


def test_decompress_same_args_handles_empty_lists() -> None:
    """Empty lists in nested args should survive round-trip."""
    grouped_args = {
        "shape": {"dims": []},
        "name": ["foo", "bar"],
    }
    compressed = {
        "shape": {"dims": []},
        "name": ["foo", "bar"],
    }

    restored = [SLP.decompress_same_args(compressed, index) for index in range(2)]

    assert restored == [
        {"shape": {"dims": []}, "name": "foo"},
        {"shape": {"dims": []}, "name": "bar"},
    ]
    assert [SLP.decompress_same_args(grouped_args, index) for index in range(2)] == restored


def test_trace_directory_loading_and_streaming_compression(tmp_path: Path) -> None:
    """Directory loading should return stats and support both compression modes."""
    trace_dir = tmp_path / "trace_dir"
    trace_dir.mkdir()
    write_rank_trace(TRACE_PATH, trace_dir / "rank0.json", rank=0)
    write_rank_trace(TRACE_PATH, trace_dir / "rank1.json", rank=1)

    load_result = Trace.from_dir_with_stats(str(trace_dir), max_workers=2)

    assert load_result.stats.file_count == 2
    assert load_result.stats.event_count > 0
    assert load_result.stats.loaded_memory_bytes > 0
    assert sorted(load_result.trace.get_ranks()) == ["0", "1"]

    compressor = TemplateCompressor()
    independent_trace, streaming_stats = compressor.inter_compress_dir(
        str(trace_dir),
        max_workers=2,
        merge_ranks=False,
    )

    merged_trace, merged_stats = TemplateCompressor().inter_compress_dir(
        str(trace_dir),
        max_workers=2,
        merge_ranks=True,
    )

    assert sorted(independent_trace.get_ranks()) == ["0", "1"]
    assert sorted(merged_trace.get_ranks()) == ["0", "1"]
    assert streaming_stats.file_count == 2
    assert streaming_stats.event_count == load_result.stats.event_count
    assert merged_stats.file_count == 2
    assert len(merged_trace.event_templates) <= len(independent_trace.event_templates)

    compressed_dir = tmp_path / "compressed_parts"
    compressed_dir.mkdir()
    for source_name in sorted(trace_dir.iterdir()):
        compressed_trace, _ = TemplateCompressor().compress_file(str(source_name))
        compressed_trace.write_file(str(compressed_dir / source_name.name))

    merged_from_parts = TemplateCompressor().merge_compressed_files(
        [str(path) for path in sorted(compressed_dir.iterdir())]
    )
    assert sorted(merged_from_parts.get_ranks()) == ["0", "1"]
    restored_dir = tmp_path / "restored_merged"
    restored_dir.mkdir()
    restored_trace = TemplateCompressor().inter_decompress(merged_from_parts)
    restored_trace.write_dir(str(restored_dir), "json")
    passed, message = compare_trace_directories_with_report(str(trace_dir), str(restored_dir))
    assert passed, message


def test_merge_small_real_subset_roundtrip(tmp_path: Path) -> None:
    """Merge should round-trip on a small subset extracted from the real failing traces."""
    compressed_dir = tmp_path / "compressed_parts"
    restored_dir = tmp_path / "restored"
    compressed_dir.mkdir()
    restored_dir.mkdir()

    for source_path in sorted(MERGE_SMALL_DIR.iterdir()):
        compressed_trace, _ = TemplateCompressor().compress_file(str(source_path))
        compressed_trace.write_file(str(compressed_dir / source_path.name))

    merged_trace = TemplateCompressor().merge_compressed_files(
        [str(path) for path in sorted(compressed_dir.iterdir())]
    )
    restored_trace = TemplateCompressor().inter_decompress(merged_trace)
    restored_trace.write_dir(str(restored_dir), "json")

    passed, message = compare_trace_directories_with_report(
        str(MERGE_SMALL_DIR),
        str(restored_dir),
    )
    assert passed, message


def test_merge_single_compressed_file_roundtrip(tmp_path: Path) -> None:
    """Merging a single compressed file should preserve the original events."""
    compressed_path = tmp_path / "part.json"
    restored_dir = tmp_path / "restored"

    compressed_trace, _ = TemplateCompressor().compress_file(str(TRACE_PATH))
    compressed_trace.write_file(str(compressed_path))

    merged_trace = TemplateCompressor().merge_compressed_files([str(compressed_path)])
    restored_trace = TemplateCompressor().inter_decompress(merged_trace)
    restored_trace.write_dir(str(restored_dir), "json")

    passed, message = compare_trace_files_with_report(
        str(TRACE_PATH),
        str(restored_dir / "rank0.json"),
    )
    assert passed, message


def test_merge_of_intermediate_merged_files_roundtrip(tmp_path: Path) -> None:
    """Merging intermediate multi-rank compressed files should preserve every rank."""
    trace_dir = tmp_path / "trace_dir"
    trace_dir.mkdir()
    for rank in range(4):
        write_rank_trace(TRACE_PATH, trace_dir / f"rank{rank}.json", rank=rank)

    part_dir = tmp_path / "parts"
    part_dir.mkdir()
    for source_name in sorted(trace_dir.iterdir()):
        compressed_trace, _ = TemplateCompressor().compress_file(str(source_name))
        compressed_trace.write_file(str(part_dir / source_name.name))

    stage_one_a = TemplateCompressor().merge_compressed_files(
        [str(part_dir / "rank0.json"), str(part_dir / "rank1.json")],
        emit_summary=False,
        emit_progress=False,
        emit_rank_logs=False,
    )
    stage_one_b = TemplateCompressor().merge_compressed_files(
        [str(part_dir / "rank2.json"), str(part_dir / "rank3.json")],
        emit_summary=False,
        emit_progress=False,
        emit_rank_logs=False,
    )

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    stage_one_a_path = stage_dir / "merge_a.json"
    stage_one_b_path = stage_dir / "merge_b.json"
    stage_one_a.write_file(str(stage_one_a_path))
    stage_one_b.write_file(str(stage_one_b_path))

    final_merged = TemplateCompressor().merge_compressed_files(
        [str(stage_one_a_path), str(stage_one_b_path)],
        emit_summary=False,
        emit_progress=False,
        emit_rank_logs=False,
    )
    restored_dir = tmp_path / "restored"
    restored_trace = TemplateCompressor().inter_decompress(final_merged)
    restored_trace.write_dir(str(restored_dir), "json")

    passed, message = compare_trace_directories_with_report(str(trace_dir), str(restored_dir))
    assert passed, message


def test_single_file_pipeline_and_verify(tmp_path: Path) -> None:
    """Single-file compression should use the same pipeline and verify cleanly."""
    compressor = TemplateCompressor()
    compressed_trace, load_stats = compressor.compress_file(str(TRACE_PATH))
    restored_trace = compressor.intra_decompress(compressed_trace)

    restored_path = tmp_path / "restored.json"
    restored_trace.write_file(str(restored_path), origin=True)

    assert load_stats.file_count == 1
    assert load_stats.event_count > 0
    assert compare_trace_files(str(TRACE_PATH), str(restored_path))


def test_verify_report_returns_reason(tmp_path: Path) -> None:
    """Verify helpers should return a readable reason on failure."""
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    payload = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    left.write_text(json.dumps(payload), encoding="utf-8")
    payload["traceEvents"] = payload["traceEvents"][:-1]
    right.write_text(json.dumps(payload), encoding="utf-8")

    passed, message = compare_trace_files_with_report(str(left), str(right))

    assert not passed
    assert "event_count mismatch" in message


def test_directory_verify_matches_rank_based_file_names(tmp_path: Path) -> None:
    """Directory verification should match equivalent rank file names."""
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    payload = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    payload.setdefault("distributedInfo", {})["rank"] = 0
    (left_dir / "profiler_0.json").write_text(json.dumps(payload), encoding="utf-8")
    (right_dir / "rank0.json").write_text(json.dumps(payload), encoding="utf-8")

    passed, message = compare_trace_directories_with_report(str(left_dir), str(right_dir))

    assert passed
    assert message == "all files match"


def test_verify_normalizes_negative_numeric_tid(tmp_path: Path) -> None:
    """Negative numeric tids should compare equal across int/string forms."""
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    payload = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    payload["traceEvents"][0]["tid"] = -123
    left.write_text(json.dumps(payload), encoding="utf-8")
    payload["traceEvents"][0]["tid"] = "-123"
    right.write_text(json.dumps(payload), encoding="utf-8")

    passed, message = compare_trace_files_with_report(str(left), str(right))

    assert passed, message


def test_decompress_linear_segment_supports_segment_blocks() -> None:
    """Segment block payloads should decompress without hitting fallback errors."""
    segments = [
        (0, 3, 2, 10, [0, 0, 0]),
        (3, 2, 1, 20, [0, 1]),
    ]

    assert SLP.decompress_linear_segment(segments, 0) == 10
    assert SLP.decompress_linear_segment(segments, 1) == 12
    assert SLP.decompress_linear_segment({"segments": segments}, 3) == 20
    assert SLP.decompress_linear_segment({"segments": segments}, 4) == 22


def test_mpi_hierarchical_merge_demo(tmp_path: Path) -> None:
    """MPI demo should pass multi-rank hierarchical merge on the small trace subset."""
    pytest.importorskip("mpi4py")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun is not available")

    command = [
        sys.executable,
        str(ROOT / "examples" / "compressed_analyze" / "compress_demo.py"),
        "--input_file",
        str(TRACE_PATH),
        "--output_file",
        str(tmp_path / "single.json"),
        "--reconstruct_file",
        str(tmp_path / "single_restored.json"),
        "--multi_rank_input_dir",
        str(MERGE_SMALL_DIR),
        "--multi_rank_output_file",
        str(tmp_path / "multi.json"),
        "--multi_rank_reconstruct_dir",
        str(tmp_path / "restored_dir"),
        "--multi_rank_executor",
        "mpi",
        "--mpi_processes",
        "2",
        "--mpi_merge_fanin",
        "2",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        combined_output = f"{exc.stdout}\n{exc.stderr}"
        if "Operation not permitted" in combined_output:
            pytest.skip("mpirun is not permitted in the current environment")
        raise

    assert "Launching MPI merge round 1 with 2 processes" in result.stdout
    assert "Running MPI verify in merged_compressed_file mode" in result.stdout
    assert "multi-rank+merge" in result.stdout
    assert "all files match" in result.stdout
