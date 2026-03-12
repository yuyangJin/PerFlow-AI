"""CLI demo for PADOC trace compression."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from perflowai.padoc import (
    CompressedTrace,
    TemplateCompressor,
    compare_trace_directories_with_report,
    compare_trace_files_with_report,
)
from perflowai.padoc._compat import asizeof
from perflowai.padoc.event import memory_breakdown_templates
from perflowai.padoc.node import collect_trace_node_stats, count_trace_nodes, print_node_stats
from perflowai.padoc.trace import TraceLoadStats, compressed_trace_core_parts


@dataclass(frozen=True)
class CompressionRunResult:
    """Summary for one demo scenario."""

    scenario: str
    source: str
    file_count: int
    event_count: int
    source_size_bytes: int
    source_memory_bytes: int
    compressed_size_bytes: int
    compressed_memory_bytes: int
    verify_passed: bool
    verify_message: str
    memory_before: Dict[str, int]
    memory_after: Dict[str, int]
    load_seconds: float
    compress_seconds: float
    store_seconds: float
    verify_seconds: float
    stats_seconds: float
    mpi_overhead_seconds: float
    duration_seconds: float


def format_size(num_bytes: int) -> str:
    """Format bytes into a readable string."""
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def compression_ratio(compressed: int, original: int) -> str:
    """Return compressed/original ratio as text."""
    if original == 0:
        return "100.00%"
    return f"{compressed / original:.2%}"


def format_duration(seconds: float) -> str:
    """Format seconds into a compact duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:d}m{seconds:02d}s"


def shorten_source_path(path: str) -> str:
    """Keep only the last two path components for display."""
    normalized = Path(path)
    parts = normalized.parts
    if len(parts) <= 2:
        return str(normalized)
    return str(Path(parts[-2]) / parts[-1])


def directory_size(paths: Iterable[str]) -> int:
    """Return total size of a file collection."""
    return sum(os.path.getsize(path) for path in paths)


def input_files(input_dir: str) -> List[str]:
    """Return sorted file list for a directory."""
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if os.path.isfile(os.path.join(input_dir, name))
    ]


def output_parts_dir(output_file: str) -> str:
    """Return the directory that stores per-file compressed outputs."""
    path = Path(output_file)
    return str(path.with_suffix("")) + "_parts"


def output_merge_dir(output_file: str) -> str:
    """Return the directory that stores intermediate merge outputs."""
    path = Path(output_file)
    return str(path.with_suffix("")) + "_merge"


def reset_output_path(path: str) -> None:
    """Remove a stale output file or directory before regenerating it."""
    target = Path(path)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def print_progress(
    stage: str,
    current: int,
    total: int,
    started_at: float,
) -> None:
    """Print a simple serial progress line with ETA."""
    elapsed = time.perf_counter() - started_at
    average = elapsed / current if current else 0.0
    remaining = average * max(total - current, 0)
    print(
        f"[{stage}] {current}/{total} completed | "
        f"elapsed={format_duration(elapsed)} | eta={format_duration(remaining)}"
    )


def print_memory_distribution(title: str, parts: Dict[str, int]) -> None:
    """Print a compact memory distribution table."""
    total = sum(parts.values())
    print(f"\n=== {title} ===")
    for name, size in sorted(parts.items(), key=lambda item: item[1], reverse=True):
        pct = (size / total * 100.0) if total > 0 else 0.0
        print(f"{name:16s}: {format_size(size):>10s} ({pct:5.1f}%)")
    print(f"{'total':16s}: {format_size(total):>10s} (100.0%)")


def print_trace_memory_distribution(scenario: str, compressed_trace: CompressedTrace) -> None:
    """Print core and template memory distribution for one compressed trace."""
    print(f"\nMemory Distribution [{scenario}]")
    print_memory_distribution("CompressedTrace Core Memory", compressed_trace_core_parts(compressed_trace))
    template_parts = memory_breakdown_templates(compressed_trace.event_templates)
    template_parts = {key: value for key, value in template_parts.items() if key != "total"}
    print_memory_distribution("Event Templates Breakdown", template_parts)


def print_trace_memory_distribution_from_parts(
    scenario: str,
    core_parts: Dict[str, int],
    template_parts: Dict[str, int],
) -> None:
    """Print aggregated memory distribution from precomputed parts."""
    print(f"\nMemory Distribution [{scenario}]")
    print_memory_distribution("CompressedTrace Core Memory", core_parts)
    print_memory_distribution("Event Templates Breakdown", template_parts)


def run_mpi_multi_rank_compression(
    input_dir: str,
    compressed_dir: str,
    mpi_processes: int,
    mpi_log_dir: str | None,
    json_indent: int,
) -> Dict[str, object]:
    """Run independent per-file compression through mpirun."""
    summary_path = os.path.join(compressed_dir, ".padoc_mpi_summary.json")
    worker_script = Path(__file__).resolve().parents[2] / "perflowai" / "padoc" / "compress_mpi_worker.py"
    command = [
        "mpirun",
        "-np",
        str(mpi_processes),
        sys.executable,
        str(worker_script),
        "--input_dir",
        input_dir,
        "--output_dir",
        compressed_dir,
        "--summary_file",
        summary_path,
    ]
    if mpi_log_dir:
        command.extend(["--log_dir", mpi_log_dir])
    command.extend(["--json_indent", str(json_indent)])

    try:
        import mpi4py  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MPI execution requires mpi4py in the active environment."
        ) from exc

    print(f"Launching MPI compression with {mpi_processes} processes")
    subprocess.run(command, check=True)
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    os.remove(summary_path)
    return summary


def run_mpi_verify(
    mode: str,
    source_dir: str,
    target_dir: str,
    mpi_processes: int,
    mpi_log_dir: str | None,
    json_indent: int,
    compressed_dir: str | None = None,
    compressed_file: str | None = None,
) -> tuple[bool, str, Dict[str, float]]:
    """Run MPI verification on a directory pair."""
    summary_path = os.path.join(target_dir, ".padoc_mpi_verify_summary.json")
    worker_script = Path(__file__).resolve().parents[2] / "perflowai" / "padoc" / "verify_mpi_worker.py"
    command = [
        "mpirun",
        "-np",
        str(mpi_processes),
        sys.executable,
        str(worker_script),
        "--mode",
        mode,
        "--source_dir",
        source_dir,
        "--target_dir",
        target_dir,
        "--summary_file",
        summary_path,
    ]
    if compressed_dir:
        command.extend(["--compressed_dir", compressed_dir])
    if compressed_file:
        command.extend(["--compressed_file", compressed_file])
    if mpi_log_dir:
        command.extend(["--log_dir", mpi_log_dir])
    command.extend(["--json_indent", str(json_indent)])

    try:
        import mpi4py  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MPI verification requires mpi4py in the active environment."
        ) from exc

    print(f"Launching MPI verify with {mpi_processes} processes")
    subprocess.run(command, check=True)
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    os.remove(summary_path)
    return (
        bool(summary["passed"]),
        str(summary["message"]),
        {
            "verify_seconds": float(summary.get("rank0_verify_seconds", 0.0)),
            "mpi_overhead_seconds": float(summary.get("mpi_overhead_seconds", 0.0)),
        },
    )


def split_into_groups(paths: List[str], group_count: int) -> List[List[str]]:
    """Split paths into stable non-empty groups."""
    if group_count <= 0:
        return []
    groups: List[List[str]] = [[] for _ in range(group_count)]
    for index, path in enumerate(sorted(paths)):
        groups[index % group_count].append(path)
    return [group for group in groups if group]


def run_mpi_multi_rank_merge(
    compressed_dir: str,
    output_file: str,
    mpi_processes: int,
    mpi_log_dir: str | None,
    merge_fanin: int,
    json_indent: int,
) -> Dict[str, float]:
    """Run hierarchical MPI merge reduction over independently compressed files."""
    try:
        import mpi4py  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MPI execution requires mpi4py in the active environment."
        ) from exc

    merge_root = output_merge_dir(output_file)
    reset_output_path(merge_root)
    os.makedirs(merge_root, exist_ok=True)

    current_files = sorted(input_files(compressed_dir))
    round_index = 1
    totals = {
        "load_seconds": 0.0,
        "compress_seconds": 0.0,
        "store_seconds": 0.0,
        "mpi_overhead_seconds": 0.0,
    }

    while len(current_files) > 1:
        if round_index == 1:
            group_count = min(mpi_processes, len(current_files))
        else:
            group_count = min(
                mpi_processes,
                max(1, math.ceil(len(current_files) / max(merge_fanin, 1))),
            )
        groups = split_into_groups(current_files, group_count)
        round_dir = os.path.join(merge_root, f"round_{round_index:02d}")
        os.makedirs(round_dir, exist_ok=True)
        manifest_path = os.path.join(round_dir, "manifest.json")
        summary_path = os.path.join(round_dir, "summary.json")
        with open(manifest_path, "w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "round_index": round_index,
                    "groups": groups,
                },
                file_obj,
                indent=2,
            )

        worker_script = Path(__file__).resolve().parents[2] / "perflowai" / "padoc" / "merge_mpi_worker.py"
        command = [
            "mpirun",
            "-np",
            str(len(groups)),
            sys.executable,
            str(worker_script),
            "--manifest",
            manifest_path,
            "--output_dir",
            round_dir,
            "--summary_file",
            summary_path,
        ]
        if mpi_log_dir:
            round_log_dir = os.path.join(mpi_log_dir, f"merge_round_{round_index:02d}")
            command.extend(["--log_dir", round_log_dir])
        command.extend(["--json_indent", str(json_indent)])

        print(
            f"Launching MPI merge round {round_index} with {len(groups)} processes",
            flush=True,
        )
        subprocess.run(command, check=True)
        with open(summary_path, "r", encoding="utf-8") as file_obj:
            summary = json.load(file_obj)

        rank0_timings = summary.get("rank0_timings", {})
        totals["load_seconds"] += float(rank0_timings.get("load_seconds", 0.0))
        totals["compress_seconds"] += float(rank0_timings.get("compress_seconds", 0.0))
        totals["store_seconds"] += float(rank0_timings.get("store_seconds", 0.0))
        totals["mpi_overhead_seconds"] += float(summary.get("mpi_overhead_seconds", 0.0))
        current_files = [str(path) for path in summary["output_files"]]
        round_index += 1

    shutil.copyfile(current_files[0], output_file)
    return totals


def print_node_statistics(scenario: str, compressed_trace: CompressedTrace) -> None:
    """Print node statistics for one completed scenario."""
    print(f"\nNode Statistics [{scenario}]")
    count_trace_nodes(compressed_trace)


def print_node_statistics_from_parts(
    scenario: str,
    counter: Dict[str, int],
    type_memory: Dict[str, int],
    type_field_memory: Dict[str, Dict[str, int]],
) -> None:
    """Print aggregated node statistics from precomputed parts."""
    print(f"\nNode Statistics [{scenario}]")
    print_node_stats(counter, "Trace Node Statistics", type_memory, type_field_memory)


def print_summary_table(results: List[CompressionRunResult]) -> None:
    """Print the final summary table."""
    headers = [
        "Scenario",
        "Source",
        "Source File",
        "Source Memory",
        "Compressed File",
        "Compressed Memory",
        "File Ratio",
        "Memory Ratio",
        "Load",
        "Compress",
        "Store",
        "Verify",
        "Stats",
        "MPI Overhead",
        "Time",
        "Verify",
        "Reason",
    ]
    rows = []
    for result in results:
        rows.append([
            result.scenario,
            shorten_source_path(result.source),
            format_size(result.source_size_bytes),
            format_size(result.source_memory_bytes),
            format_size(result.compressed_size_bytes),
            format_size(result.compressed_memory_bytes),
            compression_ratio(result.compressed_size_bytes, result.source_size_bytes),
            compression_ratio(result.compressed_memory_bytes, result.source_memory_bytes),
            format_duration(result.load_seconds),
            format_duration(result.compress_seconds),
            format_duration(result.store_seconds),
            format_duration(result.verify_seconds),
            format_duration(result.stats_seconds),
            format_duration(result.mpi_overhead_seconds),
            format_duration(result.duration_seconds),
            "PASS" if result.verify_passed else "FAIL",
            result.verify_message,
        ])

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(values: List[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    print("\nCompression Summary")
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))


def collect_single_rank_result(
    input_file: str,
    output_file: str,
    restore_file: str,
    skip_verify: bool,
    json_indent: int,
) -> CompressionRunResult:
    """Run single-rank compression and return the summary row."""
    print(f"Running single-rank compression for {input_file}")
    started_at = time.perf_counter()
    reset_output_path(output_file)
    reset_output_path(restore_file)
    compressor = TemplateCompressor()
    compressed_trace, load_stats, timings = compressor.compress_file_with_timing(input_file)
    store_start = time.perf_counter()
    compressed_trace.write_file(output_file, json_indent=json_indent)
    store_seconds = time.perf_counter() - store_start

    verify_passed = True
    verify_message = "skipped"
    verify_seconds = 0.0
    if not skip_verify:
        verify_start = time.perf_counter()
        restored_trace = compressor.intra_decompress(CompressedTrace.from_file(output_file))
        restored_trace.write_file(restore_file, origin=True, json_indent=json_indent)
        verify_passed, verify_message = compare_trace_files_with_report(input_file, restore_file)
        verify_seconds = time.perf_counter() - verify_start
    stats_start = time.perf_counter()
    print_trace_memory_distribution("single-rank", compressed_trace)
    print_node_statistics("single-rank", compressed_trace)
    stats_seconds = time.perf_counter() - stats_start

    return CompressionRunResult(
        scenario="single-rank",
        source=input_file,
        file_count=load_stats.file_count,
        event_count=load_stats.event_count,
        source_size_bytes=os.path.getsize(input_file),
        source_memory_bytes=load_stats.loaded_memory_bytes,
        compressed_size_bytes=os.path.getsize(output_file),
        compressed_memory_bytes=asizeof.asizeof(compressed_trace),
        verify_passed=verify_passed,
        verify_message=verify_message,
        memory_before=compressor.last_memory_before,
        memory_after=compressor.last_memory_after,
        load_seconds=timings["load_seconds"],
        compress_seconds=timings["compress_seconds"],
        store_seconds=store_seconds,
        verify_seconds=verify_seconds,
        stats_seconds=stats_seconds,
        mpi_overhead_seconds=0.0,
        duration_seconds=time.perf_counter() - started_at,
    )


def collect_multi_rank_result(
    input_dir: str,
    output_file: str,
    restore_dir: str,
    max_workers: int | None,
    skip_verify: bool,
    executor: str,
    mpi_processes: int,
    mpi_log_dir: str | None,
    json_indent: int,
) -> tuple[CompressionRunResult, str]:
    """Compress each input file independently and return the summary row."""
    del max_workers
    print(f"Running multi-rank compression for {input_dir}")
    started_at = time.perf_counter()
    files = input_files(input_dir)
    compressed_dir = output_parts_dir(output_file)
    reset_output_path(compressed_dir)
    reset_output_path(restore_dir)
    os.makedirs(compressed_dir, exist_ok=True)

    total_load_stats = TraceLoadStats()
    total_compressed_size = 0
    total_compressed_memory = 0
    load_seconds = 0.0
    compress_seconds = 0.0
    store_seconds = 0.0
    verify_seconds = 0.0
    mpi_overhead_seconds = 0.0
    aggregated_before: Dict[str, int] = {}
    aggregated_after: Dict[str, int] = {}
    aggregated_core_parts: Dict[str, int] = defaultdict(int)
    aggregated_template_parts: Dict[str, int] = defaultdict(int)
    aggregated_node_counter: Dict[str, int] = defaultdict(int)
    aggregated_node_type_memory: Dict[str, int] = defaultdict(int)
    aggregated_node_field_memory: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    if executor == "mpi":
        mpi_summary = run_mpi_multi_rank_compression(
            input_dir,
            compressed_dir,
            mpi_processes,
            mpi_log_dir,
            json_indent,
        )
        total_load_stats.file_count = int(mpi_summary["file_count"])
        total_load_stats.event_count = int(mpi_summary["event_count"])
        total_load_stats.source_size_bytes = int(mpi_summary["source_size_bytes"])
        total_load_stats.loaded_memory_bytes = int(mpi_summary["source_memory_bytes"])
        total_compressed_size = int(mpi_summary["compressed_size_bytes"])
        total_compressed_memory = int(mpi_summary["compressed_memory_bytes"])
        rank0_timings = mpi_summary["rank0_timings"]
        load_seconds = float(rank0_timings.get("load_seconds", 0.0))
        compress_seconds = float(rank0_timings.get("compress_seconds", 0.0))
        store_seconds = float(rank0_timings.get("store_seconds", 0.0))
        mpi_overhead_seconds = float(mpi_summary.get("mpi_overhead_seconds", 0.0))
        aggregated_before = {
            key: int(value) for key, value in mpi_summary["memory_before"].items()
        }
        aggregated_after = {
            key: int(value) for key, value in mpi_summary["memory_after"].items()
        }
        aggregated_core_parts = defaultdict(
            int,
            {key: int(value) for key, value in mpi_summary["core_parts"].items()},
        )
        aggregated_template_parts = defaultdict(
            int,
            {key: int(value) for key, value in mpi_summary["template_parts"].items()},
        )
        aggregated_node_counter = defaultdict(
            int,
            {key: int(value) for key, value in mpi_summary["node_counter"].items()},
        )
        aggregated_node_type_memory = defaultdict(
            int,
            {key: int(value) for key, value in mpi_summary["node_type_memory"].items()},
        )
        aggregated_node_field_memory = defaultdict(lambda: defaultdict(int))
        for node_type, field_values in mpi_summary["node_field_memory"].items():
            aggregated_node_field_memory[node_type] = defaultdict(
                int,
                {field: int(value) for field, value in field_values.items()},
            )
    else:
        for index, source_file in enumerate(files, start=1):
            output_path = os.path.join(compressed_dir, os.path.basename(source_file))
            compressor = TemplateCompressor()
            compressed_trace, load_stats, timings = compressor.compress_file_with_timing(
                source_file,
                emit_summary=False,
            )
            store_start = time.perf_counter()
            compressed_trace.write_file(output_path, json_indent=json_indent)
            store_seconds += time.perf_counter() - store_start

            total_load_stats.file_count += load_stats.file_count
            total_load_stats.event_count += load_stats.event_count
            total_load_stats.source_size_bytes += load_stats.source_size_bytes
            total_load_stats.loaded_memory_bytes += load_stats.loaded_memory_bytes
            total_compressed_size += os.path.getsize(output_path)
            total_compressed_memory += asizeof.asizeof(compressed_trace)
            load_seconds += timings["load_seconds"]
            compress_seconds += timings["compress_seconds"]
            for key, value in compressor.last_memory_before.items():
                aggregated_before[key] = aggregated_before.get(key, 0) + value
            for key, value in compressor.last_memory_after.items():
                aggregated_after[key] = aggregated_after.get(key, 0) + value
            for key, value in compressed_trace_core_parts(compressed_trace).items():
                aggregated_core_parts[key] += value
            for key, value in memory_breakdown_templates(compressed_trace.event_templates).items():
                if key == "total":
                    continue
                aggregated_template_parts[key] += value
            node_counter, node_type_memory, node_field_memory = collect_trace_node_stats(compressed_trace)
            for key, value in node_counter.items():
                aggregated_node_counter[key] += value
            for key, value in node_type_memory.items():
                aggregated_node_type_memory[key] += value
            for node_type, field_values in node_field_memory.items():
                for field, value in field_values.items():
                    aggregated_node_field_memory[node_type][field] += value
            print_progress("multi-rank compress", index, len(files), started_at)

    compressed_files = input_files(compressed_dir)
    verify_passed = True
    verify_message = "skipped"
    if not skip_verify:
        restored_files_dir = restore_dir
        os.makedirs(restored_files_dir, exist_ok=True)
        if executor == "mpi":
            verify_passed, verify_message, verify_timing = run_mpi_verify(
                "compressed_parts",
                input_dir,
                restored_files_dir,
                mpi_processes,
                mpi_log_dir,
                json_indent,
                compressed_dir=compressed_dir,
            )
            verify_seconds = verify_timing["verify_seconds"]
            mpi_overhead_seconds += verify_timing["mpi_overhead_seconds"]
        else:
            verify_started_at = time.perf_counter()
            for index, compressed_file in enumerate(compressed_files, start=1):
                compressed_trace = CompressedTrace.from_file(compressed_file)
                restored_trace = TemplateCompressor().inter_decompress(compressed_trace)
                restored_path = os.path.join(restored_files_dir, os.path.basename(compressed_file))
                restored_trace.write_file(restored_path, origin=True, json_indent=json_indent)
                print_progress("multi-rank verify", index, len(compressed_files), verify_started_at)

            verify_passed, verify_message = compare_trace_directories_with_report(input_dir, restored_files_dir)
            verify_seconds = time.perf_counter() - verify_started_at
    stats_start = time.perf_counter()
    print_trace_memory_distribution_from_parts(
        "multi-rank",
        dict(aggregated_core_parts),
        dict(aggregated_template_parts),
    )
    print_node_statistics_from_parts(
        "multi-rank",
        dict(aggregated_node_counter),
        dict(aggregated_node_type_memory),
        {
            node_type: dict(field_values)
            for node_type, field_values in aggregated_node_field_memory.items()
        },
    )
    stats_seconds = time.perf_counter() - stats_start
    return (
        CompressionRunResult(
            scenario="multi-rank",
            source=input_dir,
            file_count=total_load_stats.file_count,
            event_count=total_load_stats.event_count,
            source_size_bytes=directory_size(files),
            source_memory_bytes=total_load_stats.loaded_memory_bytes,
            compressed_size_bytes=total_compressed_size,
            compressed_memory_bytes=total_compressed_memory,
            verify_passed=verify_passed,
            verify_message=verify_message,
            memory_before=aggregated_before,
            memory_after=aggregated_after,
            load_seconds=load_seconds,
            compress_seconds=compress_seconds,
            store_seconds=store_seconds,
            verify_seconds=verify_seconds,
            stats_seconds=stats_seconds,
            mpi_overhead_seconds=mpi_overhead_seconds,
            duration_seconds=time.perf_counter() - started_at,
        ),
        compressed_dir,
    )


def collect_multi_rank_merge_result(
    input_dir: str,
    compressed_dir: str,
    output_file: str,
    restore_dir: str,
    baseline_memory_bytes: int,
    skip_verify: bool,
    executor: str,
    mpi_processes: int,
    mpi_log_dir: str | None,
    mpi_merge_fanin: int,
    json_indent: int,
) -> CompressionRunResult:
    """Merge previously compressed rank files and return the summary row."""
    print(f"Running multi-rank+merge for {compressed_dir}")
    started_at = time.perf_counter()
    reset_output_path(output_file)
    reset_output_path(restore_dir)
    compressor = TemplateCompressor()
    mpi_overhead_seconds = 0.0
    if executor == "mpi":
        merge_timings = run_mpi_multi_rank_merge(
            compressed_dir,
            output_file,
            mpi_processes,
            mpi_log_dir,
            mpi_merge_fanin,
            json_indent,
        )
        compressed_trace = CompressedTrace.from_file(output_file)
        store_seconds = 0.0
        mpi_overhead_seconds = merge_timings["mpi_overhead_seconds"]
    else:
        compressed_trace, merge_timings = compressor.merge_compressed_files_with_timing(
            input_files(compressed_dir)
        )
        store_start = time.perf_counter()
        compressed_trace.write_file(output_file, json_indent=json_indent)
        store_seconds = time.perf_counter() - store_start

    verify_passed = True
    verify_message = "skipped"
    verify_seconds = 0.0
    if not skip_verify:
        file_type = "json" if output_file.endswith(".json") else "bin"
        if executor == "mpi":
            verify_passed, verify_message, verify_timing = run_mpi_verify(
                "merged_compressed_file",
                input_dir,
                restore_dir,
                mpi_processes,
                mpi_log_dir,
                json_indent,
                compressed_file=output_file,
            )
            verify_seconds = verify_timing["verify_seconds"]
            mpi_overhead_seconds = verify_timing["mpi_overhead_seconds"]
        else:
            verify_start = time.perf_counter()
            compressor.inter_decompress_to_dir(
                CompressedTrace.from_file(output_file),
                restore_dir,
                file_type,
                json_indent=json_indent,
            )
            verify_passed, verify_message = compare_trace_directories_with_report(input_dir, restore_dir)
            verify_seconds = time.perf_counter() - verify_start
    stats_start = time.perf_counter()
    print_trace_memory_distribution("multi-rank+merge", compressed_trace)
    print_node_statistics("multi-rank+merge", compressed_trace)
    stats_seconds = time.perf_counter() - stats_start

    return CompressionRunResult(
        scenario="multi-rank+merge",
        source=input_dir,
        file_count=len(input_files(input_dir)),
        event_count=0,
        source_size_bytes=directory_size(input_files(input_dir)),
        source_memory_bytes=baseline_memory_bytes,
        compressed_size_bytes=os.path.getsize(output_file),
        compressed_memory_bytes=asizeof.asizeof(compressed_trace),
        verify_passed=verify_passed,
        verify_message=verify_message,
        memory_before=compressor.last_memory_before,
        memory_after=compressor.last_memory_after,
        load_seconds=merge_timings["load_seconds"],
        compress_seconds=merge_timings["compress_seconds"],
        store_seconds=store_seconds,
        verify_seconds=verify_seconds,
        stats_seconds=stats_seconds,
        mpi_overhead_seconds=mpi_overhead_seconds,
        duration_seconds=time.perf_counter() - started_at,
    )


def main() -> None:
    """Parse arguments and run demos."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="tests/example_trace/profiler_585.json")
    parser.add_argument("--output_file", default="compressed.bin")
    parser.add_argument("--reconstruct_file", default="reconstructed.bin")
    parser.add_argument("--multi_rank_input_dir")
    parser.add_argument("--multi_rank_output_file", default="compressed_multi_rank.json")
    parser.add_argument("--multi_rank_reconstruct_dir", default="reconstructed_dir")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--run_merge", action="store_true")
    parser.add_argument("--json_indent", type=int, default=0)
    parser.add_argument("--multi_rank_executor", choices=["serial", "mpi"], default="serial")
    parser.add_argument("--mpi_processes", type=int, default=4)
    parser.add_argument("--mpi_merge_fanin", type=int, default=8)
    parser.add_argument("--mpi_log_dir")
    args = parser.parse_args()

    results: List[CompressionRunResult] = []
    results.append(
        collect_single_rank_result(
            args.input_file,
            args.output_file,
            args.reconstruct_file,
            args.skip_verify,
            args.json_indent,
        )
    )

    if args.multi_rank_input_dir:
        multi_result, compressed_dir = collect_multi_rank_result(
            args.multi_rank_input_dir,
            args.multi_rank_output_file,
            args.multi_rank_reconstruct_dir,
            max_workers=args.max_workers,
            skip_verify=args.skip_verify,
            executor=args.multi_rank_executor,
            mpi_processes=args.mpi_processes,
            mpi_log_dir=args.mpi_log_dir,
            json_indent=args.json_indent,
        )
        results.append(multi_result)
        if args.run_merge:
            merged_output = f"{os.path.splitext(args.multi_rank_output_file)[0]}_merged{os.path.splitext(args.multi_rank_output_file)[1]}"
            merged_restore = f"{args.multi_rank_reconstruct_dir}_merged"
            results.append(
                collect_multi_rank_merge_result(
                    args.multi_rank_input_dir,
                    compressed_dir,
                    merged_output,
                    merged_restore,
                    baseline_memory_bytes=multi_result.source_memory_bytes,
                    skip_verify=args.skip_verify,
                    executor=args.multi_rank_executor,
                    mpi_processes=args.mpi_processes,
                    mpi_log_dir=args.mpi_log_dir,
                    mpi_merge_fanin=args.mpi_merge_fanin,
                    json_indent=args.json_indent,
                )
            )

    print_summary_table(results)


if __name__ == "__main__":
    main()
