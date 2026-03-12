"""CLI demo for PADOC trace compression."""

from __future__ import annotations

import argparse
import json
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
from perflowai.padoc.node import count_nodes, count_trace_nodes, print_node_stats
from perflowai.padoc.trace import TraceLoadStats


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


def trace_core_parts(compressed_trace: CompressedTrace) -> Dict[str, int]:
    """Return core in-memory size parts for one compressed trace."""
    return {
        "event_templates": asizeof.asizeof(compressed_trace.event_templates),
        "ranks": asizeof.asizeof(compressed_trace.ranks),
        "metadata": asizeof.asizeof(compressed_trace.metadata),
        "start_timestamp": asizeof.asizeof(compressed_trace.start_timestamp),
        "launch_indexes": asizeof.asizeof(compressed_trace._launch_indexes),
    }


def print_trace_memory_distribution(scenario: str, compressed_trace: CompressedTrace) -> None:
    """Print core and template memory distribution for one compressed trace."""
    print(f"\nMemory Distribution [{scenario}]")
    print_memory_distribution("CompressedTrace Core Memory", trace_core_parts(compressed_trace))
    template_parts = memory_breakdown_templates(compressed_trace.event_templates)
    template_parts = {key: value for key, value in template_parts.items() if key != "total"}
    print_memory_distribution("Event Templates Breakdown", template_parts)


def print_trace_memory_distribution_for_files(scenario: str, paths: List[str]) -> None:
    """Print aggregated memory distribution for independent compressed files."""
    core_parts = defaultdict(int)
    template_parts = defaultdict(int)
    for path in paths:
        compressed_trace = CompressedTrace.from_file(path)
        for key, value in trace_core_parts(compressed_trace).items():
            core_parts[key] += value
        current_template_parts = memory_breakdown_templates(compressed_trace.event_templates)
        for key, value in current_template_parts.items():
            if key == "total":
                continue
            template_parts[key] += value

    print(f"\nMemory Distribution [{scenario}]")
    print_memory_distribution("CompressedTrace Core Memory", dict(core_parts))
    print_memory_distribution("Event Templates Breakdown", dict(template_parts))


def run_mpi_multi_rank_compression(
    input_dir: str,
    compressed_dir: str,
    mpi_processes: int,
    mpi_log_dir: str | None,
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
    compressed_dir: str | None = None,
) -> tuple[bool, str]:
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
    if mpi_log_dir:
        command.extend(["--log_dir", mpi_log_dir])

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
    return bool(summary["passed"]), str(summary["message"])


def print_node_statistics(scenario: str, compressed_trace: CompressedTrace) -> None:
    """Print node statistics for one completed scenario."""
    print(f"\nNode Statistics [{scenario}]")
    count_trace_nodes(compressed_trace)


def print_node_statistics_for_files(scenario: str, paths: List[str]) -> None:
    """Print aggregated node statistics for a list of compressed trace files."""
    counter = defaultdict(int)
    type_memory = defaultdict(int)
    type_field_memory = defaultdict(lambda: defaultdict(int))
    seen = set()

    for path in paths:
        compressed_trace = CompressedTrace.from_file(path)
        for rank in compressed_trace.get_ranks():
            for _, _, _, _, node in compressed_trace.iter_nodes(rank):
                count_nodes(node, counter, type_memory, seen, type_field_memory)

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
        "Time",
        "Verify",
        "Reason",
    ]
    rows = []
    for result in results:
        rows.append([
            result.scenario,
            result.source,
            format_size(result.source_size_bytes),
            format_size(result.source_memory_bytes),
            format_size(result.compressed_size_bytes),
            format_size(result.compressed_memory_bytes),
            compression_ratio(result.compressed_size_bytes, result.source_size_bytes),
            compression_ratio(result.compressed_memory_bytes, result.source_memory_bytes),
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
) -> CompressionRunResult:
    """Run single-rank compression and return the summary row."""
    print(f"Running single-rank compression for {input_file}")
    started_at = time.perf_counter()
    reset_output_path(output_file)
    reset_output_path(restore_file)
    compressor = TemplateCompressor()
    compressed_trace, load_stats = compressor.compress_file(input_file)
    compressed_trace.write_file(output_file)

    verify_passed = True
    verify_message = "skipped"
    if not skip_verify:
        restored_trace = compressor.intra_decompress(CompressedTrace.from_file(output_file))
        restored_trace.write_file(restore_file, origin=True)
        verify_passed, verify_message = compare_trace_files_with_report(input_file, restore_file)
    print_trace_memory_distribution("single-rank", compressed_trace)
    print_node_statistics("single-rank", compressed_trace)

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
    aggregated_before: Dict[str, int] = {}
    aggregated_after: Dict[str, int] = {}

    if executor == "mpi":
        mpi_summary = run_mpi_multi_rank_compression(
            input_dir,
            compressed_dir,
            mpi_processes,
            mpi_log_dir,
        )
        total_load_stats.file_count = int(mpi_summary["file_count"])
        total_load_stats.event_count = int(mpi_summary["event_count"])
        total_load_stats.source_size_bytes = int(mpi_summary["source_size_bytes"])
        total_load_stats.loaded_memory_bytes = int(mpi_summary["source_memory_bytes"])
        total_compressed_size = int(mpi_summary["compressed_size_bytes"])
        total_compressed_memory = int(mpi_summary["compressed_memory_bytes"])
        aggregated_before = {
            key: int(value) for key, value in mpi_summary["memory_before"].items()
        }
        aggregated_after = {
            key: int(value) for key, value in mpi_summary["memory_after"].items()
        }
    else:
        for index, source_file in enumerate(files, start=1):
            output_path = os.path.join(compressed_dir, os.path.basename(source_file))
            compressor = TemplateCompressor()
            compressed_trace, load_stats = compressor.compress_file(source_file, emit_summary=False)
            compressed_trace.write_file(output_path)

            total_load_stats.file_count += load_stats.file_count
            total_load_stats.event_count += load_stats.event_count
            total_load_stats.source_size_bytes += load_stats.source_size_bytes
            total_load_stats.loaded_memory_bytes += load_stats.loaded_memory_bytes
            total_compressed_size += os.path.getsize(output_path)
            total_compressed_memory += asizeof.asizeof(compressed_trace)
            for key, value in compressor.last_memory_before.items():
                aggregated_before[key] = aggregated_before.get(key, 0) + value
            for key, value in compressor.last_memory_after.items():
                aggregated_after[key] = aggregated_after.get(key, 0) + value
            print_progress("multi-rank compress", index, len(files), started_at)

    compressed_files = input_files(compressed_dir)
    verify_passed = True
    verify_message = "skipped"
    if not skip_verify:
        restored_files_dir = restore_dir
        os.makedirs(restored_files_dir, exist_ok=True)
        if executor == "mpi":
            verify_passed, verify_message = run_mpi_verify(
                "compressed_parts",
                input_dir,
                restored_files_dir,
                mpi_processes,
                mpi_log_dir,
                compressed_dir=compressed_dir,
            )
        else:
            verify_started_at = time.perf_counter()
            for index, compressed_file in enumerate(compressed_files, start=1):
                compressed_trace = CompressedTrace.from_file(compressed_file)
                restored_trace = TemplateCompressor().inter_decompress(compressed_trace)
                restored_path = os.path.join(restored_files_dir, os.path.basename(compressed_file))
                restored_trace.write_file(restored_path, origin=True)
                print_progress("multi-rank verify", index, len(compressed_files), verify_started_at)

            verify_passed, verify_message = compare_trace_directories_with_report(input_dir, restored_files_dir)
    print_trace_memory_distribution_for_files("multi-rank", compressed_files)
    print_node_statistics_for_files("multi-rank", compressed_files)
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
) -> CompressionRunResult:
    """Merge previously compressed rank files and return the summary row."""
    print(f"Running multi-rank+merge for {compressed_dir}")
    started_at = time.perf_counter()
    reset_output_path(output_file)
    reset_output_path(restore_dir)
    compressor = TemplateCompressor()
    compressed_trace = compressor.merge_compressed_files(input_files(compressed_dir))
    compressed_trace.write_file(output_file)

    verify_passed = True
    verify_message = "skipped"
    if not skip_verify:
        file_type = "json" if output_file.endswith(".json") else "bin"
        restored_trace = compressor.inter_decompress(CompressedTrace.from_file(output_file))
        restored_trace.write_dir(restore_dir, file_type)
        if executor == "mpi":
            verify_passed, verify_message = run_mpi_verify(
                "directory_compare",
                input_dir,
                restore_dir,
                mpi_processes,
                mpi_log_dir,
            )
        else:
            verify_passed, verify_message = compare_trace_directories_with_report(input_dir, restore_dir)
    print_trace_memory_distribution("multi-rank+merge", compressed_trace)
    print_node_statistics("multi-rank+merge", compressed_trace)

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
    parser.add_argument("--multi_rank_executor", choices=["serial", "mpi"], default="serial")
    parser.add_argument("--mpi_processes", type=int, default=4)
    parser.add_argument("--mpi_log_dir")
    args = parser.parse_args()

    results: List[CompressionRunResult] = []
    results.append(
        collect_single_rank_result(
            args.input_file,
            args.output_file,
            args.reconstruct_file,
            args.skip_verify,
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
        )
        results.append(multi_result)
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
            )
        )

    print_summary_table(results)


if __name__ == "__main__":
    main()
