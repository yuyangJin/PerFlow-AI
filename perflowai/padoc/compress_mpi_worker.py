"""MPI worker for independent multi-rank PADOC compression."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from mpi4py import MPI

from perflowai.padoc import TemplateCompressor
from perflowai.padoc._compat import asizeof


def input_files(input_dir: str) -> List[str]:
    """Return sorted file list for a directory."""
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if os.path.isfile(os.path.join(input_dir, name))
    ]


def append_log(log_file: Path | None, message: str) -> None:
    """Append one line to a worker log file if enabled."""
    if log_file is None:
        return
    with log_file.open("a", encoding="utf-8") as file_obj:
        file_obj.write(message + "\n")


def configure_worker_output(rank: int, log_file: Path | None) -> None:
    """Keep only rank0 logs on the terminal; optionally redirect workers to files."""
    if rank == 0:
        return

    if log_file is None:
        sink = open(os.devnull, "w", encoding="utf-8")
    else:
        sink = log_file.open("a", encoding="utf-8")

    sys.stdout = sink
    sys.stderr = sink

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if hasattr(handler, "setStream"):
            handler.setStream(sink)


def format_duration(seconds: float) -> str:
    """Format seconds into a compact duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:d}m{seconds:02d}s"


def main() -> None:
    """Compress an independent subset of files under MPI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_file", required=True)
    parser.add_argument("--log_dir")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    files = input_files(args.input_dir)
    assigned_files = files[rank::world_size]

    log_file = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = Path(args.log_dir) / f"worker_{rank}.log"
        if log_file.exists():
            log_file.unlink()

    configure_worker_output(rank, log_file)

    if rank == 0:
        print(
            f"Running MPI multi-rank compression with {world_size} processes "
            f"for {len(files)} files"
        )

    local_summary = {
        "file_count": 0,
        "event_count": 0,
        "source_size_bytes": 0,
        "source_memory_bytes": 0,
        "compressed_size_bytes": 0,
        "compressed_memory_bytes": 0,
        "memory_before": defaultdict(int),
        "memory_after": defaultdict(int),
    }
    started_at = time.perf_counter()
    assigned_total = len(assigned_files)

    for index, source_file in enumerate(assigned_files, start=1):
        append_log(log_file, f"compressing {source_file}")
        compressor = TemplateCompressor()
        compressed_trace, load_stats = compressor.compress_file(source_file, emit_summary=False)
        output_path = os.path.join(args.output_dir, os.path.basename(source_file))
        compressed_trace.write_file(output_path)

        local_summary["file_count"] += load_stats.file_count
        local_summary["event_count"] += load_stats.event_count
        local_summary["source_size_bytes"] += load_stats.source_size_bytes
        local_summary["source_memory_bytes"] += load_stats.loaded_memory_bytes
        local_summary["compressed_size_bytes"] += os.path.getsize(output_path)
        local_summary["compressed_memory_bytes"] += asizeof.asizeof(compressed_trace)
        for key, value in compressor.last_memory_before.items():
            local_summary["memory_before"][key] += value
        for key, value in compressor.last_memory_after.items():
            local_summary["memory_after"][key] += value

        if rank == 0 and assigned_total > 0:
            elapsed = time.perf_counter() - started_at
            average = elapsed / index
            eta = average * max(assigned_total - index, 0)
            print(
                f"[mpi-rank0 compress] {index}/{assigned_total} completed | "
                f"elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
            )

    gathered = comm.gather(local_summary, root=0)
    if rank != 0:
        return

    merged = {
        "file_count": 0,
        "event_count": 0,
        "source_size_bytes": 0,
        "source_memory_bytes": 0,
        "compressed_size_bytes": 0,
        "compressed_memory_bytes": 0,
        "memory_before": {},
        "memory_after": {},
    }

    merged_before: Dict[str, int] = defaultdict(int)
    merged_after: Dict[str, int] = defaultdict(int)
    for summary in gathered:
        merged["file_count"] += summary["file_count"]
        merged["event_count"] += summary["event_count"]
        merged["source_size_bytes"] += summary["source_size_bytes"]
        merged["source_memory_bytes"] += summary["source_memory_bytes"]
        merged["compressed_size_bytes"] += summary["compressed_size_bytes"]
        merged["compressed_memory_bytes"] += summary["compressed_memory_bytes"]
        for key, value in summary["memory_before"].items():
            merged_before[key] += value
        for key, value in summary["memory_after"].items():
            merged_after[key] += value

    merged["memory_before"] = dict(merged_before)
    merged["memory_after"] = dict(merged_after)
    with open(args.summary_file, "w", encoding="utf-8") as file_obj:
        json.dump(merged, file_obj, indent=2)


if __name__ == "__main__":
    main()
