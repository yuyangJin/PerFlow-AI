"""MPI worker for hierarchical PADOC merge reduction."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from mpi4py import MPI

from perflowai.padoc import TemplateCompressor


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
    """Run one MPI merge reduction round and write one summary file on rank 0."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_file", required=True)
    parser.add_argument("--log_dir")
    parser.add_argument("--json_indent", type=int, default=2)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    log_file = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = Path(args.log_dir) / f"merge_worker_{rank}.log"
        if log_file.exists():
            log_file.unlink()

    configure_worker_output(rank, log_file)

    with open(args.manifest, "r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)

    round_index = int(manifest["round_index"])
    groups: List[List[str]] = manifest["groups"]
    assigned_paths = groups[rank] if rank < len(groups) else []

    if rank == 0:
        total_inputs = sum(len(group) for group in groups)
        print(
            f"Running MPI merge round {round_index} with {world_size} processes "
            f"for {total_inputs} files -> {len(groups)} outputs",
            flush=True,
        )

    local_summary: Dict[str, object] = {
        "output_file": "",
        "group_size": len(assigned_paths),
        "timings": {
            "load_seconds": 0.0,
            "compress_seconds": 0.0,
            "store_seconds": 0.0,
        },
    }

    started_at = time.perf_counter()
    if assigned_paths:
        append_log(log_file, f"merging {len(assigned_paths)} files")
        compressor = TemplateCompressor()
        compressed_trace, timings = compressor.merge_compressed_files_with_timing(
            assigned_paths,
            emit_summary=False,
            emit_progress=(rank == 0),
            emit_rank_logs=False,
        )
        output_path = os.path.join(args.output_dir, f"merge_{rank:05d}.json")
        store_start = time.perf_counter()
        compressed_trace.write_file(output_path, json_indent=args.json_indent)
        local_summary["output_file"] = output_path
        local_summary["timings"] = {
            "load_seconds": float(timings["load_seconds"]),
            "compress_seconds": float(timings["compress_seconds"]),
            "store_seconds": time.perf_counter() - store_start,
        }

    if rank == 0:
        elapsed = time.perf_counter() - started_at
        print(
            f"[mpi-rank0 merge] 1/1 completed | elapsed={format_duration(elapsed)} | "
            f"eta={format_duration(0.0)}",
            flush=True,
        )

    gather_start = time.perf_counter()
    gathered = comm.gather(local_summary, root=0)
    if rank != 0:
        return
    mpi_overhead_seconds = time.perf_counter() - gather_start

    output_files = [
        str(item["output_file"])
        for item in gathered
        if item["output_file"]
    ]
    rank0_timings = gathered[0]["timings"] if gathered else {
        "load_seconds": 0.0,
        "compress_seconds": 0.0,
        "store_seconds": 0.0,
    }
    summary = {
        "output_files": output_files,
        "round_outputs": len(output_files),
        "rank0_timings": rank0_timings,
        "mpi_overhead_seconds": mpi_overhead_seconds,
    }
    with open(args.summary_file, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    print(
        f"MPI merge round {round_index} finished | outputs={len(output_files)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
