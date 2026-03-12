"""MPI worker for PADOC verification tasks."""

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

from perflowai.padoc import CompressedTrace, TemplateCompressor, compare_trace_files_with_report
from perflowai.padoc.verify import _directory_trace_index


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


def sorted_keys(index: Dict[str, str]) -> List[str]:
    """Return stable keys for a trace index."""
    return sorted(index.keys())


def main() -> None:
    """Run MPI verification work and write one summary file on rank 0."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["compressed_parts", "directory_compare", "merged_compressed_file"],
        required=True,
    )
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--compressed_dir")
    parser.add_argument("--compressed_file")
    parser.add_argument("--summary_file", required=True)
    parser.add_argument("--log_dir")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    log_file = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = Path(args.log_dir) / f"verify_worker_{rank}.log"
        if log_file.exists():
            log_file.unlink()

    configure_worker_output(rank, log_file)

    source_index = _directory_trace_index(args.source_dir)
    target_index = _directory_trace_index(args.target_dir) if os.path.isdir(args.target_dir) else {}
    compressed_index = _directory_trace_index(args.compressed_dir) if args.compressed_dir else {}

    if args.mode == "compressed_parts":
        if sorted(source_index.keys()) != sorted(compressed_index.keys()):
            if rank == 0:
                with open(args.summary_file, "w", encoding="utf-8") as file_obj:
                    json.dump(
                        {
                            "passed": False,
                            "message": (
                                f"file key mismatch: {sorted(source_index.keys())} vs "
                                f"{sorted(compressed_index.keys())}"
                            ),
                        },
                        file_obj,
                        indent=2,
                    )
            return
        keys = sorted_keys(source_index)
    elif args.mode == "directory_compare":
        if sorted(source_index.keys()) != sorted(target_index.keys()):
            if rank == 0:
                with open(args.summary_file, "w", encoding="utf-8") as file_obj:
                    json.dump(
                        {
                            "passed": False,
                            "message": (
                                f"file key mismatch: {sorted(source_index.keys())} vs "
                                f"{sorted(target_index.keys())}"
                            ),
                        },
                        file_obj,
                        indent=2,
                    )
            return
        keys = sorted_keys(source_index)
    else:
        keys = sorted_keys(source_index)

    if rank == 0:
        print(
            f"Running MPI verify in {args.mode} mode with {world_size} processes "
            f"for {len(keys)} files"
        )

    assigned_keys = keys[rank::world_size]
    failures: List[str] = []
    started_at = time.perf_counter()
    assigned_total = len(assigned_keys)

    for index, file_key in enumerate(assigned_keys, start=1):
        source_path = source_index[file_key]
        if args.mode == "compressed_parts":
            target_path = os.path.join(args.target_dir, os.path.basename(source_path))
        elif args.mode == "merged_compressed_file":
            target_path = os.path.join(args.target_dir, f"rank{file_key}.json")
        else:
            target_path = target_index[file_key]
        append_log(log_file, f"verifying {source_path} vs {target_path}")

        if args.mode == "compressed_parts":
            compressed_path = compressed_index[file_key]
            compressed_trace = CompressedTrace.from_file(compressed_path)
            restored_trace = TemplateCompressor().inter_decompress(compressed_trace)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            restored_trace.write_file(target_path, origin=True)
        elif args.mode == "merged_compressed_file":
            compressed_trace = CompressedTrace.from_file(args.compressed_file)
            restored_trace = TemplateCompressor().intra_decompress(compressed_trace, file_key)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            restored_trace.write_file(target_path, file_key, origin=True)

        passed, message = compare_trace_files_with_report(source_path, target_path)
        if not passed:
            failures.append(f"{os.path.basename(source_path)}: {message}")

        if rank == 0 and assigned_total > 0:
            elapsed = time.perf_counter() - started_at
            average = elapsed / index
            eta = average * max(assigned_total - index, 0)
            print(
                f"[mpi-rank0 verify] {index}/{assigned_total} completed | "
                f"elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
            )

    gathered = comm.gather(failures, root=0)
    if rank != 0:
        return

    flat_failures = [item for chunk in gathered for item in chunk]
    summary = {
        "passed": not flat_failures,
        "message": "all files match" if not flat_failures else flat_failures[0],
    }
    with open(args.summary_file, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    print(f"MPI verify finished | passed={summary['passed']} | message={summary['message']}")


if __name__ == "__main__":
    main()
