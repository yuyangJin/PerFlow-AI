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
from perflowai.padoc.event import memory_breakdown_templates
from perflowai.padoc.node import collect_trace_node_stats
from perflowai.padoc.trace import compressed_trace_core_parts


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


def to_plain_summary(summary: Dict[str, object]) -> Dict[str, object]:
    """Convert nested defaultdict containers into plain dicts for MPI pickling."""
    return {
        "file_count": summary["file_count"],
        "event_count": summary["event_count"],
        "source_size_bytes": summary["source_size_bytes"],
        "source_memory_bytes": summary["source_memory_bytes"],
        "compressed_size_bytes": summary["compressed_size_bytes"],
        "compressed_memory_bytes": summary["compressed_memory_bytes"],
        "memory_before": dict(summary["memory_before"]),
        "memory_after": dict(summary["memory_after"]),
        "core_parts": dict(summary["core_parts"]),
        "template_parts": dict(summary["template_parts"]),
        "node_counter": dict(summary["node_counter"]),
        "node_type_memory": dict(summary["node_type_memory"]),
        "node_field_memory": {
            node_type: dict(field_values)
            for node_type, field_values in summary["node_field_memory"].items()
        },
        "timings": dict(summary["timings"]),
    }


def main() -> None:
    """Compress an independent subset of files under MPI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_file", required=True)
    parser.add_argument("--log_dir")
    parser.add_argument("--json_indent", type=int, default=2)
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
            f"for {len(files)} files",
            flush=True,
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
        "core_parts": defaultdict(int),
        "template_parts": defaultdict(int),
        "node_counter": defaultdict(int),
        "node_type_memory": defaultdict(int),
        "node_field_memory": defaultdict(lambda: defaultdict(int)),
        "timings": {
            "load_seconds": 0.0,
            "compress_seconds": 0.0,
            "store_seconds": 0.0,
        },
    }
    started_at = time.perf_counter()
    assigned_total = len(assigned_files)

    for index, source_file in enumerate(assigned_files, start=1):
        append_log(log_file, f"compressing {source_file}")
        compressor = TemplateCompressor()
        compressed_trace, load_stats, timings = compressor.compress_file_with_timing(
            source_file,
            emit_summary=False,
        )
        output_path = os.path.join(args.output_dir, os.path.basename(source_file))
        store_start = time.perf_counter()
        compressed_trace.write_file(output_path, json_indent=args.json_indent)
        store_seconds = time.perf_counter() - store_start

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
        for key, value in compressed_trace_core_parts(compressed_trace).items():
            local_summary["core_parts"][key] += value
        for key, value in memory_breakdown_templates(compressed_trace.event_templates).items():
            if key == "total":
                continue
            local_summary["template_parts"][key] += value
        node_counter, node_type_memory, node_field_memory = collect_trace_node_stats(compressed_trace)
        for key, value in node_counter.items():
            local_summary["node_counter"][key] += value
        for key, value in node_type_memory.items():
            local_summary["node_type_memory"][key] += value
        for node_type, field_values in node_field_memory.items():
            for field, value in field_values.items():
                local_summary["node_field_memory"][node_type][field] += value
        local_summary["timings"]["load_seconds"] += timings["load_seconds"]
        local_summary["timings"]["compress_seconds"] += timings["compress_seconds"]
        local_summary["timings"]["store_seconds"] += store_seconds

        if rank == 0 and assigned_total > 0:
            elapsed = time.perf_counter() - started_at
            average = elapsed / index
            eta = average * max(assigned_total - index, 0)
            print(
                f"[mpi-rank0 compress] {index}/{assigned_total} completed | "
                f"elapsed={format_duration(elapsed)} | eta={format_duration(eta)}",
                flush=True,
            )

    gather_start = time.perf_counter()
    gathered = comm.gather(to_plain_summary(local_summary), root=0)
    if rank != 0:
        return
    mpi_overhead_seconds = time.perf_counter() - gather_start

    merged = {
        "file_count": 0,
        "event_count": 0,
        "source_size_bytes": 0,
        "source_memory_bytes": 0,
        "compressed_size_bytes": 0,
        "compressed_memory_bytes": 0,
        "memory_before": {},
        "memory_after": {},
        "core_parts": {},
        "template_parts": {},
        "node_counter": {},
        "node_type_memory": {},
        "node_field_memory": {},
        "rank0_timings": {},
        "mpi_overhead_seconds": mpi_overhead_seconds,
    }

    merged_before: Dict[str, int] = defaultdict(int)
    merged_after: Dict[str, int] = defaultdict(int)
    merged_core_parts: Dict[str, int] = defaultdict(int)
    merged_template_parts: Dict[str, int] = defaultdict(int)
    merged_node_counter: Dict[str, int] = defaultdict(int)
    merged_node_type_memory: Dict[str, int] = defaultdict(int)
    merged_node_field_memory: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
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
        for key, value in summary["core_parts"].items():
            merged_core_parts[key] += value
        for key, value in summary["template_parts"].items():
            merged_template_parts[key] += value
        for key, value in summary["node_counter"].items():
            merged_node_counter[key] += value
        for key, value in summary["node_type_memory"].items():
            merged_node_type_memory[key] += value
        for node_type, field_values in summary["node_field_memory"].items():
            for field, value in field_values.items():
                merged_node_field_memory[node_type][field] += value

    merged["memory_before"] = dict(merged_before)
    merged["memory_after"] = dict(merged_after)
    merged["core_parts"] = dict(merged_core_parts)
    merged["template_parts"] = dict(merged_template_parts)
    merged["node_counter"] = dict(merged_node_counter)
    merged["node_type_memory"] = dict(merged_node_type_memory)
    merged["node_field_memory"] = {
        node_type: dict(field_values)
        for node_type, field_values in merged_node_field_memory.items()
    }
    merged["rank0_timings"] = dict(gathered[0]["timings"])
    with open(args.summary_file, "w", encoding="utf-8") as file_obj:
        json.dump(merged, file_obj, indent=2)


if __name__ == "__main__":
    main()
