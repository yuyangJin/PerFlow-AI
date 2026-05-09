"""``python -m perflowai.padoc.bench`` -- command-line interface.

Three subcommands::

    bench compress  --traces ... --compressors ... [--out-json ...] [--out-csv ...]
    bench analyze   --traces ... --compressors ... --tasks ...
    bench list      [traces|compressors|tasks]

The CLI is intentionally thin: it loads datasets via
:func:`perflowai.padoc.bench.datasets.load_dataset` (built-in name OR
on-disk path), then dispatches to :func:`run_compression_matrix` /
:func:`run_analysis_matrix` and renders the result.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

from ..baselines import available_compressors
from .datasets import (
    TraceDataset,
    builtin_datasets,
    load_dataset,
    load_dataset_manifest,
)
from .report import (
    render_analysis_markdown,
    render_compression_markdown,
    write_records_csv,
    write_records_json,
)
from .parallel import render_parallel_markdown, run_parallel_compression
from .runner import run_analysis_matrix, run_compression_matrix
from .scalability import (
    render_scalability_markdown,
    run_gpu_sweep,
    run_iteration_sweep,
    run_layer_sweep,
)
from .tasks import builtin_tasks


# ----------------------------------------------------------------------
# Argument helpers
# ----------------------------------------------------------------------


def _resolve_traces(args: argparse.Namespace) -> List[TraceDataset]:
    if getattr(args, "manifest", None):
        return load_dataset_manifest(args.manifest)

    requested: List[str] = []
    if getattr(args, "traces", None):
        requested.extend(args.traces)
    if not requested:
        traces = builtin_datasets()
        if not traces:
            raise SystemExit(
                "No traces specified and no built-in datasets available; "
                "pass --traces <path-or-name> or --manifest <file.json>."
            )
        return traces

    out: List[TraceDataset] = []
    for entry in requested:
        out.append(load_dataset(entry))
    return out


def _resolve_compressors(args: argparse.Namespace) -> List[str]:
    if getattr(args, "compressors", None):
        return list(args.compressors)
    return ["raw_msgpack", "gzip_msgpack", "tracezip", "scalatrace", "padoc"]


def _resolve_tasks(args: argparse.Namespace) -> List[str]:
    if getattr(args, "tasks", None):
        return list(args.tasks)
    return ["operator_hotspot", "stream_load_balance"]


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------


def cmd_compress(args: argparse.Namespace) -> int:
    traces = _resolve_traces(args)
    compressors = _resolve_compressors(args)

    def _progress(record):
        size = record.compressed_size_bytes
        ratio = record.compression_ratio
        print(
            f"  [done] {record.trace_name:>20s} | {record.compressor:>12s} | "
            f"size={size:>10d} ratio={ratio:5.2f}x | "
            f"compress={record.compress_seconds:6.2f}s decompress={record.decompress_seconds:6.2f}s "
            f"| verify={'Y' if record.verify_passed else ('-' if not record.verify_message else 'N')}",
            flush=True,
        )

    print(f"=== compression matrix ({len(traces)} traces x {len(compressors)} compressors) ===")
    result = run_compression_matrix(
        datasets=traces,
        compressors=compressors,
        verify=not args.no_verify,
        track_memory=not args.no_track_memory,
        progress_cb=_progress,
    )
    print()
    print(render_compression_markdown(result.records))

    if args.out_json:
        write_records_json(result.records, args.out_json)
        print(f"wrote {args.out_json}")
    if args.out_csv:
        write_records_csv(result.records, args.out_csv)
        print(f"wrote {args.out_csv}")
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("# Compression matrix\n\n")
            f.write(render_compression_markdown(result.records))
        print(f"wrote {args.out_md}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    traces = _resolve_traces(args)
    compressors = _resolve_compressors(args)
    tasks = _resolve_tasks(args)

    def _progress(record):
        status = "ok" if record.success else f"err: {record.error_message}"
        in_situ = "Y" if record.in_situ else "-"
        print(
            f"  [done] {record.trace_name:>20s} | {record.compressor:>12s} | "
            f"task={record.task:>22s} in_situ={in_situ} | "
            f"end_to_end={record.end_to_end_seconds:6.2f}s | {status}",
            flush=True,
        )

    print(
        f"=== analysis matrix ({len(traces)} traces x {len(compressors)} compressors "
        f"x {len(tasks)} tasks) ==="
    )
    result = run_analysis_matrix(
        datasets=traces,
        compressors=compressors,
        tasks=tasks,
        progress_cb=_progress,
    )
    print()
    print(render_analysis_markdown(result.records))

    if args.out_json:
        write_records_json(result.records, args.out_json)
        print(f"wrote {args.out_json}")
    if args.out_csv:
        write_records_csv(result.records, args.out_csv)
        print(f"wrote {args.out_csv}")
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("# Analysis matrix\n\n")
            f.write(render_analysis_markdown(result.records))
        print(f"wrote {args.out_md}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    target = args.target
    if target == "compressors":
        for name in available_compressors():
            print(name)
    elif target == "tasks":
        for name in builtin_tasks():
            print(name)
    elif target == "traces":
        for dataset in builtin_datasets():
            print(f"{dataset.name}\t{dataset.path}")
    else:
        raise SystemExit(f"unknown target: {target}")
    return 0


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perflowai.padoc.bench",
        description="Benchmark PADOC and baseline compressors on AI traces.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--traces",
        nargs="*",
        default=None,
        help="One or more dataset names or trace file/dir paths (defaults to local sample).",
    )
    common.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest of datasets (overrides --traces).",
    )
    common.add_argument(
        "--compressors",
        nargs="*",
        default=None,
        help="Compressors to evaluate (defaults to raw_msgpack gzip_msgpack tracezip scalatrace padoc).",
    )
    common.add_argument("--out-json", default=None, help="Write records as JSON.")
    common.add_argument("--out-csv", default=None, help="Write records as CSV.")
    common.add_argument("--out-md", default=None, help="Write a markdown summary.")

    p_compress = sub.add_parser(
        "compress",
        parents=[common],
        help="Run the compression-ratio matrix.",
    )
    p_compress.add_argument(
        "--no-verify", action="store_true", help="Skip the decompress + sig-check pass."
    )
    p_compress.add_argument(
        "--no-track-memory",
        action="store_true",
        help="Disable tracemalloc -- faster but no peak memory column.",
    )
    p_compress.set_defaults(handler=cmd_compress)

    p_analyze = sub.add_parser(
        "analyze",
        parents=[common],
        help="Run the analysis-time matrix.",
    )
    p_analyze.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Tasks to evaluate (defaults to operator_hotspot stream_load_balance).",
    )
    p_analyze.set_defaults(handler=cmd_analyze)

    p_list = sub.add_parser("list", help="List built-in compressors / tasks / traces.")
    p_list.add_argument(
        "target",
        choices=["compressors", "tasks", "traces"],
    )
    p_list.set_defaults(handler=cmd_list)

    p_scale = sub.add_parser(
        "scalability",
        help="Run synthetic-trace scalability sweeps (gpus / layers / iters).",
    )
    p_scale.add_argument(
        "--axis",
        choices=["gpus", "layers", "iterations"],
        default="gpus",
        help="Which dimension to sweep.",
    )
    p_scale.add_argument(
        "--values",
        nargs="+",
        type=int,
        required=True,
        help="Concrete values for the swept axis (e.g. --values 4 8 16 32).",
    )
    p_scale.add_argument(
        "--compressors",
        nargs="*",
        default=None,
        help="Compressors to include (defaults to all baselines + padoc).",
    )
    p_scale.add_argument("--out-md", default=None, help="Write the sweep table as markdown.")
    p_scale.set_defaults(handler=cmd_scalability)

    p_par = sub.add_parser(
        "parallel",
        help="Sweep worker count when compressing a multi-rank trace dir.",
    )
    p_par.add_argument("--trace-dir", required=True)
    p_par.add_argument("--compressor", default="padoc")
    p_par.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )
    p_par.add_argument(
        "--backend",
        choices=["process", "thread"],
        default="process",
    )
    p_par.add_argument("--out-md", default=None)
    p_par.set_defaults(handler=cmd_parallel)

    return parser


def cmd_scalability(args: argparse.Namespace) -> int:
    compressors = (
        args.compressors
        if args.compressors
        else ["raw_msgpack", "gzip_msgpack", "tracezip", "scalatrace", "padoc"]
    )
    sweep_fn = {
        "gpus": run_gpu_sweep,
        "layers": run_layer_sweep,
        "iterations": run_iteration_sweep,
    }[args.axis]
    sweep_kw = {
        "gpus": "gpu_counts",
        "layers": "layer_counts",
        "iterations": "iteration_counts",
    }[args.axis]
    points = sweep_fn(**{sweep_kw: args.values}, compressors=tuple(compressors))
    md = render_scalability_markdown(points)
    print(md)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(f"# Scalability sweep ({args.axis})\n\n")
            f.write(md)
        print(f"wrote {args.out_md}")
    return 0


def cmd_parallel(args: argparse.Namespace) -> int:
    rows = run_parallel_compression(
        args.trace_dir,
        compressor_name=args.compressor,
        workers_grid=args.workers,
        backend=args.backend,
    )
    md = render_parallel_markdown(rows)
    print(md)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(f"# Parallel compression ({args.backend})\n\n")
            f.write(md)
        print(f"wrote {args.out_md}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
