#!/usr/bin/env python3
"""PADOC compression performance breakdown.

For each input trace directory the script runs PADOC's parallel
streaming pipeline and breaks the wall time into:

* ``load_seconds``      – disk -> python events (per-rank workers)
* ``intra_seconds``     – per-rank template compression (workers)
* ``serialize_seconds`` – per-rank ``msgpack.packb`` of the compressed trace
* ``ipc_seconds``       – wall - sum(workers' load+intra+serialize)
* ``merge_seconds``     – parent process: ``_merge_independent_compressed_traces``
* ``finalize_seconds``  – parent process: ``_merge_independent_ranks`` (if ``--merge-ranks``)
* ``writeout_seconds``  – serializing the final ``CompressedTrace`` to bytes

Output is markdown + JSON.  Designed for the paper's "performance
breakdown" axis so reviewers see the dominant cost and the parallel
amortization.

Usage::

    python -u scripts/perf_breakdown.py \
        --label qwen3_subset8 --trace-dir scratch/qwen3_subset8 --workers 8

Or as a sweep::

    python -u scripts/perf_breakdown.py \
        --traces scripts/manifests/subset8.json scripts/manifests/subset32.json \
        --workers 8 16 \
        --out-md report/exp_results/E22_perf_breakdown.md \
        --out-json report/exp_results/E22_perf_breakdown.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import msgpack

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perflowai.padoc.compressor import TemplateCompressor
from perflowai.padoc.config import default_config
from perflowai.padoc.trace import (
    Trace,
    compressed_trace_bin_bytes,
    serialized_trace_file_size,
)

try:
    import zstandard as zstd  # type: ignore

    def _final_compress(data: bytes, level: int = 10) -> bytes:
        return zstd.ZstdCompressor(level=level).compress(data)

    _FINAL_NAME = "zstd"
except Exception:  # pragma: no cover - production env has zstandard
    import gzip

    def _final_compress(data: bytes, level: int = 6) -> bytes:  # type: ignore
        return gzip.compress(data, compresslevel=level)

    _FINAL_NAME = "gzip"


@dataclass
class FilePhaseTimings:
    rank: str
    load_seconds: float
    intra_seconds: float
    serialize_seconds: float
    blob_bytes: int
    event_count: int
    source_size_bytes: int


@dataclass
class TraceBreakdown:
    label: str
    workers: int
    file_count: int
    event_count: int
    source_size_bytes: int
    output_bytes: int
    output_bytes_zstd: int
    wall_seconds: float
    sum_load_seconds: float
    sum_intra_seconds: float
    sum_serialize_seconds: float
    ipc_seconds: float
    merge_seconds: float
    finalize_seconds: float
    writeout_seconds: float
    zstd_seconds: float
    template_count: int
    rank_count: int
    merge_ranks: bool
    per_file: List[FilePhaseTimings] = field(default_factory=list)


def _profile_one_file(file_path: str, config_dict: dict) -> Tuple[bytes, dict]:
    """Worker: load + intra-compress + serialize one trace file, with timings."""
    logging.getLogger("perflowai.padoc").setLevel(logging.WARNING)
    from perflowai.padoc.config import CompressorConfig

    cfg = CompressorConfig(**config_dict)

    t0 = time.perf_counter()
    file_data = Trace.load_file_data(file_path)
    t_load = time.perf_counter() - t0

    compressor = TemplateCompressor(config=cfg)
    t1 = time.perf_counter()
    compressed_trace, file_stats = compressor._compress_file_data(
        file_data, emit_summary=False
    )
    t_intra = time.perf_counter() - t1

    t2 = time.perf_counter()
    blob = compressed_trace_bin_bytes(compressed_trace)
    t_serialize = time.perf_counter() - t2

    return blob, {
        "rank": str(file_data.rank),
        "load_seconds": t_load,
        "intra_seconds": t_intra,
        "serialize_seconds": t_serialize,
        "blob_bytes": len(blob),
        "event_count": int(file_stats.event_count),
        "source_size_bytes": int(file_stats.source_size_bytes),
    }


def profile_trace(
    label: str,
    trace_dir: str,
    workers: int,
    *,
    merge_ranks: bool,
) -> TraceBreakdown:
    files = Trace._list_trace_files(trace_dir)
    if not files:
        raise SystemExit(f"no trace files under {trace_dir}")

    cfg_dict = default_config().as_dict()
    parent = TemplateCompressor(default_config())

    per_file: List[FilePhaseTimings] = []
    blobs: List[bytes] = []

    wall_start = time.perf_counter()
    if workers <= 1:
        for fp in files:
            blob, info = _profile_one_file(fp, cfg_dict)
            blobs.append(blob)
            per_file.append(FilePhaseTimings(**info))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_profile_one_file, fp, cfg_dict): fp for fp in files}
            for fut in as_completed(futs):
                blob, info = fut.result()
                blobs.append(blob)
                per_file.append(FilePhaseTimings(**info))

    parallel_section_seconds = time.perf_counter() - wall_start
    sum_load = sum(p.load_seconds for p in per_file)
    sum_intra = sum(p.intra_seconds for p in per_file)
    sum_serialize = sum(p.serialize_seconds for p in per_file)
    workers_eff = max(workers, 1)
    ipc_seconds = max(
        parallel_section_seconds - (sum_load + sum_intra + sum_serialize) / workers_eff,
        0.0,
    )

    from perflowai.padoc.trace import CompressedTrace

    independent_traces = [CompressedTrace.from_bytes(b) for b in blobs]

    t_merge_start = time.perf_counter()
    independent_result = parent._merge_independent_compressed_traces(independent_traces)
    merge_seconds = time.perf_counter() - t_merge_start

    if merge_ranks:
        t_fin_start = time.perf_counter()
        final_result = parent._merge_independent_ranks(independent_result)
        finalize_seconds = time.perf_counter() - t_fin_start
    else:
        final_result = independent_result
        finalize_seconds = 0.0

    t_write_start = time.perf_counter()
    output_blob = compressed_trace_bin_bytes(final_result)
    writeout_seconds = time.perf_counter() - t_write_start

    t_zstd_start = time.perf_counter()
    output_blob_zstd = _final_compress(output_blob)
    zstd_seconds = time.perf_counter() - t_zstd_start
    wall_seconds = time.perf_counter() - wall_start

    return TraceBreakdown(
        label=label,
        workers=workers,
        file_count=len(files),
        event_count=sum(p.event_count for p in per_file),
        source_size_bytes=sum(p.source_size_bytes for p in per_file),
        output_bytes=len(output_blob),
        output_bytes_zstd=len(output_blob_zstd),
        wall_seconds=wall_seconds,
        sum_load_seconds=sum_load,
        sum_intra_seconds=sum_intra,
        sum_serialize_seconds=sum_serialize,
        ipc_seconds=ipc_seconds,
        merge_seconds=merge_seconds,
        finalize_seconds=finalize_seconds,
        writeout_seconds=writeout_seconds,
        zstd_seconds=zstd_seconds,
        template_count=len(final_result.event_templates),
        rank_count=len(final_result.ranks),
        merge_ranks=merge_ranks,
        per_file=per_file,
    )


def _human_seconds(s: float) -> str:
    if s < 1:
        return f"{s*1000:.1f} ms"
    if s < 60:
        return f"{s:.2f} s"
    return f"{s/60:.2f} min"


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} GiB"


def render_markdown(breakdowns: List[TraceBreakdown]) -> str:
    lines: List[str] = []
    lines.append("# PADOC compression performance breakdown")
    lines.append("")
    final_name = _FINAL_NAME
    lines.append(
        f"| label | workers | files | events | source | output (msgpack) | output (+{final_name}) | "
        f"ratio (msgpack) | ratio (+{final_name}) | wall | "
        f"load (sum / per-worker) | intra (sum / per-worker) | serialize | ipc | "
        f"merge | finalize | writeout | {final_name} | templates | ranks |"
    )
    lines.append(
        "| ----- | ------: | ----: | -----: | -----: | -----: | -----: | ----: | ----: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for b in breakdowns:
        ratio = b.source_size_bytes / b.output_bytes if b.output_bytes else 0.0
        ratio_z = b.source_size_bytes / b.output_bytes_zstd if b.output_bytes_zstd else 0.0
        per_worker_load = b.sum_load_seconds / max(b.workers, 1)
        per_worker_intra = b.sum_intra_seconds / max(b.workers, 1)
        lines.append(
            f"| {b.label} | {b.workers} | {b.file_count} | {b.event_count} | "
            f"{_human_bytes(b.source_size_bytes)} | {_human_bytes(b.output_bytes)} | "
            f"{_human_bytes(b.output_bytes_zstd)} | "
            f"{ratio:.2f}x | {ratio_z:.2f}x | {_human_seconds(b.wall_seconds)} | "
            f"{_human_seconds(b.sum_load_seconds)} / {_human_seconds(per_worker_load)} | "
            f"{_human_seconds(b.sum_intra_seconds)} / {_human_seconds(per_worker_intra)} | "
            f"{_human_seconds(b.sum_serialize_seconds)} | {_human_seconds(b.ipc_seconds)} | "
            f"{_human_seconds(b.merge_seconds)} | {_human_seconds(b.finalize_seconds)} | "
            f"{_human_seconds(b.writeout_seconds)} | {_human_seconds(b.zstd_seconds)} | "
            f"{b.template_count} | {b.rank_count} |"
        )
    lines.append("")
    lines.append(
        "_load/intra are wall-summed across workers; the per-worker column "
        "is `sum / workers`, an apples-to-apples view of the parallel section. "
        "ipc = `wall - max(per-worker phases)`._"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--trace-dir", action="append", default=[])
    ap.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[8],
    )
    ap.add_argument(
        "--merge-ranks",
        action="store_true",
        help="Run the cross-rank dedup pass too (PADOC's merge_ranks=True path).",
    )
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if not args.trace_dir:
        raise SystemExit("at least one --trace-dir is required")
    if len(args.label) != len(args.trace_dir):
        # auto-label from basename
        args.label = [Path(p).name for p in args.trace_dir]

    breakdowns: List[TraceBreakdown] = []
    for label, tdir in zip(args.label, args.trace_dir):
        for w in args.workers:
            print(f"[perf-breakdown] label={label} dir={tdir} workers={w} merge_ranks={args.merge_ranks}", flush=True)
            b = profile_trace(label, tdir, w, merge_ranks=args.merge_ranks)
            breakdowns.append(b)
            print(
                f"[perf-breakdown]   wall={_human_seconds(b.wall_seconds)} "
                f"output={_human_bytes(b.output_bytes)} "
                f"templates={b.template_count}",
                flush=True,
            )

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        Path(args.out_md).write_text(render_markdown(breakdowns), encoding="utf-8")
        print(f"wrote {args.out_md}", flush=True)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps([{**asdict(b), "per_file": [asdict(p) for p in b.per_file]} for b in breakdowns], indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.out_json}", flush=True)

    print(render_markdown(breakdowns), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
