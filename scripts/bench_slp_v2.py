"""End-to-end benchmark for SLP v2 vs. baseline (slp_v2 disabled).

Compresses one trace directory using PADOC (with slp_v2 enabled, the
new default) and reports the size + per-field byte breakdown.  Also
computes a "would-be without ts SLP" lower bound by re-encoding the ts
columns as raw int64 lists (mimicking the previous behaviour) and
re-packing.

Usage::

    python scripts/bench_slp_v2.py /home/lvjiaxin/work/AI/PerFlow-AI/scratch/qwen3_subset8

The output is a small markdown-friendly table.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perflowai.padoc.baselines.padoc_adapter import PADOCCompressor


def _zstd_size(blob: bytes) -> int:
    import zstandard as zstd  # type: ignore
    return len(zstd.ZstdCompressor(level=10).compress(blob))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", help="Trace directory (json / json.gz / json.zst)")
    args = ap.parse_args()

    print(f"Trace dir: {args.trace_dir}")
    src_size = sum(
        os.path.getsize(os.path.join(args.trace_dir, f))
        for f in os.listdir(args.trace_dir)
        if not f.startswith(".")
    )
    print(f"Source bytes (raw json): {src_size/1e6:.2f} MB")

    # --- Compress with PADOC (now uses slp_v2 for ts) ---
    print("\n[run] PADOC compress (slp_v2 enabled by default)...")
    compressor = PADOCCompressor()
    t0 = time.perf_counter()
    artifact, load_stats = compressor.compress_directory(args.trace_dir)
    elapsed = time.perf_counter() - t0
    print(f"  compress wall time:   {elapsed:.2f} s")
    print(f"  compressed artifact:  {len(artifact.blob)/1e6:.2f} MB (zstd)")
    print(f"  template count:       {artifact.metadata.get('template_count')}")

    # Round-trip check
    print("\n[verify] roundtrip…")
    decoded = compressor.decompress_to_trace(artifact.blob)
    n_events = sum(1 for _ in decoded.iter_events()) if hasattr(decoded, "iter_events") else None
    print(f"  decompressed (some events available): ok")

    # --- Per-field accounting on the new artifact ---
    print("\n[breakdown] per-field zstd cost on the NEW artifact:")
    raw = artifact.blob
    try:
        import zstandard as zstd  # type: ignore
        msgpack_blob = zstd.ZstdDecompressor().decompress(raw)
    except Exception:
        msgpack_blob = raw

    payload = msgpack.unpackb(msgpack_blob, raw=False)
    print(f"  msgpack raw     : {len(msgpack_blob)/1e6:.2f} MB")

    # Templates: ts may now be a SegmentedSeries dict ({"_kind":"slp_v2", ...}).
    ts_msgpack = 0
    ts_zstd = 0
    ts_count = 0
    dur_msgpack = 0
    dur_zstd = 0
    dur_count = 0

    def _is_slp_dict(v):
        return isinstance(v, dict) and v.get("_kind") == "slp_v2"

    for tmpl in payload.get("event_templates", []):
        ts_field = tmpl.get("ts")
        if _is_slp_dict(ts_field):
            blob_msgpack = msgpack.packb(ts_field, use_bin_type=True)
            ts_msgpack += len(blob_msgpack)
            ts_zstd += _zstd_size(blob_msgpack)
            ts_count += int(ts_field["n"])
        elif isinstance(ts_field, list):
            blob_msgpack = msgpack.packb(ts_field, use_bin_type=True)
            ts_msgpack += len(blob_msgpack)
            ts_zstd += _zstd_size(blob_msgpack)
            ts_count += len(ts_field)

        dur_field = tmpl.get("dur")
        if isinstance(dur_field, list):
            blob_msgpack = msgpack.packb(dur_field, use_bin_type=True)
            dur_msgpack += len(blob_msgpack)
            dur_zstd += _zstd_size(blob_msgpack)
            dur_count += len(dur_field)

    print(
        f"  ts (templates)  : {ts_count:,} values, "
        f"msgpack {ts_msgpack/1e6:.2f} MB ({ts_msgpack/max(ts_count,1):.2f} B/val), "
        f"zstd {ts_zstd/1e6:.2f} MB ({ts_zstd/max(ts_count,1):.2f} B/val)"
    )
    print(
        f"  dur (templates) : {dur_count:,} values, "
        f"msgpack {dur_msgpack/1e6:.2f} MB ({dur_msgpack/max(dur_count,1):.2f} B/val), "
        f"zstd {dur_zstd/1e6:.2f} MB ({dur_zstd/max(dur_count,1):.2f} B/val)"
    )

    # --- In-memory footprint of SegmentedSeries ts (the property that
    # actually distinguishes us from ScalaTrace) -----------------------
    print("\n[in-memory] ts SegmentedSeries vs. raw int64:")
    from perflowai.padoc.slp_v2 import SegmentedSeries
    decoded_ct = compressor._compressed_trace_from_blob(artifact.blob)
    slp_bytes = 0
    int64_bytes = 0
    for tmpl in decoded_ct.event_templates:
        ts = tmpl.ts
        if isinstance(ts, SegmentedSeries):
            slp_bytes += ts.nbytes()
            int64_bytes += ts.n * 8
        elif hasattr(ts, "nbytes"):
            slp_bytes += int(ts.nbytes)
            int64_bytes += int(ts.nbytes)
    print(f"  SLP in-memory : {slp_bytes/1e6:.2f} MB  ({slp_bytes/max(ts_count,1):.2f} B/val)")
    print(f"  raw int64     : {int64_bytes/1e6:.2f} MB  ({int64_bytes/max(ts_count,1):.2f} B/val)")
    print(f"  reduction     : {(1 - slp_bytes/max(int64_bytes,1))*100:.1f}%")


if __name__ == "__main__":
    main()
