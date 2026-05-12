"""Audit how many bytes ts/dur arrays consume inside a PADOC artifact.

Walks the **decoded** payload (msgpack outer + pickle leaves) and
measures every leaf field's contribution by **re-packing it through
msgpack** so we get an apples-to-apples byte count vs. the actual
on-disk encoding (msgpack-then-zstd).

Reported numbers
----------------

  * raw msgpack share of each field (ts, dur, id, name, args, …)
  * zstd-after share of the same field (we recompress the field's
    msgpack alone and measure compressed size).  This is a lower
    bound — the real shared zstd dictionary is usually a bit better
    on individual fields, but the ordering / magnitude is informative.
  * Projection of how much we'd save by

      (a) int64 → int32 only for ts/dur
      (b) full SLP @ ~1.4 B per element (mix of int8/int16 + 11 B per
          ~2k-event segment header)

Run::

    python scripts/audit_ts_dur_share.py \\
        /mnt/treasure/ljx/padoc_artifacts/v2/padoc/qwen3_subset8.bin
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import msgpack


# Field names whose **list** value we treat as a numeric column to
# attribute to the ts/dur/etc. bucket.  Anything else is "structural".
_NUMERIC_FIELDS = {"ts", "dur"}
# scalatrace 用 names / payload_args 命名，padoc 用 name / args；两者都覆盖
_OTHER_NUMERIC_FIELDS = {"id", "name", "args", "names", "payload_names",
                          "payload_ts", "payload_dur", "payload_id", "payload_args",
                          "rsd", "args_keys"}


def _try_zstd(blob: bytes) -> bytes:
    try:
        import zstandard as zstd  # type: ignore
        try:
            return zstd.ZstdDecompressor().decompress(blob)
        except zstd.ZstdError:
            return blob
    except ImportError:
        return blob


def _pack(value: Any) -> bytes:
    return msgpack.packb(value, use_bin_type=True)


def _zpack(value: Any) -> int:
    """msgpack-pack ``value`` then zstd compress, return size in bytes."""
    raw = _pack(value)
    try:
        import zstandard as zstd  # type: ignore
        return len(zstd.ZstdCompressor(level=10).compress(raw))
    except ImportError:
        import zlib
        return len(zlib.compress(raw, 6))


def _walk(value: Any, path: str, sink: Dict[str, List[Tuple[str, Any]]]) -> None:
    """Collect every "interesting" leaf into ``sink`` keyed by bucket name."""
    if isinstance(value, dict):
        for k, v in value.items():
            new_path = f"{path}/{k}" if path else str(k)
            if k in _NUMERIC_FIELDS or k in _OTHER_NUMERIC_FIELDS:
                sink[k].append((new_path, v))
                continue
            _walk(v, new_path, sink)
        return
    if isinstance(value, list):
        # If list-of-dicts (e.g. event_templates), recurse; otherwise leaf.
        if value and isinstance(value[0], dict):
            for i, v in enumerate(value):
                _walk(v, f"{path}[{i}]" if i < 3 else f"{path}[…]", sink)


def audit(blob_path: str) -> None:
    with open(blob_path, "rb") as f:
        on_disk = f.read()
    raw = _try_zstd(on_disk)
    payload = msgpack.unpackb(raw, raw=False)

    # ScalaTrace stores under "streams" without pickled leaves; PADOC under "ranks" with pickle.
    ranks = payload.get("ranks", {})
    decoded_ranks: Dict[Any, Any] = {}
    for r, processes in ranks.items():
        decoded_ranks[r] = {}
        for pid, tids in processes.items():
            decoded_ranks[r][pid] = {}
            for tid, phases in tids.items():
                decoded_ranks[r][pid][tid] = {}
                for ph, leaf in phases.items():
                    if isinstance(leaf, (bytes, bytearray)):
                        sys.setrecursionlimit(max(sys.getrecursionlimit(), 200_000))
                        decoded_ranks[r][pid][tid][ph] = pickle.loads(leaf)
                    else:
                        decoded_ranks[r][pid][tid][ph] = leaf
    payload["ranks"] = decoded_ranks

    sink: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
    _walk(payload, "", sink)

    # ---- field-level msgpack/zstd accounting --------------------------------
    field_stats: Dict[str, Dict[str, int]] = {}
    for field, items in sink.items():
        # Concat all values (each is a list) into one giant list to mirror what
        # would happen if we packed them together.  This avoids per-list overhead
        # dominating for many small templates.
        concat: List[Any] = []
        nvals = 0
        for _path, v in items:
            if isinstance(v, list):
                concat.extend(v)
                nvals += len(v)
            else:
                concat.append(v)
                nvals += 1
        msgpack_bytes = len(_pack(concat))
        zstd_bytes = _zpack(concat)
        field_stats[field] = {
            "n_lists": len(items),
            "n_values": nvals,
            "msgpack_bytes": msgpack_bytes,
            "zstd_bytes": zstd_bytes,
        }

    # ---- everything else (structural) ----------------------------------------
    # Strip out the listed fields, then pack the rest.
    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in _NUMERIC_FIELDS or k in _OTHER_NUMERIC_FIELDS:
                    continue
                out[k] = _strip(v)
            return out
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    structural = _strip(payload)
    structural_msgpack = len(_pack(structural))
    structural_zstd = _zpack(structural)

    # ---- print ---------------------------------------------------------------
    print(f"\n=== {blob_path} ===")
    print(f"  on-disk (zstd)  : {len(on_disk)/1e6:8.2f} MB")
    print(f"  msgpack raw     : {len(raw)/1e6:8.2f} MB")
    print()
    print(f"  {'field':<10} {'n_values':>12} {'msgpack MB':>12} "
          f"{'zstd MB':>10} {'B/val (msgpack)':>16} {'B/val (zstd)':>14}")
    for field in ("ts", "dur", "id", "name", "args",
                   "names", "payload_ts", "payload_dur", "payload_args",
                   "rsd", "args_keys"):
        s = field_stats.get(field)
        if not s:
            continue
        nv = max(s["n_values"], 1)
        print(f"  {field:<10} {s['n_values']:>12} {s['msgpack_bytes']/1e6:>12.2f} "
              f"{s['zstd_bytes']/1e6:>10.2f} {s['msgpack_bytes']/nv:>16.2f} "
              f"{s['zstd_bytes']/nv:>14.2f}")
    print(f"  {'STRUCTURAL':<10} {'-':>12} {structural_msgpack/1e6:>12.2f} "
          f"{structural_zstd/1e6:>10.2f}")

    # Sum & shares
    total_zstd_field = sum(s["zstd_bytes"] for s in field_stats.values()) + structural_zstd
    ts = field_stats.get("ts", {})
    dur = field_stats.get("dur", {})
    ts_dur_zstd = ts.get("zstd_bytes", 0) + dur.get("zstd_bytes", 0)
    ts_dur_msgpack = ts.get("msgpack_bytes", 0) + dur.get("msgpack_bytes", 0)
    n_ts = ts.get("n_values", 0)
    n_dur = dur.get("n_values", 0)

    print()
    print(f"  ts+dur zstd     : {ts_dur_zstd/1e6:.2f} MB  "
          f"({ts_dur_zstd/max(total_zstd_field,1)*100:.1f}% of payload, "
          f"vs. on-disk {ts_dur_zstd/max(len(on_disk),1)*100:.1f}%)")
    print(f"  ts+dur msgpack  : {ts_dur_msgpack/1e6:.2f} MB")
    print()
    print(f"  Projections (raw msgpack, before zstd):")
    # int64 list in msgpack ≈ 9 B/val (1 type byte + 8 payload).
    # int32 array packing via numpy + msgpack-numpy or just stripping high bits
    # ⇒ ~5 B/val (msgpack int  type + 4 B).  Conservative estimate uses 5 B.
    int32_proj = (n_ts + n_dur) * 5
    saved_int32 = ts_dur_msgpack - int32_proj
    print(f"    int64→int32  : {int32_proj/1e6:.2f} MB  "
          f"(save {saved_int32/1e6:.2f} MB on raw, "
          f"~{saved_int32/max(ts_dur_msgpack,1)*100:.0f}% of ts+dur)")
    # SLP: 1.4 B/val mean
    slp_proj = (n_ts + n_dur) * 1.4
    saved_slp = ts_dur_msgpack - slp_proj
    print(f"    SLP @1.4B/ev : {slp_proj/1e6:.2f} MB  "
          f"(save {saved_slp/1e6:.2f} MB on raw, "
          f"~{saved_slp/max(ts_dur_msgpack,1)*100:.0f}% of ts+dur)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", nargs="+", help="Path(s) to padoc .bin artifact")
    args = ap.parse_args()
    for path in args.artifact:
        try:
            audit(path)
        except Exception as exc:  # pragma: no cover
            print(f"\n=== {path} === audit FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
