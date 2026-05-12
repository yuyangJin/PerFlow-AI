"""Probe how different residual layouts for SLP compress under zstd.

We pick one large ts column from a real PADOC artifact and try several
encodings of the same data:

  A. baseline:  msgpack varint list of int (raw int64 ts values)
  B. slp_v2 bytes:    current implementation (residuals as raw bytes)
  C. slp_v2 ints:     residuals as a msgpack list of ints
  D. raw int32 bytes: just np.int32(ts).tobytes() (no SLP)
  E. raw int32 ints:  ts as a msgpack list of ints, capped to int32

For each we report msgpack-only and msgpack+zstd sizes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perflowai.padoc.slp_v2 import slp_encode, _DTYPE_TABLE


def _zstd(blob: bytes) -> int:
    import zstandard as zstd  # type: ignore
    return len(zstd.ZstdCompressor(level=10).compress(blob))


def _pack(value) -> bytes:
    return msgpack.packb(value, use_bin_type=True)


def main() -> None:
    artifact = "/mnt/treasure/ljx/padoc_artifacts/v2/padoc/qwen3_subset8.bin"
    print(f"Loading {artifact}…")
    with open(artifact, "rb") as f:
        raw = f.read()
    import zstandard as zstd
    payload = msgpack.unpackb(zstd.ZstdDecompressor().decompress(raw), raw=False)

    # Pick the longest ts list among templates (likely a hot op).
    longest = (0, None)
    for tmpl in payload["event_templates"]:
        ts = tmpl.get("ts")
        if isinstance(ts, list) and len(ts) > longest[0]:
            longest = (len(ts), ts)
    n, ts_list = longest
    if ts_list is None:
        print("no ts found")
        return
    ts = np.asarray(ts_list, dtype=np.int64)
    print(f"\nLongest ts column: {n} values, min={ts.min()} max={ts.max()} "
          f"step≈{int(np.median(np.diff(ts))) if n>1 else '-'}")

    print(f"{'enc':<20} {'msgpack MB':>12} {'zstd MB':>12} {'msgpack B/val':>14} {'zstd B/val':>14}")

    # A. baseline msgpack varint list
    blob = _pack(ts.tolist())
    print(f"{'A. msgpack int list':<20} {len(blob)/1e6:>12.3f} {_zstd(blob)/1e6:>12.3f} "
          f"{len(blob)/n:>14.3f} {_zstd(blob)/n:>14.3f}")

    # B. slp_v2 bytes (current)
    series = slp_encode(ts)
    series_dict = series.to_dict()
    blob = _pack(series_dict)
    print(f"{'B. slp bytes':<20} {len(blob)/1e6:>12.3f} {_zstd(blob)/1e6:>12.3f} "
          f"{len(blob)/n:>14.3f} {_zstd(blob)/n:>14.3f}  segments={series.lengths.size}")

    # C. slp_v2 with residuals as msgpack int list (one big concatenated list)
    # We need to recover residual ints from the bytes blob.
    res_ints = []
    cursor = 0
    for length, dt in zip(series.lengths.tolist(), series.dtypes.tolist()):
        length = int(length)
        dt = int(dt)
        if dt == 0:
            continue
        np_dt, byte_size, _ = _DTYPE_TABLE[dt]
        nbytes = length * byte_size
        chunk = np.frombuffer(series.residuals[cursor:cursor+nbytes], dtype=np_dt).tolist()
        res_ints.extend(chunk)
        cursor += nbytes
    series_dict_c = {
        "_kind": "slp_v2_ints",
        "n": series.n,
        "lengths": series.lengths.tolist(),
        "slopes": series.slopes.tolist(),
        "intercepts": series.intercepts.tolist(),
        "dtypes": series.dtypes.tolist(),
        "residuals": res_ints,
    }
    blob = _pack(series_dict_c)
    print(f"{'C. slp int list':<20} {len(blob)/1e6:>12.3f} {_zstd(blob)/1e6:>12.3f} "
          f"{len(blob)/n:>14.3f} {_zstd(blob)/n:>14.3f}")

    # D. raw int32 bytes
    blob = _pack(ts.astype(np.int32).tobytes())
    print(f"{'D. int32 bytes':<20} {len(blob)/1e6:>12.3f} {_zstd(blob)/1e6:>12.3f} "
          f"{len(blob)/n:>14.3f} {_zstd(blob)/n:>14.3f}")

    # E. delta-encoded int32 bytes (ts[k] - ts[k-1])
    diffs = np.empty_like(ts, dtype=np.int32)
    diffs[0] = ts[0]
    diffs[1:] = (ts[1:] - ts[:-1]).astype(np.int32)
    blob = _pack(diffs.tobytes())
    print(f"{'E. delta int32 bytes':<20} {len(blob)/1e6:>12.3f} {_zstd(blob)/1e6:>12.3f} "
          f"{len(blob)/n:>14.3f} {_zstd(blob)/n:>14.3f}")

    # F. delta-encoded int16 bytes (cap deltas to int16 if possible)
    blob = _pack(diffs.astype(np.int16).tobytes())
    print(f"{'F. delta int16 bytes':<20} {len(blob)/1e6:>12.3f} {_zstd(blob)/1e6:>12.3f} "
          f"{len(blob)/n:>14.3f} {_zstd(blob)/n:>14.3f}  (LOSSY if delta>32k!)")

    # G. ts via msgpack int list encoded as numpy array directly (no list overhead)
    blob = _pack({"vals": ts.tolist()})
    print(f"{'G. dict varint list':<20} {len(blob)/1e6:>12.3f} {_zstd(blob)/1e6:>12.3f} "
          f"{len(blob)/n:>14.3f} {_zstd(blob)/n:>14.3f}")


if __name__ == "__main__":
    main()
