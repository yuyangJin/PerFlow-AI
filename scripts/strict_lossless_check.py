#!/usr/bin/env python3
"""逐字段验证 baseline 是否无损（只在小子集上跑，O(events)）。

对 ``--trace`` 给定的 trace 文件/目录，逐 baseline 跑 ``compress -> decompress``，
然后对 **每个事件** 比 ``name / ts / dur / id / cat / bp / s / args / pid / tid / ph``
是否完全一致；若不一致，打印前若干处 diff 并以非零退出。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from perflowai.padoc.baselines import (
    GzipMsgpackCompressor,
    PADOCCompressor,
    RawMsgpackCompressor,
    ScalaTraceCompressor,
    TracezipCompressor,
)
from perflowai.padoc.trace import Trace


# 要比对的字段（id 在 chrome trace 中可能为字符串或整数；统一成 str 比对）
FIELDS = ("name", "ts", "dur", "id", "cat", "bp", "s", "pid", "tid", "ph")


def _norm_event(rank: str, pid: str, tid: str, ph: str, ev) -> Tuple:
    """把一个 Event/KernelEvent 拍成可比的元组（缺字段统一为 ``None``）。"""
    norm_tid = str(tid)
    if norm_tid.startswith("stream "):
        norm_tid = norm_tid.split(" ", 1)[1]
    if norm_tid.endswith("-abnormal"):
        norm_tid = norm_tid.split("-", 1)[0]
    ev_id = getattr(ev, "id", None)
    ev_cat = getattr(ev, "cat", None)
    ev_bp = getattr(ev, "bp", None)
    ev_s = getattr(ev, "s", None)
    ev_args = getattr(ev, "args", None)
    return (
        str(rank),
        str(pid),
        norm_tid,
        str(ph),
        str(ev.name),
        int(ev.ts),
        None if ev.dur is None else int(ev.dur),
        None if ev_id is None else str(ev_id),
        ev_cat,
        ev_bp,
        ev_s,
        json.dumps(ev_args, sort_keys=True, default=str) if ev_args else None,
    )


def _gather(trace: Trace) -> Dict[Tuple[str, str], List[Tuple]]:
    """以 ``(rank, name)`` 分桶，桶内按 ``ts`` 排序后逐元素比对。"""
    out: Dict[Tuple[str, str], List[Tuple]] = defaultdict(list)
    for rank, pid, tid, ph, events in trace.iter_events():
        for ev in events:
            tup = _norm_event(rank, pid, tid, ph, ev)
            out[(str(rank), str(ev.name))].append(tup)
    for k in out:
        out[k].sort()
    return out


def _diff(orig: Dict[Tuple[str, str], List[Tuple]], dec: Dict[Tuple[str, str], List[Tuple]],
          max_diff: int = 5) -> List[str]:
    msgs: List[str] = []
    keys_orig = set(orig)
    keys_dec = set(dec)
    miss = keys_orig - keys_dec
    extra = keys_dec - keys_orig
    if miss:
        msgs.append(f"missing keys (rank,name) [{len(miss)}]: {list(sorted(miss))[:5]}")
    if extra:
        msgs.append(f"extra keys (rank,name) [{len(extra)}]: {list(sorted(extra))[:5]}")
    diffed = 0
    for key in sorted(keys_orig & keys_dec):
        a = orig[key]
        b = dec[key]
        if a == b:
            continue
        diffed += 1
        if len(msgs) < max_diff + 2:
            if len(a) != len(b):
                msgs.append(f"len mismatch at {key}: orig={len(a)} dec={len(b)}")
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        msgs.append(f"value diff at {key}[{i}]:\n   orig={x}\n    dec={y}")
                        break
        if diffed >= max_diff:
            break
    if diffed:
        msgs.insert(0, f"value-diff buckets: {diffed} (showing first {max_diff})")
    return msgs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="trace file or dir")
    p.add_argument(
        "--baselines",
        nargs="+",
        default=["raw_msgpack", "gzip_msgpack", "tracezip", "scalatrace", "padoc"],
    )
    args = p.parse_args()

    if os.path.isdir(args.trace):
        trace = Trace.from_dir(args.trace)
    else:
        trace = Trace.from_file(args.trace)

    print(f"loaded trace from {args.trace}")
    orig = _gather(trace)
    total_events = sum(len(v) for v in orig.values())
    print(f"  buckets={len(orig)} events={total_events}")

    factories = {
        "raw_msgpack": RawMsgpackCompressor,
        "gzip_msgpack": GzipMsgpackCompressor,
        "tracezip": TracezipCompressor,
        "scalatrace": ScalaTraceCompressor,
        "padoc": PADOCCompressor,
    }

    fail = False
    for name in args.baselines:
        cls = factories[name]
        c = cls()
        artifact = c.compress_trace(trace)
        decoded = c.decompress_to_trace(artifact.blob)
        dec = _gather(decoded)
        msgs = _diff(orig, dec)
        if msgs:
            fail = True
            print(f"\n=== {name}: LOSSY ===")
            for m in msgs:
                print("  " + m)
        else:
            print(f"\n=== {name}: lossless ✓ ===")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
