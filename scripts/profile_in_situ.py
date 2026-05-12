#!/usr/bin/env python3
"""Profile PADOC in-situ analysis tasks on a cached artifact.

Usage:
    python scripts/profile_in_situ.py \
        --artifact /mnt/treasure/ljx/padoc_artifacts/v2/padoc/qwen3_subset64.bin \
        --task operator_hotspot

Prints cProfile top-30 hot spots + wall time per task.  Useful for finding
the SLP-expansion / Python-loop hot path before optimising.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from typing import Any

from perflowai.padoc.baselines.padoc_adapter import PADOCCompressor
from perflowai.padoc.bench.tasks import get_task, builtin_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--task", default="operator_hotspot", choices=builtin_tasks())
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--sort", default="cumulative")
    args = parser.parse_args()

    print(f"[profile] task={args.task} artifact={args.artifact}")

    with open(args.artifact, "rb") as f:
        blob = f.read()

    compressor = PADOCCompressor()
    task = get_task(args.task)

    if not task.has_in_situ_for(compressor):
        print(f"[profile] task {args.task!r} has no in-situ implementation -- exiting")
        return

    # warm-up so JIT / first-time module loads aren't counted
    _ = task.run_in_situ(compressor, blob)

    # ----------------- Run 1: cold (no shared compressed_trace) ---------
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    summary: Any = task.run_in_situ(compressor, blob)
    profiler.disable()
    cold_wall = time.perf_counter() - t0
    print(f"[profile] COLD (re-deserialize blob each call) wall={cold_wall:.3f}s")

    out = io.StringIO()
    pstats.Stats(profiler, stream=out).sort_stats(args.sort).print_stats(args.top)
    print(out.getvalue())

    # ----------------- Run 2: hot (shared CompressedTrace) --------------
    t_setup = time.perf_counter()
    ct = compressor.load_compressed_trace(blob)
    setup = time.perf_counter() - t_setup

    profiler2 = cProfile.Profile()
    t0 = time.perf_counter()
    profiler2.enable()
    _ = task.run_in_situ(compressor, blob, compressed_trace=ct)
    profiler2.disable()
    hot_wall = time.perf_counter() - t0
    print(
        f"[profile] HOT  (shared CompressedTrace) wall={hot_wall:.3f}s   "
        f"(one-time setup paid separately: {setup:.3f}s)"
    )
    print(f"[profile] speedup analysis-only = {cold_wall / hot_wall:.2f}x")

    out2 = io.StringIO()
    pstats.Stats(profiler2, stream=out2).sort_stats(args.sort).print_stats(args.top)
    print(out2.getvalue())

    # echo summary so we know the analysis itself succeeded
    if isinstance(summary, dict):
        keys = sorted(summary.keys())[:6]
        print(f"[profile] summary keys (head 6): {keys}")


if __name__ == "__main__":
    main()
