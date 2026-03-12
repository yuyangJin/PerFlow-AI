"""CLI demo for PADOC compressed-trace analysis."""

from __future__ import annotations

import argparse
import logging
import os
import time

import pandas as pd
from pandas.testing import assert_frame_equal

from hta.configs.config import logger as hta_logger
from hta.trace_analysis import TraceAnalysis as HTATraceAnalysis

from perflowai.padoc import CompressedTrace, TemplateCompressor, Trace, TraceAnalysis


def check_exact_match(
    df_compressed: pd.DataFrame,
    df_hta: pd.DataFrame,
    name: str,
) -> bool:
    """Check whether PADOC and HTA produce identical frames."""
    try:
        assert_frame_equal(df_compressed, df_hta)
        print(f"[PASS] {name}")
        return True
    except AssertionError as error:
        print(f"[FAIL] {name}")
        print(error)
        return False


def load_compressed_trace(input_file: str, compressed_file: str | None) -> CompressedTrace:
    """Load an existing compressed trace or build one from raw input."""
    if compressed_file and os.path.exists(compressed_file):
        print(f"Loading compressed trace: {compressed_file}")
        return CompressedTrace.from_file(compressed_file)

    print(f"Loading raw trace: {input_file}")
    trace = Trace.from_file(input_file)
    print("Compressing raw trace")
    return TemplateCompressor().intra_compress(trace)


def analyze_demo(input_file: str, compressed_file: str | None = None, visualize: bool = True) -> None:
    """Run PADOC analysis and compare results with HTA."""
    start_total = time.time()
    compressed_trace = load_compressed_trace(input_file, compressed_file)
    analyzer = TraceAnalysis(compressed_trace)
    hta_analyzer = HTATraceAnalysis(trace_files={"0": input_file})

    analysis_jobs = [
        (
            "temporal_breakdown",
            analyzer.get_temporal_breakdown,
            hta_analyzer.get_temporal_breakdown,
        ),
        (
            "comm_comp_overlap",
            analyzer.get_comm_comp_overlap,
            hta_analyzer.get_comm_comp_overlap,
        ),
        (
            "gpu_kernel_breakdown",
            analyzer.get_gpu_kernel_breakdown,
            hta_analyzer.get_gpu_kernel_breakdown,
        ),
    ]

    results = {}
    timings = {}
    for name, padoc_fn, hta_fn in analysis_jobs:
        print(f"Running {name}")
        padoc_start = time.time()
        padoc_result = padoc_fn(visualize)
        padoc_end = time.time()

        hta_start = time.time()
        hta_result = hta_fn(False)
        hta_end = time.time()

        if isinstance(padoc_result, tuple):
            results[name] = check_exact_match(padoc_result[0], hta_result[0], name)
        else:
            results[name] = check_exact_match(padoc_result, hta_result, name)
        timings[name] = (padoc_end - padoc_start, hta_end - hta_start)

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name:24s} {status}")

    print("-" * 60)
    for name, (padoc_time, hta_time) in timings.items():
        print(f"{name:24s} PADOC={padoc_time:.3f}s HTA={hta_time:.3f}s")
    print("-" * 60)
    print(f"Total PADOC runtime: {time.time() - start_total:.3f}s")


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--compressed_file", default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    hta_logger.setLevel(logging.ERROR)
    analyze_demo(
        input_file=args.input_file,
        compressed_file=args.compressed_file,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
