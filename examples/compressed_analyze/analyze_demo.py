"""
Command-line tool for compressing and analyzing Torch Profiler trace files.

This script loads a trace, compresses it using TemplateCompressor unless the
input is already compressed, analyzes the (compressed) trace using TraceAnalysis,
and prints the results.

Run with:

    python analyze_demo.py \
        --input_file <trace.json> \
        [--compressed_file <compressed_trace.json>]
"""

import argparse
import os
import logging
import time
import pandas as pd
from pandas.testing import assert_frame_equal
from hta.trace_analysis import TraceAnalysis as HTATraceAnalysis
from hta.configs.config import logger as hta_logger
from perflowai.padoc import Trace, CompressedTrace, TemplateCompressor, TraceAnalysis


def check_exact_match(df_compressed: pd.DataFrame, df_hta: pd.DataFrame, name: str) -> bool:
    """
    Check if two dataframes are exactly the same.

    Args:
        df_compressed: The compressed analysis result.
        df_hta: The HTA benchmark result.
        name: The name of the analysis.

    Returns:
        None
    """
    try:
        assert_frame_equal(df_compressed, df_hta)
        print(f"✅ {name} 结果完全一致 (结构、类型、数值零差异)。")
        return True
    except AssertionError as e:
        print(f"❌ {name} 结果不一致。")
        print("\n--- 详细差异报告 ---")
        print(e)
        print("------------------------")
        return False


def analyze_demo(input_file: str, compressed_file: str = None, visualize: bool = True):
    """
    Compress or load a pre-compressed PerFlow-AI trace, then analyze it.

    This function performs the following steps:
        1. Load a Trace or CompressedTrace object from file.
        2. If a compressed file is not provided, compress the raw trace.
        3. Analyze the resulting compressed trace using TraceAnalysis.

    Args:
        input_file (str):
            Path to the input JSON trace file (always the raw trace).
        compressed_file (str, optional):
            Path to a pre-compressed trace file. If provided, this file is 
            loaded directly as CompressedTrace, and 'input_file' is used only 
            for reference/comparison.
        visualize (bool, optional):
            Whether to visualize the results.

    Returns:
        None
            The function prints analysis results directly to stdout. No value
            is returned.

    Raises:
        FileNotFoundError:
            If the required input file does not exist.
        JSONDecodeError:
            If the input trace JSON file is malformed.
        Exception:
            Any unexpected errors raised during trace loading, compression,
            or analysis.

    Example:
        >>> analyze_demo(input_file="profiler.json") # Loads raw, then compresses
        >>> analyze_demo(input_file="profiler.json", compressed_file="compressed.json") 
        # Loads compressed.json directly
    """

    # ======================================================================
    # 全局耗时统计
    # ======================================================================
    start_total_my = time.time()

    # Determine which file to analyze
    if compressed_file and os.path.exists(compressed_file):
        file_to_load = compressed_file
        print(f"📦 Loading pre-compressed trace from {file_to_load} ...")
        compressed_trace = CompressedTrace.from_json(file_to_load)
    else:
        file_to_load = input_file
        print(f"⚙️ Loading raw trace from {file_to_load} and compressing ...")

        if not os.path.exists(file_to_load):
            raise FileNotFoundError(f"Raw input file not found: {file_to_load}")

        trace = Trace.from_json(file_to_load)

        # Compress the trace
        print("⚙️ Compressing trace ...")
        compressor = TemplateCompressor()
        compressed_trace = compressor.intra_compress(trace)

    # Analyze the compressed trace
    print("\n🔍 Analyzing trace ...")
    analyzer = TraceAnalysis(compressed_trace)

    # HTA analyzer
    hta_analyzer = HTATraceAnalysis(trace_files={"0": input_file})

    results = {}
    time_stats = {}   # 存储每个分析项的时间

    # ======================================================================
    # 1. Temporal breakdown
    # ======================================================================
    print("\n=== Temporal Breakdown ===")

    start_my = time.time()
    temporal_breakdown = analyzer.get_temporal_breakdown(visualize)
    end_my = time.time()

    start_hta = time.time()
    hta_temporal_breakdown = hta_analyzer.get_temporal_breakdown(False)
    end_hta = time.time()

    print("  Temporal breakdown:")
    print(temporal_breakdown)
    print("  HTA Temporal breakdown:")
    print(hta_temporal_breakdown)

    results["temporal_breakdown"] = check_exact_match(
        temporal_breakdown, hta_temporal_breakdown, "Temporal breakdown"
    )
    time_stats["temporal_breakdown"] = (end_my - start_my, end_hta - start_hta)

    # ======================================================================
    # 2. Communication-computation overlap
    # ======================================================================
    print("\n=== Communication-Computation Overlap ===")

    start_my = time.time()
    comm_comp_overlap = analyzer.get_comm_comp_overlap(visualize)
    end_my = time.time()

    start_hta = time.time()
    hta_comm_comp_overlap = hta_analyzer.get_comm_comp_overlap(False)
    end_hta = time.time()

    print("  Communication-computation overlap:")
    print(comm_comp_overlap)
    print("  HTA Communication-computation overlap:")
    print(hta_comm_comp_overlap)

    results["comm_comp_overlap"] = check_exact_match(
        comm_comp_overlap, hta_comm_comp_overlap, "Communication-computation overlap"
    )
    time_stats["comm_comp_overlap"] = (end_my - start_my, end_hta - start_hta)

    # ======================================================================
    # 3. GPU kernel breakdown
    # ======================================================================
    print("\n=== GPU Kernel Breakdown ===")

    start_my = time.time()
    gpu_kernel_breakdown = analyzer.get_gpu_kernel_breakdown(visualize)
    end_my = time.time()

    start_hta = time.time()
    hta_gpu_kernel_breakdown = hta_analyzer.get_gpu_kernel_breakdown(False)
    end_hta = time.time()

    print("  GPU kernel breakdown:")
    print(gpu_kernel_breakdown[0])
    print(gpu_kernel_breakdown[1])
    print("  HTA GPU kernel breakdown:")
    print(hta_gpu_kernel_breakdown[0])
    print(hta_gpu_kernel_breakdown[1])

    results["gpu_kernel_breakdown"] = check_exact_match(
        gpu_kernel_breakdown[0], hta_gpu_kernel_breakdown[0], "GPU kernel breakdown"
    )
    time_stats["gpu_kernel_breakdown"] = (end_my - start_my, end_hta - start_hta)

    # ======================================================================
    # 最终整体耗时
    # ======================================================================
    end_total_my = time.time()
    total_my = end_total_my - start_total_my

    # ======================================================================
    # 🌈 Final pretty summary output
    # ======================================================================
    print("\n" + "=" * 60)
    print("🧾 最终结果汇总")
    print("=" * 60)

    for key, passed in results.items():
        name_padded = key.ljust(20)
        if passed:
            print(f"{name_padded} {"\033[92m"}✅ 通过{"\033[0m"}")
        else:
            print(f"{name_padded} {"\033[91m"}❌ 不一致{"\033[0m"}")

    print("=" * 60)
    print("🕒 分析耗时对比（秒）")
    print("=" * 60)

    for key, (my_t, hta_t) in time_stats.items():
        print(f"{key.ljust(20)}  PerFlow-AI: {my_t:.3f}   HTA: {hta_t:.3f}")

    print("-" * 60)
    print(f"PerFlow-AI 总分析耗时: {total_my:.3f} 秒")
    print("=" * 60)
    print()


def main():
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        required=True,
                        type=str,
                        help="Path to the input raw trace file (always required).")

    parser.add_argument("--compressed_file",
                        type=str,
                        default=None,
                        help="Path to a pre-compressed trace file.")
    parser.add_argument("--visualize",
                        action="store_true",
                        help="Whether to visualize the results.")

    args = parser.parse_args()

    hta_logger.setLevel(logging.ERROR)

    analyze_demo(input_file=args.input_file,
                 compressed_file=args.compressed_file,
                 visualize=args.visualize)


if __name__ == "__main__":
    main()
