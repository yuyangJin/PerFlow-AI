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
import pandas as pd
from pandas.testing import assert_frame_equal
from hta.trace_analysis import TraceAnalysis as HTATraceAnalysis
from hta.configs.config import logger as hta_logger
from perflowai.padoc import Trace, CompressedTrace, TemplateCompressor, TraceAnalysis
from perflowai.padoc import logger

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
    except AssertionError as e:
        print(f"❌ {name} 结果不一致。")
        print("\n--- 详细差异报告 ---")
        print(e)
        print("------------------------")


def analyze_demo(input_file: str, compressed_file: str = None):
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
    hta_analyzer = HTATraceAnalysis(trace_files={"0": input_file})

    # get the temporal_breakdown
    temporal_breakdown = analyzer.get_temporal_breakdown()
    hta_temporal_breakdown = hta_analyzer.get_temporal_breakdown(False)
    print("  Temporal breakdown:")
    print(temporal_breakdown)
    print("  HTA Temporal breakdown:")
    print(hta_temporal_breakdown)
    check_exact_match(temporal_breakdown, hta_temporal_breakdown, "Temporal breakdown")

    # get the comm_comp_overlap
    comm_comp_overlap = analyzer.get_comm_comp_overlap()
    hta_comm_comp_overlap = hta_analyzer.get_comm_comp_overlap(False)

    print("  Communication-computation overlap:")
    print(comm_comp_overlap)
    print("  HTA Communication-computation overlap:")
    print(hta_comm_comp_overlap)
    check_exact_match(comm_comp_overlap, hta_comm_comp_overlap, "Communication-computation overlap")


def main():
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        default="tests/example_trace/profiler_585.json",
                        type=str,
                        help="Path to the input raw trace file (always required).")

    parser.add_argument("--compressed_file",
                        type=str,
                        default=None,
                        help="Path to a pre-compressed trace file.")

    args = parser.parse_args()

    hta_logger.setLevel(logging.ERROR)

    # Pass the compressed_file argument directly
    analyze_demo(input_file=args.input_file,
                 compressed_file=args.compressed_file)


if __name__ == "__main__":
    main()
