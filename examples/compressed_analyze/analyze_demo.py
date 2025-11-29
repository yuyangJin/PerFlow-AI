"""
Command-line tool for compressing and analyzing Torch Profiler trace files.

This script loads a trace, compresses it using TemplateCompressor,
analyzes the compressed trace using TraceAnalysis, and prints the results.

Run with:

    python analyze_demo.py \
        --input_file <trace.json>
"""

import argparse
from perflowai.padoc import Trace, TemplateCompressor, TraceAnalysis

def analyze_demo(input_file: str):
    """
    Compress a PerFlow-AI trace using TemplateCompressor, then analyze the compressed trace.

    This function performs the following steps:
        1. Load a Trace object from the JSON trace file.
        2. Compress the trace using TemplateCompressor.
        3. Analyze the compressed trace using TraceAnalysis.

    Args:
        input_file (str):
            Path to the input JSON trace file.

    Returns:
        None
            The function prints analysis results directly to stdout. No value is returned.

    Raises:
        FileNotFoundError:
            If `input_file` does not exist.
        JSONDecodeError:
            If the input trace JSON file is malformed.
        Exception:
            Any unexpected errors raised during trace loading, compression,
            or analysis.

    Example:
        >>> analyze_demo(input_file="profiler.json")
    """

    print(f"📥 Loading trace from {input_file}")
    trace = Trace.from_json(input_file)

    # compress the trace
    print("⚙️ Compressing trace ...")
    compressor = TemplateCompressor()
    compressed_trace = compressor.intra_compress(trace)

    # analyze the compressed trace
    analyzer = TraceAnalysis(compressed_trace)

    print("\n🔍 Analyzing trace ...")
    # get the temporal_breakdown
    temporal_breakdown = analyzer.get_temporal_breakdown()
    print(f"  Temporal breakdown: {temporal_breakdown}")


def main():
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="tests/example_trace/profiler_585.json",
                        type=str, help="input trace file path")

    args = parser.parse_args()

    analyze_demo(args.input_file)


if __name__ == "__main__":
    main()
