"""
Command-line tool for compressing Torch Profiler trace files.

This script loads a trace, compresses it using TemplateCompressor,
writes both original and compressed outputs, reconstructs the trace,
and verifies correctness by comparing the reconstructed file against
the original.

Run with:

    python compress_demo.py --input_file <trace.json> \
        --origin_file <origin.bin> \
        --output_file <compressed.bin> \
        --reconstruct_file <reconstructed.bin>
"""

import os
import argparse
import filecmp
from typing import List
from pympler import asizeof
from perflowai.padoc import Trace, TemplateCompressor, CompressedTrace


def compress_single_rank_demo(input_file: str, origin_file: str, output_file: str, restore_file: str):
    """
    Compress a PerFlow-AI trace using TemplateCompressor, evaluate compression
    performance (file size + memory size), and verify correctness by reconstructing
    the trace and comparing it with the original.

    This function performs the following steps:
        1. Load a Trace object from the JSON trace file.
        2. Measure memory size of the original Trace object.
        3. Write the original trace to a JSON file.
        4. Compress the trace using TemplateCompressor.
        5. Measure memory and file size after compression.
        6. Save the compressed trace.
        7. Re-load the compressed trace, reconstruct the original trace, and save it.
        8. Compare the reconstructed trace file with the original file using a byte-level diff.

    Args:
        input_file (str):
            Path to the input JSON trace file.
        origin_file (str):
            Path to write the serialized original trace.
        output_file (str):
            Path to write the compressed trace file.
        restore_file (str):
            Path to write the reconstructed (decompressed) trace.

    Returns:
        None
            The function prints compression statistics and correctness test results
            directly to stdout. No value is returned.

    Raises:
        FileNotFoundError:
            If `input_file` does not exist.
        JSONDecodeError:
            If the input trace JSON file is malformed.
        Exception:
            Any unexpected errors raised during trace loading, compression,
            serialization, or comparison.

    Example:
        >>> compress_demo(
        ...     input_file="profiler.json",
        ...     origin_file="origin.bin",
        ...     output_file="compressed.bin",
        ...     restore_file="reconstructed.bin"
        ... )
    """

    print(f"📥 Loading trace from {input_file}")
    trace = Trace.from_file(input_file)

    # get the original trace memory size
    trace_size_mem = asizeof.asizeof(trace)
    print(f"🧠 Original Trace memory size: {trace_size_mem / 1024 / 1024:.2f} MB")

    # write the original trace to a file
    print(f"💾 Writing original trace to {origin_file}")
    trace.write_file(origin_file)

    # get the original file size
    origin_file_size = os.path.getsize(origin_file)
    print(f"📦 Original file size: {origin_file_size / 1024 / 1024:.2f} MB")

    # compress the trace
    print("⚙️ Compressing trace ...")
    compressor = TemplateCompressor()
    compressed_trace = compressor.intra_compress(trace)
    compressed_trace.segmented_linear_predictor_compress()

    # get the compressed trace memory size
    compressed_size_mem = asizeof.asizeof(compressed_trace)
    print(f"🧠 Compressed Trace memory size: {compressed_size_mem / 1024 / 1024:.2f} MB")

    # write the compressed trace to a file
    print(f"💾 Writing compressed trace to {output_file}")
    compressed_trace.write_file(output_file)

    # get the compressed file size
    compressed_file_size = os.path.getsize(output_file)
    print(f"📦 Compressed file size: {compressed_file_size / 1024 / 1024:.2f} MB")

    # calculate the compression ratio
    file_compression_ratio = compressed_file_size / origin_file_size if origin_file_size else 0
    mem_compression_ratio = compressed_size_mem / trace_size_mem if trace_size_mem else 0

    print("\n📊 Compression Summary:")
    print(
        f"  💾 File compression ratio: {file_compression_ratio:.2%} "
        f"(↓ {1 - file_compression_ratio:.2%})"
    )
    print(
        f"  🧠 Memory compression ratio: {mem_compression_ratio:.2%} "
        f"(↓ {1 - mem_compression_ratio:.2%})"
    )
    # test correctness
    print("\n🔍 Testing correctness...")

    # 1) load the compressed trace
    print(f"📥 Loading compressed trace from {output_file}")
    compressed_trace = CompressedTrace.from_file(output_file)
    print("✅ Loaded successfully.")

    # 2) decompress the compressed trace
    trace = compressor.intra_decompress(compressed_trace)
    print("✅ Decompressed successfully.")

    # 3) write the reconstructed trace to a file
    print(f"💾 Writing reconstructed trace to {restore_file}")
    trace.write_file(restore_file, origin=True)

    # 4) compare the reconstructed file with the original file
    print("🔎 Comparing reconstructed file with original...")

    if filecmp.cmp(origin_file, restore_file, shallow=False):
        print("✅ Correctness test PASSED: reconstructed file == original file")
    else:
        print("❌ Correctness test FAILED: reconstructed file != original file")
        print("   You should inspect differences, e.g.:")
        print(f"   diff -u {origin_file} {restore_file}")

    print("Done.")

def compress_multi_rank_demo(input_dir: str, origin_dir: str, output_file: str, restore_dir: str):
    """
    Compress multiple PerFlow-AI trace files using TemplateCompressor
    """

    print(f"📥 Loading trace from {input_dir}")
    trace = Trace.from_dir(input_dir)
    if output_file.endswith(".json"):
        file_type = "json"
    else:
        file_type = "bin"

    trace_size_mem = asizeof.asizeof(trace)
    print(f"🧠 Original Trace memory size: {trace_size_mem / 1024 / 1024:.2f} MB")

    # write the original trace to a directory
    print(f"💾 Writing original trace to {origin_dir}")
    trace.write_dir(origin_dir, file_type)

    # get the original directory size
    origin_dir_size = sum(os.path.getsize(os.path.join(origin_dir, f)) \
                          for f in os.listdir(origin_dir))
    print(f"📦 Original directory size: {origin_dir_size / 1024 / 1024:.2f} MB")

    # compress the trace
    print("⚙️ Compressing trace ...")
    compressor = TemplateCompressor()
    compressed_trace = compressor.inter_compress(trace)
    compressed_trace.segmented_linear_predictor_compress()

    # get the compressed trace memory size
    compressed_size_mem = asizeof.asizeof(compressed_trace)
    print(f"🧠 Compressed Trace memory size: {compressed_size_mem / 1024 / 1024:.2f} MB")

    # write the compressed trace to a file
    print(f"💾 Writing compressed trace to {output_file}")
    compressed_trace.write_file(output_file)

    # get the compressed file size
    compressed_file_size = os.path.getsize(output_file)
    print(f"📦 Compressed file size: {compressed_file_size / 1024 / 1024:.2f} MB")

    # calculate the compression ratio
    file_compression_ratio = compressed_file_size / origin_dir_size if origin_dir_size else 0
    mem_compression_ratio = compressed_size_mem / trace_size_mem if trace_size_mem else 0

    print("\n📊 Compression Summary:")
    print(
        f"  💾 File compression ratio: {file_compression_ratio:.2%} "
        f"(↓ {1 - file_compression_ratio:.2%})"
    )
    print(
        f"  🧠 Memory compression ratio: {mem_compression_ratio:.2%} "
        f"(↓ {1 - mem_compression_ratio:.2%})"
    )

    # test correctness
    print("\n🔍 Testing correctness...")

    # 1) load the compressed trace
    print(f"📥 Loading compressed trace from {output_file}")
    compressed_trace = CompressedTrace.from_file(output_file)
    print(f"✅ Loaded successfully, have {len(compressed_trace.get_ranks())} ranks.")

    # 2) decompress the compressed trace
    trace = compressor.inter_decompress(compressed_trace)
    print("✅ Decompressed successfully.")

    # 3) write the reconstructed trace to a directory
    print(f"💾 Writing reconstructed trace to {restore_dir}")
    trace.write_dir(restore_dir, file_type)

    # 4) compare the reconstructed file with the original file
    print("🔎 Comparing reconstructed traces with original...")

    # Get file lists from both directories
    origin_files: List[str] = sorted(os.listdir(origin_dir))
    restore_files: List[str] = sorted(os.listdir(restore_dir))

    # Check if file lists are identical (names and number)
    if origin_files != restore_files:
        print("❌ Correctness test FAILED: File lists do not match.")
        print(f"   Original files count: {len(origin_files)}")
        print(f"   Reconstructed files count: {len(restore_files)}")
        print("   Differences in file names/counts detected.")
        return # Exit the function or skip further comparison

    # Compare each corresponding file
    all_passed = True
    for filename in origin_files:
        origin_file = os.path.join(origin_dir, filename)
        restore_file = os.path.join(restore_dir, filename)

        # Check if they are files before attempting comparison
        if os.path.isfile(origin_file) and os.path.isfile(restore_file):
            if filecmp.cmp(origin_file, restore_file, shallow=False):
                print(f"  ✅ {filename} PASSED") # Optional: print success for each file
            else:
                print(f"  ❌ Correctness test FAILED: {filename} != original file")
                print("     You should inspect differences, e.g.:")
                print(f"     diff -u {origin_file} {restore_file}")
                all_passed = False
        else:
            print(f"  ⚠️ Skipping comparison for non-file item: {filename}")

    if all_passed:
        print("✅ Correctness test PASSED: All reconstructed files match original files.")
    else:
        print("❌ Correctness test FAILED: One or more files failed comparison.")

    print("Done.")



def main():
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="tests/example_trace/profiler_585.json",
                        type=str, help="input trace file path")
    parser.add_argument("--origin_file", default="origin.bin",
                        type=str, help="path to write original trace file")
    parser.add_argument("--output_file", default="compressed.bin",
                        type=str, help="output compressed trace file path")
    parser.add_argument("--reconstruct_file", default="reconstructed.bin",
                        type=str, help="path to write restored trace file")
    parser.add_argument("--multi_rank_input_dir",
                        type=str, help="input directory path for multi-rank")
    parser.add_argument("--multi_rank_origin_dir", default="origin_dir",
                        type=str, help="origin directory path for multi-rank to write original traces")
    parser.add_argument("--multi_rank_output_file", default="compressed_multi_rank.bin",
                        type=str, help="output compressed trace file path for multi-rank")
    parser.add_argument("--multi_rank_reconstruct_dir", default="reconstructed_dir",
                        type=str, help="path to write restored trace dir for multi-rank")

    args = parser.parse_args()

    print("=" * 50)
    print("Compressing a single-rank trace demo")
    print("=" * 50)
    compress_single_rank_demo(args.input_file, args.origin_file,
                              args.output_file, args.reconstruct_file)

    if args.multi_rank_input_dir and args.multi_rank_origin_dir \
        and args.multi_rank_output_file and args.multi_rank_reconstruct_dir:
        print("=" * 50)
        print("Compressing a multi-rank trace demo")
        print("=" * 50)
        compress_multi_rank_demo(args.multi_rank_input_dir, args.multi_rank_origin_dir,
                                 args.multi_rank_output_file, args.multi_rank_reconstruct_dir)
    else:
        print("No multi-rank input directory provided, skipping multi-rank compression.")
        print("To compress multi-rank traces, provide --multi_rank_input_dir, \
              --multi_rank_origin_dir, --multi_rank_output_file, and --multi_rank_reconstruct_dir.")


if __name__ == "__main__":
    main()
