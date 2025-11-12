from perflowai.padoc import Trace, Compressor, TemplateCompressor, CompressedTrace
import argparse
import os
import sys
from pympler import asizeof


def compress_demo(input_file: str, origin_file: str, output_file: str):
    print(f"📥 Loading trace from {input_file}")
    trace = Trace.from_json(input_file)

    # 计算原始对象内存占用
    trace_size_mem = asizeof.asizeof(trace)
    print(f"🧠 Original Trace memory size: {trace_size_mem / 1024 / 1024:.2f} MB")

    # 写出原始 trace 到二进制文件
    print(f"💾 Writing original trace to {origin_file}")
    trace.write_json_file(origin_file)

    # 获取原始文件大小
    origin_file_size = os.path.getsize(origin_file)
    print(f"📦 Original file size: {origin_file_size / 1024 / 1024:.2f} MB")

    # 压缩
    print("⚙️ Compressing trace ...")
    compressor = TemplateCompressor()
    compressed_trace = compressor.intra_compress(trace)

    # 计算压缩后内存占用
    compressed_size_mem = asizeof.asizeof(compressed_trace)
    print(f"🧠 Compressed Trace memory size: {compressed_size_mem / 1024 / 1024:.2f} MB")

    # 写出压缩后的文件
    print(f"💾 Writing compressed trace to {output_file}")
    compressed_trace.write_json_file(output_file)

    # 获取压缩后文件大小
    compressed_file_size = os.path.getsize(output_file)
    print(f"📦 Compressed file size: {compressed_file_size / 1024 / 1024:.2f} MB")

    # 计算压缩率
    file_compression_ratio = compressed_file_size / origin_file_size if origin_file_size else 0
    mem_compression_ratio = compressed_size_mem / trace_size_mem if trace_size_mem else 0

    print(f"\n📊 Compression Summary:")
    print(f"  💾 File compression ratio: {file_compression_ratio:.2%} (↓ {1 - file_compression_ratio:.2%})")
    print(f"  🧠 Memory compression ratio: {mem_compression_ratio:.2%} (↓ {1 - mem_compression_ratio:.2%})")

    # 测试读取压缩 trace
    print("✅ Testing reading compressed trace...")
    compressed_trace = CompressedTrace.from_json(output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="tests/example_trace/profiler_585.json", type=str, help="input trace file path")
    parser.add_argument("--origin_file", default="origin.bin", type=str, help="path to write original trace file")
    parser.add_argument("--output_file", default="compressed.bin", type=str, help="output compressed trace file path")
    args = parser.parse_args()

    compress_demo(args.input_file, args.origin_file, args.output_file)


if __name__ == "__main__":
    main()
