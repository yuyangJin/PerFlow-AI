from perflowai.padoc import Trace, Compressor, TemplateCompressor
import argparse

def compress_demo(input_file: str, origin_file: str,output_file: str):
    trace = Trace.from_json(input_file)
    # write original trace to file
    trace.write_json_file(origin_file)
    # TODO: compress trace and write to file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="tests/example_trace/out-1024.json", type=str, help="input trace file path")
    parser.add_argument("--origin_file", default="origin.json", type=str, help="path to write original trace file")
    parser.add_argument("--output_file", default="compressed.json", type=str, help="output compressed trace file path")
    args = parser.parse_args()
    compress_demo(args.input_file, args.origin_file, args.output_file)

if __name__ == "__main__":
    main()