#!/usr/bin/env bash
#
# 一夜流水线 v2：
#   1) 把所有 trace 用所有 compressor 压一次，blob 落盘到 /mnt/treasure/ljx/padoc_artifacts/
#   2) 用 --artifact-dir 跑 analyze（不再重新压缩）；
#      PADOC 现在对 5/7 task 已经是 in-situ（operator_hotspot, stream_load_balance,
#      layer_operator_balance, parallel_group, plus we still keep HTA-backed
#      gpu_kernel_breakdown / comm_comp_overlap / temporal_breakdown going
#      through one shared decompress, not per-task).
#
# 全部用 scripts/exp/run_exp.sh 包裹到 tmux:padoc 里，**不会因为本地电脑关机断**。
# 每个 step 在前一个完成后才启动，串成 chain，所以这是一条「**一个长任务**」。
#
# 用法：
#   bash scripts/exp/run_overnight_v2.sh
#

set -euo pipefail
cd /home/lvjiaxin/work/AI/PerFlow-AI

ART_DIR=/mnt/treasure/ljx/padoc_artifacts/v2
mkdir -p "$ART_DIR"

ACTIVATE='source /mnt/treasure/ljx/miniconda3/etc/profile.d/conda.sh && conda activate PerFlow-AI'
NWIN=10

# ----------------------------------------------------------------------
# Step 1 - 压缩并落盘
# ----------------------------------------------------------------------

# 子集（5 个 baseline 都压，便于跑 5 baseline × 7 task 的对比）
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 0 "$NWIN")" scripts/exp/run_exp.sh F1_compress_subset8_cache -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/subset8.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --padoc-workers 16 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F1_compress_subset8_cache.md \
  --out-json report/exp_results/F1_compress_subset8_cache.json \
  --out-csv  report/exp_results/F1_compress_subset8_cache.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 1 "$NWIN")" scripts/exp/run_exp.sh F2_compress_subset32_cache -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/subset32.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --padoc-workers 16 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F2_compress_subset32_cache.md \
  --out-json report/exp_results/F2_compress_subset32_cache.json \
  --out-csv  report/exp_results/F2_compress_subset32_cache.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 2 "$NWIN")" scripts/exp/run_exp.sh F3_compress_subset64_cache -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/subset64.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --padoc-workers 16 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F3_compress_subset64_cache.md \
  --out-json report/exp_results/F3_compress_subset64_cache.json \
  --out-csv  report/exp_results/F3_compress_subset64_cache.csv
"

# 全量 trace：只压 padoc（其它 baseline 在千卡级 trace 上有过 OOM 风险，
# 而且论文里其它 baseline 不需要在 full-trace 上做 analyze 对比）
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 3 "$NWIN")" scripts/exp/run_exp.sh F4_compress_qwen3_full_cache -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/full_qwen3.json \
  --compressors padoc \
  --padoc-workers 32 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F4_compress_qwen3_full_cache.md \
  --out-json report/exp_results/F4_compress_qwen3_full_cache.json \
  --out-csv  report/exp_results/F4_compress_qwen3_full_cache.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 4 "$NWIN")" scripts/exp/run_exp.sh F5_compress_llama_full_cache -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/full_llama.json \
  --compressors padoc \
  --padoc-workers 32 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F5_compress_llama_full_cache.md \
  --out-json report/exp_results/F5_compress_llama_full_cache.json \
  --out-csv  report/exp_results/F5_compress_llama_full_cache.csv
"

# ----------------------------------------------------------------------
# Step 2 - 用 cache 跑 analyze（这里全部带 --artifact-dir）
# ----------------------------------------------------------------------

# 因为 step 1 / step 2 都进 tmux:padoc 同一会话，但 run_exp.sh 各自新开
# window，互相不阻塞。下面给 step 2 加一个简单的「等 step 1 全完」barrier，
# 用 inotify/sleep 都能做；这里用 sleep+ls 的轻量方法。
#
# 真要严格串行的话，可以把 step 2 都放在一个 bash -c 里，先 wait artifact
# 文件就绪再跑。下面就这么做，更稳。

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 5 "$NWIN")" scripts/exp/run_exp.sh F6_analyze_subset8_full_cache -- bash -c "
$ACTIVATE
# 等 step 1 的对应 artifact 完整出现
need=( '$ART_DIR/padoc/qwen3_subset8.bin' '$ART_DIR/padoc/llama_subset8.bin' '$ART_DIR/scalatrace/qwen3_subset8.bin' '$ART_DIR/tracezip/qwen3_subset8.bin' '$ART_DIR/raw_msgpack/qwen3_subset8.bin' '$ART_DIR/gzip_msgpack/qwen3_subset8.bin' )
for f in \"\${need[@]}\"; do
  while [ ! -s \"\$f\" ]; do
    echo \"waiting for \$f ...\"; sleep 30
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/subset8.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F6_analyze_subset8_full_cache.md \
  --out-json report/exp_results/F6_analyze_subset8_full_cache.json \
  --out-csv  report/exp_results/F6_analyze_subset8_full_cache.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 6 "$NWIN")" scripts/exp/run_exp.sh F7_analyze_subset32_full_cache -- bash -c "
$ACTIVATE
need=( '$ART_DIR/padoc/qwen3_subset32.bin' '$ART_DIR/padoc/llama_subset32.bin' '$ART_DIR/scalatrace/qwen3_subset32.bin' '$ART_DIR/tracezip/qwen3_subset32.bin' '$ART_DIR/raw_msgpack/qwen3_subset32.bin' '$ART_DIR/gzip_msgpack/qwen3_subset32.bin' )
for f in \"\${need[@]}\"; do
  while [ ! -s \"\$f\" ]; do
    echo \"waiting for \$f ...\"; sleep 30
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/subset32.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F7_analyze_subset32_full_cache.md \
  --out-json report/exp_results/F7_analyze_subset32_full_cache.json \
  --out-csv  report/exp_results/F7_analyze_subset32_full_cache.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 7 "$NWIN")" scripts/exp/run_exp.sh F8_analyze_subset64_full_cache -- bash -c "
$ACTIVATE
need=( '$ART_DIR/padoc/qwen3_subset64.bin' '$ART_DIR/padoc/llama_subset64.bin' '$ART_DIR/scalatrace/qwen3_subset64.bin' '$ART_DIR/tracezip/qwen3_subset64.bin' '$ART_DIR/raw_msgpack/qwen3_subset64.bin' '$ART_DIR/gzip_msgpack/qwen3_subset64.bin' )
for f in \"\${need[@]}\"; do
  while [ ! -s \"\$f\" ]; do
    echo \"waiting for \$f ...\"; sleep 30
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/subset64.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F8_analyze_subset64_full_cache.md \
  --out-json report/exp_results/F8_analyze_subset64_full_cache.json \
  --out-csv  report/exp_results/F8_analyze_subset64_full_cache.csv
"

# 全量 trace 只跑 padoc × 7 task；用 cache + 单次解压共享
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 8 "$NWIN")" scripts/exp/run_exp.sh F9_analyze_qwen3_full_cache -- bash -c "
$ACTIVATE
need=( '$ART_DIR/padoc/qwen3_full.bin' )
for f in \"\${need[@]}\"; do
  while [ ! -s \"\$f\" ]; do
    echo \"waiting for \$f ...\"; sleep 60
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/full_qwen3.json \
  --compressors padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F9_analyze_qwen3_full_cache.md \
  --out-json report/exp_results/F9_analyze_qwen3_full_cache.json \
  --out-csv  report/exp_results/F9_analyze_qwen3_full_cache.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 9 "$NWIN")" scripts/exp/run_exp.sh F10_analyze_llama_full_cache -- bash -c "
$ACTIVATE
need=( '$ART_DIR/padoc/llama_full.bin' )
for f in \"\${need[@]}\"; do
  while [ ! -s \"\$f\" ]; do
    echo \"waiting for \$f ...\"; sleep 60
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/full_llama.json \
  --compressors padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/F10_analyze_llama_full_cache.md \
  --out-json report/exp_results/F10_analyze_llama_full_cache.json \
  --out-csv  report/exp_results/F10_analyze_llama_full_cache.csv
"

echo
echo "all jobs queued under tmux:padoc -- safe to disconnect / shut down local laptop."
echo "monitor with:"
echo "  python scripts/exp/status.py --filter F"
echo
echo "artifacts will accumulate under $ART_DIR/<compressor>/<trace>.bin"
