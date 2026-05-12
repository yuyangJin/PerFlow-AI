#!/usr/bin/env bash
#
# G-series:
#   重跑 analyze 部分（**复用 F1-F5 已经写好的 artifact cache**），但用的是
#   修复了 in-situ setup-share bug 之后的代码——每个 (dataset, padoc) 对只
#   load CompressedTrace 一次，跨 4 个 in-situ task 共享。在 subset8 上测出
#   in-situ analysis 加速 100-300x，所以 subset64 / full 这边的 PADOC 总 wall
#   会从 ~7 min 跌到 ~30-90 s 级别。
#
# 顺序：
#   G1  subset8       (raw + gzip + tracezip + scalatrace + padoc) × 7 task
#   G2  subset32      同上
#   G3  subset64      同上
#   G4  qwen3_full    只跑 padoc × 7 task（baseline 在 full trace 上 7 GiB+，
#                     太重，子集已经足够把 baseline vs PADOC 比清楚）
#   G5  llama_full    同上
#
# 全部用 scripts/exp/run_exp.sh 挂到 tmux:padoc，**电脑可以关机**。
#
# 用法：
#   bash scripts/exp/run_g_series.sh

set -euo pipefail
cd /home/lvjiaxin/work/AI/PerFlow-AI

ART_DIR=/mnt/treasure/ljx/padoc_artifacts/v2

ACTIVATE='source /mnt/treasure/ljx/miniconda3/etc/profile.d/conda.sh && conda activate PerFlow-AI'

# ----------------------------------------------------------------------
# Sanity check: F1-F5 artifacts must already exist (they do; verified ls)
# ----------------------------------------------------------------------
for f in \
    "$ART_DIR/padoc/qwen3_subset8.bin" \
    "$ART_DIR/padoc/qwen3_subset32.bin" \
    "$ART_DIR/padoc/qwen3_subset64.bin" \
    "$ART_DIR/padoc/qwen3_full.bin" \
    "$ART_DIR/padoc/llama_subset8.bin" \
    "$ART_DIR/padoc/llama_subset32.bin" \
    "$ART_DIR/padoc/llama_subset64.bin" \
    "$ART_DIR/padoc/llama_full.bin" \
    "$ART_DIR/raw_msgpack/qwen3_subset8.bin" \
    "$ART_DIR/scalatrace/qwen3_subset8.bin" \
    "$ART_DIR/tracezip/qwen3_subset8.bin"; do
  if [ ! -s "$f" ]; then
    echo "missing artifact: $f"
    exit 1
  fi
done
echo "[g-series] all required artifacts present, queueing tmux jobs ..."

NWIN=5

# ----------------------------------------------------------------------
# G1 - subset8 full 5x7
# ----------------------------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 0 "$NWIN")" scripts/exp/run_exp.sh G1_analyze_subset8_fixed -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/subset8.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/G1_analyze_subset8_fixed.md \
  --out-json report/exp_results/G1_analyze_subset8_fixed.json \
  --out-csv  report/exp_results/G1_analyze_subset8_fixed.csv
"

# ----------------------------------------------------------------------
# G2 - subset32 full 5x7
# ----------------------------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 1 "$NWIN")" scripts/exp/run_exp.sh G2_analyze_subset32_fixed -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/subset32.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/G2_analyze_subset32_fixed.md \
  --out-json report/exp_results/G2_analyze_subset32_fixed.json \
  --out-csv  report/exp_results/G2_analyze_subset32_fixed.csv
"

# ----------------------------------------------------------------------
# G3 - subset64 full 5x7
# ----------------------------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 2 "$NWIN")" scripts/exp/run_exp.sh G3_analyze_subset64_fixed -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/subset64.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/G3_analyze_subset64_fixed.md \
  --out-json report/exp_results/G3_analyze_subset64_fixed.json \
  --out-csv  report/exp_results/G3_analyze_subset64_fixed.csv
"

# ----------------------------------------------------------------------
# G4 - qwen3_full padoc-only 7 task
# ----------------------------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 3 "$NWIN")" scripts/exp/run_exp.sh G4_analyze_qwen3_full_fixed -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/full_qwen3.json \
  --compressors padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/G4_analyze_qwen3_full_fixed.md \
  --out-json report/exp_results/G4_analyze_qwen3_full_fixed.json \
  --out-csv  report/exp_results/G4_analyze_qwen3_full_fixed.csv
"

# ----------------------------------------------------------------------
# G5 - llama_full padoc-only 7 task
# ----------------------------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 4 "$NWIN")" scripts/exp/run_exp.sh G5_analyze_llama_full_fixed -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/full_llama.json \
  --compressors padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/G5_analyze_llama_full_fixed.md \
  --out-json report/exp_results/G5_analyze_llama_full_fixed.json \
  --out-csv  report/exp_results/G5_analyze_llama_full_fixed.csv
"

echo
echo "[g-series] G1..G5 queued under tmux:padoc"
echo "monitor with:"
echo "  python scripts/exp/status.py --filter G"
