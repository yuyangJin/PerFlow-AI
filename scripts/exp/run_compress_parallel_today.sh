#!/usr/bin/env bash
#
# 今日并行压缩矩阵：在 tmux:padoc 中同时开多个 window，各自写独立
# report/exp/<TS>__<NAME>/ 与 report/exp_results/*.md|json|csv。
#
# 覆盖：
#   · subset8 / 32 / 64  — 5 baseline × 2 trace（与 F 系列一致）
#   · leworldmodel / unifolm — 5 baseline（H 系列数据）
#   · qwen3_full / llama_full — 仅 padoc（全量避免 raw baseline OOM）
#
# 用法：
#   bash scripts/exp/run_compress_parallel_today.sh
#
# CPU：各 window 经 `run_exp.sh` 用 `taskset` 绑在不重叠的核上（本机 nproc 均分）。
# 若需完全关闭绑核：`export PERFLOW_BENCH_CPUSET=off` 后再跑（不推荐多路并行时关）。
#
# 查看：
#   tmux ls
#   tmux attach -t padoc
#   ls -lt report/exp | head
#   python scripts/exp/status.py --filter 'C20'

set -euo pipefail
cd /home/lvjiaxin/work/AI/PerFlow-AI

ART_DIR=/mnt/treasure/ljx/padoc_artifacts/v2
mkdir -p "$ART_DIR" report/exp_results

ACTIVATE='source /mnt/treasure/ljx/miniconda3/etc/profile.d/conda.sh && conda activate PerFlow-AI'
COMP5='raw_msgpack gzip_msgpack tracezip scalatrace padoc'
NWIN=7

# --- 子集：5 baseline，多 worker ------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 0 "$NWIN")" scripts/exp/run_exp.sh C20_compress_subset8 -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/subset8.json \
  --compressors $COMP5 \
  --padoc-workers 16 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/C20_compress_subset8.md \
  --out-json report/exp_results/C20_compress_subset8.json \
  --out-csv  report/exp_results/C20_compress_subset8.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 1 "$NWIN")" scripts/exp/run_exp.sh C21_compress_subset32 -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/subset32.json \
  --compressors $COMP5 \
  --padoc-workers 16 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/C21_compress_subset32.md \
  --out-json report/exp_results/C21_compress_subset32.json \
  --out-csv  report/exp_results/C21_compress_subset32.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 2 "$NWIN")" scripts/exp/run_exp.sh C22_compress_subset64 -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/subset64.json \
  --compressors $COMP5 \
  --padoc-workers 16 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/C22_compress_subset64.md \
  --out-json report/exp_results/C22_compress_subset64.json \
  --out-csv  report/exp_results/C22_compress_subset64.csv
"

# --- 新 trace：5 baseline -------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 3 "$NWIN")" scripts/exp/run_exp.sh C23_compress_leworldmodel -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/leworldmodel.json \
  --compressors $COMP5 \
  --padoc-workers 8 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/C23_compress_leworldmodel.md \
  --out-json report/exp_results/C23_compress_leworldmodel.json \
  --out-csv  report/exp_results/C23_compress_leworldmodel.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 4 "$NWIN")" scripts/exp/run_exp.sh C24_compress_unifolm -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/unifolm.json \
  --compressors $COMP5 \
  --padoc-workers 8 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/C24_compress_unifolm.md \
  --out-json report/exp_results/C24_compress_unifolm.json \
  --out-csv  report/exp_results/C24_compress_unifolm.csv
"

# --- 全量：仅 PADOC（刷新当前代码路径下的 blob）---------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 5 "$NWIN")" scripts/exp/run_exp.sh C25_compress_qwen3_full_padoc -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/full_qwen3.json \
  --compressors padoc \
  --padoc-workers 32 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/C25_compress_qwen3_full_padoc.md \
  --out-json report/exp_results/C25_compress_qwen3_full_padoc.json \
  --out-csv  report/exp_results/C25_compress_qwen3_full_padoc.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 6 "$NWIN")" scripts/exp/run_exp.sh C26_compress_llama_full_padoc -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/full_llama.json \
  --compressors padoc \
  --padoc-workers 32 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/C26_compress_llama_full_padoc.md \
  --out-json report/exp_results/C26_compress_llama_full_padoc.json \
  --out-csv  report/exp_results/C26_compress_llama_full_padoc.csv
"

echo
echo "[compress-parallel] queued 7 tmux windows on session padoc:"
echo "  C20 subset8   C21 subset32  C22 subset64"
echo "  C23 leworldmodel  C24 unifolm"
echo "  C25 qwen3_full(padoc)  C26 llama_full(padoc)"
echo "artifact cache: $ART_DIR"
echo "attach: tmux attach -t padoc"
