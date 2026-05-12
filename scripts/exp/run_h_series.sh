#!/usr/bin/env bash
#
# H-series: leworldmodel (2 ranks) + unifolm-world-model (4 ranks)
#
# Both new traces are small enough (4M / 80M events total) to:
#   - compress with **all 5 baselines** (raw/gzip/tracezip/scalatrace/padoc)
#   - run all 7 analysis tasks
#
# Files use new formats:
#   - leworldmodel:    *.json.gz   (gzip-compressed JSON)
#   - unifolm:         *.json.zst  (zstandard-compressed JSON)
# Both are now read transparently by `Trace._read_trace_payload` /
# `_list_trace_files` (the loader change in this commit batch).
#
# Layout:
#   H1: compress leworldmodel × 5 baselines  → cache to artifact-dir
#   H2: compress unifolm × 5 baselines       → cache to artifact-dir
#   H3: analyze leworldmodel × 5 × 7 tasks   (waits on H1)
#   H4: analyze unifolm × 5 × 7 tasks        (waits on H2)
#
# H1 and H2 run **in parallel** with the still-running G3 (subset64) / G5
# (llama_full PADOC).  Both traces are small so the cpu/mem footprint is
# moderate.

set -euo pipefail
cd /home/lvjiaxin/work/AI/PerFlow-AI

ART_DIR=/mnt/treasure/ljx/padoc_artifacts/v2

ACTIVATE='source /mnt/treasure/ljx/miniconda3/etc/profile.d/conda.sh && conda activate PerFlow-AI'
NWIN=4

# ----------------------------------------------------------------------
# H1 — leworldmodel compress (2 ranks)
# ----------------------------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 0 "$NWIN")" scripts/exp/run_exp.sh H1_compress_leworldmodel_cache -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/leworldmodel.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --padoc-workers 8 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/H1_compress_leworldmodel.md \
  --out-json report/exp_results/H1_compress_leworldmodel.json \
  --out-csv  report/exp_results/H1_compress_leworldmodel.csv
"

# ----------------------------------------------------------------------
# H2 — unifolm-world-model compress (4 ranks)
# ----------------------------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 1 "$NWIN")" scripts/exp/run_exp.sh H2_compress_unifolm_cache -- bash -c "
$ACTIVATE
python -u -m perflowai.padoc.bench compress \
  --manifest scripts/manifests/unifolm.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --padoc-workers 8 --no-verify --no-track-memory \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/H2_compress_unifolm.md \
  --out-json report/exp_results/H2_compress_unifolm.json \
  --out-csv  report/exp_results/H2_compress_unifolm.csv
"

# ----------------------------------------------------------------------
# H3 — leworldmodel × 5 × 7 analyze (waits on H1)
# ----------------------------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 2 "$NWIN")" scripts/exp/run_exp.sh H3_analyze_leworldmodel_full_cache -- bash -c "
$ACTIVATE
need=( '$ART_DIR/padoc/leworldmodel_full.bin' '$ART_DIR/scalatrace/leworldmodel_full.bin' '$ART_DIR/tracezip/leworldmodel_full.bin' '$ART_DIR/raw_msgpack/leworldmodel_full.bin' '$ART_DIR/gzip_msgpack/leworldmodel_full.bin' )
for f in \"\${need[@]}\"; do
  while [ ! -s \"\$f\" ]; do
    echo \"waiting for \$f ...\"; sleep 20
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/leworldmodel.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/H3_analyze_leworldmodel_full.md \
  --out-json report/exp_results/H3_analyze_leworldmodel_full.json \
  --out-csv  report/exp_results/H3_analyze_leworldmodel_full.csv
"

# ----------------------------------------------------------------------
# H4 — unifolm × 5 × 7 analyze (waits on H2)
# ----------------------------------------------------------------------
PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 3 "$NWIN")" scripts/exp/run_exp.sh H4_analyze_unifolm_full_cache -- bash -c "
$ACTIVATE
need=( '$ART_DIR/padoc/unifolm_full.bin' '$ART_DIR/scalatrace/unifolm_full.bin' '$ART_DIR/tracezip/unifolm_full.bin' '$ART_DIR/raw_msgpack/unifolm_full.bin' '$ART_DIR/gzip_msgpack/unifolm_full.bin' )
for f in \"\${need[@]}\"; do
  while [ ! -s \"\$f\" ]; do
    echo \"waiting for \$f ...\"; sleep 30
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/unifolm.json \
  --compressors raw_msgpack gzip_msgpack tracezip scalatrace padoc \
  --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
  --artifact-dir $ART_DIR \
  --out-md   report/exp_results/H4_analyze_unifolm_full.md \
  --out-json report/exp_results/H4_analyze_unifolm_full.json \
  --out-csv  report/exp_results/H4_analyze_unifolm_full.csv
"

echo
echo "[h-series] H1..H4 queued under tmux:padoc"
echo "monitor with:"
echo "  python scripts/exp/status.py --filter '[GH]'"
