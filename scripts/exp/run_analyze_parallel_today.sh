#!/usr/bin/env bash
# 并行排队 analyze：每个 window 内先 wait blob，再跑 7 task。
# 可与 C20–C26 压缩同时启动（wait 会阻塞到压缩写完）。
set -euo pipefail
cd /home/lvjiaxin/work/AI/PerFlow-AI

ART=/mnt/treasure/ljx/padoc_artifacts/v2
A='source /mnt/treasure/ljx/miniconda3/etc/profile.d/conda.sh && conda activate PerFlow-AI'
C='raw_msgpack gzip_msgpack tracezip scalatrace padoc'
T='operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group'
NWIN=7

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 0 "$NWIN")" scripts/exp/run_exp.sh D20_analyze_subset8 -- bash -c "
set -euo pipefail
$A
cd /home/lvjiaxin/work/AI/PerFlow-AI
ART='$ART'
for c in raw_msgpack gzip_msgpack tracezip scalatrace padoc; do
  for t in qwen3_subset8 llama_subset8; do
    f=\"\$ART/\$c/\$t.bin\"
    while [[ ! -s \"\$f\" ]]; do echo waiting \"\$f\"; sleep 25; done
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/subset8.json --compressors $C --tasks $T \
  --artifact-dir \"$ART\" \
  --out-md report/exp_results/D20_analyze_subset8.md \
  --out-json report/exp_results/D20_analyze_subset8.json \
  --out-csv report/exp_results/D20_analyze_subset8.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 1 "$NWIN")" scripts/exp/run_exp.sh D21_analyze_subset32 -- bash -c "
set -euo pipefail
$A
cd /home/lvjiaxin/work/AI/PerFlow-AI
ART='$ART'
for c in raw_msgpack gzip_msgpack tracezip scalatrace padoc; do
  for t in qwen3_subset32 llama_subset32; do
    f=\"\$ART/\$c/\$t.bin\"
    while [[ ! -s \"\$f\" ]]; do echo waiting \"\$f\"; sleep 25; done
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/subset32.json --compressors $C --tasks $T \
  --artifact-dir \"$ART\" \
  --out-md report/exp_results/D21_analyze_subset32.md \
  --out-json report/exp_results/D21_analyze_subset32.json \
  --out-csv report/exp_results/D21_analyze_subset32.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 2 "$NWIN")" scripts/exp/run_exp.sh D22_analyze_subset64 -- bash -c "
set -euo pipefail
$A
cd /home/lvjiaxin/work/AI/PerFlow-AI
ART='$ART'
for c in raw_msgpack gzip_msgpack tracezip scalatrace padoc; do
  for t in qwen3_subset64 llama_subset64; do
    f=\"\$ART/\$c/\$t.bin\"
    while [[ ! -s \"\$f\" ]]; do echo waiting \"\$f\"; sleep 25; done
  done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/subset64.json --compressors $C --tasks $T \
  --artifact-dir \"$ART\" \
  --out-md report/exp_results/D22_analyze_subset64.md \
  --out-json report/exp_results/D22_analyze_subset64.json \
  --out-csv report/exp_results/D22_analyze_subset64.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 3 "$NWIN")" scripts/exp/run_exp.sh D23_analyze_leworldmodel -- bash -c "
set -euo pipefail
$A
cd /home/lvjiaxin/work/AI/PerFlow-AI
ART='$ART'
for c in raw_msgpack gzip_msgpack tracezip scalatrace padoc; do
  f=\"\$ART/\$c/leworldmodel_full.bin\"
  while [[ ! -s \"\$f\" ]]; do echo waiting \"\$f\"; sleep 25; done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/leworldmodel.json --compressors $C --tasks $T \
  --artifact-dir \"$ART\" \
  --out-md report/exp_results/D23_analyze_leworldmodel.md \
  --out-json report/exp_results/D23_analyze_leworldmodel.json \
  --out-csv report/exp_results/D23_analyze_leworldmodel.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 4 "$NWIN")" scripts/exp/run_exp.sh D24_analyze_unifolm -- bash -c "
set -euo pipefail
$A
cd /home/lvjiaxin/work/AI/PerFlow-AI
ART='$ART'
for c in raw_msgpack gzip_msgpack tracezip scalatrace padoc; do
  f=\"\$ART/\$c/unifolm_full.bin\"
  while [[ ! -s \"\$f\" ]]; do echo waiting \"\$f\"; sleep 25; done
done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/unifolm.json --compressors $C --tasks $T \
  --artifact-dir \"$ART\" \
  --out-md report/exp_results/D24_analyze_unifolm.md \
  --out-json report/exp_results/D24_analyze_unifolm.json \
  --out-csv report/exp_results/D24_analyze_unifolm.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 5 "$NWIN")" scripts/exp/run_exp.sh D25_analyze_qwen3_full_padoc -- bash -c "
set -euo pipefail
$A
cd /home/lvjiaxin/work/AI/PerFlow-AI
ART='$ART'
f=\"\$ART/padoc/qwen3_full.bin\"
while [[ ! -s \"\$f\" ]]; do echo waiting \"\$f\"; sleep 45; done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/full_qwen3.json --compressors padoc --tasks $T \
  --artifact-dir \"$ART\" \
  --out-md report/exp_results/D25_analyze_qwen3_full_padoc.md \
  --out-json report/exp_results/D25_analyze_qwen3_full_padoc.json \
  --out-csv report/exp_results/D25_analyze_qwen3_full_padoc.csv
"

PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh 6 "$NWIN")" scripts/exp/run_exp.sh D26_analyze_llama_full_padoc -- bash -c "
set -euo pipefail
$A
cd /home/lvjiaxin/work/AI/PerFlow-AI
ART='$ART'
f=\"\$ART/padoc/llama_full.bin\"
while [[ ! -s \"\$f\" ]]; do echo waiting \"\$f\"; sleep 60; done
python -u -m perflowai.padoc.bench analyze \
  --manifest scripts/manifests/full_llama.json --compressors padoc --tasks $T \
  --artifact-dir \"$ART\" \
  --out-md report/exp_results/D26_analyze_llama_full_padoc.md \
  --out-json report/exp_results/D26_analyze_llama_full_padoc.json \
  --out-csv report/exp_results/D26_analyze_llama_full_padoc.csv
"

echo "[analyze-parallel] queued D20..D26"
echo "results: report/exp_results/D20_*.md … D26_*.md"
