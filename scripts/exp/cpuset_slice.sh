#!/usr/bin/env bash
# 按「本机逻辑 CPU 数」把 CPU 均分给 n 个并行 job，打印第 idx 个 job 的 taskset 列表。
#
# 用法：
#   PERFLOW_BENCH_CPUSET="$(scripts/exp/cpuset_slice.sh <idx> <n_parallel>)"
#   idx 为 0 .. n_parallel-1
#
# 例：7 路并行压缩
#   "$(scripts/exp/cpuset_slice.sh 0 7)"  → 可能 0-4
#   "$(scripts/exp/cpuset_slice.sh 6 7)" → 最后一段
set -euo pipefail

idx="${1:?need index (0..n-1)}"
nwin="${2:?need n_parallel jobs}"

n="$(nproc)"
if [[ "$n" -lt 1 ]]; then
  n=1
fi

span=$((n / nwin))
if [[ "$span" -lt 1 ]]; then
  span=1
fi

start=$((idx * span))
end=$((start + span - 1))

if [[ "$end" -ge "$n" ]]; then
  end=$((n - 1))
fi
if [[ "$start" -ge "$n" ]]; then
  start=$((n - 1))
  end=$((n - 1))
fi
if [[ "$start" -gt "$end" ]]; then
  end=$start
fi

echo "${start}-${end}"
