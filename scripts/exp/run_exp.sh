#!/usr/bin/env bash
# Usage: scripts/exp/run_exp.sh <NAME> -- <CMD ...>
#
# CPU 亲和：默认用 taskset 把子进程绑在固定核上，减轻与其它任务抢 CPU。
#   · 未设置时：nproc≥8 → 0-7，否则 0..(nproc-1)
#   · PERFLOW_BENCH_CPUSET=3,5,7-11  → 显式列表（传给 taskset -c）
#   · PERFLOW_BENCH_CPUSET=off|no   → 不绑核
#
# 多路并行脚本（如 run_compress_parallel_today.sh）会为每个 window 设置不同的
# PERFLOW_BENCH_CPUSET（见 scripts/exp/cpuset_slice.sh）。
#
# Spawns the command inside tmux session "padoc" (window=NAME), writing:
#   report/exp/<TS>__<NAME>/cmd.sh        <- exact reproducible command
#   report/exp/<TS>__<NAME>/_wrap.sh      <- internal wrapper (timestamps + meta update)
#   report/exp/<TS>__<NAME>/stdout.log    <- full timestamped stdout/stderr
#   report/exp/<TS>__<NAME>/meta.json     <- {status, exit_code, start/end, elapsed}
#
# Survives tab close because it lives in tmux; reattach with:
#   tmux attach -t padoc \; select-window -t padoc:<NAME>
#
# Prints the experiment directory on stdout.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <NAME> -- <CMD...>" >&2
  exit 2
fi

NAME="$1"; shift
if [[ "${1:-}" == "--" ]]; then shift; fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
EXP_DIR="$REPO_ROOT/report/exp/${TS}__${NAME}"
mkdir -p "$EXP_DIR"

# 1) save reproducible command
{
  echo "#!/usr/bin/env bash"
  echo "# Reproduce experiment '$NAME' (recorded $(date -u -Iseconds))"
  echo "set -euo pipefail"
  echo "source /mnt/treasure/ljx/miniconda3/etc/profile.d/conda.sh"
  echo "conda activate PerFlow-AI"
  echo "cd $REPO_ROOT"
  echo
  printf '%q ' "$@"
  echo
} > "$EXP_DIR/cmd.sh"
chmod +x "$EXP_DIR/cmd.sh"

# 2) save initial meta
HOST="$(hostname)"
START_EPOCH="$(date +%s)"
START_ISO="$(date -u -Iseconds)"
python3 -c "
import json
json.dump({
    'name': '$NAME',
    'exp_dir': '$EXP_DIR',
    'cmd_file': '$EXP_DIR/cmd.sh',
    'host': '$HOST',
    'start_epoch': $START_EPOCH,
    'start_iso': '$START_ISO',
    'tmux_session': 'padoc',
    'tmux_window': '$NAME',
    'status': 'running',
    'exit_code': None,
    'end_epoch': None,
    'end_iso': None,
    'elapsed_seconds': None,
}, open('$EXP_DIR/meta.json', 'w'), indent=2)
"

# 3) wrapper that runs cmd.sh, timestamps every line, updates meta on exit
WRAP="$EXP_DIR/_wrap.sh"
cat > "$WRAP" <<EOF
#!/usr/bin/env bash
set -uo pipefail
# Force unbuffered Python; line-buffered awk; tee writes after each line.
export PYTHONUNBUFFERED=1
START=\$(date +%s)
exec > >(stdbuf -oL awk '{ printf "[%s] %s\n", strftime("%H:%M:%S"), \$0; fflush() }' | tee "$EXP_DIR/stdout.log") 2>&1
echo "+++ exp=$NAME"
echo "+++ start=\$(date -u -Iseconds) host=\$(hostname) pid=\$\$"
echo "+++ cmd=$EXP_DIR/cmd.sh"
echo "----- begin command output -----"
_nproc=\$(nproc)
_cset="\${PERFLOW_BENCH_CPUSET:-}"
if [[ "\$_cset" == "off" || "\$_cset" == "no" ]]; then
  echo "+++ cpuset: disabled (PERFLOW_BENCH_CPUSET=off|no)"
  bash "$EXP_DIR/cmd.sh"
elif [[ -n "\$_cset" ]]; then
  echo "+++ cpuset: taskset -c \$_cset (PERFLOW_BENCH_CPUSET)"
  taskset -c "\$_cset" bash "$EXP_DIR/cmd.sh"
else
  if [[ "\$_nproc" -ge 8 ]]; then _def="0-7"; else _def="0-\$((_nproc - 1))"; fi
  echo "+++ cpuset: taskset -c \${_def} (auto; nproc=\${_nproc})"
  taskset -c "\$_def" bash "$EXP_DIR/cmd.sh"
fi
EC=\$?
END=\$(date +%s)
ELAPSED=\$((END - START))
echo "----- end command output -----"
echo "+++ exit_code=\$EC elapsed=\${ELAPSED}s end=\$(date -u -Iseconds)"
python3 -c "
import json
m = json.load(open('$EXP_DIR/meta.json'))
m['status'] = 'done' if \$EC == 0 else 'failed'
m['exit_code'] = \$EC
m['end_epoch'] = \$END
m['end_iso'] = '\$(date -u -Iseconds)'
m['elapsed_seconds'] = \$ELAPSED
json.dump(m, open('$EXP_DIR/meta.json', 'w'), indent=2)
"
EOF
chmod +x "$WRAP"

# 4) launch in tmux (detached, persistent)
if ! tmux has-session -t padoc 2>/dev/null; then
  tmux new-session -d -s padoc -n _idle 'while :; do sleep 3650; done'
fi
# unique window name (tmux requires unique names per session)
WIN="${NAME}_${TS}"
tmux new-window -d -t padoc: -n "$WIN" "bash '$WRAP'"

echo "$EXP_DIR"
