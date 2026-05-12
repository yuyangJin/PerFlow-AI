#!/usr/bin/env python3
"""并行跑多条子进程命令（用于 Llama / Qwen3 等不同 trace 同时跑 bench）。

示例::

    conda run -n PerFlow-AI python scripts/padoc_parallel_driver.py --max-workers 2 \\
      -c "python -m perflowai.padoc.bench compress --manifest scripts/paper_llama_only.json --compressors padoc --no-verify --no-track-memory --out-md /tmp/llama.md" \\
      -c "python -m perflowai.padoc.bench compress --manifest scripts/paper_qwen_only.json --compressors padoc --no-verify --no-track-memory --out-md /tmp/qwen.md"

或从文件读入（每行一条 shell，``#`` 开头为注释）::

    python scripts/padoc_parallel_driver.py --max-workers 4 --commands-file scripts/paper_wave1.txt
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Sequence, Tuple


def _run_one(cmd: Sequence[str]) -> Tuple[int, str]:
    proc = subprocess.run(list(cmd), capture_output=True, text=True)
    tail = (proc.stdout or "")[-4000:] + (proc.stderr or "")[-4000:]
    return proc.returncode, tail


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="并行执行多条命令（子进程相互独立，勿对同一输出文件双写）。"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="进程池大小。",
    )
    parser.add_argument(
        "-c",
        "--command",
        action="append",
        dest="commands",
        default=[],
        help="一条完整命令（可多次指定）。",
    )
    parser.add_argument(
        "--commands-file",
        default=None,
        help="每行一条命令；# 开头行为注释。",
    )
    args = parser.parse_args(argv)

    cmds: List[List[str]] = []
    if args.commands_file:
        with open(args.commands_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cmds.append(shlex.split(line))
    for c in args.commands:
        cmds.append(shlex.split(c))

    if not cmds:
        print("请用 -c 或 --commands-file 提供至少一条命令。", file=sys.stderr)
        return 2

    ok = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run_one, c): c for c in cmds}
        for fut in as_completed(futures):
            cmd = futures[fut]
            code, tail = fut.result()
            label = " ".join(cmd[:8]) + ("..." if len(cmd) > 8 else "")
            print(f"\n=== exit {code}: {label} ===\n{tail}")
            if code == 0:
                ok += 1
    print(f"\n完成 {ok}/{len(cmds)} 条命令成功。")
    return 0 if ok == len(cmds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
