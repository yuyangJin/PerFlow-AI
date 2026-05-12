#!/usr/bin/env python3
"""Watch experiment load and dequeue optional jobs into tmux:padoc.

Use when the main C20–C26 / D20–D26 batch is running: this process polls
``report/exp/*/meta.json`` for ``status == running``, and when the count
drops below ``--max-running``, starts the next line from a queue file via
``scripts/exp/run_exp.sh``.

Queue file format (``scripts/exp/extra_queue.queue`` by default)::

    # comment or blank line
    LOGICAL_NAME|bash -lc 'cd /repo && ...'

* ``LOGICAL_NAME`` is passed as the first argument to ``run_exp.sh`` (must
  not collide with another *running* experiment of the same ``name`` in
  meta.json).
* The part after the first ``|`` is the shell command run as ``run_exp.sh
  NAME -- <that command>`` (typically ``bash -lc '...'`` with conda + bench).

Examples::

    python3 scripts/exp/watch_fill.py --interval 45 --max-running 5
    python3 scripts/exp/watch_fill.py --queue scripts/exp/my_backlog.queue --dry-run

Logs append to ``report/exp/.watch_fill.log``.  Successfully started logical
names are appended to ``report/exp/.watch_fill_done`` (delete lines there to
allow re-queueing the same name).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "report" / "exp"
RUN_EXP = ROOT / "scripts" / "exp" / "run_exp.sh"
LOG_PATH = EXP_ROOT / ".watch_fill.log"
DONE_PATH = EXP_ROOT / ".watch_fill_done"


def count_running() -> int:
    n = 0
    if not EXP_ROOT.is_dir():
        return 0
    for meta_path in EXP_ROOT.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("status") == "running":
            n += 1
    return n


def running_logical_names() -> set[str]:
    out: set[str] = set()
    for meta_path in EXP_ROOT.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("status") != "running":
            continue
        name = meta.get("name")
        if isinstance(name, str) and name:
            out.add(name)
    return out


def parse_queue_line(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if "|" not in s:
        return None
    name, cmd = s.split("|", 1)
    name = name.strip()
    cmd = cmd.strip()
    if not name or not cmd:
        return None
    return name, cmd


def read_queue(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    jobs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = parse_queue_line(line)
        if p:
            jobs.append(p)
    return jobs


def loadavg_block(cpu_count: int | None, factor: float) -> bool:
    """Return True if we should NOT start (machine too busy)."""
    if cpu_count is None or cpu_count <= 0:
        return False
    try:
        la1, _, _ = os.getloadavg()
    except OSError:
        return False
    return la1 > cpu_count * factor


def load_filled_done() -> set[str]:
    if not DONE_PATH.is_file():
        return set()
    out: set[str] = set()
    try:
        for line in DONE_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                out.add(s)
    except OSError:
        pass
    return out


def mark_filled_done(name: str) -> None:
    try:
        EXP_ROOT.mkdir(parents=True, exist_ok=True)
        with DONE_PATH.open("a", encoding="utf-8") as f:
            f.write(name + "\n")
    except OSError:
        pass


def append_log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        EXP_ROOT.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
    print(line, end="")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "scripts" / "exp" / "extra_queue.queue",
        help="queue file path",
    )
    ap.add_argument("--interval", type=float, default=40.0, help="poll interval (seconds)")
    ap.add_argument(
        "--max-running",
        type=int,
        default=10,
        help="global ceiling: start queue jobs while total running meta.json count < this",
    )
    ap.add_argument(
        "--load-factor",
        type=float,
        default=0.92,
        help="if load1 > cpus * factor, do not start (use 0 to disable)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print actions only")
    args = ap.parse_args()

    load_guard = args.load_factor > 0
    cpus = os.cpu_count()

    append_log(
        f"watch_fill start queue={args.queue} interval={args.interval} "
        f"max_running={args.max_running} load_factor={args.load_factor} cpus={cpus}"
    )

    tick = 0
    last_guard_log = 0.0
    last_empty_log = 0.0

    while True:
        try:
            tick += 1
            now = time.time()
            running_n = count_running()
            busy = running_logical_names()
            filled_done = load_filled_done()
            queue = read_queue(args.queue)

            if load_guard and cpus and loadavg_block(cpus, args.load_factor):
                if now - last_guard_log > 600:
                    append_log(f"skip tick: load guard (running={running_n})")
                    last_guard_log = now
                time.sleep(args.interval)
                continue

            slots = args.max_running - running_n
            if slots <= 0:
                time.sleep(args.interval)
                continue

            started = 0
            for name, cmd in queue:
                if started >= slots:
                    break
                if name in busy or name in filled_done:
                    continue
                if args.dry_run:
                    print(f"[dry-run] would start {name!r}: {cmd[:120]}...")
                    started += 1
                    continue
                append_log(f"START {name} (running was {running_n}, slots {slots})")
                try:
                    subprocess.run(
                        ["bash", str(RUN_EXP), name, "--", "bash", "-lc", cmd],
                        cwd=str(ROOT),
                        check=False,
                    )
                except OSError as e:
                    append_log(f"ERROR spawn {name}: {e}")
                    time.sleep(args.interval)
                    continue
                mark_filled_done(name)
                filled_done.add(name)
                busy.add(name)
                started += 1
                running_n += 1
                slots -= 1

            if not queue and (now - last_empty_log > 900):
                append_log("queue empty or missing; sleeping")
                last_empty_log = now
            time.sleep(args.interval)
        except KeyboardInterrupt:
            append_log("watch_fill interrupted")
            return 0


if __name__ == "__main__":
    sys.exit(main())
