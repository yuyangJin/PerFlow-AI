#!/usr/bin/env python3
"""Show status + ETA for all experiments under report/exp/.

Usage:
    scripts/exp/status.py                  # one-shot table
    scripts/exp/status.py --watch 5        # refresh every 5s
    scripts/exp/status.py --filter NAME    # name substring filter
    scripts/exp/status.py --running-only --watch 10   # only active jobs

ETA is derived from the bench harness ``[done] ...`` log lines:
- Total work units = ``X traces x Y compressors`` / ``... x Z tasks`` parsed
  from the ``=== ... ===`` banner that the CLI prints.
- Done count = number of ``[done]`` lines so far.
- ETA = (total - done) * (elapsed / max(done,1)).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "report" / "exp"

BANNER_RE = re.compile(r"===\s*(?:PADOC ablation \w+|[\w ]+matrix|[\w ]+sweep)[^=]*?\(([^)]+)\)\s*===")
DONE_RE = re.compile(r"\[done\]")
NUM_RE = re.compile(r"(\d+)\s*(?:traces?|compressors?|tasks?|presets?)", re.IGNORECASE)
NUM_RE_LOOSE = re.compile(r"(\d+)")


def fmt_secs(s: Optional[float]) -> str:
    if s is None or s < 0:
        return "  -  "
    s = int(s)
    if s < 60:
        return f"{s:>3d}s"
    if s < 3600:
        return f"{s // 60:>2d}m{s % 60:02d}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h{m:02d}m"


def parse_banner_total(line: str) -> Optional[int]:
    m = BANNER_RE.search(line)
    if not m:
        return None
    inside = m.group(1)
    nums = [int(n) for n in NUM_RE.findall(inside)]
    if not nums:
        nums = [int(n) for n in NUM_RE_LOOSE.findall(inside)]
    if not nums:
        return None
    total = 1
    for n in nums:
        total *= n
    return total


def scan_log(log_path: Path) -> tuple[Optional[int], int, str]:
    """Return (total, done, last_progress_line)."""
    total: Optional[int] = None
    done = 0
    last = ""
    if not log_path.exists():
        return total, done, last
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if total is None:
                    t = parse_banner_total(line)
                    if t is not None:
                        total = t
                if DONE_RE.search(line):
                    done += 1
                    last = line.rstrip()
    except Exception as exc:  # pragma: no cover
        last = f"(scan error: {exc})"
    return total, done, last


def collect_rows(filter_substr: Optional[str], running_only: bool = False) -> list[dict]:
    rows: list[dict] = []
    if not EXP_ROOT.exists():
        return rows
    for d in sorted(EXP_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if filter_substr and filter_substr not in d.name:
            continue
        meta_path = d / "meta.json"
        log_path = d / "stdout.log"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        total, done, last = scan_log(log_path)
        now = time.time()
        start = meta.get("start_epoch") or now
        if meta["status"] == "running":
            elapsed = now - start
            if done > 0 and total and total > done:
                eta = (total - done) * (elapsed / done)
            else:
                eta = None
        else:
            elapsed = meta.get("elapsed_seconds") or 0
            eta = 0
        row = {
            "name": meta.get("name", d.name),
            "status": meta.get("status", "?"),
            "exit_code": meta.get("exit_code"),
            "elapsed": elapsed,
            "eta": eta,
            "done": done,
            "total": total,
            "last": last,
            "exp_dir": str(d),
            "tmux_window": meta.get("tmux_window"),
        }
        if running_only and row["status"] != "running":
            continue
        rows.append(row)
    return rows


def render(rows: list[dict]) -> str:
    if not rows:
        return "(no experiments under report/exp/)"
    lines = []
    header = f"{'STATUS':<8} {'NAME':<32} {'PROGRESS':<14} {'ELAPSED':<8} {'ETA':<8} {'EXIT':<5}  TAIL"
    lines.append(header)
    lines.append("-" * 130)
    for r in rows:
        prog = f"{r['done']}/{r['total']}" if r['total'] else f"{r['done']}/?"
        exit_str = "" if r['exit_code'] is None else str(r['exit_code'])
        last = r['last'][:60] if r['last'] else ""
        lines.append(
            f"{r['status']:<8} {r['name'][:32]:<32} {prog:<14} "
            f"{fmt_secs(r['elapsed']):<8} {fmt_secs(r['eta']):<8} "
            f"{exit_str:<5}  {last}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0, help="refresh interval (s); 0 = one-shot")
    ap.add_argument("--filter", default=None, help="substring filter on exp dir name")
    ap.add_argument("--running-only", action="store_true", help="only rows with status=running")
    args = ap.parse_args()
    if args.watch <= 0:
        print(render(collect_rows(args.filter, running_only=args.running_only)))
        return 0
    try:
        while True:
            os.system("clear")
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print(render(collect_rows(args.filter, running_only=args.running_only)))
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
