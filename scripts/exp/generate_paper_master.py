#!/usr/bin/env python3
"""Generate ``report/PAPER_EXPERIMENTS_MASTER.md`` — one file for paper + repro.

Re-run after new ``report/exp_results/*.json`` land:

    python3 scripts/exp/generate_paper_master.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "report" / "exp"
RES = ROOT / "report" / "exp_results"
OUT = ROOT / "report" / "PAPER_EXPERIMENTS_MASTER.md"

LOGICAL_ORDER = [
    "C20_compress_subset8",
    "C21_compress_subset32",
    "C22_compress_subset64",
    "C23_compress_leworldmodel",
    "C23b_compress_leworldmodel",
    "C24_compress_unifolm",
    "C25_compress_qwen3_full_padoc",
    "C26_compress_llama_full_padoc",
    "D20_analyze_subset8",
    "D21_analyze_subset32",
    "D22_analyze_subset64",
    "D23_analyze_leworldmodel",
    "D24_analyze_unifolm",
    "D25_analyze_qwen3_full_padoc",
    "D26_analyze_llama_full_padoc",
]

STATUS_RANK = {"done": 4, "running": 3, "failed": 2, "cancelled": 1}


def collect_latest_meta() -> dict[str, dict]:
    by_name: dict[str, dict] = {}
    for d in EXP.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        m = re.match(r"(\d{8}T\d{6}Z)__(.+)", d.name)
        if not m:
            continue
        ts, _rest = m.group(1), m.group(2)
        mp = d / "meta.json"
        if not mp.is_file():
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta["_ts"] = ts
        meta["_dir"] = d.name
        key = str(meta.get("name", _rest))
        old = by_name.get(key)
        if old is None:
            by_name[key] = meta
            continue
        r_new = STATUS_RANK.get(meta.get("status"), 0)
        r_old = STATUS_RANK.get(old.get("status"), 0)
        if r_new > r_old or (r_new == r_old and ts > old["_ts"]):
            by_name[key] = meta
    return by_name


def read_cmd_sh(exp_dir_name: str) -> str:
    p = EXP / exp_dir_name / "cmd.sh"
    if not p.is_file():
        return "(cmd.sh missing)\n"
    return p.read_text(encoding="utf-8", errors="replace")


def fmt_ratio_table(rows: list[dict]) -> str:
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_t[str(r.get("trace_name", "?"))].append(r)
    lines = ["| trace | PADOC | ScalaTrace | TraceZip | gzip+mp | raw+mp |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    order = ["padoc", "scalatrace", "tracezip", "gzip_msgpack", "raw_msgpack"]
    for tn in sorted(by_t.keys()):
        m = {x["compressor"]: x.get("compression_ratio") for x in by_t[tn]}
        cells = [f"{m.get(c, float('nan')):8.2f}×".replace("nan×", "—") for c in order]
        lines.append(f"| {tn} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_analyze_totals(rows: list[dict]) -> str:
    """Sum end_to_end per (trace, compressor)."""
    agg: dict[tuple[str, str], float] = defaultdict(float)
    for r in rows:
        if not r.get("success", True):
            continue
        agg[(r["trace_name"], r["compressor"])] += float(r.get("end_to_end_seconds") or 0)
    lines = ["| trace | padoc | raw | gzip | tracezip | scalatrace |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    traces = sorted({t for t, _ in agg})
    cs = ["padoc", "raw_msgpack", "gzip_msgpack", "tracezip", "scalatrace"]
    for tr in traces:
        vals = []
        for c in cs:
            v = agg.get((tr, c))
            vals.append("—" if v is None or v == 0 else f"{v:.1f}")
        lines.append(f"| {tr} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    meta_by = collect_latest_meta()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# PADOC 论文实验总表（数据 + 可复现命令）",
        "",
        "> **尚未 `done` 的行**（见 §1）由 `tmux` 会话 `padoc` 中的任务继续跑；全部 JSON 落盘后请再执行：  \n"
        "> `python3 scripts/exp/generate_paper_master.py` 以刷新本文件中的分析表与状态列。",
        "",
        f"_本文件由 `python3 scripts/exp/generate_paper_master.py` 于 **{now}** 自动生成；"
        f"原始 JSON/CSV/MD 在 `report/exp_results/`，实验包装日志在 `report/exp/<ts>__<name>/`。**改数据请先重跑 bench，再重新执行该脚本。**_",
        "",
        "## 1. 实验清单与状态",
        "",
        "| 逻辑名 | 状态 | exit | wall(s) | 实验目录 | 结果 JSON |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in LOGICAL_ORDER:
        m = meta_by.get(name)
        if m is None:
            lines.append(f"| `{name}` | (无记录) | | | | |")
            continue
        st = m.get("status", "?")
        ec = m.get("exit_code")
        ec_s = "" if ec is None else str(ec)
        el = m.get("elapsed_seconds")
        el_s = "—" if el is None else str(int(el))
        jn = name + ".json"
        jp = RES / jn
        if name == "C23b_compress_leworldmodel":
            alt = RES / "C23_compress_leworldmodel.json"
            if alt.is_file():
                jcell = "`report/exp_results/C23_compress_leworldmodel.json`（C23b 成功覆盖）"
            elif jp.is_file():
                jcell = f"`report/exp_results/{jn}`"
            else:
                jcell = "—（未落盘）"
        elif jp.is_file():
            jcell = f"`report/exp_results/{jn}`"
        else:
            jcell = "—（未落盘）"
        lines.append(
            f"| `{name}` | {st} | {ec_s} | {el_s} | `{m['_dir']}` | {jcell} |"
        )

    lines += [
        "",
        "**说明：** 首轮 `C23_compress_leworldmodel` 曾 `failed`（TraceZip）；以 **`C23b_compress_leworldmodel`** 与 `report/exp_results/C23_compress_leworldmodel.*`（同名覆盖）为准。",
        "",
        "---",
        "",
        "## 2. 一键复现（与当时并行脚本等价）",
        "",
        "在仓库根目录、已激活 `PerFlow-AI` conda 环境后：",
        "",
        "```bash",
        "cd /home/lvjiaxin/work/AI/PerFlow-AI",
        "bash scripts/exp/run_compress_parallel_today.sh   # C20–C26 压缩（tmux: padoc）",
        "bash scripts/exp/run_analyze_parallel_today.sh # D20–D26 分析（可与压缩并行）",
        "```",
        "",
        "单独重跑某一格，请用 **`scripts/exp/run_exp.sh <逻辑名> -- bash -c '...'`**；每次运行会新建 `report/exp/<新时间戳>__<逻辑名>/`，其中 **`cmd.sh` 为精确可复现命令**（见下节）。",
        "",
        "---",
        "",
        "## 3. 各实验 `cmd.sh`（精确复现）",
        "",
    ]

    for name in LOGICAL_ORDER:
        m = meta_by.get(name)
        if not m:
            continue
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"- **目录:** `report/exp/{m['_dir']}/`")
        lines.append(f"- **stdout / meta:** `stdout.log`, `meta.json`")
        lines.append("")
        lines.append("```bash")
        lines.append(read_cmd_sh(m["_dir"]).rstrip())
        lines.append("```")
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. 论文用数值（来自已落盘 JSON）",
        "",
    ]

    # Compression blocks
    for tag in ["C20_compress_subset8", "C21_compress_subset32", "C22_compress_subset64", "C23_compress_leworldmodel", "C25_compress_qwen3_full_padoc", "C26_compress_llama_full_padoc"]:
        jp = RES / f"{tag}.json"
        if not jp.is_file():
            lines.append(f"### {tag}")
            lines.append("_（JSON 尚未生成）_\n")
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        lines.append(f"### 压缩 — `{tag}`")
        lines.append("")
        lines.append(fmt_ratio_table(data))
        lines.append("")

    jp = RES / "C24_compress_unifolm.json"
    if jp.is_file():
        data = json.loads(jp.read_text(encoding="utf-8"))
        lines.append("### 压缩 — `C24_compress_unifolm`")
        lines.append("")
        lines.append(fmt_ratio_table(data))
        lines.append("")

    # Analyze blocks
    for tag in ["D20_analyze_subset8", "D21_analyze_subset32", "D22_analyze_subset64", "D23_analyze_leworldmodel", "D24_analyze_unifolm", "D25_analyze_qwen3_full_padoc", "D26_analyze_llama_full_padoc"]:
        jp = RES / f"{tag}.json"
        if not jp.is_file():
            lines.append(f"### 分析 — `{tag}`")
            lines.append("_（JSON 尚未生成；对应 analyze 仍在跑或等待上游压缩）_\n")
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        lines.append(f"### 分析 — `{tag}`")
        lines.append("")
        lines.append("**七任务 `end_to_end_seconds` 之和（秒，按 trace×compressor）：**")
        lines.append("")
        lines.append(fmt_analyze_totals(data))
        lines.append("")
        lines.append("逐任务原始字段（含 `decompress_seconds` / `analysis_seconds`）见该 JSON。")
        lines.append("")

    lines += [
        "---",
        "",
        "## 5. 计时与公平性（写论文时必读）",
        "",
        "- **TraceZip / ScalaTrace / raw / gzip**：对同一 `(trace, compressor)`，bench **只完整解码一次**；解压时间记在**第一次需要内存 `Trace` 的任务行**的 `decompress_seconds`，并与 `analysis_seconds` 相加得该行 `end_to_end_seconds`；后续任务 `decompress_seconds` 多为 0。",
        "- **PADOC**：in-situ 任务与需全量解码的 HTA 任务混排；某一行的「慢」常为 **首次全量解码** 而非单任务算法本身。详见 `perflowai/padoc/bench/runner.py` 中 `decoded_trace_cache` 注释。",
        "",
        "---",
        "",
        "## 6. 产物路径速查",
        "",
        "| 类型 | 路径 |",
        "| --- | --- |",
        "| Artifact blobs | `/mnt/treasure/ljx/padoc_artifacts/v2/<compressor>/<dataset>.bin` |",
        "| Bench 表 | `report/exp_results/C*.md`, `D*.md` 及对应 `.json` / `.csv` |",
        "| 实时 meta 总览 | `report/exp/LIVE_SNAPSHOT.md`（`refresh_snapshot.py` 刷新） |",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
