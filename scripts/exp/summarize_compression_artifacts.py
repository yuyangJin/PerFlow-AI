#!/usr/bin/env python3
"""Write report/COMPRESSION_DATA_ALL.md — every non-empty *.bin under artifact-dir."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = Path("/mnt/treasure/ljx/padoc_artifacts/v2")
OUT = ROOT / "report" / "COMPRESSION_DATA_ALL.md"
RES = ROOT / "report" / "exp_results"


def main() -> int:
    rows: list[tuple[str, str, int, str]] = []
    if ART.is_dir():
        for bin_path in sorted(ART.rglob("*.bin")):
            rel = bin_path.relative_to(ART)
            parts = rel.parts
            if len(parts) != 2:
                continue
            comp, name = parts[0], parts[1]
            try:
                sz = bin_path.stat().st_size
            except OSError:
                continue
            if sz <= 0:
                continue
            m = datetime.fromtimestamp(bin_path.stat().st_mtime, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            rows.append((comp, name[:-4] if name.endswith(".bin") else name, sz, m))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# 压缩产物汇总（artifact `.bin`）",
        "",
        f"_生成时间: {now}. 路径根: `{ART}`_",
        "",
        "| compressor | dataset | size (bytes) | size (human) | mtime (UTC) |",
        "| --- | --- | ---: | --- | --- |",
    ]

    def hum2(n: int) -> str:
        if n < 1024:
            return f"{n} B"
        if n < 1024**2:
            return f"{n / 1024:.2f} KiB"
        if n < 1024**3:
            return f"{n / 1024**2:.2f} MiB"
        return f"{n / 1024**3:.2f} GiB"

    for comp, ds, sz, m in sorted(rows, key=lambda r: (r[1], r[0])):
        lines.append(f"| `{comp}` | `{ds}` | {sz} | {hum2(sz)} | {m} |")

    lines += ["", "## 已有 bench JSON（压缩矩阵落盘）", ""]
    if RES.is_dir():
        for p in sorted(RES.glob("*compress*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list) or not data:
                continue
            lines.append(f"- `{p.relative_to(ROOT)}` — {len(data)} rows")
    else:
        lines.append("_（无 exp_results）_")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(rows)} blobs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
