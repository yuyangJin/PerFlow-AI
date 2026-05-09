"""Report formatting for the bench harness.

Three output formats are supported:

* JSON -- raw record dump (for downstream plotting / tables).
* CSV  -- one row per record (Excel-friendly).
* Markdown table -- ready to paste in the paper.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Iterable, List, Sequence

from .metrics import AnalysisRecord, CompressionRecord


# ----------------------------------------------------------------------
# Number formatting helpers
# ----------------------------------------------------------------------


def _format_bytes(num: float) -> str:
    if num < 0:
        return "-"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(num):d} {unit}"
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} TiB"


def _format_seconds(num: float) -> str:
    if num < 0:
        return "-"
    if num < 1.0:
        return f"{num * 1000:.1f} ms"
    if num < 60.0:
        return f"{num:.2f} s"
    minutes = num / 60
    return f"{minutes:.1f} min"


def _format_ratio(num: float) -> str:
    if num <= 0:
        return "-"
    return f"{num:.2f}x"


# ----------------------------------------------------------------------
# Compression matrix renderers
# ----------------------------------------------------------------------


def render_compression_markdown(records: Sequence[CompressionRecord]) -> str:
    if not records:
        return "_no compression records_\n"

    headers = [
        "trace",
        "compressor",
        "events",
        "source",
        "compressed",
        "ratio",
        "compress",
        "decompress",
        "verify",
        "templates",
    ]
    rows: List[List[str]] = [headers]
    for record in records:
        templates = record.metadata.get("template_count")
        if templates is None:
            templates = (
                record.metadata.get("name_count")
                or record.metadata.get("srt_path_count")
                or "-"
            )
        rows.append(
            [
                record.trace_name,
                record.compressor,
                str(record.event_count),
                _format_bytes(record.source_size_bytes),
                _format_bytes(record.compressed_size_bytes),
                _format_ratio(record.compression_ratio),
                _format_seconds(record.compress_seconds),
                _format_seconds(record.decompress_seconds),
                "Y" if record.verify_passed else ("-" if not record.verify_message else "N"),
                str(templates),
            ]
        )
    return _render_markdown_table(rows)


def render_analysis_markdown(records: Sequence[AnalysisRecord]) -> str:
    if not records:
        return "_no analysis records_\n"

    headers = [
        "trace",
        "compressor",
        "task",
        "in_situ",
        "decompress",
        "analysis",
        "end_to_end",
        "peak_mem",
        "status",
    ]
    rows: List[List[str]] = [headers]
    for record in records:
        rows.append(
            [
                record.trace_name,
                record.compressor,
                record.task,
                "Y" if record.in_situ else "-",
                _format_seconds(record.decompress_seconds),
                _format_seconds(record.analysis_seconds),
                _format_seconds(record.end_to_end_seconds),
                _format_bytes(record.peak_memory_bytes),
                "ok" if record.success else f"err: {record.error_message}",
            ]
        )
    return _render_markdown_table(rows)


def _render_markdown_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = ["| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(rows[0])) + " |", sep]
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Generic JSON / CSV writers
# ----------------------------------------------------------------------


def write_records_json(records: Sequence[Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = []
    for record in records:
        if hasattr(record, "as_dict"):
            payload.append(record.as_dict())
        else:
            payload.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default_json)


def write_records_csv(records: Sequence[Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = []
    for record in records:
        if hasattr(record, "as_dict"):
            payload.append(record.as_dict())
        elif isinstance(record, dict):
            payload.append(record)
    if not payload:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys: List[str] = []
    seen: set[str] = set()
    for entry in payload:
        for key in entry.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for entry in payload:
            row = {key: _flatten_value(entry.get(key)) for key in keys}
            writer.writerow(row)


def _default_json(value: Any) -> Any:
    try:
        return value.__dict__
    except AttributeError:
        return str(value)


def _flatten_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=_default_json)
    return value
