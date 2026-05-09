from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perflowai.padoc.extract_slp_dataset import build_slp_dataset, write_slp_dataset
from perflowai.padoc._compat import asizeof
from perflowai.padoc.event import Event, MergeEvent
from perflowai.padoc.slp import SegmentedLinearPredictorCompressor as SLP
from perflowai.padoc.utils import to_json_safe
import msgpack


TRACE_PATH = Path("tests/example_trace/out-1024.json")
ARTIFACTS_DIR = Path("tests/padoc_test/artifacts")
METRICS_PATH = ARTIFACTS_DIR / "slp_metrics.json"
METRICS_TABLE_MD_PATH = ARTIFACTS_DIR / "slp_metrics_table.md"
METRICS_TABLE_CSV_PATH = ARTIFACTS_DIR / "slp_metrics_table.csv"
METRICS_PNG_PATH = ARTIFACTS_DIR / "slp_metrics_chart.png"
DATASET_PATH = ARTIFACTS_DIR / "slp_dataset.json"
_DATASET: dict[str, Any] | None = None
_METRICS: dict[str, dict[str, float]] = {}


def _load_dataset() -> dict[str, Any]:
    global _DATASET
    if _DATASET is None:
        _DATASET = build_slp_dataset(str(TRACE_PATH))
    return _DATASET


def _ratio(compressed: int, original: int) -> float:
    if original == 0:
        return 1.0
    return compressed / original


def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_metrics() -> None:
    _ensure_artifacts_dir()
    METRICS_PATH.write_text(
        json.dumps(_METRICS, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_metrics_table()
    _write_metrics_chart()


def _write_metrics_table() -> None:
    headers = [
        "field",
        "memory",
        "memory_ratio",
        "json",
        "json_ratio",
        "binary_msgpack",
        "binary_ratio",
    ]

    rows = []
    for field in sorted(_METRICS.keys()):
        metrics = _METRICS[field]
        rows.append(
            {
                "field": field,
                "memory": (
                    f"{metrics['compressed_memory_human']} / "
                    f"{metrics['original_memory_human']}"
                ),
                "memory_ratio": f"{metrics['memory_ratio']:.2%}",
                "json": (
                    f"{metrics['compressed_json_human']} / "
                    f"{metrics['original_json_human']}"
                ),
                "json_ratio": f"{metrics['json_ratio']:.2%}",
                "binary_msgpack": (
                    f"{metrics['compressed_binary_human']} / "
                    f"{metrics['original_binary_human']}"
                ),
                "binary_ratio": f"{metrics['binary_ratio']:.2%}",
            }
        )

    with METRICS_TABLE_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# SLP Metrics Summary",
        "",
        "| Field | Memory (compressed / original) | Memory Ratio | JSON (compressed / original) | JSON Ratio | Binary Msgpack (compressed / original) | Binary Ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['field']} | {row['memory']} | {row['memory_ratio']} | "
            f"{row['json']} | {row['json_ratio']} | {row['binary_msgpack']} | "
            f"{row['binary_ratio']} |"
        )

    METRICS_TABLE_MD_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _write_metrics_chart() -> None:
    fields = sorted(_METRICS.keys())
    if not fields:
        return

    memory_ratios = [_METRICS[field]["memory_ratio"] * 100 for field in fields]
    json_ratios = [_METRICS[field]["json_ratio"] * 100 for field in fields]
    binary_ratios = [_METRICS[field]["binary_ratio"] * 100 for field in fields]

    x = range(len(fields))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], memory_ratios, width=width, label="Memory")
    ax.bar(x, json_ratios, width=width, label="JSON")
    ax.bar([i + width for i in x], binary_ratios, width=width, label="Msgpack")

    ax.set_title("SLP Compression Ratios by Field")
    ax.set_ylabel("Compressed / Original (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fields)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(METRICS_PNG_PATH, dpi=160)
    plt.close(fig)


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _measure_sizes(stem: str, original_obj: Any, compressed_obj: Any) -> dict[str, float]:
    _ensure_artifacts_dir()
    original_mem = asizeof.asizeof(original_obj)
    compressed_mem = asizeof.asizeof(compressed_obj)

    original_json = ARTIFACTS_DIR / f"{stem}_original.json"
    compressed_json = ARTIFACTS_DIR / f"{stem}_compressed.json"
    original_json_obj = to_json_safe(original_obj)
    compressed_json_obj = to_json_safe(compressed_obj)
    original_json.write_text(json.dumps(original_json_obj, indent=2), encoding="utf-8")
    compressed_json.write_text(json.dumps(compressed_json_obj, indent=2), encoding="utf-8")

    original_json_bytes = len(json.dumps(original_json_obj, separators=(",", ":")).encode("utf-8"))
    compressed_json_bytes = len(json.dumps(compressed_json_obj, separators=(",", ":")).encode("utf-8"))

    original_bin = ARTIFACTS_DIR / f"{stem}_original.bin"
    compressed_bin = ARTIFACTS_DIR / f"{stem}_compressed.bin"
    original_bin.write_bytes(msgpack.packb(original_json_obj, use_bin_type=True))
    compressed_bin.write_bytes(msgpack.packb(compressed_json_obj, use_bin_type=True))

    metrics = {
        "original_memory_bytes": original_mem,
        "compressed_memory_bytes": compressed_mem,
        "original_memory_human": _format_size(original_mem),
        "compressed_memory_human": _format_size(compressed_mem),
        "memory_ratio": _ratio(compressed_mem, original_mem),
        "original_json_bytes": original_json_bytes,
        "compressed_json_bytes": compressed_json_bytes,
        "original_json_human": _format_size(original_json_bytes),
        "compressed_json_human": _format_size(compressed_json_bytes),
        "json_ratio": _ratio(compressed_json_bytes, original_json_bytes),
        "original_binary_bytes": original_bin.stat().st_size,
        "compressed_binary_bytes": compressed_bin.stat().st_size,
        "original_binary_human": _format_size(original_bin.stat().st_size),
        "compressed_binary_human": _format_size(compressed_bin.stat().st_size),
        "binary_ratio": _ratio(compressed_bin.stat().st_size, original_bin.stat().st_size),
    }

    compressed_bin.unlink()
    original_bin.unlink()

    _METRICS[stem] = metrics
    _write_metrics()

    print(
        f"{stem}: "
        f"memory={_format_size(metrics['compressed_memory_bytes'])}/"
        f"{_format_size(metrics['original_memory_bytes'])} "
        f"({metrics['memory_ratio']:.2%}), "
        f"json={_format_size(metrics['compressed_json_bytes'])}/"
        f"{_format_size(metrics['original_json_bytes'])} "
        f"({metrics['json_ratio']:.2%}), "
        f"binary(msgpack)={_format_size(metrics['compressed_binary_bytes'])}/"
        f"{_format_size(metrics['original_binary_bytes'])} "
        f"({metrics['binary_ratio']:.2%})"
    )
    return metrics


def _roundtrip_groups(
    groups: list[Any],
    compress_fn: Callable[[Any], Any],
    decompress_fn: Callable[[Any, int], Any],
) -> list[dict[str, Any]]:
    reports = []
    for idx, group in enumerate(groups):
        compressed = compress_fn(copy.deepcopy(group))
        restored = [decompress_fn(compressed, i) for i in range(len(group))]
        assert restored == group, f"group {idx} mismatch"
        reports.append({"original": group, "compressed": compressed})
    return reports


def _roundtrip_name_groups(name_groups: list[list[list[int]]], patterns: list[str]) -> None:
    reports = []
    for idx, (group, pattern) in enumerate(zip(name_groups, patterns)):
        compressed_names, compressed_pattern = SLP.compress_names(copy.deepcopy(group), pattern)
        restored = [
            SLP.decompress_names(compressed_names, compressed_pattern, i)
            for i in range(len(group))
        ]
        assert restored == [
            _format_name(pattern, name_nums)
            for name_nums in group
        ], f"name group {idx} mismatch"
        reports.append(
            {
                "original": {"pattern": pattern, "values": group},
                "compressed": {"pattern": compressed_pattern, "values": compressed_names},
            }
        )

    original_obj = [item["original"] for item in reports]
    compressed_obj = [item["compressed"] for item in reports]
    _measure_sizes("name_nums", original_obj, compressed_obj)


def _format_name(pattern: str, nums: list[int]) -> str:
    result = []
    it = iter(nums)
    for ch in pattern:
        if ch == "0":
            result.append(str(next(it)))
        else:
            result.append(ch)
    return "".join(result)


def test_extract_slp_dataset(tmp_path: Path) -> None:
    dataset = _load_dataset()

    assert dataset["group_count"] > 0
    assert len(dataset["ts"]) == dataset["group_count"]
    assert len(dataset["name_nums"]) == dataset["group_count"]
    assert len(dataset["args"]) == dataset["group_count"]

    output = write_slp_dataset(str(TRACE_PATH), str(DATASET_PATH))
    assert output.exists()

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["group_count"] == dataset["group_count"]


def test_slp_ts_roundtrip_and_compression(tmp_path: Path) -> None:
    dataset = _load_dataset()
    groups = [group for group in dataset["ts"] if group]

    reports = _roundtrip_groups(
        groups,
        SLP.segment_linear_compress,
        SLP.decompress_linear_segment,
    )
    _measure_sizes("ts", [item["original"] for item in reports], [item["compressed"] for item in reports])


def test_slp_dur_roundtrip_and_compression(tmp_path: Path) -> None:
    dataset = _load_dataset()
    groups = [group for group in dataset["dur"] if group]

    reports = _roundtrip_groups(
        groups,
        SLP.segment_linear_compress,
        SLP.decompress_linear_segment,
    )
    _measure_sizes("dur", [item["original"] for item in reports], [item["compressed"] for item in reports])


def test_slp_args_roundtrip_and_compression(tmp_path: Path) -> None:
    dataset = _load_dataset()
    groups = [group for group in dataset["args"] if group]

    reports = []
    for idx, group in enumerate(groups):
        compressed = copy.deepcopy(group)
        SLP.compress_same_args(compressed)
        restored = [SLP.decompress_same_args(compressed, i) for i in range(_group_len(group))]
        original = [SLP.decompress_same_args(group, i) for i in range(_group_len(group))]
        assert restored == original, f"args group {idx} mismatch"
        reports.append({"original": group, "compressed": compressed})

    _measure_sizes("args", [item["original"] for item in reports], [item["compressed"] for item in reports])


def test_slp_name_roundtrip_and_compression(tmp_path: Path) -> None:
    dataset = _load_dataset()
    _roundtrip_name_groups(dataset["name_nums"], dataset["name_pattern"])


def test_slp_id_roundtrip_and_compression(tmp_path: Path) -> None:
    dataset = _load_dataset()
    groups = [group for group in dataset["id"] if group]
    groups.append(["f100", "f101", "f102", "f120"])

    reports = _roundtrip_groups(
        groups,
        SLP.compress_ids,
        SLP.decompress_ids,
    )
    _measure_sizes("id", [item["original"] for item in reports], [item["compressed"] for item in reports])


def test_merge_event_prefixed_id_roundtrip() -> None:
    events = [
        Event({"name": "foo", "ts": 1, "dur": 3, "id": "f100"}),
        Event({"name": "foo", "ts": 2, "dur": 4, "id": "f101"}),
    ]

    merged = MergeEvent(events)
    merged.compress_values()
    restored = MergeEvent.from_dict(to_json_safe(merged.to_dict()))

    assert restored.get_event_by_index(0).id == "f100"
    assert restored.get_event_by_index(1).id == "f101"


def _group_len(group: dict[str, Any]) -> int:
    for value in group.values():
        if isinstance(value, dict):
            nested = _group_len(value)
            if nested:
                return nested
        elif isinstance(value, list):
            if not value:
                continue
            first = value[0]
            if isinstance(first, dict):
                nested = _group_len(first)
                if nested:
                    return nested
                continue
            if isinstance(first, list):
                nested = _group_len({"_": first})
                if nested:
                    return nested
                continue
            return len(value)
        elif hasattr(value, "shape"):
            return len(value)
    return 0
