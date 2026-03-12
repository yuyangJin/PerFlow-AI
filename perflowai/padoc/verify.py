"""Utilities for validating raw trace equivalence."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import msgpack

from .utils import to_json_safe


def _load_trace_payload(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    with open(path, "rb") as file_obj:
        return msgpack.load(file_obj, strict_map_key=False)


def _normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    normalized = to_json_safe(event)
    for key in ("pid", "tid"):
        value = normalized.get(key)
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
            normalized[key] = int(value)
    return normalized


def _sort_trace_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_events = [_normalize_event(event) for event in events]
    return sorted(
        normalized_events,
        key=lambda event: (
            event.get("ts", 0),
            event.get("ph", ""),
            event.get("name", ""),
            event.get("dur", 0),
            event.get("pid", 0),
            event.get("tid", 0),
        ),
    )


def _normalized_trace_payload(path: str) -> Dict[str, Any]:
    payload = _load_trace_payload(path)
    payload = dict(payload)
    payload["traceEvents"] = _sort_trace_events(payload.get("traceEvents", []))
    return payload


def compare_trace_files(left_path: str, right_path: str) -> bool:
    """Compare two raw trace files after normalizing event order."""
    return compare_trace_files_with_report(left_path, right_path)[0]


def compare_trace_files_with_report(left_path: str, right_path: str) -> tuple[bool, str]:
    """Compare two raw trace files and return a diagnostic message."""
    left_payload = _normalized_trace_payload(left_path)
    right_payload = _normalized_trace_payload(right_path)
    left_events = left_payload.get("traceEvents", [])
    right_events = right_payload.get("traceEvents", [])
    if len(left_events) != len(right_events):
        return (
            False,
            f"event_count mismatch: {len(left_events)} vs {len(right_events)}",
        )
    for index, (left_event, right_event) in enumerate(zip(left_events, right_events)):
        if left_event != right_event:
            return (
                False,
                f"first mismatch at event {index}: {left_event} != {right_event}",
            )
    return True, "events match"


def compare_trace_directories(left_dir: str, right_dir: str) -> bool:
    """Compare two trace directories file by file."""
    return compare_trace_directories_with_report(left_dir, right_dir)[0]


def _directory_trace_index(directory: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for file_name in sorted(
        candidate for candidate in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, candidate))
    ):
        match = re.search(r"(\d+)(?=\.[^.]+$)", file_name)
        key = match.group(1) if match else file_name
        index[key] = os.path.join(directory, file_name)
    return index


def compare_trace_directories_with_report(left_dir: str, right_dir: str) -> tuple[bool, str]:
    """Compare two trace directories and return a diagnostic message."""
    left_index = _directory_trace_index(left_dir)
    right_index = _directory_trace_index(right_dir)
    if sorted(left_index.keys()) != sorted(right_index.keys()):
        return (
            False,
            f"file key mismatch: {sorted(left_index.keys())} vs {sorted(right_index.keys())}",
        )

    for file_key in sorted(left_index.keys()):
        passed, message = compare_trace_files_with_report(
            left_index[file_key],
            right_index[file_key],
        )
        if not passed:
            return False, f"{os.path.basename(left_index[file_key])}: {message}"
    return True, "all files match"
