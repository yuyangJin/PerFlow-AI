from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from perflowai.padoc.event import Event, is_same_event
from perflowai.padoc.trace import Trace
from perflowai.padoc.utils import to_json_safe


def _parse_name(name: str) -> tuple[str, List[int]]:
    nums: List[int] = []
    pattern_parts: List[str] = []
    last_idx = 0

    for match in re.finditer(r"\d+", name):
        start, end = match.start(), match.end()
        pattern_parts.append(name[last_idx:start])

        digit_seq = match.group()
        i = 0
        while i < len(digit_seq):
            if digit_seq[i] == "0":
                nums.append(0)
                pattern_parts.append("0")
                i += 1
            else:
                nums.append(int(digit_seq[i:]))
                pattern_parts.append("0")
                break

        last_idx = end

    pattern_parts.append(name[last_idx:])
    return "".join(pattern_parts), nums


def _append_arg(dst: Any, src: Any) -> Any:
    """Append one raw arg value into the grouped args structure."""
    if isinstance(src, dict):
        grouped = {} if dst is None else dst
        for key, value in src.items():
            grouped[key] = _append_arg(grouped.get(key), value)
        return grouped

    if isinstance(src, list):
        grouped = [] if dst is None else dst
        if len(grouped) < len(src):
            grouped.extend([None] * (len(src) - len(grouped)))
        for index, value in enumerate(src):
            grouped[index] = _append_arg(grouped[index], value)
        return grouped

    grouped = [] if dst is None else dst
    grouped.append(src)
    return grouped


def _add_args(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key, value in src.items():
        dst[key] = _append_arg(dst.get(key), value)


def _new_group(event: Event, rank: str, pid: str, tid: str, ph: str) -> Dict[str, Any]:
    name_pattern, name_nums = _parse_name(event.name)
    group = {
        "meta": {
            "rank": rank,
            "pid": pid,
            "tid": tid,
            "ph": ph,
            "name_pattern": name_pattern,
            "cat": event.cat,
            "bp": event.bp,
            "s": event.s,
        },
        "rep": event,
        "name_nums": [name_nums],
        "ts": [event.ts],
        "dur": [],
        "id": [],
        "args": None,
    }
    if event.dur is not None:
        group["dur"].append(event.dur)
    if event.id is not None:
        group["id"].append(event.id)
    if event.args is not None:
        group["args"] = {}
        _add_args(group["args"], event.args)
    return group


def build_slp_dataset(trace_path: str) -> Dict[str, Any]:
    trace = Trace.from_file(trace_path)
    grouped_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rank, pid, tid, ph, events in trace.iter_events():
        for event in events:
            normalized_name = re.sub(r"\d+", "", event.get_name())
            candidates = grouped_by_name[normalized_name]
            matched = None
            for group in candidates:
                if is_same_event(group["rep"], event):
                    matched = group
                    break

            if matched is None:
                matched = _new_group(event, rank, pid, tid, ph)
                candidates.append(matched)
                continue

            _, name_nums = _parse_name(event.name)
            matched["name_nums"].append(name_nums)
            matched["ts"].append(event.ts)
            if event.dur is not None:
                matched["dur"].append(event.dur)
            if event.id is not None:
                matched["id"].append(event.id)
            if event.args is not None:
                if matched["args"] is None:
                    matched["args"] = {}
                _add_args(matched["args"], event.args)

    groups = [group for same_name_groups in grouped_by_name.values() for group in same_name_groups]

    return {
        "source_trace": str(trace_path),
        "group_count": len(groups),
        "meta": [group["meta"] for group in groups],
        "name_nums": [group["name_nums"] for group in groups],
        "name_pattern": [group["meta"]["name_pattern"] for group in groups],
        "ts": [group["ts"] for group in groups],
        "dur": [group["dur"] for group in groups],
        "id": [group["id"] for group in groups],
        "args": [group["args"] for group in groups],
    }


def write_slp_dataset(trace_path: str, output_path: str) -> Path:
    dataset = build_slp_dataset(trace_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_json_safe(dataset), indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract grouped PADOC event fields for SLP tests.")
    parser.add_argument("--input-file", required=True, help="Input trace JSON/BIN file.")
    parser.add_argument("--output-file", required=True, help="Output JSON path.")
    args = parser.parse_args()

    output = write_slp_dataset(args.input_file, args.output_file)
    print(output)


if __name__ == "__main__":
    main()
