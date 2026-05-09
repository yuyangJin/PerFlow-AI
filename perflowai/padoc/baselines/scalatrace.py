"""ScalaTrace baseline adapted to AI profiler traces.

ScalaTrace (Noeth et al., SC'07; Ratn et al., 2008) compresses MPI call
sequences using Regular Section Descriptors (RSD) and their nested form
PRSD.  We adapt it to chrome-trace style profiler events as follows:

* For every ``(rank, pid, tid, ph)`` stream we project events into a
  sequence of *event templates*, where one template = one
  ``(name, set(args.keys), cat, bp, s)`` signature.  This lets us detect
  loops at the event-call level (e.g. one transformer layer = same
  template sequence repeated many times across micro-batches).
* The template sequence is then folded with a greedy RSD detector that
  finds, at each position, the period that maximizes the number of
  consecutive repeats.  Repeats are stored as ``RSD(period, count)``.
* The fold step is applied iteratively to produce PRSDs (loops of
  loops).
* Per-event payload (ts, dur, id, args values) is stored as parallel
  arrays; we store *all* values verbatim and let zstd handle numeric
  redundancy.  We deliberately do *not* invoke PADOC's SLP -- this keeps
  the baseline honest.

Final blob = ``msgpack`` -> optional ``zstd``.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import msgpack

try:
    import zstandard as zstd  # type: ignore

    def _final_compress(data: bytes) -> bytes:
        return zstd.ZstdCompressor(level=10).compress(data)

    def _final_decompress(data: bytes) -> bytes:
        return zstd.ZstdDecompressor().decompress(data)

except ImportError:  # pragma: no cover
    import gzip

    def _final_compress(data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=6)

    def _final_decompress(data: bytes) -> bytes:
        return gzip.decompress(data)

from ..event import Event
from ..trace import Trace
from ..utils import to_json_safe
from .base import BaselineCompressor, CompressArtifact, register_compressor


_DIGIT_RE = re.compile(r"\d+")


def _normalize_name(name: str) -> str:
    """Collapse digit runs in event names so closely-related ops share a template."""
    return _DIGIT_RE.sub("0", name)


def _template_signature(event: Event) -> Tuple[Any, ...]:
    """The fingerprint identifying a ScalaTrace template.

    Two events with the same fingerprint are considered "the same call"
    and become the same RSD token.  We keep the args *keyset* in the
    fingerprint (paper's ``MPI_Call`` already includes argument count);
    args *values* are emitted as per-instance payload.
    """
    args_keys: Tuple[str, ...] = ()
    if event.args is not None:
        args_keys = tuple(sorted(str(k) for k in event.args.keys()))
    return (
        _normalize_name(event.name),
        event.cat,
        event.bp,
        event.s,
        args_keys,
    )


# ----------------------------------------------------------------------
# RSD / PRSD data structures
# ----------------------------------------------------------------------


@dataclass
class _Literal:
    """Single-event leaf inside the (P)RSD tree."""

    template_id: int


@dataclass
class _RSD:
    """Periodic block: ``body`` repeated ``count`` times."""

    body: List["_Node"]
    count: int


_Node = Union[_Literal, _RSD]


def _node_template_count(node: _Node) -> int:
    """Number of literal leaves represented by a single (P)RSD subtree."""
    if isinstance(node, _Literal):
        return 1
    return node.count * sum(_node_template_count(child) for child in node.body)


def _flatten_template_sequence(nodes: Iterable[_Node]) -> List[int]:
    """Re-expand a (P)RSD forest into a flat template-id list."""
    out: List[int] = []
    for node in nodes:
        if isinstance(node, _Literal):
            out.append(node.template_id)
        else:
            for _ in range(node.count):
                out.extend(_flatten_template_sequence(node.body))
    return out


# ----------------------------------------------------------------------
# Greedy RSD folding
# ----------------------------------------------------------------------


def _fold_one_pass(
    sequence: List[int],
    max_period: int,
    min_repeat: int,
) -> List[_Node]:
    """One greedy pass of RSD folding over a flat template-id sequence.

    At each position we scan periods ``1..max_period`` and pick the one
    that maximizes the number of consecutive repeats.  Ties are broken by
    longer total span (so RSDs swallow more events).
    """
    out: List[_Node] = []
    i = 0
    n = len(sequence)
    while i < n:
        best_period = 0
        best_count = 0
        best_span = 0

        max_p = min(max_period, (n - i) // min_repeat)
        for period in range(1, max_p + 1):
            count = 1
            j = i + period
            while j + period <= n and sequence[j : j + period] == sequence[i : i + period]:
                count += 1
                j += period
            if count < min_repeat:
                continue
            span = period * count
            if span > best_span:
                best_span = span
                best_period = period
                best_count = count

        if best_period > 0 and best_count >= min_repeat:
            body = [_Literal(template_id=tid) for tid in sequence[i : i + best_period]]
            out.append(_RSD(body=body, count=best_count))
            i += best_period * best_count
        else:
            out.append(_Literal(template_id=sequence[i]))
            i += 1
    return out


def _fold_prsd(
    sequence: List[int],
    max_period: int = 16,
    min_repeat: int = 2,
    max_levels: int = 3,
) -> List[_Node]:
    """Iteratively fold RSDs into PRSDs.

    Convergence: stop when one pass produces a forest that is identical
    to the previous one.  Bound the number of nesting levels with
    ``max_levels`` to keep compress time predictable on long sequences.
    """
    if not sequence:
        return []

    # Level 1: fold the raw integer sequence.
    folded = _fold_one_pass(sequence, max_period, min_repeat)
    for _level in range(max_levels - 1):
        # Treat each top-level node as an integer token by hashing its
        # canonical form, then fold those tokens.
        canonical_ids = [_node_canonical_id(node) for node in folded]
        next_folded = _fold_one_pass(canonical_ids, max_period, min_repeat)
        if _forest_shape(next_folded) == _forest_shape(folded):
            break
        # Re-attach: each top-level _Literal's template_id is a synthetic
        # canonical id; resolve it back to the original subtree.
        id_to_node: Dict[int, _Node] = {}
        for node, canonical in zip(folded, canonical_ids):
            id_to_node[canonical] = node
        rebuilt: List[_Node] = []
        for node in next_folded:
            rebuilt.append(_resolve_canonical(node, id_to_node))
        folded = rebuilt
    return folded


def _forest_shape(forest: List[_Node]) -> Tuple[Any, ...]:
    return tuple(_node_canonical_id(node) for node in forest)


_CANONICAL_CACHE: Dict[Any, int] = {}


def _node_canonical_id(node: _Node) -> int:
    """Stable integer id assigned to each unique (P)RSD subtree."""
    if isinstance(node, _Literal):
        key = ("L", node.template_id)
    else:
        key = ("R", node.count, tuple(_node_canonical_id(child) for child in node.body))
    cached = _CANONICAL_CACHE.get(key)
    if cached is None:
        cached = len(_CANONICAL_CACHE)
        _CANONICAL_CACHE[key] = cached
    return cached


def _resolve_canonical(node: _Node, id_to_node: Dict[int, _Node]) -> _Node:
    if isinstance(node, _Literal):
        if node.template_id in id_to_node:
            return id_to_node[node.template_id]
        return node
    return _RSD(
        body=[_resolve_canonical(child, id_to_node) for child in node.body],
        count=node.count,
    )


# ----------------------------------------------------------------------
# Compressor
# ----------------------------------------------------------------------


@register_compressor
class ScalaTraceCompressor(BaselineCompressor):
    """RSD/PRSD-style sequence compression for AI traces."""

    name = "scalatrace"
    supports_in_situ_analysis = False

    def __init__(
        self,
        max_period: int = 16,
        min_repeat: int = 2,
        max_levels: int = 3,
        post_zstd: bool = True,
    ) -> None:
        self.max_period = max_period
        self.min_repeat = min_repeat
        self.max_levels = max_levels
        self.post_zstd = post_zstd

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress_trace(self, trace: Trace) -> CompressArtifact:
        compress_start = time.perf_counter()

        templates: List[Dict[str, Any]] = []
        signature_to_id: Dict[Tuple[Any, ...], int] = {}

        per_stream_records: Dict[str, Dict[str, Any]] = {}

        for rank, pid, tid, ph, events in trace.iter_events():
            template_seq: List[int] = []
            payload_names: List[str] = []
            payload_ts: List[int] = []
            payload_dur: List[Optional[int]] = []
            payload_id: List[Optional[Any]] = []
            payload_args: List[List[Any]] = []  # per-event values aligned to template's args_keys

            for event in events:
                signature = _template_signature(event)
                tid_index = signature_to_id.get(signature)
                if tid_index is None:
                    tid_index = len(signature_to_id)
                    signature_to_id[signature] = tid_index
                    templates.append(
                        {
                            "name_pattern": _normalize_name(event.name),
                            "cat": event.cat,
                            "bp": event.bp,
                            "s": event.s,
                            "args_keys": list(signature[4]),
                        }
                    )

                template_seq.append(tid_index)
                payload_names.append(event.name)
                payload_ts.append(int(event.ts))
                payload_dur.append(None if event.dur is None else int(event.dur))
                payload_id.append(None if event.id is None else to_json_safe(event.id))

                tmpl = templates[tid_index]
                if event.args is not None and tmpl["args_keys"]:
                    payload_args.append(
                        [to_json_safe(event.args.get(k)) for k in tmpl["args_keys"]]
                    )
                else:
                    payload_args.append([])

            folded = _fold_prsd(
                template_seq,
                max_period=self.max_period,
                min_repeat=self.min_repeat,
                max_levels=self.max_levels,
            )

            per_stream_records["::".join((str(rank), str(pid), str(tid), str(ph)))] = {
                "rsd": _serialize_forest(folded),
                "names": payload_names,
                "ts": payload_ts,
                "dur": payload_dur,
                "id": payload_id,
                "args": payload_args,
                "expanded_length": len(template_seq),
            }

        payload: Dict[str, Any] = {
            "templates": templates,
            "metadata": dict(trace.get_metadata()),
            "rank_start_timestamp": dict(trace.get_start_time()),
            "streams": per_stream_records,
            "config": {
                "max_period": self.max_period,
                "min_repeat": self.min_repeat,
                "max_levels": self.max_levels,
            },
        }

        blob = msgpack.packb(payload, use_bin_type=True)
        if self.post_zstd:
            blob = _final_compress(blob)

        elapsed = time.perf_counter() - compress_start
        meta = {
            "template_count": len(templates),
            "stream_count": len(per_stream_records),
            "post_zstd": self.post_zstd,
            "rsd_total_nodes": sum(
                _serialized_size_count(stream["rsd"]) for stream in per_stream_records.values()
            ),
            "expanded_event_count": sum(
                stream["expanded_length"] for stream in per_stream_records.values()
            ),
        }
        return CompressArtifact(blob=blob, metadata=meta, compress_seconds=elapsed)

    # ------------------------------------------------------------------
    # Decompression
    # ------------------------------------------------------------------

    def decompress_to_trace(self, blob: bytes) -> Trace:
        if self.post_zstd:
            blob = _final_decompress(blob)
        payload = msgpack.unpackb(blob, raw=False, strict_map_key=False)

        templates = payload.get("templates", [])
        rank_start_timestamp = payload.get("rank_start_timestamp", {}) or {}
        events_per_rank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for stream_key, stream in (payload.get("streams") or {}).items():
            rank, pid, tid, ph = stream_key.split("::")
            forest = _deserialize_forest(stream.get("rsd", []))
            template_ids = _flatten_template_sequence(forest)

            name_list = stream.get("names", [])
            ts_list = stream.get("ts", [])
            dur_list = stream.get("dur", [])
            id_list = stream.get("id", [])
            args_list = stream.get("args", [])

            assert len(template_ids) == len(ts_list), (
                f"RSD expansion mismatch: {len(template_ids)} vs {len(ts_list)}"
            )
            assert len(template_ids) == len(name_list), (
                f"RSD/name mismatch: {len(template_ids)} vs {len(name_list)}"
            )

            base_ts = int(rank_start_timestamp.get(rank, 0))
            normalized_tid = (
                tid.split(" ")[1] if isinstance(tid, str) and tid.startswith("stream ") else tid
            )

            for index, template_id in enumerate(template_ids):
                tmpl = templates[template_id]
                event_dict: Dict[str, Any] = {
                    "name": name_list[index],
                    "ts": int(ts_list[index]) + base_ts,
                    "pid": pid,
                    "tid": normalized_tid,
                    "ph": ph,
                }
                if tmpl.get("cat") is not None:
                    event_dict["cat"] = tmpl["cat"]
                if tmpl.get("bp") is not None:
                    event_dict["bp"] = tmpl["bp"]
                if tmpl.get("s") is not None:
                    event_dict["s"] = tmpl["s"]
                if dur_list[index] is not None:
                    event_dict["dur"] = int(dur_list[index])
                if id_list[index] is not None:
                    event_dict["id"] = id_list[index]
                if tmpl.get("args_keys"):
                    args_values = args_list[index]
                    event_dict["args"] = {
                        key: args_values[i] for i, key in enumerate(tmpl["args_keys"])
                    }
                events_per_rank[rank].append(event_dict)

        return Trace(
            events=dict(events_per_rank),
            metadata=dict(payload.get("metadata", {}) or {}),
            start_timestamp=dict(rank_start_timestamp),
        )


# ----------------------------------------------------------------------
# Forest serialization
# ----------------------------------------------------------------------


def _serialize_forest(forest: List[_Node]) -> List[Any]:
    out: List[Any] = []
    for node in forest:
        if isinstance(node, _Literal):
            out.append(["L", int(node.template_id)])
        else:
            out.append(["R", int(node.count), _serialize_forest(node.body)])
    return out


def _deserialize_forest(raw: List[Any]) -> List[_Node]:
    out: List[_Node] = []
    for entry in raw:
        if entry[0] == "L":
            out.append(_Literal(template_id=int(entry[1])))
        elif entry[0] == "R":
            out.append(_RSD(body=_deserialize_forest(entry[2]), count=int(entry[1])))
        else:
            raise ValueError(f"Unknown forest node tag: {entry[0]!r}")
    return out


def _serialized_size(forest: List[_Node]) -> int:
    """Count storage units for the forest (used for stats only)."""
    return _serialized_size_count(_serialize_forest(forest))


def _serialized_size_count(raw: List[Any]) -> int:
    count = 0
    for entry in raw:
        count += 1
        if entry[0] == "R":
            count += _serialized_size_count(entry[2])
    return count
