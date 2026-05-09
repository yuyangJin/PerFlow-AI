"""No-compression reference points (raw JSON / msgpack)."""

from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from typing import Any, Dict, List

import msgpack

from ..trace import Trace
from ..utils import to_json_safe
from .base import BaselineCompressor, CompressArtifact, register_compressor


def _serialize_trace_payload(trace: Trace) -> Dict[str, Any]:
    """Reconstruct a chrome-trace style payload from a Trace.

    The payload is structured as ``{"traceEvents": [...], "metadata": {...}}``
    and is sufficient to round-trip back into a :class:`Trace`.
    """
    events: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    rank_metadata = trace.get_metadata()

    for rank, pid, tid, ph, ev_list in trace.iter_events():
        start_ts = trace.get_start_time().get(rank, 0)
        for event in ev_list:
            event_dict = event.to_dict().copy()
            event_dict["pid"] = pid
            normalized_tid = tid
            if normalized_tid.startswith("stream "):
                normalized_tid = normalized_tid.split(" ", maxsplit=1)[1]
            if normalized_tid.endswith("-abnormal"):
                normalized_tid = normalized_tid.split("-", maxsplit=1)[0]
            event_dict["tid"] = normalized_tid
            event_dict["ph"] = ph
            event_dict["ts"] = event_dict.get("ts", 0) + start_ts
            event_dict.setdefault("rank", rank)
            events.append(to_json_safe(event_dict))

    payload: Dict[str, Any] = {
        "traceEvents": sorted(
            events,
            key=lambda x: (
                x.get("ts", 0),
                x.get("ph", ""),
                x.get("name", ""),
                x.get("dur", 0),
                x.get("rank", ""),
            ),
        ),
    }
    if rank_metadata:
        payload["padoc_metadata"] = rank_metadata
    if trace.get_start_time():
        payload["padoc_start_timestamp"] = dict(trace.get_start_time())
    return payload


def _trace_from_payload(payload: Dict[str, Any]) -> Trace:
    metadata = payload.get("padoc_metadata", {})
    start_timestamp = payload.get("padoc_start_timestamp", {})
    raw_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in payload.get("traceEvents", []):
        rank = str(event.pop("rank", "0"))
        raw_events[rank].append(event)
    return Trace(
        events=dict(raw_events),
        metadata=dict(metadata),
        start_timestamp=dict(start_timestamp),
    )


@register_compressor
class RawJsonCompressor(BaselineCompressor):
    """Serialize the trace as compact JSON; no compression at all."""

    name = "raw_json"

    def compress_trace(self, trace: Trace) -> CompressArtifact:
        start = time.perf_counter()
        payload = _serialize_trace_payload(trace)
        blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        elapsed = time.perf_counter() - start
        return CompressArtifact(
            blob=blob,
            metadata={"event_count": len(payload.get("traceEvents", []))},
            compress_seconds=elapsed,
        )

    def decompress_to_trace(self, blob: bytes) -> Trace:
        payload = json.loads(blob.decode("utf-8"))
        return _trace_from_payload(payload)


@register_compressor
class RawMsgpackCompressor(BaselineCompressor):
    """Serialize the trace as msgpack; no compression beyond binary packing."""

    name = "raw_msgpack"

    def compress_trace(self, trace: Trace) -> CompressArtifact:
        start = time.perf_counter()
        payload = _serialize_trace_payload(trace)
        blob = msgpack.packb(payload, use_bin_type=True)
        elapsed = time.perf_counter() - start
        return CompressArtifact(
            blob=blob,
            metadata={"event_count": len(payload.get("traceEvents", []))},
            compress_seconds=elapsed,
        )

    def decompress_to_trace(self, blob: bytes) -> Trace:
        payload = msgpack.unpackb(blob, raw=False, strict_map_key=False)
        return _trace_from_payload(payload)
