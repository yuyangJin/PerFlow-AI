"""TraceZip baseline adapted to AI profiler traces.

Implements the Span Retrieval Tree (SRT) compression scheme from
"Tracezip: Efficient Distributed Tracing via Trace Compression"
(Chen et al., ISSTA'25, arXiv:2502.06318).

We treat every PyTorch profiler event as a span:

* ``Event.name``                          -> span Name (groups SRT subtrees)
* ``Event.cat / bp / s / args.*``         -> universal-field candidates
* ``Event.ts / dur / id / high-cardinality args.*`` -> local fields

Implemented design points from the paper:

1. Span Retrieval Tree (SRT) keyed by ``(name, [(field_key, value)])``.
2. SRT restructuring (Section 3.4.1): per-Name fields are reordered
   ascending by distinct-value count; fields whose distinct count exceeds
   ``psi`` are demoted to local fields.
3. Dictionary + key tokenization (Section 3.4.2): universal-field keys
   are split by ``.`` / ``_`` and frequent tokens / values are mapped to
   short alphanumeric ids.
4. Time base (Section 3.3): per stream we re-anchor a ``time_base`` so
   the local ``ts`` field is stored as a small offset.

Deliberately not borrowed from PADOC:

* No structural compression (no SameCPUNode / RefNode / anchor alignment).
* No name-pattern + name-nums column packing.
* No SLP-style numeric column compression.

Final blob = ``msgpack`` -> optional ``zstd``.
"""

from __future__ import annotations

import re
import string
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import msgpack

try:
    import zstandard as zstd  # type: ignore

    def _final_compress(data: bytes) -> bytes:
        return zstd.ZstdCompressor(level=10).compress(data)

    def _final_decompress(data: bytes) -> bytes:
        return zstd.ZstdDecompressor().decompress(data)

except ImportError:  # pragma: no cover - exercised only if dep missing
    import gzip

    def _final_compress(data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=6)

    def _final_decompress(data: bytes) -> bytes:
        return gzip.decompress(data)

from ..event import Event
from ..trace import Trace
from ..utils import to_json_safe
from .base import BaselineCompressor, CompressArtifact, register_compressor


_TOKEN_SPLIT_RE = re.compile(r"([._])")
_ALPHABET = "0123456789" + string.ascii_lowercase + string.ascii_uppercase


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _flatten_args(args: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested args dicts using ``-`` separator (paper convention)."""
    flat: Dict[str, Any] = {}
    if not isinstance(args, dict):
        return flat
    for key, value in args.items():
        compound = f"{prefix}-{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_args(value, compound))
        elif isinstance(value, list):
            try:
                flat[compound] = tuple(value)
            except TypeError:
                flat[compound] = repr(value)
        else:
            flat[compound] = value
    return flat


def _unflatten_args(flat: Dict[str, Any]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for compound, value in flat.items():
        parts = compound.split("-")
        cursor = nested
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        leaf_key = parts[-1]
        if isinstance(value, tuple):
            cursor[leaf_key] = list(value)
        else:
            cursor[leaf_key] = value
    return nested


def _event_to_kv(event: Event) -> Dict[str, Any]:
    """Project one Event into the flat KV bag SRT operates on.

    ``ts``, ``dur``, ``id`` are explicitly *not* included -- they are the
    canonical local fields and are emitted alongside the SRT record.
    """
    kv: Dict[str, Any] = {}
    if event.cat is not None:
        kv["cat"] = event.cat
    if event.bp is not None:
        kv["bp"] = event.bp
    if event.s is not None:
        kv["s"] = event.s
    flat_args = _flatten_args(event.args or {})
    for key, value in flat_args.items():
        kv[f"args.{key}"] = value
    return kv


def _alnum_id(index: int) -> str:
    if index < 0:
        raise ValueError(index)
    base = len(_ALPHABET)
    if index < base:
        return _ALPHABET[index]
    out: List[str] = []
    n = index
    while n > 0:
        out.append(_ALPHABET[n % base])
        n //= base
    return "".join(reversed(out))


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _serialize_local(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_serialize_local(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_local(v) for k, v in value.items()}
    return to_json_safe(value)


def _apply_dict_to_token_string(value: str, dictionary: Dict[str, str]) -> str:
    """Tokenize ``value`` by ``.``/``_`` and replace each token via dict."""
    if not isinstance(value, str) or not value:
        return value
    if value in dictionary:
        return dictionary[value]
    if "." not in value and "_" not in value:
        return value
    parts = _TOKEN_SPLIT_RE.split(value)
    out: List[str] = []
    for part in parts:
        if part in (".", "_"):
            out.append(part)
        else:
            out.append(dictionary.get(part, part))
    return "".join(out)


def _apply_value_dict(value: Any, dictionary: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return _apply_dict_to_token_string(value, dictionary)
    return value


# ----------------------------------------------------------------------
# Per-Name SRT bucket
# ----------------------------------------------------------------------


@dataclass
class _NameBucket:
    name: str
    events: List[Tuple[Tuple[str, str, str, str], Dict[str, Any], Event]] = field(
        default_factory=list
    )
    universal_keys: List[str] = field(default_factory=list)
    local_keys: List[str] = field(default_factory=list)
    field_distinct_count: Dict[str, int] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Compressor
# ----------------------------------------------------------------------


@register_compressor
class TracezipCompressor(BaselineCompressor):
    """SRT-based AI-trace adaptation of TraceZip."""

    name = "tracezip"
    supports_in_situ_analysis = False

    def __init__(
        self,
        psi: int = 1000,
        post_zstd: bool = True,
        time_base_period_ns: int = 1_000_000,
    ) -> None:
        self.psi = psi
        self.post_zstd = post_zstd
        self.time_base_period_ns = time_base_period_ns

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress_trace(self, trace: Trace) -> CompressArtifact:
        compress_start = time.perf_counter()

        per_stream_events = self._collect_events(trace)
        name_buckets = self._group_by_name(per_stream_events)
        for bucket in name_buckets.values():
            self._classify_fields(bucket)

        srt = self._build_srt(name_buckets)
        token_dict = self._build_token_dictionary(srt, name_buckets)
        srt_body = self._serialize_srt(srt, name_buckets, token_dict)
        encoded_streams = self._encode_streams(per_stream_events, name_buckets, srt)

        payload: Dict[str, Any] = {
            "psi": self.psi,
            "metadata": dict(trace.get_metadata()),
            "rank_start_timestamp": dict(trace.get_start_time()),
            "srt": srt_body,
            "token_dict": token_dict,
            "streams": encoded_streams,
        }

        blob = msgpack.packb(payload, use_bin_type=True)
        if self.post_zstd:
            blob = _final_compress(blob)

        elapsed = time.perf_counter() - compress_start
        meta = {
            "psi": self.psi,
            "name_count": len(name_buckets),
            "srt_path_count": sum(len(paths) for paths in srt.values()),
            "token_dict_size": len(token_dict),
            "post_zstd": self.post_zstd,
            "stream_count": len(encoded_streams),
            "event_count": sum(len(events) for events in per_stream_events.values()),
        }
        return CompressArtifact(blob=blob, metadata=meta, compress_seconds=elapsed)

    # ------------------------------------------------------------------
    # Decompression
    # ------------------------------------------------------------------

    def decompress_to_trace(self, blob: bytes) -> Trace:
        if self.post_zstd:
            blob = _final_decompress(blob)
        payload = msgpack.unpackb(blob, raw=False, strict_map_key=False)

        token_dict = payload.get("token_dict", {}) or {}
        reverse_dict = {v: k for k, v in token_dict.items()}

        srt_section = payload.get("srt", {}) or {}
        per_name: Dict[str, Tuple[List[str], List[str], Dict[int, List[Any]]]] = {}
        for name, body in srt_section.items():
            universal_keys = [
                _apply_dict_to_token_string(k, reverse_dict)
                for k in body.get("u_keys", [])
            ]
            local_keys = [
                _apply_dict_to_token_string(k, reverse_dict)
                for k in body.get("l_keys", [])
            ]
            paths: Dict[int, List[Any]] = {}
            for path_id, values in body.get("paths", []):
                decoded = []
                for value in values:
                    if isinstance(value, str):
                        decoded.append(_apply_dict_to_token_string(value, reverse_dict))
                    else:
                        decoded.append(value)
                paths[int(path_id)] = decoded
            per_name[name] = (universal_keys, local_keys, paths)

        events_per_rank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        rank_start_timestamp = payload.get("rank_start_timestamp", {}) or {}

        for stream_key, records in (payload.get("streams") or {}).items():
            rank, pid, tid, ph = stream_key.split("::")
            for record in records:
                (
                    name,
                    path_id,
                    time_base,
                    ts_offset,
                    dur,
                    id_value,
                    local_values,
                ) = record
                universal_keys, local_keys, paths = per_name[name]
                universal_values = paths[int(path_id)]

                kv: Dict[str, Any] = {}
                for key, value in zip(universal_keys, universal_values):
                    kv[key] = value
                for key, value in zip(local_keys, local_values):
                    kv[key] = value

                event_dict: Dict[str, Any] = {
                    "name": name,
                    "ts": int(time_base) + int(ts_offset) + int(rank_start_timestamp.get(rank, 0)),
                    "pid": pid,
                    "tid": tid.split(" ")[1] if tid.startswith("stream ") else tid,
                    "ph": ph,
                }
                if dur is not None:
                    event_dict["dur"] = int(dur)
                if id_value is not None:
                    event_dict["id"] = id_value

                args_flat: Dict[str, Any] = {}
                for key, value in kv.items():
                    if key in ("cat", "bp", "s"):
                        event_dict[key] = value
                    elif key.startswith("args."):
                        args_flat[key[len("args.") :]] = value
                if args_flat:
                    event_dict["args"] = _unflatten_args(args_flat)

                events_per_rank[rank].append(event_dict)

        return Trace(
            events=dict(events_per_rank),
            metadata=dict(payload.get("metadata", {}) or {}),
            start_timestamp=dict(rank_start_timestamp),
        )

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _collect_events(
        self,
        trace: Trace,
    ) -> Dict[Tuple[str, str, str, str], List[Event]]:
        per_stream: Dict[Tuple[str, str, str, str], List[Event]] = defaultdict(list)
        for rank, pid, tid, ph, events in trace.iter_events():
            per_stream[(str(rank), str(pid), str(tid), str(ph))].extend(events)
        return per_stream

    def _group_by_name(
        self,
        per_stream_events: Dict[Tuple[str, str, str, str], List[Event]],
    ) -> Dict[str, _NameBucket]:
        buckets: Dict[str, _NameBucket] = {}
        for stream_key, events in per_stream_events.items():
            for event in events:
                bucket = buckets.get(event.name)
                if bucket is None:
                    bucket = _NameBucket(name=event.name)
                    buckets[event.name] = bucket
                bucket.events.append((stream_key, _event_to_kv(event), event))
        return buckets

    def _classify_fields(self, bucket: _NameBucket) -> None:
        """Restructure (Section 3.4.1) and split universal/local fields."""
        distinct_values: Dict[str, set] = defaultdict(set)
        for _, kv, _ in bucket.events:
            for key, value in kv.items():
                distinct_values[key].add(_hashable(value))

        bucket.field_distinct_count = {k: len(v) for k, v in distinct_values.items()}
        sorted_keys = sorted(
            bucket.field_distinct_count.items(),
            key=lambda item: (item[1], item[0]),
        )
        bucket.universal_keys = [k for k, count in sorted_keys if count <= self.psi]
        bucket.local_keys = [k for k, count in sorted_keys if count > self.psi]

    def _build_srt(
        self,
        name_buckets: Dict[str, _NameBucket],
    ) -> Dict[str, Dict[Tuple[Tuple[str, Any], ...], int]]:
        srt: Dict[str, Dict[Tuple[Tuple[str, Any], ...], int]] = {}
        for name, bucket in name_buckets.items():
            paths: Dict[Tuple[Tuple[str, Any], ...], int] = {}
            for _, kv, _ in bucket.events:
                key_tuple = tuple(
                    (key, _hashable(kv.get(key))) for key in bucket.universal_keys
                )
                if key_tuple not in paths:
                    paths[key_tuple] = len(paths)
            srt[name] = paths
        return srt

    def _build_token_dictionary(
        self,
        srt: Dict[str, Dict[Tuple[Tuple[str, Any], ...], int]],
        name_buckets: Dict[str, _NameBucket],
    ) -> Dict[str, str]:
        """Assign short alphanumeric ids to frequent universal-field tokens."""
        token_counter: Counter[str] = Counter()
        for paths in srt.values():
            for key_tuple in paths:
                for key, value in key_tuple:
                    for token in _TOKEN_SPLIT_RE.split(str(key)):
                        if token and token not in (".", "_"):
                            token_counter[token] += 1
                    if isinstance(value, str) and value:
                        token_counter[value] += 1
        for bucket in name_buckets.values():
            for key in bucket.local_keys:
                for token in _TOKEN_SPLIT_RE.split(str(key)):
                    if token and token not in (".", "_"):
                        token_counter[token] += 1

        dictionary: Dict[str, str] = {}
        ranked = sorted(
            ((count, token) for token, count in token_counter.items() if count >= 2),
            reverse=True,
        )
        for index, (_count, token) in enumerate(ranked):
            short = _alnum_id(index)
            if len(short) >= len(token):
                continue
            dictionary[token] = short
        return dictionary

    def _serialize_srt(
        self,
        srt: Dict[str, Dict[Tuple[Tuple[str, Any], ...], int]],
        name_buckets: Dict[str, _NameBucket],
        token_dict: Dict[str, str],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, paths in srt.items():
            bucket = name_buckets[name]
            tokenized_universal_keys = [
                _apply_dict_to_token_string(k, token_dict) for k in bucket.universal_keys
            ]
            tokenized_local_keys = [
                _apply_dict_to_token_string(k, token_dict) for k in bucket.local_keys
            ]
            serialized_paths: List[List[Any]] = []
            for key_tuple, path_id in paths.items():
                values = [_apply_value_dict(v, token_dict) for _, v in key_tuple]
                serialized_paths.append([path_id, values])
            out[name] = {
                "u_keys": tokenized_universal_keys,
                "l_keys": tokenized_local_keys,
                "paths": serialized_paths,
            }
        return out

    def _encode_streams(
        self,
        per_stream_events: Dict[Tuple[str, str, str, str], List[Event]],
        name_buckets: Dict[str, _NameBucket],
        srt: Dict[str, Dict[Tuple[Tuple[str, Any], ...], int]],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for stream_key, events in per_stream_events.items():
            time_base = 0
            records: List[List[Any]] = []
            for event in events:
                bucket = name_buckets[event.name]
                kv = _event_to_kv(event)
                key_tuple = tuple(
                    (key, _hashable(kv.get(key))) for key in bucket.universal_keys
                )
                path_id = srt[event.name][key_tuple]

                if (
                    self.time_base_period_ns > 0
                    and (event.ts - time_base) >= self.time_base_period_ns
                ):
                    time_base = int(event.ts)
                ts_offset = int(event.ts) - int(time_base)

                local_values = [
                    _serialize_local(kv.get(key)) for key in bucket.local_keys
                ]

                records.append(
                    [
                        event.name,
                        int(path_id),
                        int(time_base),
                        int(ts_offset),
                        None if event.dur is None else int(event.dur),
                        None if event.id is None else _serialize_local(event.id),
                        local_values,
                    ]
                )
            out["::".join(stream_key)] = records
        return out
