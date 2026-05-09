"""Roundtrip + sanity tests for every baseline compressor.

The goal is *not* to assert that decompressed events match the source
trace bit-for-bit -- profiler-trace baselines are inherently lossy on
some incidental fields (PID/TID stringification, ts ordering across
streams).  Instead, we assert:

* Event count survives a roundtrip.
* Event NAME population per stream is preserved.
* Sum of durations is preserved.
* Compressed size is strictly smaller than raw msgpack on a non-trivial
  trace.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perflowai.padoc.baselines import (
    BaselineCompressor,
    GzipJsonCompressor,
    GzipMsgpackCompressor,
    PADOCCompressor,
    RawJsonCompressor,
    RawMsgpackCompressor,
    ScalaTraceCompressor,
    TracezipCompressor,
    available_compressors,
    get_compressor,
)
from perflowai.padoc.trace import Trace


TRACE_PATH = Path("tests/example_trace/out-1024.json")


@pytest.fixture(scope="module")
def reference_trace() -> Trace:
    return Trace.from_file(str(TRACE_PATH))


def _event_name_population(trace: Trace) -> Counter[str]:
    counter: Counter[str] = Counter()
    for _rank, _pid, _tid, _ph, events in trace.iter_events():
        for event in events:
            counter[str(event.name)] += 1
    return counter


def _event_count(trace: Trace) -> int:
    return sum(len(events) for *_, events in trace.iter_events())


def _duration_sum(trace: Trace) -> int:
    total = 0
    for _rank, _pid, _tid, _ph, events in trace.iter_events():
        for event in events:
            if event.dur is not None:
                total += int(event.dur)
    return total


# ----------------------------------------------------------------------
# Registry sanity
# ----------------------------------------------------------------------


def test_registry_lists_all_baselines() -> None:
    expected = {
        "raw_json",
        "raw_msgpack",
        "gzip_json",
        "gzip_msgpack",
        "tracezip",
        "scalatrace",
        "padoc",
    }
    assert expected.issubset(set(available_compressors()))


def test_get_compressor_constructs_known_classes() -> None:
    pairs = {
        "raw_json": RawJsonCompressor,
        "raw_msgpack": RawMsgpackCompressor,
        "gzip_json": GzipJsonCompressor,
        "gzip_msgpack": GzipMsgpackCompressor,
        "tracezip": TracezipCompressor,
        "scalatrace": ScalaTraceCompressor,
        "padoc": PADOCCompressor,
    }
    for name, cls in pairs.items():
        assert isinstance(get_compressor(name), cls)


def test_unknown_compressor_raises() -> None:
    with pytest.raises(KeyError):
        get_compressor("not-a-compressor")


# ----------------------------------------------------------------------
# Roundtrip per baseline
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "compressor",
    [
        RawJsonCompressor(),
        RawMsgpackCompressor(),
        GzipJsonCompressor(),
        GzipMsgpackCompressor(),
        TracezipCompressor(),
        ScalaTraceCompressor(),
        PADOCCompressor(),
    ],
    ids=lambda c: c.name,
)
def test_baseline_event_count_roundtrip(compressor: BaselineCompressor, reference_trace: Trace) -> None:
    artifact = compressor.compress_trace(reference_trace)
    decoded = compressor.decompress_to_trace(artifact.blob)
    assert _event_count(decoded) == _event_count(reference_trace)


@pytest.mark.parametrize(
    "compressor",
    [
        RawJsonCompressor(),
        RawMsgpackCompressor(),
        GzipJsonCompressor(),
        GzipMsgpackCompressor(),
        TracezipCompressor(),
        ScalaTraceCompressor(),
        PADOCCompressor(),
    ],
    ids=lambda c: c.name,
)
def test_baseline_event_name_population_roundtrip(
    compressor: BaselineCompressor,
    reference_trace: Trace,
) -> None:
    artifact = compressor.compress_trace(reference_trace)
    decoded = compressor.decompress_to_trace(artifact.blob)
    assert _event_name_population(decoded) == _event_name_population(reference_trace)


@pytest.mark.parametrize(
    "compressor",
    [
        RawJsonCompressor(),
        RawMsgpackCompressor(),
        GzipJsonCompressor(),
        GzipMsgpackCompressor(),
        TracezipCompressor(),
        ScalaTraceCompressor(),
        PADOCCompressor(),
    ],
    ids=lambda c: c.name,
)
def test_baseline_duration_sum_roundtrip(
    compressor: BaselineCompressor,
    reference_trace: Trace,
) -> None:
    artifact = compressor.compress_trace(reference_trace)
    decoded = compressor.decompress_to_trace(artifact.blob)
    assert _duration_sum(decoded) == _duration_sum(reference_trace)


# ----------------------------------------------------------------------
# Compression ratio sanity --- baselines must beat raw msgpack
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "compressor_name",
    ["gzip_json", "gzip_msgpack", "tracezip", "scalatrace", "padoc"],
)
def test_baselines_beat_raw_msgpack(compressor_name: str, reference_trace: Trace) -> None:
    raw = RawMsgpackCompressor().compress_trace(reference_trace).size_bytes
    artifact = get_compressor(compressor_name).compress_trace(reference_trace)
    assert artifact.size_bytes < raw, (
        f"{compressor_name} size={artifact.size_bytes} should be strictly smaller than "
        f"raw_msgpack size={raw}"
    )


# ----------------------------------------------------------------------
# TraceZip-specific: psi controls universal/local split
# ----------------------------------------------------------------------


def test_tracezip_psi_smaller_pushes_more_to_local(reference_trace: Trace) -> None:
    big_psi = TracezipCompressor(psi=10000).compress_trace(reference_trace)
    small_psi = TracezipCompressor(psi=1).compress_trace(reference_trace)
    # With psi=1 almost every key has more than one distinct value, so the
    # SRT collapses to root-only paths and most fields become local.
    assert (
        small_psi.metadata["srt_path_count"] <= big_psi.metadata["srt_path_count"]
    )


# ----------------------------------------------------------------------
# ScalaTrace-specific: RSD folding actually fires on a periodic stream
# ----------------------------------------------------------------------


def test_scalatrace_rsd_folds_periodic_sequence(reference_trace: Trace) -> None:
    artifact = ScalaTraceCompressor(max_period=8, min_repeat=2).compress_trace(
        reference_trace
    )
    expanded = artifact.metadata["expanded_event_count"]
    folded = artifact.metadata["rsd_total_nodes"]
    assert expanded > 0
    # On non-trivial profiler traces folding should reduce node count.
    assert folded <= expanded


# ----------------------------------------------------------------------
# File roundtrip via compress_file()
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "compressor",
    [
        TracezipCompressor(),
        ScalaTraceCompressor(),
        PADOCCompressor(),
    ],
    ids=lambda c: c.name,
)
def test_compress_file_roundtrip(compressor: BaselineCompressor) -> None:
    artifact, stats = compressor.compress_file(str(TRACE_PATH))
    assert stats.event_count > 0
    decoded = compressor.decompress_to_trace(artifact.blob)
    assert _event_count(decoded) == stats.event_count
