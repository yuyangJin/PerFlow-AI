"""Common abstraction shared by every baseline compressor.

The bench harness only ever sees ``BaselineCompressor`` instances.  Each
implementation must provide a roundtrip pair (``compress_trace`` /
``decompress_to_trace``) plus optional in-situ analysis support.

Design notes
------------
* ``compress_trace`` always returns a :class:`CompressArtifact` so the
  harness can inspect compressor-internal stats (template counts, SRT
  size, dictionary size, ...) without re-parsing the blob.
* The blob is a single ``bytes`` buffer.  Sub-stream layout (rank-by-rank,
  whole-directory) is encoded inside the blob; the harness deals only
  with files.  This keeps the storage measurement comparable.
* ``compress_file`` / ``compress_directory`` are convenience wrappers that
  reuse the file loading code of :class:`Trace`.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from ..trace import Trace, TraceLoadStats


@dataclass
class CompressArtifact:
    """Output of one compression call."""

    blob: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Compressor-specific stats: template count, SRT size, etc."""

    compress_seconds: float = 0.0

    @property
    def size_bytes(self) -> int:
        return len(self.blob)


class BaselineCompressor(ABC):
    """Common interface for every paper-level compressor."""

    name: str = "base"

    #: ``True`` only for compressors that can run analyses without first
    #: rebuilding a raw :class:`Trace`.  Currently only PADOC supports
    #: this; everything else has to ``decompress_to_trace`` first.
    supports_in_situ_analysis: bool = False

    @abstractmethod
    def compress_trace(self, trace: Trace) -> CompressArtifact:
        """Compress one already-loaded :class:`Trace` into a blob."""

    @abstractmethod
    def decompress_to_trace(self, blob: bytes) -> Trace:
        """Reconstruct a :class:`Trace` equivalent to the original."""

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def compress_file(self, path: str) -> Tuple[CompressArtifact, TraceLoadStats]:
        """Load one trace file and compress it."""
        load_start = time.perf_counter()
        load_result = Trace.from_file_with_stats(path)
        load_seconds = time.perf_counter() - load_start

        artifact = self.compress_trace(load_result.trace)
        artifact.metadata.setdefault("load_seconds", load_seconds)
        return artifact, load_result.stats

    def compress_directory(
        self,
        directory: str,
    ) -> Tuple[CompressArtifact, TraceLoadStats]:
        """Load and compress an entire trace directory.

        The default implementation builds one big :class:`Trace` and runs
        :meth:`compress_trace` on it.  Streaming-friendly subclasses may
        override this method.
        """
        load_start = time.perf_counter()
        load_result = Trace.from_dir_with_stats(directory)
        load_seconds = time.perf_counter() - load_start

        artifact = self.compress_trace(load_result.trace)
        artifact.metadata.setdefault("load_seconds", load_seconds)
        return artifact, load_result.stats

    # ------------------------------------------------------------------
    # File-system helpers
    # ------------------------------------------------------------------

    def write_blob(self, blob: bytes, path: str) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(blob)
        return len(blob)

    @staticmethod
    def read_blob(path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()


# ----------------------------------------------------------------------
# Registry --- bench harness uses this to look up compressors by name.
# ----------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BaselineCompressor]] = {}


def register_compressor(cls: Type[BaselineCompressor]) -> Type[BaselineCompressor]:
    """Decorator that registers a compressor class by its ``name`` attribute."""
    if not issubclass(cls, BaselineCompressor):
        raise TypeError(f"{cls!r} must subclass BaselineCompressor")
    if not cls.name or cls.name == "base":
        raise ValueError(f"{cls!r} must define a unique non-'base' name")
    _REGISTRY[cls.name] = cls
    return cls


def get_compressor(name: str, **kwargs: Any) -> BaselineCompressor:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown compressor {name!r}. Available: {available}")
    return _REGISTRY[name](**kwargs)


def available_compressors() -> List[str]:
    return sorted(_REGISTRY)


# ----------------------------------------------------------------------
# Lightweight roundtrip helper
# ----------------------------------------------------------------------


def trace_event_count(trace: Trace) -> int:
    return sum(
        len(events)
        for _rank, _pid, _tid, _ph, events in trace.iter_events()
    )


def collect_event_signatures(trace: Trace) -> Iterable[Tuple[str, str, str, str, str, int]]:
    """Yield ``(rank, pid, tid, ph, event_name, ts)`` for every event.

    This is intentionally cheap and is used by tests to confirm that a
    decompressed trace contains the same population of events as the
    original, regardless of in-memory ordering.
    """
    for rank, pid, tid, ph, events in trace.iter_events():
        for event in events:
            yield rank, str(pid), str(tid), str(ph), str(event.name), int(event.ts)
