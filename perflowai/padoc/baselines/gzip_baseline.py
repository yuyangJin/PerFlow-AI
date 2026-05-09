"""Generic gzip baselines on top of raw JSON / msgpack payloads."""

from __future__ import annotations

import gzip
import time

from ..trace import Trace
from .base import CompressArtifact, register_compressor
from .raw import RawJsonCompressor, RawMsgpackCompressor


@register_compressor
class GzipJsonCompressor(RawJsonCompressor):
    """Raw chrome trace JSON piped through gzip (level 6)."""

    name = "gzip_json"

    def __init__(self, level: int = 6) -> None:
        self.level = level

    def compress_trace(self, trace: Trace) -> CompressArtifact:
        start = time.perf_counter()
        artifact = super().compress_trace(trace)
        blob = gzip.compress(artifact.blob, compresslevel=self.level)
        elapsed = time.perf_counter() - start
        artifact.blob = blob
        artifact.compress_seconds = elapsed
        artifact.metadata["gzip_level"] = self.level
        return artifact

    def decompress_to_trace(self, blob: bytes) -> Trace:
        return super().decompress_to_trace(gzip.decompress(blob))


@register_compressor
class GzipMsgpackCompressor(RawMsgpackCompressor):
    """Raw msgpack payload piped through gzip (level 6)."""

    name = "gzip_msgpack"

    def __init__(self, level: int = 6) -> None:
        self.level = level

    def compress_trace(self, trace: Trace) -> CompressArtifact:
        start = time.perf_counter()
        artifact = super().compress_trace(trace)
        blob = gzip.compress(artifact.blob, compresslevel=self.level)
        elapsed = time.perf_counter() - start
        artifact.blob = blob
        artifact.compress_seconds = elapsed
        artifact.metadata["gzip_level"] = self.level
        return artifact

    def decompress_to_trace(self, blob: bytes) -> Trace:
        return super().decompress_to_trace(gzip.decompress(blob))
