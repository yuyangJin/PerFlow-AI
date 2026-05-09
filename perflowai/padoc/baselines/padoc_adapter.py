"""Adapter that wraps :class:`TemplateCompressor` as a baseline."""

from __future__ import annotations

import time
from typing import Any, Dict

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

from ..compressor import TemplateCompressor
from ..config import CompressorConfig, default_config
from ..trace import (
    CompressedTrace,
    Trace,
    compressed_trace_bin_bytes,
    compressed_trace_payload,
)
from .base import BaselineCompressor, CompressArtifact, register_compressor


@register_compressor
class PADOCCompressor(BaselineCompressor):
    """PADOC behind the same baseline interface as TraceZip / ScalaTrace."""

    name = "padoc"
    supports_in_situ_analysis = True

    def __init__(
        self,
        merge_ranks: bool = False,
        post_zstd: bool = True,
        config: CompressorConfig | None = None,
    ) -> None:
        self.merge_ranks = merge_ranks
        self.post_zstd = post_zstd
        self.config = config or default_config()
        # Cache of recently produced CompressedTrace objects so the harness
        # can run in-situ analyses without repeating decompression.
        self._last_compressed: CompressedTrace | None = None

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress_trace(self, trace: Trace) -> CompressArtifact:
        compressor = TemplateCompressor(config=self.config)
        start = time.perf_counter()
        if self.merge_ranks:
            compressed_trace = compressor.inter_compress(trace, merge_ranks=True)
        elif len(trace.get_ranks()) <= 1:
            compressed_trace = compressor.intra_compress(trace, emit_summary=False)
        else:
            compressed_trace = compressor.inter_compress(trace, merge_ranks=False)
        compress_seconds = time.perf_counter() - start

        blob = compressed_trace_bin_bytes(compressed_trace)
        if self.post_zstd:
            blob = _final_compress(blob)

        self._last_compressed = compressed_trace

        meta: Dict[str, Any] = {
            "template_count": len(compressed_trace.event_templates),
            "rank_count": len(compressed_trace.ranks),
            "post_zstd": self.post_zstd,
            "merge_ranks": self.merge_ranks,
            "config": self.config.as_dict(),
        }
        return CompressArtifact(
            blob=blob,
            metadata=meta,
            compress_seconds=compress_seconds,
        )

    def decompress_to_trace(self, blob: bytes) -> Trace:
        compressed_trace = self._compressed_trace_from_blob(blob)
        return TemplateCompressor().inter_decompress(compressed_trace)

    # ------------------------------------------------------------------
    # In-situ analysis hook
    # ------------------------------------------------------------------

    def load_compressed_trace(self, blob: bytes) -> CompressedTrace:
        return self._compressed_trace_from_blob(blob)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compressed_trace_from_blob(self, blob: bytes) -> CompressedTrace:
        if self.post_zstd:
            blob = _final_decompress(blob)
        return CompressedTrace.from_bytes(blob)
