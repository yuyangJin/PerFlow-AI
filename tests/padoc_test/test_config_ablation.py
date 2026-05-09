"""Ablation tests for :class:`CompressorConfig`.

Each preset must:
1. Run without raising,
2. Produce a roundtrip-equal event count,
3. Produce a non-empty blob.

We also verify that ``no_name_pattern`` produces strictly more templates
than the default (digit collapsing actively dedups), and that
``minimal`` produces a strictly different blob than ``default``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from perflowai.padoc.baselines import PADOCCompressor
from perflowai.padoc.config import (
    CompressorConfig,
    all_ablation_presets,
    default_config,
)
from perflowai.padoc.trace import Trace


TRACE_PATH = Path("tests/example_trace/profiler_585.json")


@pytest.fixture(scope="module")
def trace() -> Trace:
    if not TRACE_PATH.exists():
        pytest.skip(f"missing trace {TRACE_PATH}")
    return Trace.from_file(str(TRACE_PATH))


def _count_events(trace: Trace) -> int:
    return sum(len(events) for *_rest, events in trace.iter_events())


@pytest.mark.parametrize(
    "label,config",
    list(all_ablation_presets().items()),
    ids=list(all_ablation_presets().keys()),
)
def test_ablation_preset_roundtrips(trace: Trace, label: str, config: CompressorConfig) -> None:
    src = _count_events(trace)
    compressor = PADOCCompressor(config=config)
    artifact = compressor.compress_trace(trace)
    assert artifact.size_bytes > 0
    decoded = compressor.decompress_to_trace(artifact.blob)
    dst = _count_events(decoded)
    assert dst == src, f"{label}: event count drifted ({dst} != {src})"


def test_no_name_pattern_grows_template_count(trace: Trace) -> None:
    base = PADOCCompressor(config=default_config()).compress_trace(trace)
    no_pat = PADOCCompressor(
        config=CompressorConfig(enable_name_pattern=False, label="no_name_pattern")
    ).compress_trace(trace)
    assert (
        no_pat.metadata["template_count"] > base.metadata["template_count"]
    ), "Disabling name-pattern should cause templates to fragment"


def test_minimal_differs_from_default(trace: Trace) -> None:
    base = PADOCCompressor(config=default_config()).compress_trace(trace)
    minimal = PADOCCompressor(
        config=CompressorConfig(
            enable_structural=False,
            enable_anchor_matching=False,
            enable_slp=False,
            enable_args_dedup=False,
            enable_kernel_links=False,
            enable_name_pattern=False,
            label="minimal",
        )
    ).compress_trace(trace)
    assert base.blob != minimal.blob, "Minimal config should produce a different blob"


def test_disabling_kernel_links_drops_kernel_launch_nodes(trace: Trace) -> None:
    """When kernel links are disabled the rank tree must not contain KernelLaunchNode."""
    from perflowai.padoc.compressor import TemplateCompressor
    from perflowai.padoc.node import KernelLaunchNode

    cfg = CompressorConfig(enable_kernel_links=False, label="no_kernel_links")
    ct = TemplateCompressor(config=cfg).intra_compress(trace, emit_summary=False)

    seen = []

    def _walk(node):
        if isinstance(node, KernelLaunchNode):
            seen.append(node)
        for child in (getattr(node, "children", None) or []):
            _walk(child)
        slots = getattr(node, "slots", None)
        if slots:
            for slot in slots:
                if isinstance(slot, list):
                    for entry in slot:
                        _walk(entry)
                else:
                    _walk(slot)

    for _rank, processes in ct.ranks.items():
        for _pid, threads in processes.items():
            for _tid, phases in threads.items():
                for _ph, root in phases.items():
                    _walk(root)

    assert not seen, f"Found {len(seen)} KernelLaunchNode(s) despite ablation flag"
