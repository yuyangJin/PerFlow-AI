"""Compression configuration / ablation switches.

These flags let the bench harness toggle individual PADOC techniques to
produce ablation rows for the paper.  Each flag corresponds to one of
the building blocks described in the design doc:

* ``enable_structural`` -- :class:`SameCPUNode` formation in
  :meth:`TemplateCompressor._compress_node_new` /
  :meth:`TemplateCompressor._build_template` /
  :meth:`TemplateCompressor._extract_anchors`.
  When ``False`` the compressor skips template-style child grouping;
  every CPUNode keeps its own children verbatim and we fall back to a
  template-per-instance layout.
* ``enable_anchor_matching`` -- the LCS-style anchor extraction inside
  :meth:`_build_template`.  When ``False`` we still merge children
  whose ``template_index`` already matches but stop trying to align
  partially-matching child sequences across instances.
* ``enable_slp`` -- numeric SLP compression of ts / dur / ids inside
  every :class:`MergeEvent`.  When ``False`` the merged events keep
  raw python lists.
* ``enable_args_dedup`` -- the per-template args dedup applied by
  :func:`SLP.compress_same_args`.  When ``False`` we keep one copy of
  each event's args.
* ``enable_kernel_links`` -- correlation-id-based ``KernelLaunchNode``
  pairing between CPU launch events and GPU kernel events.  When
  ``False`` GPU events become independent leaves on their stream and
  no soft-link edges are written.
* ``enable_name_pattern`` -- name -> pattern + nums splitting inside
  :class:`MergeEvent`.  When ``False`` the merged event keeps every
  raw event name (still deduplicated via the template, but not
  digit-collapsed).

A single instance is shared by everything in PADOC.  The defaults
reproduce the production behaviour; ablation runs flip individual
flags via :class:`PADOCCompressor(merge_ranks=..., config=...)`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CompressorConfig:
    """Toggleable knobs for :class:`TemplateCompressor` ablation studies."""

    enable_structural: bool = True
    enable_anchor_matching: bool = True
    enable_slp: bool = True
    enable_args_dedup: bool = True
    enable_kernel_links: bool = True
    enable_name_pattern: bool = True

    label: str = "default"
    """Free-form label used by the bench report to identify the ablation row."""

    def with_label(self, label: str) -> "CompressorConfig":
        return replace(self, label=label)

    def replace(self, **changes: Any) -> "CompressorConfig":
        return replace(self, **changes)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_default(self) -> bool:
        return (
            self.enable_structural
            and self.enable_anchor_matching
            and self.enable_slp
            and self.enable_args_dedup
            and self.enable_kernel_links
            and self.enable_name_pattern
        )


# Convenience presets used by the ablation runner -------------------------------


def default_config() -> CompressorConfig:
    return CompressorConfig()


def ablation_no_structural() -> CompressorConfig:
    return CompressorConfig(
        enable_structural=False,
        enable_anchor_matching=False,
        label="no_structural",
    )


def ablation_no_anchor_matching() -> CompressorConfig:
    return CompressorConfig(enable_anchor_matching=False, label="no_anchor")


def ablation_no_slp() -> CompressorConfig:
    return CompressorConfig(enable_slp=False, label="no_slp")


def ablation_no_args_dedup() -> CompressorConfig:
    return CompressorConfig(enable_args_dedup=False, label="no_args_dedup")


def ablation_no_kernel_links() -> CompressorConfig:
    return CompressorConfig(enable_kernel_links=False, label="no_kernel_links")


def ablation_no_name_pattern() -> CompressorConfig:
    return CompressorConfig(enable_name_pattern=False, label="no_name_pattern")


def ablation_minimal() -> CompressorConfig:
    """Disable every optional technique; keeps only template dedup."""
    return CompressorConfig(
        enable_structural=False,
        enable_anchor_matching=False,
        enable_slp=False,
        enable_args_dedup=False,
        enable_kernel_links=False,
        enable_name_pattern=False,
        label="minimal",
    )


def all_ablation_presets() -> Dict[str, CompressorConfig]:
    """Standard ablation rows for the paper."""
    return {
        cfg.label: cfg
        for cfg in [
            default_config(),
            ablation_no_structural(),
            ablation_no_anchor_matching(),
            ablation_no_slp(),
            ablation_no_args_dedup(),
            ablation_no_kernel_links(),
            ablation_no_name_pattern(),
            ablation_minimal(),
        ]
    }
