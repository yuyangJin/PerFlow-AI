"""Compressed-trace storage profiling.

For the paper we need to know **where the bytes go** in a PADOC blob.
This module computes both:

* per-component on-disk sizes (after msgpack serialization), and
* per-component in-memory sizes (via :func:`asizeof.asizeof`)

for the three coarse buckets the paper reports:

1. ``templates`` -- the merged event template dictionary
   (further broken down by ``name_pattern``, ``name_nums``, ``ts``,
   ``dur``, ``args``, ``cat`` / ``bp`` / ``s``, ``id``, ``pid``/``tid``/``ph``).
2. ``structure`` -- the rank-keyed call tree of nodes
   (CPUNode / SameCPUNode / KernelLaunchNode / GPUNode / RefNode).
   This is the storage cost of the *soft-link edges*: every event leaf
   in the tree stores ``(template_index, instance_index)`` references.
3. ``metadata`` -- the trace-level metadata, ``rank_start_timestamp``.

The output is a :class:`StorageBreakdown` dataclass that the bench
report picks up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import msgpack

from ._compat import asizeof
from .event import MergeEvent, MergeKernelEvent
from .trace import CompressedTrace, compressed_trace_payload


_TEMPLATE_KEYS = (
    "name_pattern",
    "name_nums",
    "ts",
    "dur",
    "id",
    "args",
    "cat",
    "bp",
    "s",
    "pid",
    "tid",
    "ph",
)


@dataclass
class StorageBreakdown:
    """Detailed storage profile for one :class:`CompressedTrace`."""

    total_serialized_bytes: int = 0
    templates_serialized_bytes: int = 0
    structure_serialized_bytes: int = 0
    metadata_serialized_bytes: int = 0

    template_field_serialized: Dict[str, int] = field(default_factory=dict)

    template_count: int = 0
    cpu_template_count: int = 0
    gpu_template_count: int = 0

    structural_node_counts: Dict[str, int] = field(default_factory=dict)
    soft_link_edge_count: int = 0
    """Total number of (template_index, instance_index) references in the tree."""

    in_memory_bytes: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def template_share(self) -> float:
        if self.total_serialized_bytes <= 0:
            return 0.0
        return self.templates_serialized_bytes / self.total_serialized_bytes

    def structure_share(self) -> float:
        if self.total_serialized_bytes <= 0:
            return 0.0
        return self.structure_serialized_bytes / self.total_serialized_bytes

    def render_markdown(self) -> str:
        lines: List[str] = []
        lines.append("| component | serialized bytes | share |")
        lines.append("| --- | ---: | ---: |")
        rows = [
            ("templates", self.templates_serialized_bytes),
            ("structure (soft-link edges + tree)", self.structure_serialized_bytes),
            ("metadata", self.metadata_serialized_bytes),
            ("**total**", self.total_serialized_bytes),
        ]
        for label, value in rows:
            share = (
                value / self.total_serialized_bytes if self.total_serialized_bytes else 0
            )
            lines.append(f"| {label} | {value} | {share:.1%} |")

        if self.template_field_serialized:
            lines.append("")
            lines.append("| template field | serialized bytes |")
            lines.append("| --- | ---: |")
            for key in _TEMPLATE_KEYS:
                if key in self.template_field_serialized:
                    lines.append(f"| {key} | {self.template_field_serialized[key]} |")

        if self.structural_node_counts:
            lines.append("")
            lines.append("| node kind | count |")
            lines.append("| --- | ---: |")
            for key in sorted(self.structural_node_counts):
                lines.append(f"| {key} | {self.structural_node_counts[key]} |")

        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Implementation
# ----------------------------------------------------------------------


def _msgpack_size(value: Any) -> int:
    return len(msgpack.packb(value, use_bin_type=True))


def _per_template_field_payloads(template: MergeEvent) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for key in _TEMPLATE_KEYS:
        value = getattr(template, key, None)
        if value is None:
            continue
        # Numpy arrays / nested structures need a JSON-friendly conversion
        # before msgpack -- reuse the trace serializer's logic.
        from .utils import to_json_safe
        fields[key] = to_json_safe(value)
    return fields


def _count_structural_nodes(rank_tree: Any, counts: Dict[str, int]) -> None:
    if rank_tree is None:
        return
    type_name = type(rank_tree).__name__
    counts[type_name] = counts.get(type_name, 0) + 1

    children = getattr(rank_tree, "children", None)
    if children:
        for child in children:
            _count_structural_nodes(child, counts)
    slots = getattr(rank_tree, "slots", None)
    if slots:
        for slot in slots:
            if isinstance(slot, list):
                for entry in slot:
                    _count_structural_nodes(entry, counts)
            else:
                _count_structural_nodes(slot, counts)
    events = getattr(rank_tree, "events", None)
    if events:
        for event in events:
            counts.setdefault("Event", 0)
            counts["Event"] += 1


def _count_soft_link_edges(node: Any) -> int:
    """Count ``(template_index, instance_index)`` references in the tree."""
    if node is None:
        return 0
    edges = 0

    template_index = getattr(node, "template_index", None)
    if template_index is not None:
        # GPUNode stores arrays of indices; SameCPUNode also has a list of instances.
        if hasattr(template_index, "__len__"):
            edges += int(len(template_index))
        else:
            edges += 1
    if hasattr(node, "instance_index") and node.instance_index is not None:
        if hasattr(node.instance_index, "__len__"):
            edges += int(len(node.instance_index))
        else:
            edges += 1

    children = getattr(node, "children", None)
    if children:
        for child in children:
            edges += _count_soft_link_edges(child)
    slots = getattr(node, "slots", None)
    if slots:
        for slot in slots:
            if isinstance(slot, list):
                for entry in slot:
                    edges += _count_soft_link_edges(entry)
            else:
                edges += _count_soft_link_edges(slot)
    return edges


def measure_storage(compressed: CompressedTrace) -> StorageBreakdown:
    """Compute a detailed storage breakdown for ``compressed``."""
    payload = compressed_trace_payload(compressed)

    templates_payload = payload["event_templates"]
    structure_payload = payload["ranks"]
    metadata_payload = {
        "metadata": payload["metadata"],
        "rank_start_timestamp": payload["rank_start_timestamp"],
    }

    breakdown = StorageBreakdown()
    breakdown.templates_serialized_bytes = _msgpack_size(templates_payload)
    breakdown.structure_serialized_bytes = _msgpack_size(structure_payload)
    breakdown.metadata_serialized_bytes = _msgpack_size(metadata_payload)
    breakdown.total_serialized_bytes = (
        breakdown.templates_serialized_bytes
        + breakdown.structure_serialized_bytes
        + breakdown.metadata_serialized_bytes
    )

    breakdown.template_count = len(compressed.event_templates)
    breakdown.cpu_template_count = sum(
        1 for t in compressed.event_templates if isinstance(t, MergeEvent)
    )
    breakdown.gpu_template_count = sum(
        1 for t in compressed.event_templates if isinstance(t, MergeKernelEvent)
    )

    # Per-template-field byte breakdown.
    field_totals: Dict[str, int] = {key: 0 for key in _TEMPLATE_KEYS}
    for tmpl_dict in templates_payload:
        for key in _TEMPLATE_KEYS:
            if key in tmpl_dict:
                field_totals[key] += _msgpack_size(tmpl_dict[key])
    breakdown.template_field_serialized = {
        key: value for key, value in field_totals.items() if value > 0
    }

    # Structural node counts + soft-link edge count.
    structural_counts: Dict[str, int] = {}
    soft_link_edges = 0
    for _rank, processes in compressed.ranks.items():
        for _pid, threads in processes.items():
            for _tid, phases in threads.items():
                for _ph, root in phases.items():
                    _count_structural_nodes(root, structural_counts)
                    soft_link_edges += _count_soft_link_edges(root)
    breakdown.structural_node_counts = structural_counts
    breakdown.soft_link_edge_count = soft_link_edges

    # In-memory sizes (using pympler.asizeof when available).
    breakdown.in_memory_bytes = {
        "templates": int(asizeof.asizeof(compressed.event_templates)),
        "structure": int(asizeof.asizeof(compressed.ranks)),
        "metadata": int(asizeof.asizeof(metadata_payload)),
    }

    return breakdown
