"""Structural statistics for compressed trace trees.

These metrics quantify the *shape* of the call-tree forest after PADOC
compression -- the paper uses them in the "compression information"
section to justify why structural compression matters.

Headline numbers:

* ``depth`` -- min / max / mean / median tree depth across all roots.
* ``branching`` -- min / max / mean number of children per non-leaf node.
* ``samecpu_multiplier`` -- distribution of ``len(instance_index)`` for
  every :class:`SameCPUNode` (i.e. how many physical events one
  template instance covers).
* ``unique_subtree_shapes`` -- number of distinct subtree fingerprints
  (proxy for the size of the implicit subtree dictionary that PADOC
  builds; this is the "tree shape" count the paper asks for).
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .node import (
    CPUNode,
    GPUNode,
    KernelLaunchNode,
    KernelsLaunchNode,
    Node,
    SameCPUNode,
)
from .trace import CompressedTrace


@dataclass
class _StatVec:
    count: int = 0
    minimum: int = 0
    maximum: int = 0
    mean: float = 0.0
    median: float = 0.0

    @classmethod
    def from_values(cls, values: List[int]) -> "_StatVec":
        if not values:
            return cls()
        return cls(
            count=len(values),
            minimum=int(min(values)),
            maximum=int(max(values)),
            mean=float(statistics.fmean(values)),
            median=float(statistics.median(values)),
        )


@dataclass
class TreeStatistics:
    total_roots: int = 0
    total_nodes: int = 0

    depth: _StatVec = field(default_factory=_StatVec)
    branching: _StatVec = field(default_factory=_StatVec)
    samecpu_multiplier: _StatVec = field(default_factory=_StatVec)

    unique_subtree_shapes: int = 0
    """Number of distinct subtree fingerprints (= 'tree shape' count)."""

    samecpu_node_count: int = 0
    cpunode_count: int = 0
    kernel_launch_count: int = 0
    kernels_launch_count: int = 0
    gpunode_count: int = 0
    refnode_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Implementation
# ----------------------------------------------------------------------


def _children_of(node: Any) -> List[Any]:
    out: List[Any] = []
    children = getattr(node, "children", None)
    if children:
        out.extend(children)
    slots = getattr(node, "slots", None)
    if slots:
        for slot in slots:
            if isinstance(slot, list):
                out.extend(slot)
            else:
                out.append(slot)
    return out


def _subtree_depth(node: Any) -> int:
    children = _children_of(node)
    if not children:
        return 1
    return 1 + max(_subtree_depth(child) for child in children)


def _walk_collect(node: Any, stats: TreeStatistics, branching_values: List[int]) -> None:
    stats.total_nodes += 1
    type_name = type(node).__name__
    if isinstance(node, SameCPUNode):
        stats.samecpu_node_count += 1
    elif isinstance(node, CPUNode):
        stats.cpunode_count += 1
    elif isinstance(node, KernelLaunchNode):
        stats.kernel_launch_count += 1
    elif isinstance(node, KernelsLaunchNode):
        stats.kernels_launch_count += 1
    elif isinstance(node, GPUNode):
        stats.gpunode_count += 1
    elif "Ref" in type_name:
        stats.refnode_count += 1

    children = _children_of(node)
    if children:
        branching_values.append(len(children))
        for child in children:
            _walk_collect(child, stats, branching_values)


def _subtree_fingerprint(node: Any) -> Tuple[Any, ...]:
    """A hashable structural fingerprint of one subtree.

    We use only structural attributes: type name, the *shape* of its
    children, and the per-node multiplier (``len(instance_index)`` for
    :class:`SameCPUNode`).  Concrete event template indices are
    intentionally omitted so two subtrees with identical shape but
    different leaf templates still compare equal -- this counts the
    number of distinct **shapes**, exactly what the paper asks for.
    """
    type_name = type(node).__name__
    multiplier: int = 1
    if isinstance(node, SameCPUNode) and node.instance_index is not None:
        multiplier = int(len(node.instance_index))
    elif isinstance(node, GPUNode) and getattr(node, "template_index", None) is not None:
        multiplier = int(len(node.template_index))
    children = _children_of(node)
    return (
        type_name,
        multiplier,
        tuple(_subtree_fingerprint(child) for child in children),
    )


def measure_tree_statistics(compressed: CompressedTrace) -> TreeStatistics:
    """Compute structural statistics for ``compressed`` (over every rank)."""
    stats = TreeStatistics()
    branching_values: List[int] = []
    depth_values: List[int] = []
    samecpu_multipliers: List[int] = []
    fingerprints: set = set()

    for _rank, processes in compressed.ranks.items():
        for _pid, threads in processes.items():
            for _tid, phases in threads.items():
                for _ph, root in phases.items():
                    if root is None:
                        continue
                    stats.total_roots += 1
                    depth_values.append(_subtree_depth(root))
                    _walk_collect(root, stats, branching_values)
                    _collect_multipliers(root, samecpu_multipliers)
                    fingerprints.add(_subtree_fingerprint(root))
                    # Also fingerprint every internal subtree so we measure
                    # the implicit dictionary, not just the per-rank roots.
                    _collect_subtree_fingerprints(root, fingerprints)

    stats.depth = _StatVec.from_values(depth_values)
    stats.branching = _StatVec.from_values(branching_values)
    stats.samecpu_multiplier = _StatVec.from_values(samecpu_multipliers)
    stats.unique_subtree_shapes = len(fingerprints)
    return stats


def _collect_multipliers(node: Any, out: List[int]) -> None:
    if isinstance(node, SameCPUNode) and node.instance_index is not None:
        out.append(int(len(node.instance_index)))
    for child in _children_of(node):
        _collect_multipliers(child, out)


def _collect_subtree_fingerprints(node: Any, out: set) -> None:
    out.add(_subtree_fingerprint(node))
    for child in _children_of(node):
        _collect_subtree_fingerprints(child, out)
