"""Synthetic AI-trace generator used by the bench scalability sweeps.

The paper's scalability tables (compression / analysis time vs. GPUs,
layers, iterations) are normally produced from real trace files
collected on the cluster.  Locally we cannot ship those, but we can
*simulate* them with a parameterised synthetic generator that produces
chrome-trace style payloads exhibiting:

* the rank/pid/tid hierarchy expected by :class:`Trace`,
* repeating transformer-layer call patterns (so PADOC's structural
  compression actually fires),
* CPU launch + GPU kernel correlation pairs (so the kernel-link
  pathway is exercised),
* parallel-collective annotations on a configurable TP / DP / PP /
  EP topology (so the parallel-group analysis task has something to
  detect).

The generator is fully deterministic given a seed.  The output can be
written either as a single chrome-trace JSON file (easiest local use)
or as a multi-rank directory (suitable for the inter-rank PADOC path).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils import logger


@dataclass
class SyntheticTraceSpec:
    """All knobs that control the synthetic trace shape."""

    name: str = "synthetic"

    # Topology
    gpus: int = 8
    tp: int = 1
    dp: int = 8
    pp: int = 1
    ep: int = 1

    # Workload
    layers: int = 4
    iterations: int = 4
    micro_batches: int = 2

    # Per-layer cost
    forward_kernels: int = 4
    backward_kernels: int = 4
    optimizer_kernels: int = 2
    kernel_dur_us: int = 100

    # Collectives (one per iteration per kind, by default).
    enable_allreduce: bool = True
    enable_allgather: bool = False
    enable_alltoall: bool = False

    seed: int = 0
    base_ts_us: int = 1_000_000

    output_dir: bool = False
    """If True, write per-rank .json files; else single combined JSON."""

    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sanity: tp * dp * pp * ep should equal gpus when explicitly set.
        product = max(self.tp * self.dp * self.pp * self.ep, 1)
        if self.gpus != product and product != 1:
            # Auto-adjust dp to fill the topology.
            extra = self.gpus // (self.tp * self.pp * self.ep)
            if extra > 0:
                self.dp = extra


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _rank_topology(spec: SyntheticTraceSpec) -> Dict[int, Dict[str, int]]:
    """Map every rank id -> ``{tp, dp, pp, ep}`` coordinates."""
    out: Dict[int, Dict[str, int]] = {}
    rank = 0
    for pp_idx in range(max(spec.pp, 1)):
        for dp_idx in range(max(spec.dp, 1)):
            for ep_idx in range(max(spec.ep, 1)):
                for tp_idx in range(max(spec.tp, 1)):
                    if rank >= spec.gpus:
                        return out
                    out[rank] = {
                        "tp": tp_idx,
                        "dp": dp_idx,
                        "pp": pp_idx,
                        "ep": ep_idx,
                    }
                    rank += 1
    return out


def _collective_group_ranks(
    topology: Dict[int, Dict[str, int]],
    axis: str,
) -> Dict[Any, List[int]]:
    """Group ranks by every coordinate except ``axis``.

    For an all-reduce on the ``axis`` dimension, ranks that share all
    OTHER coordinates form one collective group.
    """
    groups: Dict[Any, List[int]] = {}
    for rank, coord in topology.items():
        key = tuple((k, v) for k, v in coord.items() if k != axis)
        groups.setdefault(key, []).append(rank)
    return {k: sorted(v) for k, v in groups.items()}


def _emit_layer_events(
    rng: random.Random,
    spec: SyntheticTraceSpec,
    rank: int,
    layer_id: int,
    micro_batch: int,
    iteration: int,
    cpu_pid: int,
    cpu_tid: int,
    gpu_tid: int,
    cursor_us: int,
    correlation_base: int,
    events: List[Dict[str, Any]],
) -> int:
    """Emit one forward+backward layer; return updated cursor_us."""

    layer_name_prefix = f"transformer.layers.{layer_id}"

    # -- forward annotation
    fwd_start = cursor_us
    cursor_us += 5
    # CPU "layer scope" annotation (used by layer_operator_balance task).
    events.append(
        {
            "ph": "X", "pid": cpu_pid, "tid": cpu_tid, "name": f"{layer_name_prefix}.forward",
            "ts": fwd_start, "dur": 0,
            "cat": "user_annotation",
            "args": {"layer": layer_id, "iter": iteration, "micro_batch": micro_batch},
        }
    )

    for k in range(spec.forward_kernels):
        kernel_dur = spec.kernel_dur_us + rng.randint(-10, 10)
        cpu_dur = max(2, kernel_dur // 5)
        gpu_offset = rng.randint(1, 5)
        corr = correlation_base + k

        # CPU launch event
        events.append(
            {
                "ph": "X", "pid": cpu_pid, "tid": cpu_tid,
                "name": f"{layer_name_prefix}.forward.cudaLaunchKernel",
                "ts": cursor_us, "dur": cpu_dur,
                "cat": "cuda_runtime",
                "args": {"correlation": corr, "External id": corr},
            }
        )
        # GPU kernel event (on the same rank, gpu stream)
        events.append(
            {
                "ph": "X", "pid": cpu_pid, "tid": gpu_tid,
                "name": f"{layer_name_prefix}.forward.matmul",
                "ts": cursor_us + gpu_offset, "dur": kernel_dur,
                "cat": "kernel",
                "args": {"correlation": corr, "stream": gpu_tid},
            }
        )
        cursor_us += max(cpu_dur, gpu_offset + kernel_dur) + 1

    # Close the layer "annotation" duration (so it covers the kernels).
    fwd_dur = cursor_us - fwd_start
    events[-1 - 2 * spec.forward_kernels]["dur"] = fwd_dur  # update annotation dur

    correlation_base += spec.forward_kernels
    return cursor_us


def _emit_backward(
    rng: random.Random,
    spec: SyntheticTraceSpec,
    rank: int,
    layer_id: int,
    micro_batch: int,
    iteration: int,
    cpu_pid: int,
    cpu_tid: int,
    gpu_tid: int,
    cursor_us: int,
    correlation_base: int,
    events: List[Dict[str, Any]],
) -> int:
    layer_name_prefix = f"transformer.layers.{layer_id}"
    for k in range(spec.backward_kernels):
        kernel_dur = spec.kernel_dur_us + rng.randint(-10, 10)
        cpu_dur = max(2, kernel_dur // 5)
        gpu_offset = rng.randint(1, 5)
        corr = correlation_base + k
        events.append(
            {
                "ph": "X", "pid": cpu_pid, "tid": cpu_tid,
                "name": f"{layer_name_prefix}.backward.cudaLaunchKernel",
                "ts": cursor_us, "dur": cpu_dur,
                "cat": "cuda_runtime",
                "args": {"correlation": corr, "External id": corr},
            }
        )
        events.append(
            {
                "ph": "X", "pid": cpu_pid, "tid": gpu_tid,
                "name": f"{layer_name_prefix}.backward.matmul_grad",
                "ts": cursor_us + gpu_offset, "dur": kernel_dur,
                "cat": "kernel",
                "args": {"correlation": corr, "stream": gpu_tid},
            }
        )
        cursor_us += max(cpu_dur, gpu_offset + kernel_dur) + 1
    return cursor_us


def _emit_collective(
    spec: SyntheticTraceSpec,
    name: str,
    group_ranks: List[int],
    rank: int,
    cpu_pid: int,
    cpu_tid: int,
    gpu_tid: int,
    cursor_us: int,
    iteration: int,
    correlation_base: int,
    events: List[Dict[str, Any]],
    kind_label: str,
) -> int:
    if rank not in group_ranks:
        return cursor_us
    corr = correlation_base
    coll_dur = max(spec.kernel_dur_us, 50)
    events.append(
        {
            "ph": "X", "pid": cpu_pid, "tid": cpu_tid,
            "name": f"nccl.{name}.cudaLaunchKernel",
            "ts": cursor_us, "dur": 5,
            "cat": "cuda_runtime",
            "args": {
                "correlation": corr,
                "group_size": len(group_ranks),
                "group_name": f"{kind_label}_size{len(group_ranks)}",
                "iter": iteration,
            },
        }
    )
    events.append(
        {
            "ph": "X", "pid": cpu_pid, "tid": gpu_tid,
            "name": f"nccl.{name}",
            "ts": cursor_us + 1, "dur": coll_dur,
            "cat": "kernel",
            "args": {
                "correlation": corr,
                "stream": gpu_tid,
                "group_size": len(group_ranks),
                "group_name": f"{kind_label}_size{len(group_ranks)}",
            },
        }
    )
    cursor_us += coll_dur + 5
    return cursor_us


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def generate_trace(spec: SyntheticTraceSpec) -> Dict[str, List[Dict[str, Any]]]:
    """Generate a per-rank ``{rank: traceEvents}`` mapping."""
    rng = random.Random(spec.seed)
    topology = _rank_topology(spec)

    tp_groups = _collective_group_ranks(topology, "tp") if spec.enable_allreduce else {}
    dp_groups = _collective_group_ranks(topology, "dp") if spec.enable_allreduce else {}
    pp_groups = _collective_group_ranks(topology, "pp") if spec.enable_allgather else {}
    ep_groups = _collective_group_ranks(topology, "ep") if spec.enable_alltoall else {}

    per_rank_events: Dict[str, List[Dict[str, Any]]] = {}
    for rank in topology:
        rng_rank = random.Random(spec.seed * 1009 + rank)
        cpu_pid = 1000
        cpu_tid = 100
        gpu_tid = 200
        cursor_us = spec.base_ts_us
        correlation = rank * 10_000_000

        events: List[Dict[str, Any]] = []
        # Process metadata so PADOC keeps the rank string.
        events.append(
            {
                "ph": "M",
                "pid": cpu_pid,
                "tid": cpu_tid,
                "name": "process_name",
                "args": {"name": f"rank{rank}"},
                "ts": cursor_us,
            }
        )

        for it in range(spec.iterations):
            for mb in range(spec.micro_batches):
                for layer in range(spec.layers):
                    cursor_us = _emit_layer_events(
                        rng_rank, spec, rank, layer, mb, it,
                        cpu_pid, cpu_tid, gpu_tid,
                        cursor_us, correlation, events,
                    )
                    correlation += spec.forward_kernels
                # backward in reverse layer order
                for layer in range(spec.layers - 1, -1, -1):
                    cursor_us = _emit_backward(
                        rng_rank, spec, rank, layer, mb, it,
                        cpu_pid, cpu_tid, gpu_tid,
                        cursor_us, correlation, events,
                    )
                    correlation += spec.backward_kernels

            # Collectives at end of iteration
            for kind, groups in (
                ("tp", tp_groups),
                ("dp", dp_groups),
                ("pp", pp_groups),
                ("ep", ep_groups),
            ):
                if not groups:
                    continue
                coll_name = (
                    "all_reduce" if kind in ("tp", "dp")
                    else "all_gather" if kind == "pp"
                    else "all_to_all"
                )
                for group_ranks in groups.values():
                    cursor_us = _emit_collective(
                        spec, coll_name, group_ranks, rank,
                        cpu_pid, cpu_tid, gpu_tid,
                        cursor_us, it, correlation, events, kind_label=kind,
                    )
                    correlation += 1

            # Optimizer step
            for k in range(spec.optimizer_kernels):
                kernel_dur = spec.kernel_dur_us + rng_rank.randint(-5, 5)
                corr = correlation + k
                events.append(
                    {
                        "ph": "X", "pid": cpu_pid, "tid": cpu_tid,
                        "name": "optimizer.step.cudaLaunchKernel",
                        "ts": cursor_us, "dur": 4, "cat": "cuda_runtime",
                        "args": {"correlation": corr, "External id": corr},
                    }
                )
                events.append(
                    {
                        "ph": "X", "pid": cpu_pid, "tid": gpu_tid,
                        "name": "optimizer.step.adamw",
                        "ts": cursor_us + 1, "dur": kernel_dur, "cat": "kernel",
                        "args": {"correlation": corr, "stream": gpu_tid},
                    }
                )
                cursor_us += kernel_dur + 1
            correlation += spec.optimizer_kernels

        per_rank_events[str(rank)] = events
    return per_rank_events


def write_trace(
    spec: SyntheticTraceSpec,
    output_path: str,
) -> str:
    """Generate and write a synthetic trace.

    If ``spec.output_dir`` is True, ``output_path`` is treated as a
    directory and one ``rank<N>.json`` file is written per rank.
    Otherwise a single chrome-trace JSON with all ranks' events is
    written.
    """
    per_rank = generate_trace(spec)

    if spec.output_dir:
        os.makedirs(output_path, exist_ok=True)
        for rank, events in per_rank.items():
            payload = {
                "schemaVersion": 1,
                "distributedInfo": {"rank": int(rank)},
                "traceEvents": events,
            }
            file_path = os.path.join(output_path, f"rank{rank}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"))
        logger.info(
            "Synthetic trace dir %s | gpus=%d layers=%d iters=%d events=%d",
            output_path,
            spec.gpus,
            spec.layers,
            spec.iterations,
            sum(len(events) for events in per_rank.values()),
        )
        return output_path

    flat_events: List[Dict[str, Any]] = []
    for rank, events in per_rank.items():
        for event in events:
            entry = dict(event)
            entry.setdefault("args", {})
            if isinstance(entry["args"], dict):
                entry["args"]["rank"] = int(rank)
            flat_events.append(entry)

    payload = {
        "schemaVersion": 1,
        "distributedInfo": {"rank": 0},
        "traceEvents": flat_events,
    }
    parent = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    logger.info(
        "Synthetic trace %s | gpus=%d layers=%d iters=%d events=%d",
        output_path,
        spec.gpus,
        spec.layers,
        spec.iterations,
        len(flat_events),
    )
    return output_path
