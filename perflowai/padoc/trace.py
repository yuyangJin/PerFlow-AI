"""Trace structures for compression, serialization, and reconstruction.

This module defines three main trace representations used in the profiling
and compression pipeline:

- Trace: Raw, uncompressed hierarchical trace structure organized by
rank → pid → tid, where each thread is represented by a `Node` containing
its list of `Event` objects.

- CompressedTrace: A template-based compressed representation that reduces
storage by deduplicating repeated subtrees (e.g., Transformer layers,
decode steps). Threads may contain `Node` or `RefNode` entries referencing
shared `TemplateNode` objects.

- BaseTrace: Abstract interface implemented by all trace types, providing
unified metadata access, node traversal utilities, and serialization
interfaces.

This module serves as the core data model for the PADoC/trace compression
pipeline, enabling both efficient storage and faithful reconstruction.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import os
from collections import defaultdict
import msgpack
from pympler import asizeof

from perflowai.padoc.utils import logger, analyze_node_dict
from perflowai.padoc.event import Event, MergeEvent, KernelEvent, MergeKernelEvent, memory_breakdown_templates
from perflowai.padoc.node import (
    CPUNode,
    GPUNode,
    node_from_dict,
    count_trace_nodes,
    LaunchSubtreeIndex,
)

class BaseTrace(ABC):
    """Abstract base class for all trace types in the trace tree.

    This interface defines a unified API for concrete trace implementations
    (Trace, TemplateTrace, CompressedTrace). Any subclass must implement metadata
    access, node access, and serialization.

    Attributes:
    metadata (Dict[str, Any]): Metadata about the trace such as version,
    model name, or profiling configuration.
    ranks (Dict[str, Dict[str, Dict[str, BaseNode]]]): Hierarchical mapping
    of rank → pid → tid → Node.
    """

    def __init__(self, metadata: Dict[str, Dict[str, Any]] | None = None,
                 start_timestamp: Dict[str, int] | None = None):
        self.metadata: Dict[str, Dict[str, Any]] = metadata if metadata else {}
        # rank → pid → tid → ph -> List[Event]
        self.ranks: Dict[str, Dict[str, Dict[str, Dict[str, List[Event]]]]] = {}
        self.start_timestamp = start_timestamp if start_timestamp else {}

    def get_metadata(self) -> Dict[str, Any]:
        """Return the trace metadata."""
        return self.metadata

    def get_ranks(self) -> List[str]:
        """Return the list of ranks in the trace."""
        return list(self.ranks.keys())

    def get_pids(self, rank: str) -> List[str]:
        """Return the list of pids in the specified rank."""
        return list(self.ranks.get(rank, {}).keys())

    def get_tids(self, rank: str, pid: str) -> List[str]:
        """Return the list of tids in the specified rank and pid."""
        return list(self.ranks.get(rank, {}).get(pid, {}).keys())

    def get_phs(self, rank: str, pid: str, tid: str) -> List[str]:
        """Return the list of phases in the specified rank, pid, and tid."""
        return list(self.ranks.get(rank, {}).get(pid, {}).get(tid, {}).keys())

    def set_ranks(self, ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]]):
        """Set the trace ranks."""
        self.ranks = ranks

    def get_node(self, rank: str, pid: str, tid: str, ph: str) -> Optional[BaseNode]:
        """Return the node for the specified rank, pid, tid and phase."""
        return self.ranks.get(rank, {}).get(pid, {}).get(tid, {}).get(ph)

    def set_node(self, rank: str, pid: str, tid: str, ph: str, node: BaseNode):
        """Set the node for the specified rank, pid, tid and phase."""
        self.ranks.setdefault(rank, {}).setdefault(pid, {}).setdefault(tid, {})[ph] = node

    def get_start_time(self) -> Dict[str, int]:
        """Return the start time of each rank."""
        return self.start_timestamp

    def iter_events(self, rank: Optional[str] = None):
        """Iterate over all events in the trace, optionally filtering by rank."""
        rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]
        for r, processes in rank_items:
            for pid, threads in processes.items():
                for tid, phases in threads.items():
                    for ph, events in phases.items():
                        yield r, pid, tid, ph, events

    @classmethod
    @abstractmethod
    def from_file(cls, path: str) -> BaseTrace:
        """Create a trace from a JSON file."""
        return None

    @abstractmethod
    def write_file(self, path: str, rank: str = "0", origin: bool = False):
        """Write the trace to a file."""
        return

class Trace(BaseTrace):
    """Concrete uncompressed trace type.

    This class represents the raw profiler trace, structured hierarchically by
    rank → pid → tid. Each thread is represented by a `Node` containing its
    chronological list of `Event` objects.

    Responsibilities:
    - Constructing the tree structure from event lists
    - Providing raw access to nodes and metadata
    - Serializing to JSON/msgpack in both raw and structured forms
    """

    def __init__(self, events: Dict[str, List[Dict[str, Any]]] | None = None,
                 metadata: Dict[str, Dict[str, Any]] | None = None,
                 start_timestamp: Dict[str, int] | None = None):
        super().__init__(metadata, start_timestamp)

        if events:
            for rank, rank_events in events.items():
                rank_events = sorted(rank_events, key=lambda x: x["ts"])

                start_timestamp = rank_events[0]["ts"]
                self.start_timestamp[rank] = start_timestamp
                self.ranks[rank] = {}

                for e in rank_events:
                    pid = e.pop("pid", 0)
                    tid = e.pop("tid", 0)
                    ph = e.pop("ph", "X")
                    e["ts"] -= start_timestamp
                    args = e.get("args", {})
                    stream_id = args.get("stream", None)
                    if stream_id is not None:
                        tid = f"stream {stream_id}"
                    else:
                        category = e.get("cat", "")
                        if category == "gpu_user_annotation":
                            tid = f"stream {tid}"

                    rank_layer = self.ranks.setdefault(rank, {})
                    pid_layer = rank_layer.setdefault(str(pid), {})
                    tid_layer = pid_layer.setdefault(str(tid), {})
                    ph_layer = tid_layer.setdefault(str(ph), [])

                    ph_layer.append(Event(e))

    @classmethod
    def from_file(cls, path: str) -> 'Trace':
        """Load trace from a single JSON/msgpack file."""
        rank, events, metadata = cls._load_single_file_data(path)

        return cls({rank: events}, {rank: metadata})

    @classmethod
    def from_dir(cls, path: str) -> 'Trace':
        """Load a trace from a directory of JSON/msgpack files."""
        all_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        all_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            rank, events_list, metadata_dict = cls._load_single_file_data(file_path)
            all_events[rank].extend(events_list)
            all_metadata[rank].update(metadata_dict)

        return cls(dict(all_events), dict(all_metadata))

    @staticmethod
    def _load_single_file_data(path: str) -> Dict[str, Any]:
        """Load data from a single JSON or msgpack file."""
        data: Dict[str, Any] = {}
        if path.endswith(".json"):
            with open(path, 'r', encoding="utf-8") as f:
                data = json.load(f)
        elif path.endswith(".bin"):
            with open(path, 'rb') as f:
                data = msgpack.load(f, strict_map_key=False)
        else:
            logger.warning("Unsupported trace file format: %s", path)

        rank = data.get("distributedInfo", {}).get("rank", "0")
        events: List[Dict[str, Any]] = data.get("traceEvents", [])
        logger.info("Loaded %d events from %s", len(events), path)
        metadata: Dict[str, Any] = \
            {k: v for k, v in data.items() if k != "traceEvents"}

        return str(rank), events, metadata

    def write_file(self, path: str, rank: str = "", origin: bool = False):
        out = {}
        if rank == "":
            rank = self.get_ranks()[0]

        if origin:
            trace_events = []

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]

            for r, processes in rank_items:
                for pid, tids in processes.items():
                    for tid, phases in tids.items():
                        for ph, events in phases.items():
                            for e in events:
                                event_dict = e.to_dict().copy()
                                if "pid" not in event_dict:
                                    event_dict["pid"] = pid
                                t = tid
                                if t.startswith("stream "):
                                    t = t.split(" ")[1]
                                if t.endswith("-abnormal"):
                                    t = t.split("-")[0]

                                if "tid" not in event_dict:
                                    event_dict["tid"] = t
                                if "ph" not in event_dict:
                                    event_dict["ph"]  = ph
                                event_dict["ts"] += self.start_timestamp[r]

                                trace_events.append(event_dict)

            trace_events = sorted(
                trace_events,
                key=lambda x: (
                    x["ts"],
                    x.get("ph", ""),
                    x.get("name", ""),
                    x.get("dur", 0)
                )
            )

            out = {"traceEvents": trace_events}
            out.update(self.metadata[rank])

        else:
            out["metadata"] = self.metadata[rank]
            out["ranks"] = {}
            out["start_timestamp"] = self.start_timestamp

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]
            for r, processes in rank_items:
                out["ranks"][r] = {}
                for pid, tids in processes.items():
                    out["ranks"][r][pid] = {}
                    for tid, phases in tids.items():
                        out["ranks"][r][pid][tid] = {}
                        for ph, events in phases.items():
                            out["ranks"][r][pid][tid][ph] = [e.to_dict() for e in events]

        ext = os.path.splitext(path)[1].lower()

        if ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        else:
            with open(path, "wb") as f:
                msgpack.dump(out, f)

    def write_dir(self, path: str, file_type: str, origin: bool = True):
        """Write the trace to a directory of JSON/msgpack files.

        Args:
            - path: directory path to write to
            - type: file format to write, now support "json" and "bin"(msgpack)
            - origin: whether to write the original trace format or the compressed format
        """

        if len(self.ranks) <= 1:
            logger.warning("Trace contains only one rank, writing to single file.")

        if file_type not in ["json", "bin"]:
            logger.warning("Unsupported trace file format: %s, writing as JSON.", file_type)
            file_type = "json"

        # Make sure the directory exists
        os.makedirs(path, exist_ok=True)

        for rank in self.ranks:
            file_path = os.path.join(path, f"rank{rank}.{file_type}")
            self.write_file(file_path, rank, origin)


class CompressedTrace(BaseTrace):
    """Trace representation using template-based compression.

    This compressed form stores:
    - `templates`: a dictionary of TemplateNode objects representing
    canonical subtrees.
    - `ranks`: same hierarchy as `Trace`, but nodes may reference
    template IDs via `RefNode` instead of storing full children.

    This format dramatically reduces storage for repetitive transformer
    layers, decode steps, or repeated micro-batches.
    """

    def __init__(self, event_templates: List[MergeEvent],
                 ranks: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
                 metadata: Dict[str, Any] | None = None,
                 start_timestamp: Dict[str, int] | None = None):

        super().__init__(metadata, start_timestamp)
        self.ranks = ranks

        self.event_templates: List[MergeEvent] = event_templates
        self._launch_indexes: Dict[tuple, LaunchSubtreeIndex] = {}
        if self.ranks:
            self.build_launch_indexes()

    def show_memory(self) -> None:
        core_parts = {
            "event_templates": asizeof.asizeof(self.event_templates),
            "ranks": asizeof.asizeof(self.ranks),
            "metadata": asizeof.asizeof(self.metadata),
            "start_timestamp": asizeof.asizeof(self.start_timestamp),
            "launch_indexes": asizeof.asizeof(self._launch_indexes),
        }
        total_core = sum(core_parts.values())

        print("\n=== CompressedTrace Core Memory ===")
        for name, size in sorted(core_parts.items(), key=lambda x: x[1], reverse=True):
            pct = (size / total_core * 100.0) if total_core > 0 else 0.0
            print(f"{name:16s}: {size / 1024 / 1024:8.2f} MB ({pct:5.1f}%)")
        print(f"{'total':16s}: {total_core / 1024 / 1024:8.2f} MB (100.0%)")

        tmpl_mem = memory_breakdown_templates(self.event_templates)
        tmpl_total = tmpl_mem.get("total", 0)
        print("\n=== Event Templates Breakdown ===")
        for name, size in sorted(tmpl_mem.items()):
            if name == "total":
                continue
            pct = (size / tmpl_total * 100.0) if tmpl_total > 0 else 0.0
            print(f"{name:16s}: {size / 1024 / 1024:8.2f} MB ({pct:5.1f}%)")
        print(f"{'total':16s}: {tmpl_total / 1024 / 1024:8.2f} MB (100.0%)")

        count_trace_nodes(self)

    def set_ranks(self, ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]]):
        super().set_ranks(ranks)
        self.clear_launch_indexes()
        if self.ranks:
            self.build_launch_indexes()

    def iter_nodes(self, rank: Optional[str] = None):
        """Iterate over all nodes in the trace, optionally filtering by rank."""
        rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]
        for r, processes in rank_items:
            for pid, threads in processes.items():
                for tid, phases in threads.items():
                    for ph, node in phases.items():
                        yield r, pid, tid, ph, node

    def clear_launch_indexes(self) -> None:
        self._launch_indexes = {}

    def build_launch_indexes(self, rank: Optional[str] = None) -> Dict[tuple, LaunchSubtreeIndex]:
        for r, pid, tid, ph, node in self.iter_nodes(rank):
            self._launch_indexes[(r, pid, tid, ph)] = LaunchSubtreeIndex(node)
        return self._launch_indexes

    def get_launch_index(self, rank: str, pid: str, tid: str, ph: str) -> LaunchSubtreeIndex:
        key = (rank, pid, tid, ph)
        if key in self._launch_indexes:
            return self._launch_indexes[key]

        node = self.get_node(rank, pid, tid, ph)
        if node is None:
            raise KeyError(f"Node not found for key {key}")

        index = LaunchSubtreeIndex(node)
        self._launch_indexes[key] = index
        return index

    def get_launch_nodes(
        self,
        rank: str,
        pid: str,
        tid: str,
        ph: str,
        node: Optional[Any] = None,
    ):
        return self.get_launch_index(rank, pid, tid, ph).get_launch_nodes(node)

    def iter_launch_nodes(
        self,
        rank: str,
        pid: str,
        tid: str,
        ph: str,
        node: Optional[Any] = None,
    ):
        yield from self.get_launch_index(rank, pid, tid, ph).iter_launch_nodes(node)

    @classmethod
    def from_file(cls, path: str) -> CompressedTrace:
        if path.endswith(".json"):
            with open(path, 'r', encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        else:
            with open(path, 'rb') as f:
                data: Dict[str, Any] = msgpack.load(f, strict_map_key=False)

        event_templates: List[MergeEvent] = []
        ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]] = {}
        metadata: Dict[str, Any] = {}

        assert "event_templates" in data, "Invalid trace format"
        assert "ranks" in data, "Invalid trace format"
        assert "metadata" in data, "Invalid trace format"

        for e in data["event_templates"]:
            if "pid" in e:
                event_templates.append(MergeKernelEvent.from_dict(e))
            else:
                event_templates.append(MergeEvent.from_dict(e))

        for rank, process_dict in data["ranks"].items():
            ranks[rank] = {}
            for pid, thread_dict in process_dict.items():
                ranks[rank][pid] = {}
                for tid, phase_dict in thread_dict.items():
                    ranks[rank][pid][tid] = {}
                    for ph, node_dict in phase_dict.items():
                        if (
                            isinstance(tid, str)
                            and tid.startswith("stream ")
                            and isinstance(node_dict, dict)
                            and "k" not in node_dict
                            and "t" in node_dict
                            and "i" in node_dict
                        ):
                            ranks[rank][pid][tid][ph] = GPUNode.from_dict(node_dict)
                        else:
                            ranks[rank][pid][tid][ph] = node_from_dict(node_dict)

        metadata = data["metadata"]
        start_timestamp = data.get("rank_start_timestamp", {})

        return cls(event_templates, ranks, metadata, start_timestamp)

    def write_file(self, path: str, rank: str = "", origin: bool = False):
        out = {}

        out["metadata"] = self.metadata
        out["event_templates"] = [m.to_dict() for m in self.event_templates]
        out["ranks"] = {}
        out["rank_start_timestamp"] = self.start_timestamp

        for r, processes in self.ranks.items():
            out["ranks"][r] = {}
            for pid, tids in processes.items():
                out["ranks"][r][pid] = {}
                for tid, phases in tids.items():
                    out["ranks"][r][pid][tid] = {}
                    for ph, node in phases.items():
                        out["ranks"][r][pid][tid][ph] = node.to_dict()

        ext = os.path.splitext(path)[1].lower()

        if ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        else:
            with open(path, "wb") as f:
                msgpack.dump(out, f)
