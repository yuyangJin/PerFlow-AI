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
import msgpack
from .event import Event
from .node import BaseNode, Node, TemplateNode, RefNode

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

    def __init__(self, metadata: Dict[str, Any] | None = None):
        self.metadata: Dict[str, Any] = metadata if metadata else {}
        # rank → pid → tid → ph -> Node
        self.ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]] = {}

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

    def get_node(self, rank: str, pid: str, tid: str, ph: str) -> Optional[BaseNode]:
        """Return the node for the specified rank, pid, tid and phase."""
        return self.ranks.get(rank, {}).get(pid, {}).get(tid, {}).get(ph)

    def set_node(self, rank: str, pid: str, tid: str, ph: str, node: BaseNode):
        """Set the node for the specified rank, pid, tid and phase."""
        self.ranks.setdefault(rank, {}).setdefault(pid, {}).setdefault(tid, {})[ph] = node

    def iter_nodes(self, rank: Optional[str] = None):
        """Iterate over all nodes in the trace, optionally filtering by rank."""
        rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]
        for r, processes in rank_items:
            for pid, threads in processes.items():
                for tid, phases in threads.items():
                    for ph, node in phases.items():
                        yield r, pid, tid, ph, node

    @classmethod
    @abstractmethod
    def from_json(cls, path: str) -> BaseTrace:
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

    def __init__(self, events: List[Dict[str, Any]] | None = None,
                 metadata: Dict[str, Any] | None = None):
        super().__init__(metadata)

        if events:
            self.add_events(events)

    @classmethod
    def from_json(cls, path: str) -> Trace:
        with open(path, 'r', encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        events: List[Dict[str, Any]] = data.get("traceEvents", [])
        metadata: Dict[str, Any] = {k: v for k, v in data.items() if k != "traceEvents"}
        return cls(events, metadata)

    def add_events(self, events: List[Dict[str, Any]], rank: str = "0"):
        """Add events to the trace."""
        if not events:
            return

        if rank not in self.ranks:
            self.ranks[rank] = {}

        for e in events:
            pid = e.pop("pid", 0)
            tid = e.pop("tid", 0)
            ph = e.pop("ph", "X")
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
            node = tid_layer.setdefault(str(ph), Node())

            node.add_events([Event(e)])

    def write_file(self, path: str, rank: str = "0", origin: bool = True):
        out = {}

        if origin:
            trace_events = []

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]

            for r, processes in rank_items:
                for pid, tids in processes.items():
                    for tid, phases in tids.items():
                        for ph, node in phases.items():
                            for e in node.get_events():
                                event_dict = e.to_dict().copy()
                                event_dict["pid"] = pid
                                if tid.startswith("stream "):
                                    event_dict["tid"] = tid.split(" ")[1]
                                else:
                                    event_dict["tid"] = tid
                                event_dict["ph"]  = ph
                                trace_events.append(event_dict)

            trace_events = sorted(trace_events, key=lambda x: x["ts"])

            out = {"traceEvents": trace_events}
            out.update(self.metadata)

        else:
            out["metadata"] = self.metadata
            out["ranks"] = {}

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]
            for r, processes in rank_items:
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

    def __init__(self, templates: Dict[str, TemplateNode],
                 ranks: Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]],
                 metadata: Dict[str, Any] | None = None):

        super().__init__(metadata)
        self.ranks = ranks

        self.templates: Dict[str, TemplateNode] = templates

    def segmented_linear_predictor_compress(self):
        """Segmented linear predictor compression for templates."""
        for node in self.templates.values():
            node.segmented_linear_predictor_compress()

    @classmethod
    def from_json(cls, path: str) -> CompressedTrace:
        if path.endswith(".json"):
            with open(path, 'r', encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        else:
            with open(path, 'rb') as f:
                data: Dict[str, Any] = msgpack.load(f, strict_map_key=False)

        templates: Dict[str, TemplateNode] = {}
        ranks: Dict[str, Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]]] = {}
        metadata: Dict[str, Any] = {}

        assert "templates" in data, "Invalid trace format"
        assert "ranks" in data, "Invalid trace format"
        assert "metadata" in data, "Invalid trace format"

        for rank, process_dict in data["ranks"].items():
            ranks[rank] = {}
            for pid, thread_dict in process_dict.items():
                ranks[rank][pid] = {}
                for tid, phase_dict in thread_dict.items():
                    ranks[rank][pid][tid] = {}
                    for ph, node_dict in phase_dict.items():
                        if "ref_node_id" in node_dict:
                            ranks[rank][pid][tid][ph] = \
                                RefNode.from_dict(node_dict, data["templates"], templates)
                        else:
                            ranks[rank][pid][tid][ph] = \
                                Node.from_dict(node_dict, data["templates"], templates)

        metadata = data["metadata"]

        return cls(templates, ranks, metadata)

    def write_file(self, path: str, rank: str = "0", origin: bool = False):
        out = {}

        if origin:
            out.update(self.metadata)
            trace_events = []

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]

            for r, processes in rank_items:
                for pid, tids in processes.items():
                    for tid, phases in tids.items():
                        for ph, node in phases.items():
                            for e in node.get_all_events():
                                event_dict = e.to_dict().copy()
                                event_dict["pid"] = pid
                                if tid.startswith("stream "):  
                                    event_dict["tid"] = tid.split(" ")[1]
                                else:
                                    event_dict["tid"] = tid
                                event_dict["ph"]  = ph
                                trace_events.append(event_dict)

            trace_events = sorted(trace_events, key=lambda x: x["ts"])

            out = {"traceEvents": trace_events}
            out.update(self.metadata)

        else:
            out["metadata"] = self.metadata
            out["templates"] = {}
            out["ranks"] = {}

            for i, template in self.templates.items():
                out["templates"][i] = template.to_dict()

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]
            for r, processes in rank_items:
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
