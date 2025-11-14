from __future__ import annotations
import json
from .event import Event
from .node import BaseNode, Node, TemplateNode, RefNode
from .utils import logger
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import msgpack
import os
import tqdm

class BaseTrace:

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_ranks(self) -> List[str]:
        pass

    @abstractmethod
    def get_pids(self, rank: str) -> List[str]:
        pass

    @abstractmethod
    def get_tids(self, rank: str, pid: str) -> List[str]:
        pass

    @abstractmethod
    def get_node(self, rank: str, pid: str, tid: str) -> Optional[BaseNode]:
        pass

    @abstractmethod
    def set_node(self, rank: str, pid: str, tid: str, node):
        pass

    @abstractmethod
    def iter_nodes(self, rank: Optional[str] = None):
        pass
    
    @classmethod
    @abstractmethod
    def from_json(cls, path: str) -> BaseTrace:
        pass

    @abstractmethod
    def write_json_file(self, path: str, rank: str = "0", origin: bool = False):
        pass

class Trace(BaseTrace):
    def __init__(self, events: Optional[List[Dict[str, Any]]] = None, metadata: Optional[Dict[str, Any]] = None):
        # rank -> pid -> tid -> node
        self.ranks: Dict[str, Dict[str, Dict[str, Node]]] = {}
        self.metadata: Dict[str, Any] = metadata if metadata else {}

        if events:
            self.add_events(events)

    @classmethod
    def from_json(cls, path: str) -> Trace:
        with open(path, 'r') as f:
            data: Dict[str, Any] = json.load(f)
        events: List[Dict[str, Any]] = data.get("traceEvents", [])
        metadata: Dict[str, Any] = {k: v for k, v in data.items() if k != "traceEvents"}
        return cls(events, metadata)
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def get_ranks(self) -> List[str]:
        return list(self.ranks.keys())

    def get_pids(self, rank: str) -> List[str]:
        return list(self.ranks.get(rank, {}).keys())

    def get_tids(self, rank: str, pid: str) -> List[str]:
        return list(self.ranks.get(rank, {}).get(pid, {}).keys())

    def get_node(self, rank: str, pid: str, tid: str) -> Optional[Node]:
        return self.ranks.get(rank, {}).get(pid, {}).get(tid)

    def set_node(self, rank: str, pid: str, tid: str, node: Node):
        self.ranks.setdefault(rank, {}).setdefault(pid, {})[tid] = node

    def iter_nodes(self, rank: Optional[str] = None):
        rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]
        for r, processes in rank_items:
            for pid, tids in processes.items():
                for tid, node in tids.items():
                    yield r, pid, tid, node
    
    def add_events(self, events: List[Dict[str, Any]], rank: str = "0"):
        if not events:
            return

        if rank not in self.ranks:
            self.ranks[rank] = {}

        for e in events:
            pid = e.pop("pid", 0)
            tid = e.pop("tid", 0)
            e.pop("rank", None)

            rank_layer = self.ranks.setdefault(rank, {})
            pid_layer = rank_layer.setdefault(str(pid), {})
            node = pid_layer.setdefault(str(tid), Node())

            node.add_events([Event(e)])

    def write_json_file(self, path: str, rank: str = "0", origin: bool = True):
        out = {}

        if origin:
            trace_events = []

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]

            for r, processes in rank_items:
                for pid, tids in processes.items():
                    for tid, node in tids.items():
                        for e in node.get_events():
                            event_dict = e.to_dict().copy()
                            event_dict["pid"] = pid
                            event_dict["tid"] = tid
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
                    for tid, node in tids.items():
                        out["ranks"][r][pid][tid] = node.to_dict()

        ext = os.path.splitext(path)[1].lower()

        if ext == ".json":
            with open(path, "w") as f:
                json.dump(out, f)
        else:
            with open(path, "wb") as f:
                msgpack.dump(out, f)


class CompressedTrace(BaseTrace):
    def __init__(self, templates: Dict[str, TemplateNode], ranks: Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]], metadata: Dict[str, Any] = {}):

        self.templates: Dict[str, TemplateNode] = templates

        # rank -> pid -> tid -> node
        self.ranks: Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]] = ranks
        self.metadata: Dict[str, Any] = metadata if metadata else {}

    @classmethod
    def from_json(cls, path: str) -> CompressedTrace:
        if path.endswith(".json"):
            with open(path, 'r') as f:
                data: Dict[str, Any] = json.load(f)
        else:
            with open(path, 'rb') as f:
                data: Dict[str, Any] = msgpack.load(f, strict_map_key=False)

        templates: Dict[str, TemplateNode] = {}
        ranks: Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]] = {}
        metadata: Dict[str, Any] = {}

        assert "templates" in data, "Invalid trace format"
        assert "ranks" in data, "Invalid trace format"
        assert "metadata" in data, "Invalid trace format"

        for rank, process_dict in data["ranks"].items():
            ranks[rank] = {}
            for pid, thread_dict in process_dict.items():
                ranks[rank][pid] = {}
                for tid, node_dict in thread_dict.items():
                    if "ref_node_id" in node_dict:
                        ranks[rank][pid][tid] = RefNode.from_dict(node_dict, data["templates"], templates)
                    else:
                        ranks[rank][pid][tid] = Node.from_dict(node_dict, data["templates"], templates)

        metadata = data["metadata"]

        return cls(templates, ranks, metadata)

    def write_json_file(self, path: str, rank: str = "0", origin: bool = False):
        out = {}

        if origin:
            out.update(self.metadata)
            trace_events = []

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]

            for r, processes in rank_items:
                for pid, tids in processes.items():
                    for tid, node in tids.items():
                        for e in node.get_all_events():
                            event_dict = e.to_dict().copy()
                            event_dict["pid"] = pid
                            event_dict["tid"] = tid
                            trace_events.append(event_dict)

            trace_events = sorted(trace_events, key=lambda x: x["ts"])

            out = {"traceEvents": trace_events}
            out.update(self.metadata)

        else:
            out["metadata"] = self.metadata
            out["templates"] = {}
            out["ranks"] = {}

            for id, template in self.templates.items():
                out["templates"][id] = template.to_dict()

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]
            for r, processes in rank_items:
                out["ranks"][r] = {}
                for pid, tids in processes.items():
                    out["ranks"][r][pid] = {}
                    for tid, node in tids.items():
                        out["ranks"][r][pid][tid] = node.to_dict()

        ext = os.path.splitext(path)[1].lower()

        if ext == ".json":
            with open(path, "w") as f:
                json.dump(out, f)
        else:
            with open(path, "wb") as f:
                msgpack.dump(out, f)
