from __future__ import annotations
import json
from .event import Event
from .node import BaseNode, Node
from .utils import logger
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

class BaseTrace:

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_ranks(self) -> List[int]:
        pass

    @abstractmethod
    def get_pids(self, rank: int) -> List[int]:
        pass

    @abstractmethod
    def get_tids(self, rank: int, pid: int) -> List[int]:
        pass

    @abstractmethod
    def get_node(self, rank: int, pid: int, tid: int) -> Optional[BaseNode]:
        pass

    @abstractmethod
    def set_node(self, rank: int, pid: int, tid: int, node: BaseNode):
        pass

    @abstractmethod
    def iter_nodes(self, rank: Optional[int] = None):
        pass
    
    @classmethod
    @abstractmethod
    def from_json(cls, path: str) -> BaseTrace:
        pass

    @abstractmethod
    def write_json_file(self, path: str, rank: int = 0, origin: bool = False):
        pass

class Trace(BaseTrace):
    def __init__(self, events: Optional[List[Dict[str, Any]]] = None, metadata: Optional[Dict[str, Any]] = None):
        # rank -> pid -> tid -> node
        self.ranks: Dict[int, Dict[int, Dict[int, Node]]] = {}
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

    def get_ranks(self) -> List[int]:
        return list(self.ranks.keys())

    def get_pids(self, rank: int) -> List[int]:
        return list(self.ranks.get(rank, {}).keys())

    def get_tids(self, rank: int, pid: int) -> List[int]:
        return list(self.ranks.get(rank, {}).get(pid, {}).keys())

    def get_node(self, rank: int, pid: int, tid: int) -> Optional[Node]:
        return self.ranks.get(rank, {}).get(pid, {}).get(tid)

    def set_node(self, rank: int, pid: int, tid: int, node: Node):
        self.ranks.setdefault(rank, {}).setdefault(pid, {})[tid] = node

    def iter_nodes(self, rank: Optional[int] = None):
        rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]
        for r, processes in rank_items:
            for pid, tids in processes.items():
                for tid, node in tids.items():
                    yield r, pid, tid, node
    
    def add_events(self, events: List[Dict[str, Any]], rank: int = 0):
        if not events:
            return

        if rank not in self.ranks:
            self.ranks[rank] = {}

        for e in events:
            pid = e.pop("pid", 0)
            tid = e.pop("tid", 0)
            e.pop("rank", None)

            rank_layer = self.ranks.setdefault(rank, {})
            pid_layer = rank_layer.setdefault(pid, {})
            node = pid_layer.setdefault(tid, Node())

            node.add_events([Event(e)])

    def write_json_file(self, path: str, rank: int = 0, origin: bool = False):
        out = {}

        if origin:
            trace_events = []

            rank_items = self.ranks.items() if rank is None else [(rank, self.ranks.get(rank, {}))]

            for r, processes in rank_items:
                for pid, tids in processes.items():
                    for tid, node in tids.items():
                        for e in node.events:
                            event_dict = e.to_dict().copy()
                            event_dict["pid"] = pid
                            event_dict["tid"] = tid
                            trace_events.append(event_dict)

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

        with open(path, "w") as f:
            json.dump(out, f, indent=2)


class CompressedTrace(BaseTrace):
    def __init__(self, ranks: Dict[int, Dict[int, Dict[int, BaseNode]]], metadata: Dict[str, Any] = None):
        # rank -> pid -> tid -> node
        self.ranks: Dict[int, Dict[int, Dict[int, BaseNode]]] = ranks
        self.metadata: Dict[str, Any] = metadata if metadata else {}

    @classmethod
    def from_json(cls, path: str) -> CompressedTrace:
        # TODO:
        logger.error("CompressedTrace.write_json_file is not implemented yet.")

    def write_json_file(self, path: str, rank: int = 0, origin: bool = False):
        # TODO:
        logger.error("CompressedTrace.write_json_file is not implemented yet.")