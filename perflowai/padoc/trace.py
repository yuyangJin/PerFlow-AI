from __future__ import annotations
import json
from .event import Event
from .node import BaseNode, Node
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

class BaseTrace:
    
    @classmethod
    @abstractmethod
    def from_json(cls, path: str) -> BaseTrace:
        pass

    @abstractmethod
    def write_json_file(self, path: str):
        pass

class Trace(BaseTrace):
    def __init__(self, events: Optional[List[Dict[str, Any]]] = None):
        self.root: BaseNode = Node([Event(e) for e in events]) if events else Node()
        # TODO: 增加metadata，比如deviceinfo等

    @classmethod
    def from_json(cls, path: str) -> Trace:
        with open(path, 'r') as f:
            data: Dict[str, Any] = json.load(f)
        events: List[Dict[str, Any]] = data.get("traceEvents", [])
        return cls(events)

    def write_json_file(self, path: str):
        with open(path, 'w') as f:
            json.dump({"traceEvents": [e.to_dict() for e in self.root.events]}, f, indent=2)


class CompressedTrace(BaseTrace):
    def __init__(self, root: BaseNode):
        self.root: BaseNode = root

    @classmethod
    def from_json(cls, path: str) -> CompressedTrace:
        # TODO:
        pass

    def write_json_file(self, path):
        with open(path, 'w') as f:
            json.dump(self.root.to_dict(), f, indent=2)