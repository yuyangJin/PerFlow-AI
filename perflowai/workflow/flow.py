from __future__ import annotations

'''
@module workflow
'''

'''
@class FlowNode
A FlowNode is a node in a flow graph.
'''

from abc import ABC
from typing import Any, Optional


class Parameter:
    """A flexible value descriptor passed between FlowNodes.

    Historically, `Parameter` was used primarily for operator-level simulation
    (shape/dtype driven). For graph-level / parallel-level simulation, we also
    need to attach placement and sharding semantics without forcing those
    concepts into the operator simulator.

    Compatibility:
    - The original constructor args (name/dtype/shape/value/trainable) are kept.
    - Additional optional fields are provided via keyword args.
    """

    def __init__(
        self,
        name: str,
        dtype: str,
        shape: Optional[tuple[int, ...]] = None,
        value: Any = None,
        trainable: bool = False,
        *,
        kind: str = "tensor",
        meta: Optional[dict[str, Any]] = None,
        placement: Any = None,
        sharding: Any = None,
        **extra_meta: Any,
    ):
        """Create a Parameter.

        Args:
            name: Identifier of the value (e.g., "Q", "weight", "kv_cache").
            dtype: Data type name (kept as a string; see perflowai.util.tensor).
            shape: Tensor shape, when applicable.
            value: Optional concrete value.
            trainable: Whether this value is trainable (e.g., weights).
            kind: Semantic kind (e.g., "tensor", "activation", "weight", "grad", "control", "input", "output", etc). // TODO: define enum?
            meta: Arbitrary metadata for graph-/parallel-level simulation. // TODO: define class?
            placement: Optional placement info (device/stage/rank/etc.). // TODO: define enum?
            sharding: Optional sharding info (tp/dp/pp partitioning, layouts, etc.).
            **extra_meta: Convenience for adding extra metadata keys.
        """

        self.name = str(name)
        self.dtype = str(dtype)
        self.shape = shape
        self.value = value
        self.trainable = bool(trainable)

        self.kind = str(kind)
        self.placement = placement
        self.sharding = sharding

        self.meta: dict[str, Any] = dict(meta) if meta is not None else {}
        if extra_meta:
            self.meta.update(extra_meta)

    def get_meta(self, key: str, default: Any = None) -> Any:
        return self.meta.get(key, default)

    def set_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    def __repr__(self) -> str:
        core = f"name={self.name}, dtype={self.dtype}, shape={self.shape}, trainable={self.trainable}, kind={self.kind}"
        if self.placement is None and self.sharding is None and not self.meta:
            return f"Parameter({core})"
        return f"Parameter({core}, placement={self.placement}, sharding={self.sharding}, meta_keys={sorted(self.meta.keys())})"

class FlowNode(ABC):
    def __init__(self, name: str, id: str, inputs: list[Parameter], outputs: list[Parameter]):
        self.m_name = name
        self.m_id = id
        self.m_inputs = inputs
        self.m_outputs = outputs

    def __str__(self):
        return f"FlowNode({self.m_name})"

    def set_inputs(self, inputs: list[Parameter]):
        self.m_inputs = inputs

    def set_outputs(self, outputs: list[Parameter]):
        self.m_outputs = outputs

    def get_inputs(self) -> list[Parameter]:
        return self.m_inputs

    def get_outputs(self) -> list[Parameter]:
        return self.m_outputs

    # @abstractmethod
    def run(self):
        print('FlowNode runs virtially.')
        pass

'''
@class FlowGraph
A FlowGraph is a diagram of tasks.
'''

class FlowGraph:
    def __init__(self):

        '''
        Dict<int, FlowNode> m_nodes
        The key is the node id, the value is the node.
        '''
        self.m_nodes = dict()
        
        '''
        Dict<int, List<int>>> m_edges
        The first int is the source node id, 
        the second List is the destination node ids.
        '''
        self.m_edges = dict()

    def get_node_by_id(self, id):
        return self.m_nodes[id]
    
    def get_next_nodelist_by_id(self, id):
        return self.m_edges[id]

    def add_node(self, node):
        self.m_nodes[node.m_id] = node
    
    def add_edge(self, src_node, dst_node):
        if src_node.m_id not in self.m_edges:
            self.m_edges[src_node.m_id] = []
        self.m_edges[src_node.m_id].append(dst_node.m_id)

    ''' 
    check the graph to ensure the output of 
    one node is the input of another node 
    '''
    def check(self):
        '''
        Not implemented yet
        '''
        pass

    def run(self, *args, **kwargs):
        # visited = set()

        # def traverse(node_id):
        #     if node_id in visited:
        #     return
        #     visited.add(node_id)
        #     node = self.get_node_by_id(node_id)
        #     print(f"Visiting node: {node}")
        #     for next_node_id in self.get_next_nodelist_by_id(node_id):
        #     traverse(next_node_id)

        # for node_id in self.m_nodes:
        #     traverse(node_id)

        for node in self.m_nodes.values():
            node.run(*args, **kwargs)