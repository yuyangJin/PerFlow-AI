from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Optional

from perflowai.core import DeviceConfig
from perflowai.simulator.kernel.kernel_simulator import Parameter
from perflowai.util.checks import require
from perflowai.util.tensor import dtype_bytes, numel
from perflowai.util.units import inter_node_bandwidth_Bps, intra_node_bandwidth_Bps
from perflowai.workflow.flow import FlowNode


class NetworkScope(Enum):
    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"


@dataclass(frozen=True)
class GroupTopology:
    """Describe how a collective group is laid out across nodes.

    Example: 2 machines, 8 ranks each => GroupTopology(num_nodes=2, ranks_per_node=8)
    and the total group size is 16.

    TODO: Multi node communication simulator models.
    """

    num_nodes: int
    ranks_per_node: int

    def group_size(self) -> int:
        return int(self.num_nodes) * int(self.ranks_per_node)


@dataclass(frozen=True)
class NetworkWorkload:
    bytes_per_rank: int
    total_bytes: int


@dataclass(frozen=True)
class NetworkSimulationResult:
    op: str
    scope: str
    group_size: int
    bytes_per_rank: int
    total_bytes: int
    comm_time_s: float
    time_s: float
    achieved_bytes_per_s: float
    achieved_GBps: float

    def to_dict(self) -> dict:
        return asdict(self)


class BaseNetworkSimulator(FlowNode, ABC):
    """Base class for network/communication simulators driven by Parameters.

    This is a lightweight cost model for communication time dominated by bandwidth.
    The bytes model intentionally matches the comm-bench scripts under
    `examples/comm-bench/` (bytes_per_rank/total_bytes definitions).

    TODO: Multi node communication simulator models.
    """

    def __init__(
            self,
            name: str,
            id: int | str,
            inputs: list[Parameter],
            outputs: list[Parameter],
            device_config: DeviceConfig,
            topology: GroupTopology,
            comm_coeff: float = 0.8,
            sm_usage: float = 0.0,
            workload_aspect: Optional[Callable[["BaseNetworkSimulator", NetworkWorkload], NetworkWorkload]] = None,
    ):
        super().__init__(name=name, id=id, inputs=inputs, outputs=outputs)
        self.m_device_config = device_config
        self.topology = topology
        self.group_size = int(self.topology.group_size())
        self.comm_coeff = float(comm_coeff)
        self.sm_usage = float(sm_usage)
        self.workload_aspect = workload_aspect
        self._cached_workload: NetworkWorkload | None = None
        self._validate_common()
        self.validate_parameters()

    def _validate_common(self) -> None:
        require(self.comm_coeff > 0.0, "comm_coeff must be > 0")

        require(self.topology.num_nodes > 0, "topology.num_nodes must be > 0")
        require(self.topology.ranks_per_node > 0, "topology.ranks_per_node must be > 0")
        require(self.group_size > 0, "topology.group_size() must be > 0")

        # Validate bandwidth availability.
        require(
            intra_node_bandwidth_Bps(self.m_device_config) > 0.0,
            "device_config.intra_node_bandwidth must be > 0",
        )
        if self.topology.num_nodes > 1:
            require(
                inter_node_bandwidth_Bps(self.m_device_config) > 0.0,
                "device_config.inter_node_bandwidth must be > 0",
            )

    def _comm_bandwidth_Bps(self, scope: NetworkScope) -> float:
        if scope == NetworkScope.INTRA_NODE:
            return intra_node_bandwidth_Bps(self.m_device_config)
        if scope == NetworkScope.INTER_NODE:
            return inter_node_bandwidth_Bps(self.m_device_config)
        raise ValueError(f"Unknown scope: {scope}")

    def _comm_latency_s(self, scope: NetworkScope) -> float:
        """Get latency for a given scope from microarch config."""
        if self.m_device_config.microarch is None:
            return 0.0

        latency_us = 0.0
        if scope == NetworkScope.INTRA_NODE:
            latency_us = self.m_device_config.microarch.comm.intra_node_latency_us
        elif scope == NetworkScope.INTER_NODE:
            latency_us = self.m_device_config.microarch.comm.inter_node_latency_us

        return latency_us * 1e-6

    @property
    @abstractmethod
    def op_type(self) -> str:
        """Return the canonical operation name (e.g. 'all_gather') for config lookups."""

    @property
    def effective_efficiency(self) -> float:
        """Get collective efficiency from microarch or fall back to comm_coeff."""
        if self.m_device_config.microarch and self.m_device_config.microarch.comm.collective_efficiency:
            effs = self.m_device_config.microarch.comm.collective_efficiency
            # Use self.op_type to look up efficiency
            if self.op_type in effs:
                return effs[self.op_type]
        return self.comm_coeff

    def _segments_total_bytes(self, workload: NetworkWorkload) -> list[tuple[NetworkScope, int]]:
        """Return a list of (scope, bytes) segments.

        - If topology spans multiple nodes: subclasses may override to provide a
          better intra/inter breakdown. Default is conservative: everything inter-node.
        """

        if self.topology.num_nodes <= 1:
            return [(NetworkScope.INTRA_NODE, int(workload.total_bytes))]

        # Conservative default: if a collective spans nodes and we don't have a
        # specific hierarchical model, assume traffic is dominated by inter-node.
        return [(NetworkScope.INTER_NODE, int(workload.total_bytes))]

    def _comm_time_s(self, workload: NetworkWorkload) -> float:
        total_time = 0.0
        eff = self.effective_efficiency
        for scope, bytes_total in self._segments_total_bytes(workload):
            bw = self._comm_bandwidth_Bps(scope) * eff
            latency = self._comm_latency_s(scope)

            # Transfer time + latency per segment
            # Note: We add latency per segment, modeling the startup cost of each phase
            transfer_time = float(bytes_total) / bw if bw > 0 else 0.0
            total_time += transfer_time + latency

        return float(total_time)

    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate that inputs/outputs Parameters match this collective op."""

    @abstractmethod
    def _workload(self) -> tuple[int, int]:
        """Return (bytes_per_rank, total_bytes)."""

    def workload(self) -> NetworkWorkload:
        if self._cached_workload is None:
            bpr, total = self._workload()
            if self.workload_aspect is not None:
                modified = self.workload_aspect(self, NetworkWorkload(bpr, total))
                bpr = modified.bytes_per_rank
                total = modified.total_bytes
            self._cached_workload = NetworkWorkload(int(bpr), int(total))
        return self._cached_workload

    def bytes_per_rank(self) -> int:
        return self.workload().bytes_per_rank

    def total_bytes(self) -> int:
        return self.workload().total_bytes

    def simulate(self) -> NetworkSimulationResult:
        wl = self.workload()
        total = float(wl.total_bytes)
        comm_time = self._comm_time_s(wl)
        time_s = comm_time
        achieved_bps = (total / time_s) if time_s > 0 else 0.0
        achieved_GBps = achieved_bps / 1e9

        return NetworkSimulationResult(
            op=str(self.m_name),
            scope="intra_node" if self.topology.num_nodes <= 1 else "hierarchical",
            group_size=int(self.group_size),
            bytes_per_rank=int(wl.bytes_per_rank),
            total_bytes=int(wl.total_bytes),
            comm_time_s=float(comm_time),
            time_s=float(time_s),
            achieved_bytes_per_s=float(achieved_bps),
            achieved_GBps=float(achieved_GBps),
        )


class AllGatherNetworkSimulator(BaseNetworkSimulator):
    """AllGather cost model.

    Matches comm-bench definition:
    - bytes_per_rank = tensor_bytes(x)
    - total_bytes = bytes_per_rank * group_size

    TODO: Multi node communication simulator models.
    """

    @property
    def op_type(self) -> str:
        return "all_gather"

    def __init__(
            self,
            device_config: DeviceConfig,
            x: Parameter,
            *,
            topology: GroupTopology,
            name: str = "all_gather",
            id: int | str = 0,
            comm_coeff: float = 0.8,
            workload_aspect: Optional[Callable[["BaseNetworkSimulator", NetworkWorkload], NetworkWorkload]] = None,
    ):
        super().__init__(
            name=name,
            id=id,
            inputs=[x],
            outputs=[],
            device_config=device_config,
            topology=topology,
            comm_coeff=comm_coeff,
            workload_aspect=workload_aspect,
        )

    def validate_parameters(self) -> None:
        require(len(self.m_inputs) == 1, "all_gather requires exactly 1 input tensor")
        x = self.m_inputs[0]
        require(x.shape is not None, "input tensor must have shape")

    def _workload(self) -> tuple[int, int]:
        x = self.m_inputs[0]
        bpr = x.get_size()
        total = bpr * self.group_size
        return bpr, total

    def _segments_total_bytes(self, workload: NetworkWorkload) -> list[tuple[NetworkScope, int]]:
        # If the group spans multiple nodes, model a simple 3-phase hierarchical allgather:
        # 1) intra-node gather within each node (p ranks): total bytes = bpr * p, across n nodes => bpr*p*n
        # 2) inter-node allgather among node leaders (n nodes) with payload size bpr*p per leader => bpr*p*n
        # 3) intra-node broadcast to all ranks within node of the full gathered result (size bpr*group_size)
        #    per node total bytes = (bpr*group_size)*p, across n nodes => bpr*group_size*p*n
        if self.topology.num_nodes <= 1:
            return [(NetworkScope.INTRA_NODE, int(workload.total_bytes))]

        bpr = int(workload.bytes_per_rank)
        n = int(self.topology.num_nodes)
        p = int(self.topology.ranks_per_node)
        g = int(self.group_size)

        intra_gather = bpr * p * n
        inter_allgather = bpr * p * n
        intra_bcast = bpr * g * p * n
        return [
            (NetworkScope.INTRA_NODE, intra_gather),
            (NetworkScope.INTER_NODE, inter_allgather),
            (NetworkScope.INTRA_NODE, intra_bcast),
        ]


class AllReduceNetworkSimulator(BaseNetworkSimulator):
    """AllReduce cost model.

    Matches comm-bench definition (simple and consistent across ops):
    - bytes_per_rank = tensor_bytes(x)
    - total_bytes = bytes_per_rank * group_size

    Note: This does not attempt to model ring/tree algorithm step counts.

    TODO: Multi node communication simulator models.
    """

    @property
    def op_type(self) -> str:
        return "all_reduce"

    def __init__(
            self,
            device_config: DeviceConfig,
            x: Parameter,
            *,
            topology: GroupTopology,
            name: str = "all_reduce",
            id: int | str = 0,
            comm_coeff: float = 0.8,
            workload_aspect: Optional[Callable[["BaseNetworkSimulator", NetworkWorkload], NetworkWorkload]] = None,
    ):
        super().__init__(
            name=name,
            id=id,
            inputs=[x],
            outputs=[],
            device_config=device_config,
            topology=topology,
            comm_coeff=comm_coeff,
            workload_aspect=workload_aspect,
        )

    def validate_parameters(self) -> None:
        require(len(self.m_inputs) == 1, "all_reduce requires exactly 1 input tensor")
        x = self.m_inputs[0]
        require(x.shape is not None, "input tensor must have shape")

    def _workload(self) -> tuple[int, int]:
        x = self.m_inputs[0]
        bpr = x.get_size()
        total = bpr * self.group_size
        return bpr, total

    def _segments_total_bytes(self, workload: NetworkWorkload) -> list[tuple[NetworkScope, int]]:
        # Simple hierarchical all-reduce model (size does not change across reduce):
        # 1) intra-node reduce: per node total bytes = bpr * p, across n nodes => bpr*p*n
        # 2) inter-node all-reduce among leaders: total bytes = bpr * n
        # 3) intra-node broadcast: per node total bytes = bpr * p, across n nodes => bpr*p*n
        if self.topology.num_nodes <= 1:
            return [(NetworkScope.INTRA_NODE, int(workload.total_bytes))]

        bpr = int(workload.bytes_per_rank)
        n = int(self.topology.num_nodes)
        p = int(self.topology.ranks_per_node)

        intra_reduce = bpr * p * n
        inter_reduce = bpr * n
        intra_bcast = bpr * p * n
        return [
            (NetworkScope.INTRA_NODE, intra_reduce),
            (NetworkScope.INTER_NODE, inter_reduce),
            (NetworkScope.INTRA_NODE, intra_bcast),
        ]


class AllToAllNetworkSimulator(BaseNetworkSimulator):
    """AllToAll cost model.

    Matches comm-bench behavior for non-divisible numel:
    - chunk_numel = max(numel // group_size, 1)
    - effective_numel = chunk_numel * group_size
    - bytes_per_rank = effective_numel * elem_size
    - total_bytes = bytes_per_rank * group_size

    TODO: Multi node communication simulator models.
    """

    @property
    def op_type(self) -> str:
        return "all_to_all"

    def __init__(
            self,
            device_config: DeviceConfig,
            x: Parameter,
            *,
            topology: GroupTopology,
            name: str = "all_to_all",
            id: int | str = 0,
            comm_coeff: float = 0.8,
            workload_aspect: Optional[Callable[["BaseNetworkSimulator", NetworkWorkload], NetworkWorkload]] = None,
    ):
        super().__init__(
            name=name,
            id=id,
            inputs=[x],
            outputs=[],
            device_config=device_config,
            topology=topology,
            comm_coeff=comm_coeff,
            workload_aspect=workload_aspect,
        )

    def validate_parameters(self) -> None:
        require(len(self.m_inputs) == 1, "all_to_all requires exactly 1 input tensor")
        x = self.m_inputs[0]
        require(x.shape is not None, "input tensor must have shape")

    def _workload(self) -> tuple[int, int]:
        x = self.m_inputs[0]
        base_numel = numel(x.shape)
        chunk_numel = max(base_numel // self.group_size, 1)
        effective_numel = chunk_numel * self.group_size
        bpr = effective_numel * dtype_bytes(x.dtype)
        total = bpr * self.group_size
        return bpr, total


class ReduceScatterNetworkSimulator(BaseNetworkSimulator):
    """ReduceScatter cost model.

    Uses the same effective-numel logic as comm-bench:
    - chunk_numel = max(numel // group_size, 1)
    - effective_numel = chunk_numel * group_size
    - bytes_per_rank = effective_numel * elem_size
    - total_bytes = bytes_per_rank * group_size

    TODO: Multi node communication simulator models.
    """

    @property
    def op_type(self) -> str:
        return "reduce_scatter"

    def __init__(
            self,
            device_config: DeviceConfig,
            x: Parameter,
            *,
            topology: GroupTopology,
            name: str = "reduce_scatter",
            id: int | str = 0,
            comm_coeff: float = 0.8,
            workload_aspect: Optional[Callable[["BaseNetworkSimulator", NetworkWorkload], NetworkWorkload]] = None,
    ):
        super().__init__(
            name=name,
            id=id,
            inputs=[x],
            outputs=[],
            device_config=device_config,
            topology=topology,
            comm_coeff=comm_coeff,
            workload_aspect=workload_aspect,
        )

    def validate_parameters(self) -> None:
        require(len(self.m_inputs) == 1, "reduce_scatter requires exactly 1 input tensor")
        x = self.m_inputs[0]
        require(x.shape is not None, "input tensor must have shape")

    def _workload(self) -> tuple[int, int]:
        x = self.m_inputs[0]
        base_numel = numel(x.shape)
        chunk_numel = max(base_numel // self.group_size, 1)
        effective_numel = chunk_numel * self.group_size
        bpr = effective_numel * dtype_bytes(x.dtype)
        total = bpr * self.group_size
        return bpr, total


class P2PSimulator(BaseNetworkSimulator):
    """
    Simulates a Point-to-Point transfer.
    For simplicity, modeled as a collective of size 2 (Src, Dst),
    but we purely calculate duration based on BW and size.
    """

    @property
    def op_type(self) -> str:
        return "p2p"

    def __init__(self,
                 device_config: DeviceConfig,
                 size_bytes: int,
                 is_inter_node: bool,
                 name: str = "p2p",
                 id: "int | str" = 0):
        # We construct a dummy topology
        topo = GroupTopology(num_nodes=2 if is_inter_node else 1, ranks_per_node=1)

        super().__init__(
            name=name,
            id=id,
            inputs=[],
            outputs=[],
            device_config=device_config,
            topology=topo
        )
        self.size_bytes = size_bytes
        self.is_inter_node = is_inter_node

    def validate_parameters(self) -> None:
        pass

    def _workload(self) -> tuple[int, int]:
        return self.size_bytes, self.size_bytes

    def _segments_total_bytes(self, workload: NetworkWorkload) -> list[tuple[NetworkScope, int]]:
        scope = NetworkScope.INTER_NODE if self.is_inter_node else NetworkScope.INTRA_NODE
        return [(scope, workload.total_bytes)]
