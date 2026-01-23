from dataclasses import dataclass, field
from enum import Enum
from typing import Union, List, Optional, Tuple, Dict


class DeviceType(Enum):
    GPU = "GPU"
    CPU = "CPU"
    ACCELERATOR = "ACCELERATOR"


class Topology(Enum):
    UNKNOWN = "UNKNOWN"
    RING = "RING"
    TREE = "TREE"
    MESH_2D = "MESH_2D"
    TORUS_2D = "TORUS_2D"
    FULLY_CONNECTED = "FULLY_CONNECTED"


@dataclass
class TensorCoreConfig:
    """
    Tensor/matrix engine capability.
    - tile_mnk: the native MMA tile shape (M, N, K) for a single MMA op.
    - peak_tflops: OPTIONAL override. If None, peak is derived from other fields if you choose to.
    """
    enabled: bool = False
    tile_mnk: Tuple[int, int, int] = (16, 16, 16)

    # If you prefer "peak at chip-level" (common in public specs), set this directly.
    # Units: TFLOPs (10^12 FLOPs/s) at max clocks for a given dtype.
    peak_tflops_by_dtype: Dict[str, float] = field(default_factory=dict)

    # If you prefer "per cycle" style modeling, you can set this instead:
    # Units: FLOPs/cycle at chip-level for a given dtype.
    peak_flops_per_cycle_by_dtype: Dict[str, float] = field(default_factory=dict)


@dataclass
class VectorCoreConfig:
    """
    Scalar/vector (non-tensor) compute capability.
    """
    peak_tflops_by_dtype: Dict[str, float] = field(default_factory=dict)
    peak_flops_per_cycle_by_dtype: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComputeConfig:
    """
    Compute-side microarchitecture.
    - cluster_count: e.g., SM count (NVIDIA), CU count (AMD), NPU-core clusters, etc.
    - warp_size: optional. Mostly useful if you model occupancy / scheduling.
    """
    cluster_count: int = 0
    warp_size: int = 32

    tensor: TensorCoreConfig = field(default_factory=TensorCoreConfig)
    vector: VectorCoreConfig = field(default_factory=VectorCoreConfig)

    # Optional ceiling factors if you want quick, coarse realism without deep modeling.
    # Typical range: 0.5~1.0, default 1.0 (no penalty).
    issue_efficiency: float = 1.0  # instruction issue / scheduling effectiveness
    pipeline_efficiency: float = 1.0  # pipeline utilization for steady-state kernels


@dataclass
class CacheLevelConfig:
    """
    Generic cache level model.
    - size_kb: capacity (KB)
    - bandwidth_gbs: sustainable bandwidth (GB/s)
    - latency_cycles: optional if you want latency-aware modeling
    """
    size_kb: int = 0
    bandwidth_gbs: float = 0.0
    latency_cycles: int = 0


@dataclass
class SharedMemoryConfig:
    """
    On-chip scratchpad / shared memory.
    """
    size_kb_per_cluster: int = 0  # KB per SM/CU/cluster
    bandwidth_gbs: float = 0.0  # chip-level effective GB/s (or per-cluster if you prefer, but be consistent)
    bank_count: int = 0  # optional (for bank conflict modeling)
    bank_width_bytes: int = 0  # optional


@dataclass
class DramConfig:
    """
    Off-chip memory.
    """
    capacity_mb: int = 0
    bandwidth_gbs: float = 0.0
    latency_cycles: int = 0  # optional; can be left 0 if you do bandwidth-only


@dataclass
class MemoryHierarchyConfig:
    """
    Memory hierarchy.
    """
    dram: DramConfig = field(default_factory=DramConfig)
    l2: CacheLevelConfig = field(default_factory=CacheLevelConfig)
    l1: CacheLevelConfig = field(default_factory=CacheLevelConfig)
    shared: SharedMemoryConfig = field(default_factory=SharedMemoryConfig)

    # Memory-level parallelism proxy. Helps bandwidth-only models behave better for small ops.
    max_outstanding_loads_per_cluster: int = 0


@dataclass
class DataMovementConfig:
    """
    Specialized data-movement engines (e.g., TMA / async copy / DMA engines).
    """
    # Tensor Memory Accelerator-like (TMA) features
    tma_enabled: bool = False
    tma_buffer_kb_per_cluster: int = 0
    tma_max_inflight_per_cluster: int = 0

    # Effective bandwidth for bulk copies into shared/on-chip (GB/s).
    # If you don't know, leave 0 and fall back to shared/l2 bandwidth in your model.
    tma_copy_bandwidth_gbs: float = 0.0

    supports_async_copy: bool = False
    supports_multicast: bool = False
    supports_layout_transform: bool = False  # swizzle/transpose-like assist


@dataclass
class InterconnectConfig:
    """
    On-chip interconnect / NoC / cross-cluster communication.
    """
    topology: Topology = Topology.UNKNOWN
    noc_bandwidth_gbs: float = 0.0
    noc_latency_cycles: int = 0

    # Useful if you model reductions or collective-like ops on-chip.
    reduction_bandwidth_gbs: float = 0.0


@dataclass
class CommConfig:
    """
    Inter-device / inter-node communication.
    Keep units consistent with other bandwidth fields (GB/s).
    Latency optional but very useful for small tensors / sync-heavy workloads.
    """
    intra_node_bandwidth_gbs: float = 0.0
    inter_node_bandwidth_gbs: float = 0.0

    intra_node_latency_us: float = 0.0
    inter_node_latency_us: float = 0.0

    topology: Topology = Topology.UNKNOWN

    # Optional efficiency factors for collectives (0~1). If absent, assume 1.
    collective_efficiency: Dict[str, float] = field(default_factory=dict)
    # e.g. {"allreduce": 0.75, "allgather": 0.80}


@dataclass
class MicroArchConfig:
    """
    Full microarchitecture config (L1).
    """
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    memory: MemoryHierarchyConfig = field(default_factory=MemoryHierarchyConfig)
    datamove: DataMovementConfig = field(default_factory=DataMovementConfig)
    interconnect: InterconnectConfig = field(default_factory=InterconnectConfig)
    comm: CommConfig = field(default_factory=CommConfig)

    # Calibration knobs (empirical). Keep it simple:
    # - bandwidth_scale: multiplies memory-related bandwidths
    # - compute_scale: multiplies compute throughput
    calibration: Dict[str, float] = field(default_factory=dict)
    # e.g. {"bandwidth_scale": 0.85, "compute_scale": 0.90}


# ----------------------------
# Backward-compatible wrapper
# ----------------------------

@dataclass
class DeviceConfig:
    id: Union[int, str]
    type: DeviceType

    # L0 fields (kept for compatibility / simple configs)
    memory_capacity: int  # in MB
    memory_bandwidth: float  # in GB/s
    compute_flops: float  # FLOPs per cycle (chip-level coarse)

    intra_node_bandwidth: float = 0.0  # in GB/s
    inter_node_bandwidth: float = 0.0  # in GB/s

    # L1 (microarchitecture)
    microarch: Optional[MicroArchConfig] = None

    # Metadata (optional but convenient)
    vendor: str = ""
    model: str = ""
    notes: str = ""

    def effective_intra_node_bandwidth(self) -> float:
        if self.microarch is not None and self.microarch.comm.intra_node_bandwidth_gbs > 0:
            return self.microarch.comm.intra_node_bandwidth_gbs
        return self.intra_node_bandwidth

    def effective_inter_node_bandwidth(self) -> float:
        if self.microarch is not None and self.microarch.comm.inter_node_bandwidth_gbs > 0:
            return self.microarch.comm.inter_node_bandwidth_gbs
        return self.inter_node_bandwidth


class NodeConfig:
    def __init__(self,
                 host: DeviceConfig,
                 gpu_devices: List[DeviceConfig] = None,
                 ):
        if gpu_devices is None:
            gpu_devices = []

        assert (host.type == DeviceType.CPU)
        for gpu in gpu_devices:
            assert (gpu.type == DeviceType.GPU)
        self.host = host
        self.gpu_devices = gpu_devices

    @property
    def id(self):
        return self.host.id
