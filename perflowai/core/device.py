from dataclasses import dataclass
from enum import Enum

class DeviceType(Enum):
    GPU = "GPU"
    CPU = "CPU"

@dataclass
class DeviceConfig:
    id: int
    type: DeviceType
    memory_capacity: int  # in MB
    memory_bandwidth: float      # in GB/s
    compute_flops: float   # FLOPs per cycle

    # Network communication bandwidth.
    # Keep units consistent with memory_bandwidth (GB/s).
    # Defaults preserve backward-compatibility with existing examples/tests.
    intra_node_bandwidth: float = 0.0  # in GB/s (within a node)
    inter_node_bandwidth: float = 0.0  # in GB/s (across nodes)
