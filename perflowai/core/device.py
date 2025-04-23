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
