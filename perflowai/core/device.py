from dataclasses import dataclass
from enum import Enum

class DeviceType(Enum):
    GPU = "GPU"
    CPU = "CPU"

@dataclass
class Device:
    id: int
    type: DeviceType
    memory_capacity: int  # in MB
    bandwidth: float      # in GB/s
    compute_perf: float   # FLOPs per cycle
