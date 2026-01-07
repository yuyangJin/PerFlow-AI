from dataclasses import dataclass
from enum import Enum
from typing import Union, List


class DeviceType(Enum):
    GPU = "GPU"
    CPU = "CPU"


@dataclass
class DeviceConfig:
    id: Union[int, str]
    type: DeviceType
    memory_capacity: int  # in MB
    memory_bandwidth: float  # in GB/s
    compute_flops: float  # FLOPs per cycle

    # Network communication bandwidth.
    # Keep units consistent with memory_bandwidth (GB/s).
    # Defaults preserve backward-compatibility with existing examples/tests.
    intra_node_bandwidth: float = 0.0  # in GB/s (within a node)
    inter_node_bandwidth: float = 0.0  # in GB/s (across nodes)


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
