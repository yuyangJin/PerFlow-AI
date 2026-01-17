from typing import List, Union

from perflowai import DeviceConfig, NodeConfig
from perflowai.simulator.kernel.kernel_simulator import Parameter


class DeviceInstance:
    def __init__(self,
                 config: DeviceConfig,
                 ):
        self.config = config
        self.holding_parameter: List[Parameter] = []

    def add_parameter(self, parameter: Parameter):
        self.holding_parameter.append(parameter)

    def get_parameter(self, id) -> Union[Parameter, None]:
        for parameter in self.holding_parameter:
            if parameter.id == id:
                return parameter
        return None

    def remove_parameter(self, parameter: Parameter):
        self.holding_parameter = [
            it for it in self.holding_parameter if parameter.id != it.id
        ]

    @property
    def id(self):
        return self.config.id

    def get_memory_usage_bytes(self) -> int:
        return sum(p.get_size() for p in self.holding_parameter)


class NodeInstance:
    def __init__(self, nodeConfig: NodeConfig):
        self.host = DeviceInstance(nodeConfig.host)
        self.gpu_devices = [DeviceInstance(gpu) for gpu in nodeConfig.gpu_devices]

    @property
    def id(self):
        return self.host.id
