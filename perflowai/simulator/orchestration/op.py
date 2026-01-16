from typing import Optional, Callable

from perflowai import BaseKernelSimulator, DeviceConfig
from perflowai.simulator.kernel.kernel_simulator import Parameter, Workload, KernelSimulationResult


class MallocSimulator(BaseKernelSimulator):
    def __init__(
            self,
            device_config: DeviceConfig,
            size: int,
            allocated: Parameter,
            *,
            name: str = "malloc",
            memory_coeff: float = 0.8,
            workload_aspect: Optional[Callable[["BaseKernelSimulator", Workload], Workload]] = None,
    ):
        super().__init__(
            name=name,
            id=name,
            inputs=[],
            outputs=[allocated],
            device_config=device_config,
            compute_coeff=1,
            memory_coeff=memory_coeff,
            workload_aspect=workload_aspect,
        )
        self.size = size

    def validate_parameters(self) -> None:
        pass

    def _workload(self) -> tuple[int, int, int]:
        return 0, 0, 0

    def simulate(self) -> KernelSimulationResult:
        return KernelSimulationResult(
            flops=0,
            bytes_accessed=0,
            peak_memory_bytes=self.size,
            compute_time_s=0,
            memory_time_s=0,
            time_s=0,
            achieved_bytes_per_s=0,
            achieved_flops_per_s=0,
            bottleneck="none",
        )


class FreeSimulator(BaseKernelSimulator):
    def __init__(
            self,
            device_config: DeviceConfig,
            size: int,
            allocated: Parameter,
            *,
            name: str = "malloc",
            memory_coeff: float = 0.8,
            workload_aspect: Optional[Callable[["BaseKernelSimulator", Workload], Workload]] = None,
    ):
        super().__init__(
            name=name,
            id=name,
            inputs=[allocated],
            outputs=[],
            device_config=device_config,
            compute_coeff=1,
            memory_coeff=memory_coeff,
            workload_aspect=workload_aspect,
        )
        self.size = size

    def validate_parameters(self) -> None:
        pass

    def _workload(self) -> tuple[int, int, int]:
        return 0, 0, 0

    def simulate(self) -> KernelSimulationResult:
        return KernelSimulationResult(
            flops=0,
            bytes_accessed=0,
            peak_memory_bytes=self.size,
            compute_time_s=0,
            memory_time_s=0,
            time_s=0,
            achieved_bytes_per_s=0,
            achieved_flops_per_s=0,
            bottleneck="none",
        )
