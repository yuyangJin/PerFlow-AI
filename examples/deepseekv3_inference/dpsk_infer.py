import heapq
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import numpy as np

from perflowai.core import Request, Task, TaskPool, TaskType, Tasks
from perflowai.parallel import InferGraph
from perflowai.simulator import InferSimulator
from perflowai.core import DeviceConfig, DeviceType
from perflowai.core import Scheduler
from perflowai.core import ModelConfig
from perflowai.visualizer import TraceVisualizer, MemoryFootprintVisualizer

class ParallelStrategy(Enum):
    DATA = "data"
    MODEL = "model"
    EXPERT = "expert"

@dataclass
class ParallelConfig:
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1

# @dataclass
# class ModelConfig:
#     num_layers: int
#     hidden_size: int
#     num_experts: int = 0
#     moe_layers: List[int] = field(default_factory=list)
#     parallel_strategy: ParallelStrategy = ParallelStrategy.DATA



class FIFOScheduler(Scheduler):
    def __init__(self, max_num_tasks: int, pool: TaskPool):
        super().__init__()
        # max num tasks per dp instance
        self.max_num_tasks_per_dp = max_num_tasks
        # ndevs = get_dp_group().group_size
    
    def schedule(self, pool: TaskPool, ndevs):
        # # Simple round-robin assignment
        # assignment = {i: [] for i in range(len(devices))}
        # for i, req in enumerate(requests):
        #     assignment[i % len(devices)].append(req)
        #     print(f"Request {req.req_id} assigned to Device {i % len(devices)}")
        # return assignment

        task_lists = pool.get_prefill_task(self.max_num_tasks_per_dp, ndevs)

        if task_lists is not None: # prefill tasks
            return TaskType.PREFILL ,task_lists
        else:
            task_lists = pool.get_decode_task(self.max_num_tasks_per_dp, ndevs)

            if task_lists is not None: # decode tasks
                return TaskType.DECODE, task_lists

        # FIXME: currently we need all dp instances process same number of tasks
        # num = max(len(task_lists[i]) for i in range(ndevs))
        # assert all(
        #     len(task_lists[i]) == num for i in range(ndevs)
        # ), f"task lists {task_lists} do not have same length"
        return  TaskType.EMPTY, None

# class PerformanceSimulator:
#     def __init__(self, model_cfg: ModelConfig, devices: List[Device]):
#         self.model_cfg = model_cfg
#         self.devices = devices
#         self.memory_usage = {d.id: [] for d in devices}
#         self.timeline = []

#     def _calculate_memory(self, phase: PhaseType, batch_size: int) -> int:
#         # Simplified memory calculation
#         base_mem = 1024  # MB
#         if phase == PhaseType.PREFILL:
#             return base_mem * batch_size
#         elif phase == PhaseType.DECODE:
#             return base_mem * batch_size // 2
#         return base_mem // 4

#     # def _calculate_memory(self, phase: PhaseType, batch_size: int) -> int:
#     #     # More accurate memory estimation
#     #     hidden = self.model_cfg.hidden_size
#     #     if self.model_cfg.parallel_strategy == ParallelStrategy.MODEL:
#     #         per_device_layers = self.model_cfg.num_layers // len(self.devices)
#     #         layer_mem = 4 * hidden * hidden * 2  # 2 matrices per layer
#     #         return (layer_mem * per_device_layers) // (1024*1024)  # Convert to MB
        
#     #     # Similar calculations for other parallel strategies...

#     def _simulate_layer(self, device: Device, phase: PhaseType, batch_size: int) -> float:
#         # Simplified timing model
#         if phase == PhaseType.PREFILL:
#             return batch_size * 0.01  # 10ms per token
#         return 0.002  # 2ms per token

#     def simulate_request(self, request: Request, scheduler: Scheduler):
#         current_time = request.arrival_time
#         device_assignment = scheduler.schedule([request], self.devices)

#         for device_id, reqs in device_assignment.items():
#             device = self.devices[device_id]
#             for req in reqs:
#                 # Prefill phase
#                 prefill_time = self._simulate_layer(device, PhaseType.PREFILL, req.input_len)
#                 mem_usage = self._calculate_memory(PhaseType.PREFILL, req.input_len)
                
#                 self.timeline.append({
#                     "device": device.id,
#                     "phase": PhaseType.PREFILL,
#                     "start": current_time,
#                     "end": current_time + prefill_time,
#                     "req_id": req.req_id
#                 })
#                 print (f"Device {device.id} Prefill: {current_time} -> {current_time + prefill_time}")
#                 # Memory usage tracking
#                 self.memory_usage[device.id].append((current_time, mem_usage))
                
#                 current_time += prefill_time
                
#                 # Decode phase
#                 for _ in range(req.output_len):
#                     decode_time = self._simulate_layer(device, PhaseType.DECODE, 1)
#                     self.timeline.append({
#                         "device": device.id,
#                         "phase": PhaseType.DECODE,
#                         "start": current_time,
#                         "end": current_time + decode_time,
#                         "req_id": req.req_id
#                     })
#                     current_time += decode_time
#                 print (f"Device {device.id} Decode: {current_time - decode_time} -> {current_time}")
#                 # Memory usage tracking
#                 # Memory release
#                 self.memory_usage[device.id].append((current_time, 0))

def generate_requests(num_requests: int,   
                      max_input_len: int = 100,   
                      max_output_len: int = 100,   
                      max_arrival_time: float = 100.0) -> List[Request]:  
    requests = []  
    for req_id in range(num_requests):  
        input_len = random.randint(1, max_input_len)          # 随机输入长度  
        output_len = random.randint(1, max_output_len)        # 随机输出长度  
        start_time = round(random.uniform(0, max_arrival_time), 2)  # 随机到达时间  
        requests.append(Request(req_id, input_len, output_len, start_time))  
    
    return requests  



# Usage example
if __name__ == "__main__":
    # Configuration
    model_cfg = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2,
        num_experts=8,
        moe_layers=[4, 8, 16, 20],
    )

    device = DeviceConfig(id=0, type=DeviceType.GPU, memory_capacity=16384, memory_bandwidth=900, compute_flops=1e12)

    # Generate sample requests
    # requests = [
    #     Request(req_id=0, input_len=512, output_len=2, start_time=0),
    #     Request(req_id=1, input_len=256, output_len=10, start_time=2)
    # ]

    requests = generate_requests(num_requests=100, max_input_len=256, max_output_len=2048, max_arrival_time=3000.0) 

    print(f"Generated {len(requests)} requests")

    # 1. Create request node
    infer_graph = InferGraph(ndevs = 1)
    infer_graph.generate_nodes(requests)


    # Run simulation
    simulator = InferSimulator(infer_graph, requests)
    pool = simulator.get_task_pool()
    scheduler = FIFOScheduler(2048, pool)

    trace = simulator.simulate(scheduler, 
                                model_config = model_cfg,
                                device_config = device)

    print(simulator.get_metrics())

    # Visualization
    TraceVisualizer(trace).visualize()
    
    MemoryFootprintVisualizer(trace.get_memory_footprint()).visualize()
