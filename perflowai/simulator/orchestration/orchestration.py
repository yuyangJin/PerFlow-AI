from typing import List, Union

from perflowai.workflow import FlowNode
from perflowai.simulator.orchestration.device_info import DeviceInstance, NodeInstance
from perflowai.simulator.orchestration.op import FreeSimulator, MallocSimulator
from perflowai.simulator.orchestration.task import Task, Assignment


def _workload_time_s(workload: FlowNode) -> float:
    """Best-effort: extract an estimated runtime from a workload FlowNode.

    Supported conventions in this repo:
    - kernel/network simulators expose .simulate().time_s
    - core Event has .get_duration(), but some wrappers may forward that.

    Raise ValueError if no time information is available.
    """

    if hasattr(workload, "simulate"):
        res = workload.simulate()
        if hasattr(res, "time_s"):
            return float(res.time_s)

    if hasattr(workload, "get_duration"):
        return float(workload.get_duration())

    raise ValueError(
        f"Cannot extract estimated runtime from workload={type(workload)}. "
        f"Expected workload.simulate().time_s or workload.get_duration()."
    )


def check_node_id_unique(nodes: List[NodeInstance]):
    ids = set()
    for n in nodes:
        if n.id in ids:
            raise ValueError(f"Duplicate NodeInstance id found: '{n.id}'")
        ids.add(n.id)

        for g in n.gpu_devices:
            if g.id in ids:
                raise ValueError(f"Duplicate DeviceInstance id found: '{g.id}'")
            ids.add(g.id)


def check_task_id_unique(tasks):
    task_ids = set()
    workload_ids = set()
    for t in tasks:
        if t.id in task_ids:
            raise ValueError(f"Duplicate Task id found: '{t.id}'")
        task_ids.add(t.id)

        if t.workload.id in workload_ids:
            raise ValueError(f"Duplicate Workload id found: '{t.workload.id}'")
        workload_ids.add(t.workload.id)


def is_memory_exists(device: DeviceInstance, t: Task):
    mems = t.workload.get_inputs() + t.workload.get_outputs()
    for mem in mems:
        if device.get_parameter(mem.id) is None:
            return False
    return True


def check_memory_exists(device: DeviceInstance, t: Task):
    if not is_memory_exists(device, t):
        raise ValueError(f"Device {device.id} does not have parameter {t.id} required by task {t.id}")


def check_device_capacity(device: DeviceInstance, t: Task):
    total_mem = sum(p.get_size() for p in device.holding_parameter)
    mems = t.workload.get_inputs() + t.workload.get_outputs()
    req_mem = sum(mem.get_size() for mem in mems)
    if total_mem + req_mem > device.config.memory_capacity * 1024 * 1024:
        raise ValueError(f"Device {device.id} exceeds memory capacity when running task {t.id}: "
                         f"holding {total_mem} + required {req_mem} > capacity {device.config.memory_capacity_bytes}")


class OrchestrationEstimateResult:
    def __init__(self):
        self.time_s: float = 0
        self.max_memory_usage_bytes: dict[str, int] = {}

    def update_max_memory_usage(self, device_id: str, usage_bytes: int):
        if device_id not in self.max_memory_usage_bytes:
            self.max_memory_usage_bytes[device_id] = usage_bytes
        else:
            self.max_memory_usage_bytes[device_id] = max(self.max_memory_usage_bytes[device_id], usage_bytes)


class OrchestrationResult:
    def __init__(self,
                 nodes: List[NodeInstance],
                 tasks: List[Task],
                 assignments: List[Assignment]
                 ):
        check_node_id_unique(nodes)
        check_task_id_unique(tasks)

        for assignment in assignments:
            if not any(t.id == assignment.task_id for t in tasks):
                raise ValueError(f"Assignment references unknown task_id '{assignment.task_id}'")

            if not any(d.id == assignment.device_id for n in nodes for d in [n.host] + n.gpu_devices):
                raise ValueError(f"Assignment references unknown device_id '{assignment.device_id}'")

        self.nodes = nodes
        self.tasks = tasks
        self.assignments = assignments

    def task_by_id(self) -> dict[str, Task]:
        return {t.id: t for t in self.tasks}

    def device_by_task_id(self) -> dict[str, Union[int, str]]:
        return {a.task_id: a.device_id for a in self.assignments}

    def get_device_by_device_id(self, device_id: Union[int, str]) -> DeviceInstance:
        for n in self.nodes:
            if n.host.id == device_id:
                return n.host
            for gpu in n.gpu_devices:
                if gpu.id == device_id:
                    return gpu
        raise ValueError(f"Device with id '{device_id}' not found in orchestration result.")

    def estimate_makespan_s(self) -> OrchestrationEstimateResult:
        """Estimate end-to-end runtime (makespan) from an OrchestrationResult.

        Assumptions (matching your description):
        - device_id is GPU-granularity
        - one stream per device => tasks on the same device execute serially
        - cross-device overlap is allowed (different device_ids can run in parallel)
        - dependencies are Task.run_after (finish-to-start)

        Returns: makespan in seconds.

        TODO: compute/comm overlap
        TODO: multi task per GPU stream
        TODO: strong collective semantics
        TODO: stall / recompute
        """

        tasks = self.task_by_id()
        device_of = self.device_by_task_id()

        # Validate references early.
        for tid, t in tasks.items():
            for dep in t.run_after:
                if dep not in tasks:
                    raise ValueError(f"Task '{tid}' depends on unknown task '{dep}'")
            if tid not in device_of:
                raise ValueError(f"Task '{tid}' has no Assignment (device_id missing)")

        # Topological processing with Kahn's algorithm.
        in_deg: dict[str, int] = {tid: 0 for tid in tasks}
        succ: dict[str, list[str]] = {tid: [] for tid in tasks}
        for tid, t in tasks.items():
            for dep in t.run_after:
                succ[dep].append(tid)
                in_deg[tid] += 1

        ready = [tid for tid, d in in_deg.items() if d == 0]

        # For deterministic output.
        ready.sort()

        earliest_finish: dict[str, float] = {}
        device_available: dict[Union[int, str], float] = {}

        processed = 0
        ret = OrchestrationEstimateResult()
        while ready:
            tid = ready.pop(0)
            t = tasks[tid]

            dep_ready_time = 0.0
            if t.run_after:
                dep_ready_time = max(earliest_finish[d] for d in t.run_after)

            device_id = device_of[tid]
            device = self.get_device_by_device_id(device_id)

            dev_ready_time = float(device_available.get(device_id, 0.0))
            start = max(dep_ready_time, dev_ready_time)

            if isinstance(t.workload, FreeSimulator):
                inputs = t.workload.get_inputs()
                for mem in inputs:
                    device.remove_parameter(mem)
            if isinstance(t.workload, MallocSimulator):
                outputs = t.workload.get_outputs()
                for mem in outputs:
                    device.add_parameter(mem)

            check_device_capacity(device, t)

            if isinstance(t.workload, FreeSimulator):
                exists = is_memory_exists(device, t)
                if exists:
                    raise ValueError(f"Task {t.id} is Free but parameters still exist on device {device.id}")
            elif isinstance(t.workload, MallocSimulator):
                exists = is_memory_exists(device, t)
                if not exists:
                    raise ValueError(f"Task {t.id} is Malloc but parameters do not exist on device {device.id}")
            else:
                check_memory_exists(device, t)

            dur = _workload_time_s(t.workload)
            finish = start + float(dur)

            earliest_finish[tid] = finish
            device_available[device_id] = finish
            ret.update_max_memory_usage(device_id, device.get_memory_usage_bytes())

            processed += 1
            for nxt in succ[tid]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    ready.append(nxt)
            ready.sort()

        if processed != len(tasks):
            # Cycle exists.
            remaining = [tid for tid, d in in_deg.items() if d > 0]
            raise ValueError(f"Task dependency graph has cycles; remaining: {remaining}")

        ret.time_s = max(earliest_finish.values(), default=0.0)
        return ret
