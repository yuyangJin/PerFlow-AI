from typing import List, Union

from perflowai import FlowNode
from perflowai.core.device import NodeConfig, DeviceConfig, DeviceType
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

    def remove_parameter(self, id: int):
        self.holding_parameter = [
            parameter for parameter in self.holding_parameter if parameter.id != id
        ]

    @property
    def id(self):
        return self.config.id


class NodeInstance:
    def __init__(self, nodeConfig: NodeConfig):
        self.host = DeviceInstance(nodeConfig.host)
        self.gpu_devices = [DeviceInstance(gpu) for gpu in nodeConfig.gpu_devices]

    @property
    def id(self):
        return self.host.id


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


class Task:
    def __init__(self,
                 id: str,
                 workload: FlowNode,
                 run_after: List[str] = None
                 ):
        if run_after is None:
            run_after = []

        self.id = id
        self.workload = workload
        self.run_after = run_after


class Assignment:
    def __init__(self,
                 task_id: str,
                 device_id: Union[int, str],
                 ):
        self.task_id = task_id
        self.device_id = device_id


class OrchestrationResult:
    def __init__(self,
                 nodes: List[NodeInstance],
                 tasks: List[Task],
                 assignments: List[Assignment]
                 ):
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

    def estimate_makespan_s(self) -> float:
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
        while ready:
            tid = ready.pop(0)
            t = tasks[tid]

            dep_ready_time = 0.0
            if t.run_after:
                dep_ready_time = max(earliest_finish[d] for d in t.run_after)

            dev = device_of[tid]
            dev_ready_time = float(device_available.get(dev, 0.0))
            start = max(dep_ready_time, dev_ready_time)
            dur = _workload_time_s(t.workload)
            finish = start + float(dur)

            earliest_finish[tid] = finish
            device_available[dev] = finish

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

        return max(earliest_finish.values(), default=0.0)
