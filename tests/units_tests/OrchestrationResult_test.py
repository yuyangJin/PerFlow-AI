from __future__ import annotations

from dataclasses import dataclass
from typing import List

from perflowai import DeviceConfig, DeviceType, NodeConfig
from perflowai.simulator.orchestration.orchestration import (
    Assignment,
    OrchestrationResult,
    Task, DeviceInstance, NodeInstance,
)
from perflowai.workflow.flow import FlowNode
import secrets
import string


def random_id(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))


def make_node():
    ret: List[NodeInstance] = []

    for i in range(2):
        host_cfg = DeviceConfig(id=str(i), type=DeviceType.CPU, memory_capacity=1024_000, memory_bandwidth=1000.0,
                                compute_flops=1e9)
        gpu_cfg: List[DeviceConfig] = []
        for j in range(8):
            gpu_cfg.append(
                DeviceConfig(id=random_id(), type=DeviceType.GPU, memory_capacity=80_000, memory_bandwidth=1000.0,
                             compute_flops=100e12)
            )

        node_cfg = NodeConfig(
            host=host_cfg,
            gpu_devices=gpu_cfg,
        )

        ret.append(
            NodeInstance(
                nodeConfig=node_cfg,
            )
        )

    return ret


@dataclass
class _FakeSimResult:
    time_s: float


class _FakeWorkload(FlowNode):
    def __init__(self, time_s: float):
        super().__init__(name="fake", id=random_id(8), inputs=[], outputs=[])
        self._time_s = float(time_s)

    def simulate(self):
        return _FakeSimResult(time_s=self._time_s)


def test_estimate_makespan_serial_single_device():
    # A -> B on same device => serial
    a = Task(id="A", workload=_FakeWorkload(2.0))
    b = Task(id="B", workload=_FakeWorkload(3.0), run_after=["A"])

    res = OrchestrationResult(
        nodes=make_node(),
        tasks=[a, b],
        assignments=[Assignment("A", "0"), Assignment("B", "0")],
    )

    assert res.estimate_makespan_s() == 5.0


def test_estimate_makespan_parallel_multi_device():
    # A and B independent on different devices; C waits for both
    a = Task(id="A", workload=_FakeWorkload(2.0))
    b = Task(id="B", workload=_FakeWorkload(5.0))
    c = Task(id="C", workload=_FakeWorkload(1.0), run_after=["A", "B"])

    res = OrchestrationResult(
        nodes=make_node(),
        tasks=[a, b, c],
        assignments=[Assignment("A", "0"), Assignment("B", "1"), Assignment("C", "0")],
    )

    # A finishes at 2 on dev0, B finishes at 5 on dev1; C is on dev0, so start=max(5, dev0_avail=2)=5
    assert res.estimate_makespan_s() == 6.0


def test_estimate_makespan_detect_cycle():
    a = Task(id="A", workload=_FakeWorkload(1.0), run_after=["B"])
    b = Task(id="B", workload=_FakeWorkload(1.0), run_after=["A"])

    res = OrchestrationResult(
        nodes=make_node(),
        tasks=[a, b],
        assignments=[Assignment("A", "0"), Assignment("B", "0")],
    )

    try:
        res.estimate_makespan_s()
        assert False, "Expected ValueError due to cycle"
    except ValueError as e:
        assert "cycles" in str(e)
