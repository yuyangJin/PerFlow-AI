"""Unit tests for network/collective simulators."""

import pytest

from perflowai.core.device import DeviceConfig, DeviceType
from perflowai.simulator.kernel.kernel_simulator import Parameter
from perflowai.simulator.comm import (
    AllGatherNetworkSimulator,
    AllReduceNetworkSimulator,
    AllToAllNetworkSimulator,
    ReduceScatterNetworkSimulator,
    GroupTopology,
)


def _device() -> DeviceConfig:
    return DeviceConfig(
        id=0,
        type=DeviceType.GPU,
        memory_capacity=80_000,
        memory_bandwidth=1000.0,
        compute_flops=100e12,
        intra_node_bandwidth=100.0,
        inter_node_bandwidth=25.0,
    )


def test_all_gather_bytes_and_time_intra():
    x = Parameter(name="x", dtype="float32", shape=(1024,))
    topo = GroupTopology(num_nodes=1, ranks_per_node=4)
    sim = AllGatherNetworkSimulator(
        _device(),
        x=x,
        topology=topo,
        comm_coeff=1.0,
    )
    r = sim.simulate()

    assert r.bytes_per_rank == 1024 * 4
    assert r.total_bytes == (1024 * 4) * 4

    # intra_node_bandwidth = 100 GB/s => 100e9 B/s
    expected_time = r.total_bytes / (100.0 * 1e9)
    assert r.time_s == pytest.approx(expected_time, rel=1e-12, abs=0.0)
    assert r.achieved_GBps == pytest.approx(100.0, rel=1e-12, abs=0.0)


def test_all_reduce_uses_inter_bandwidth():
    x = Parameter(name="x", dtype="float16", shape=(2048,))
    # Represent a group spanning 8 nodes with 1 rank each => pure inter-node.
    topo = GroupTopology(num_nodes=8, ranks_per_node=1)
    sim = AllReduceNetworkSimulator(
        _device(),
        x=x,
        topology=topo,
        comm_coeff=1.0,
    )
    r = sim.simulate()

    # float16 => 2 bytes/elem
    assert r.bytes_per_rank == 2048 * 2
    assert r.total_bytes == (2048 * 2) * 8


def test_all_reduce_hierarchical_two_nodes():
    # 2 nodes, 8 ranks per node => group_size=16
    topo = GroupTopology(num_nodes=2, ranks_per_node=8)
    x = Parameter(name="x", dtype="float32", shape=(1024,))
    sim = AllReduceNetworkSimulator(
        _device(),
        x=x,
        topology=topo,
        comm_coeff=1.0,
    )
    r = sim.simulate()

    bpr = 1024 * 4
    # intra reduce + intra bcast: 2 * bpr * p * n
    intra_bytes = 2 * bpr * 8 * 2
    # inter reduce among leaders: bpr * n
    inter_bytes = bpr * 2

    expected = intra_bytes / (100.0 * 1e9) + inter_bytes / (25.0 * 1e9)
    assert r.time_s == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_all_to_all_effective_numel_rounding():
    # numel not divisible by group_size: 10 // 4 = 2, effective_numel = 8
    x = Parameter(name="x", dtype="float32", shape=(10,))
    topo = GroupTopology(num_nodes=1, ranks_per_node=4)
    sim = AllToAllNetworkSimulator(
        _device(),
        x=x,
        topology=topo,
        comm_coeff=1.0,
    )
    r = sim.simulate()

    assert r.bytes_per_rank == 8 * 4
    assert r.total_bytes == (8 * 4) * 4


def test_reduce_scatter_min_chunk_numel_one():
    # numel smaller than group_size: max(1 // 8, 1) => 1, effective_numel = 8
    x = Parameter(name="x", dtype="int8", shape=(1,))
    topo = GroupTopology(num_nodes=1, ranks_per_node=8)
    sim = ReduceScatterNetworkSimulator(
        _device(),
        x=x,
        topology=topo,
        comm_coeff=0.5,
    )
    r = sim.simulate()

    assert r.bytes_per_rank == 8 * 1
    assert r.total_bytes == (8 * 1) * 8

    # comm_coeff=0.5 => achieved = bw * coeff
    assert r.achieved_GBps == pytest.approx(100.0 * 0.5, rel=1e-12, abs=0.0)


def test_network_workload_aspect_receives_simulator():
    x = Parameter(name="x", dtype="float32", shape=(1024,))
    topo = GroupTopology(num_nodes=1, ranks_per_node=4)
    called = {"ok": False}

    def aspect(simulator, workload):
        # Validate we receive simulator instance.
        called["ok"] = simulator is not None
        return workload

    sim = AllGatherNetworkSimulator(
        _device(),
        x=x,
        topology=topo,
        comm_coeff=1.0,
        workload_aspect=aspect,
    )
    _ = sim.simulate()
    assert called["ok"] is True
