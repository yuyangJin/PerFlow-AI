from perflowai.core.device import DeviceConfig, DeviceType, NodeConfig
from perflowai.simulator.kernel.kernel_simulator import GEMMKernelSimulator, Parameter
from perflowai.simulator.comm import AllReduceNetworkSimulator, GroupTopology
from perflowai.simulator.orchestration.task import Task, Assignment
from perflowai.simulator.orchestration.orchestration import OrchestrationResult
from perflowai.simulator.orchestration.device_info import NodeInstance

def _device_gpu(id="gpu0"):
    return DeviceConfig(
        id=id,
        type=DeviceType.GPU,
        memory_capacity=80_000,
        memory_bandwidth=1000.0,
        compute_flops=100e12,
        intra_node_bandwidth=100.0,
        inter_node_bandwidth=25.0,
    )

def _node(gpus):
    return NodeInstance(
        NodeConfig(
            host=DeviceConfig(id="cpu0", type=DeviceType.CPU, memory_capacity=1000, memory_bandwidth=100, compute_flops=1e9),
            gpu_devices=gpus
        )
    )

def test_dual_stream_overlap():
    dev = _device_gpu()
    node = _node([dev])

    # 1. Compute Task (GEMM)
    # A * B where A=(4096, 4096), B=(4096, 4096).
    a = Parameter(name='A', dtype='float16', shape=(4096, 4096))
    b = Parameter(name='B', dtype='float16', shape=(4096, 4096))
    gemm = GEMMKernelSimulator(dev, a=a, b=b, id="gemm_op")
    t_comp = gemm.simulate().time_s

    # 2. Network Task (AllReduce)
    x_shape = (25 * 1024 * 1024, )
    x = Parameter(name='X', dtype='float16', shape=x_shape)
    topo = GroupTopology(num_nodes=2, ranks_per_node=1)
    net = AllReduceNetworkSimulator(dev, x=x, topology=topo, id="net_op")
    t_comm = net.simulate().time_s

    # Ensure parameters are on device (simulate allocation)
    gpu0 = node.gpu_devices[0]
    gpu0.add_parameter(a)
    gpu0.add_parameter(b)
    gpu0.add_parameter(x)

    # Check they computed correctly (non-zero)
    assert t_comp > 0
    assert t_comm > 0

    # Create Orchestratration
    t1 = Task(id="comp", workload=gemm)
    t2 = Task(id="comm", workload=net)

    # Both assigned to same GPU
    res = OrchestrationResult(
        nodes=[node],
        tasks=[t1, t2],
        assignments=[Assignment("comp", dev.id), Assignment("comm", dev.id)]
    )

    est = res.estimate_makespan()

    # Ideally should be max(t_comp, t_comm) because one uses compute stream, one uses copy stream
    # And copy stream doesn't block compute stream.
    expected = max(t_comp, t_comm)

    # Allow small float diff
    assert abs(est.time_s - expected) < 1e-9, \
        f"Expected overlap time {expected}, got {est.time_s}. (comp={t_comp}, comm={t_comm})"


def test_dual_stream_serialization_with_sm_usage():
    dev = _device_gpu()
    node = _node([dev])

    # 1. Compute Task
    a = Parameter(name='A', dtype='float16', shape=(4096, 4096))
    b = Parameter(name='B', dtype='float16', shape=(4096, 4096))
    gemm = GEMMKernelSimulator(dev, a=a, b=b, id="gemm_op_serial")
    t_comp = gemm.simulate().time_s

    # 2. Network Task with sm_usage=1.0 (blocks compute)
    x = Parameter(name='X', dtype='float16', shape=(25 * 1024 * 1024, ))
    topo = GroupTopology(num_nodes=2, ranks_per_node=1)
    # Assuming AllReduceNetworkSimulator passes **kwargs or explicit arg for sm_usage
    net = AllReduceNetworkSimulator(dev, x=x, topology=topo, id="net_op_serial")
    net.sm_usage = 1.0
    t_comm = net.simulate().time_s

    # Ensure parameters are on device
    gpu0 = node.gpu_devices[0]
    gpu0.add_parameter(a)
    gpu0.add_parameter(b)
    gpu0.add_parameter(x)

    t1 = Task(id="comp", workload=gemm)
    t2 = Task(id="comm", workload=net)

    res = OrchestrationResult(
        nodes=[node],
        tasks=[t1, t2],
        assignments=[Assignment("comp", dev.id), Assignment("comm", dev.id)]
    )

    est = res.estimate_makespan()

    # Should serialize because network task claims SM usage.
    # Total time = t_comp + t_comm
    expected = t_comp + t_comm

    assert abs(est.time_s - expected) < 1e-9, \
        f"Expected serial time {expected}, got {est.time_s}. (comp={t_comp}, comm={t_comm})"

