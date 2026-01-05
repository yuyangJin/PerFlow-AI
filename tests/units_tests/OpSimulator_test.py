'''
test KernelSimulator classes
'''

import pytest

from perflowai.core.device import DeviceConfig, DeviceType
from perflowai.simulator.kernel.kernel_simulator import GEMMKernelSimulator, AttentionKernelSimulator, Conv2dKernelSimulator, Parameter, SoftmaxKernelSimulator, Workload


def _device():
    # memory_bandwidth is documented as GB/s in DeviceConfig
    return DeviceConfig(id=0, type=DeviceType.GPU, memory_capacity=80_000, memory_bandwidth=1000.0, compute_flops=100e12)


def test_gemm_simulate_ok():
    a = Parameter(name='A', dtype='float16', shape=(128, 64))
    b = Parameter(name='B', dtype='float16', shape=(64, 256))
    sim = GEMMKernelSimulator(_device(), a=a, b=b)
    r = sim.simulate().to_dict()
    assert r['flops'] == 2 * 128 * 256 * 64
    assert r['peak_memory_bytes'] > 0
    assert r['achieved_flops_per_s'] > 0
    assert sim.get_memory_size_bytes() == sim.peak_memory_bytes()


def test_gemm_validation_mismatch_k():
    a = Parameter(name='A', dtype='float16', shape=(128, 63))
    b = Parameter(name='B', dtype='float16', shape=(64, 256))
    with pytest.raises(ValueError):
        GEMMKernelSimulator(_device(), a=a, b=b)


def test_attention_simulate_ok():
    q = Parameter(name='Q', dtype='float16', shape=(2, 128, 256))
    k = Parameter(name='K', dtype='float16', shape=(2, 128, 256))
    v = Parameter(name='V', dtype='float16', shape=(2, 128, 256))
    sim = AttentionKernelSimulator(_device(), q=q, k=k, v=v, num_heads=8)
    r = sim.simulate().to_dict()
    assert r['flops'] == 4 * 2 * 8 * 128 * 128 * (256 // 8)
    assert r['peak_memory_bytes'] > 0


def test_attention_validation_heads_divisible():
    q = Parameter(name='Q', dtype='float16', shape=(2, 128, 255))
    k = Parameter(name='K', dtype='float16', shape=(2, 128, 255))
    v = Parameter(name='V', dtype='float16', shape=(2, 128, 255))
    with pytest.raises(ValueError):
        AttentionKernelSimulator(_device(), q=q, k=k, v=v, num_heads=8)


def test_conv2d_simulate_ok():
    x = Parameter(name='X', dtype='float16', shape=(1, 64, 56, 56))
    w = Parameter(name='W', dtype='float16', shape=(128, 64, 3, 3))
    sim = Conv2dKernelSimulator(_device(), x=x, w=w, stride=1, padding=1)
    r = sim.simulate().to_dict()
    assert r['flops'] == 2 * 1 * 128 * 56 * 56 * 64 * 3 * 3
    assert r['peak_memory_bytes'] > 0


def test_softmax_simulate_ok():
    x = Parameter(name='X', dtype='float16', shape=(2, 4, 8))
    sim = SoftmaxKernelSimulator(_device(), x=x, axis=-1)
    r = sim.simulate().to_dict()
    # groups = 2*4=8, n=8 => flops = groups*(4n-2)
    assert r['flops'] == 8 * (4 * 8 - 2)
    assert r['peak_memory_bytes'] > 0


def test_softmax_validation_axis_oob():
    x = Parameter(name='X', dtype='float16', shape=(2, 4, 8))
    with pytest.raises(ValueError):
        SoftmaxKernelSimulator(_device(), x=x, axis=3)


def test_softmax_validation_dtype():
    x = Parameter(name='X', dtype='int32', shape=(2, 4, 8))
    with pytest.raises(ValueError):
        SoftmaxKernelSimulator(_device(), x=x, axis=-1)


def test_gemm_with_workload_aspect():
    a = Parameter(name='A', dtype='float16', shape=(128, 64))
    b = Parameter(name='B', dtype='float16', shape=(64, 256))
    def modify_workload(workload):
        return Workload(flops=workload.flops // 2, bytes_accessed=workload.bytes_accessed, peak_memory_bytes=workload.peak_memory_bytes)
    sim = GEMMKernelSimulator(_device(), a=a, b=b, workload_aspect=modify_workload)
    r = sim.simulate().to_dict()
    assert r['flops'] == (2 * 128 * 256 * 64) // 2


def test_attention_with_workload_aspect():
    q = Parameter(name='Q', dtype='float16', shape=(2, 128, 256))
    k = Parameter(name='K', dtype='float16', shape=(2, 128, 256))
    v = Parameter(name='V', dtype='float16', shape=(2, 128, 256))
    def modify_workload(workload):
        return Workload(flops=workload.flops // 2, bytes_accessed=workload.bytes_accessed, peak_memory_bytes=workload.peak_memory_bytes)
    sim = AttentionKernelSimulator(_device(), q=q, k=k, v=v, num_heads=8, workload_aspect=modify_workload)
    r = sim.simulate().to_dict()
    assert r['flops'] == (4 * 2 * 8 * 128 * 128 * (256 // 8)) // 2


def test_conv2d_with_workload_aspect():
    x = Parameter(name='X', dtype='float16', shape=(1, 64, 56, 56))
    w = Parameter(name='W', dtype='float16', shape=(128, 64, 3, 3))
    def modify_workload(workload):
        return Workload(flops=workload.flops // 2, bytes_accessed=workload.bytes_accessed, peak_memory_bytes=workload.peak_memory_bytes)
    sim = Conv2dKernelSimulator(_device(), x=x, w=w, stride=1, padding=1, workload_aspect=modify_workload)
    r = sim.simulate().to_dict()
    assert r['flops'] == (2 * 1 * 128 * 56 * 56 * 64 * 3 * 3) // 2


def test_softmax_with_workload_aspect():
    x = Parameter(name='X', dtype='float16', shape=(2, 4, 8))
    def modify_workload(workload):
        return Workload(flops=workload.flops // 2, bytes_accessed=workload.bytes_accessed, peak_memory_bytes=workload.peak_memory_bytes)
    sim = SoftmaxKernelSimulator(_device(), x=x, axis=-1, workload_aspect=modify_workload)
    r = sim.simulate().to_dict()
    assert r['flops'] == (8 * (4 * 8 - 2)) // 2
