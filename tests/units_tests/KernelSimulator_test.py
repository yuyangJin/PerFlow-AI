'''
test KernelSimulator classes
'''

import pytest

from perflowai.core.device import DeviceConfig, DeviceType, MicroArchConfig, ComputeConfig, VectorCoreConfig, TensorCoreConfig
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
    def modify_workload(simulator, workload):
        return Workload(flops=workload.flops // 2, bytes_accessed=workload.bytes_accessed, peak_memory_bytes=workload.peak_memory_bytes)
    sim = GEMMKernelSimulator(_device(), a=a, b=b, workload_aspect=modify_workload)
    r = sim.simulate().to_dict()
    assert r['flops'] == (2 * 128 * 256 * 64) // 2


def test_attention_with_workload_aspect():
    q = Parameter(name='Q', dtype='float16', shape=(2, 128, 256))
    k = Parameter(name='K', dtype='float16', shape=(2, 128, 256))
    v = Parameter(name='V', dtype='float16', shape=(2, 128, 256))
    def modify_workload(simulator, workload):
        return Workload(flops=workload.flops // 2, bytes_accessed=workload.bytes_accessed, peak_memory_bytes=workload.peak_memory_bytes)
    sim = AttentionKernelSimulator(_device(), q=q, k=k, v=v, num_heads=8, workload_aspect=modify_workload)
    r = sim.simulate().to_dict()
    assert r['flops'] == (4 * 2 * 8 * 128 * 128 * (256 // 8)) // 2


def test_conv2d_with_workload_aspect():
    x = Parameter(name='X', dtype='float16', shape=(1, 64, 56, 56))
    w = Parameter(name='W', dtype='float16', shape=(128, 64, 3, 3))
    def modify_workload(simulator, workload):
        return Workload(flops=workload.flops // 2, bytes_accessed=workload.bytes_accessed, peak_memory_bytes=workload.peak_memory_bytes)
    sim = Conv2dKernelSimulator(_device(), x=x, w=w, stride=1, padding=1, workload_aspect=modify_workload)
    r = sim.simulate().to_dict()
    assert r['flops'] == (2 * 1 * 128 * 56 * 56 * 64 * 3 * 3) // 2


def test_softmax_with_workload_aspect():
    x = Parameter(name='X', dtype='float16', shape=(2, 4, 8))
    def modify_workload(simulator, workload):
        return Workload(flops=workload.flops // 2, bytes_accessed=workload.bytes_accessed, peak_memory_bytes=workload.peak_memory_bytes)
    sim = SoftmaxKernelSimulator(_device(), x=x, axis=-1, workload_aspect=modify_workload)
    r = sim.simulate().to_dict()
    assert r['flops'] == (8 * (4 * 8 - 2)) // 2


def test_gemm_tensor_core_speedup():
    """Verify that enabling Tensor Cores (via MicroArchConfig) yields higher performance."""

    # 1. Base device without explicit microarch (uses compute_flops baseline of 10 TFLOPs)
    base_flops = 10e12
    dev_base = DeviceConfig(
        id=0, type=DeviceType.GPU,
        memory_capacity=80_000,
        memory_bandwidth=1000.0,
        compute_flops=base_flops
    )

    # 2. Device with Tensor Cores (e.g., 100 TFLOPs for float16)
    # Note: The simulator looks up microarch.compute.tensor.peak_tflops_by_dtype["float16"]
    tensor_flops_val = 100.0  # TFLOPs
    microarch = MicroArchConfig(
        compute=ComputeConfig(
            vector=VectorCoreConfig(peak_tflops_by_dtype={"float32": 10.0}),
            tensor=TensorCoreConfig(
                enabled=True,
                peak_tflops_by_dtype={"float16": tensor_flops_val}
            )
        )
    )
    dev_tensor = DeviceConfig(
        id=1, type=DeviceType.GPU,
        memory_capacity=80_000,
        memory_bandwidth=1000.0,
        compute_flops=base_flops,
        microarch=microarch
    )

    # Workload: FP16 GEMM
    a = Parameter(name='A', dtype='float16', shape=(2048, 2048))
    b = Parameter(name='B', dtype='float16', shape=(2048, 2048))

    # Disable memory bottleneck effect by setting memory_coeff very high if needed,
    # or just rely on FLOPs being the bottleneck for large GEMM on slow compute.
    # Here we stick to defaults.
    # Base: 10 TFLOPs vs Tensor: 100 TFLOPs.

    sim_base = GEMMKernelSimulator(dev_base, a=a, b=b)
    sim_tensor = GEMMKernelSimulator(dev_tensor, a=a, b=b)

    res_base = sim_base.simulate()
    res_tensor = sim_tensor.simulate()

    # Higher compute capability => Lower compute time => Lower time_s (assuming compute bound)
    print(f"Base Time: {res_base.time_s}, Tensor Time: {res_tensor.time_s}")

    # Ensure Tensor Core version is faster
    assert res_tensor.time_s < res_base.time_s, \
        f"Tensor Core enabled device should be faster. Base={res_base.time_s}, Tensor={res_tensor.time_s}"

    # Verify that the simulator actually picked up the specific tensor FLOPs
    # achieved_flops_per_s should be close to (peak * coeff)
    # coeff default is 0.6. Peak is 100e12. Expect ~60e12.
    expected_achieved = tensor_flops_val * 1e12 * sim_tensor.compute_coeff
    # Allow some margin (e.g. if memory bound, it would be lower, but for large GEMM usually compute bound)
    # Check if it was compute bound first.
    if res_tensor.bottleneck == "compute":
        # It should be exactly expected_achieved because compute time = flops / effective_peak
        # achieved = flops / time = effective_peak
        assert abs(res_tensor.achieved_flops_per_s - expected_achieved) < 1e9


def test_sm_usage_parameter():
    """Verify that sm_usage parameter is correctly stored."""
    a = Parameter(name='A', dtype='float16', shape=(128, 64))
    b = Parameter(name='B', dtype='float16', shape=(64, 256))

    # Default sm_usage
    sim_default = GEMMKernelSimulator(_device(), a=a, b=b)
    assert sim_default.sm_usage == 1.0

    # Custom sm_usage
    sim_custom = GEMMKernelSimulator(_device(), a=a, b=b, sm_usage=0.5)
    assert sim_custom.sm_usage == 0.5


def test_kernel_no_tensor_op_fallback():
    """Verify that a non-TensorOp kernel (like Softmax) does not use Tensor Core peaks."""

    # Device with very high Tensor Core perf, but low Vector/Base perf
    base_flops = 10e12
    tensor_flops_val = 1000.0  # Huge number if used
    microarch = MicroArchConfig(
        compute=ComputeConfig(
            vector=VectorCoreConfig(peak_tflops_by_dtype={"float16": 10.0}), # Same as base
            tensor=TensorCoreConfig(
                enabled=True,
                peak_tflops_by_dtype={"float16": tensor_flops_val}
            )
        )
    )
    dev_tensor = DeviceConfig(
        id=2, type=DeviceType.GPU,
        memory_capacity=80_000,
        memory_bandwidth=1000.0,
        compute_flops=base_flops,
        microarch=microarch
    )

    # Softmax is NOT a tensor op (is_tensor_op returns False)
    x = Parameter(name='X', dtype='float16', shape=(4096, 4096))
    sim = SoftmaxKernelSimulator(dev_tensor, x=x)

    res = sim.simulate()

    # It should use Vector path (10 TFLOPs) or Base path, NOT Tensor path (1000 TFLOPs).
    # Expected Peak = 10 TFLOPs * coeff (0.6) = 6 TFLOPs
    # If it used Tensor: 1000 * 0.6 = 600 TFLOPs

    if res.bottleneck == "compute":
        achieved = res.achieved_flops_per_s
        # Should be much closer to 6 TFLOPs than 600 TFLOPs
        assert achieved < 20e12, f"Softmax should not use Tensor Cores! Achieved {achieved/1e12} TFLOPs"
