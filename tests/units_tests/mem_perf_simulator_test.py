'''
Integration tests for memory and performance simulators
'''

import pytest
from perflowai.simulator.model import ModelMemSimulator, ModelPerfSimulator
from perflowai.core import ModelConfig, DeviceConfig, DeviceType, EventType
from perflowai.core import Request, Task, Tasks


# Mock event class for testing
class MockEvent:
    def __init__(self, event_type, tasks):
        self._type = event_type
        self._tasks = tasks
    
    def get_type(self):
        return self._type
    
    def get_tasks(self):
        return self._tasks


# Mock task class for testing
class MockTask:
    def __init__(self, req, decode_iters=0):
        self.req = req
        self.decode_iters = decode_iters


# Mock request class for testing
class MockRequest:
    def __init__(self, input_len, output_len):
        self.input_len = input_len
        self.output_len = output_len


# Mock tasks container
class MockTasks:
    def __init__(self, task_list):
        self._tasks = task_list
    
    def get(self):
        return self._tasks
    
    def get_num_task(self):
        return len(self._tasks)


def test_mem_simulator_prefill():
    """Test memory simulator for prefill events"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    mem_sim = ModelMemSimulator(model_config)
    
    # Create a mock prefill event with one task
    req = MockRequest(input_len=100, output_len=10)
    task = MockTask(req)
    tasks = MockTasks([task])
    event = MockEvent(EventType.PRF, tasks)
    
    # Calculate KVCache
    kvcache_size = mem_sim.kvcache(event)
    
    # Expected: 64 layers * 2 (K,V) * 32 heads * 100 tokens * 128 head_dim * 2 bytes
    expected = 64 * 2 * 32 * 100 * 128 * 2
    assert kvcache_size == expected, f"Expected {expected}, got {kvcache_size}"
    assert kvcache_size == 104857600, "Should be 100 MB"


def test_mem_simulator_decode():
    """Test memory simulator for decode events"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    mem_sim = ModelMemSimulator(model_config)
    
    # Create a mock decode event
    # After prefill of 100 tokens, decode iteration 0 (generating 1st token)
    req = MockRequest(input_len=100, output_len=10)
    task = MockTask(req, decode_iters=0)
    tasks = MockTasks([task])
    event = MockEvent(EventType.DCD, tasks)
    
    # Calculate KVCache
    kvcache_size = mem_sim.kvcache(event)
    
    # Expected: 64 layers * 2 (K,V) * 32 heads * 101 tokens * 128 head_dim * 2 bytes
    # (100 input + 1 current token)
    expected = 64 * 2 * 32 * 101 * 128 * 2
    assert kvcache_size == expected, f"Expected {expected}, got {kvcache_size}"


def test_mem_simulator_decode_multiple_iterations():
    """Test memory simulator for decode with multiple iterations"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    mem_sim = ModelMemSimulator(model_config)
    
    # Create a mock decode event at iteration 5
    req = MockRequest(input_len=100, output_len=10)
    task = MockTask(req, decode_iters=5)
    tasks = MockTasks([task])
    event = MockEvent(EventType.DCD, tasks)
    
    # Calculate KVCache
    kvcache_size = mem_sim.kvcache(event)
    
    # Expected: 64 layers * 2 (K,V) * 32 heads * 106 tokens * 128 head_dim * 2 bytes
    # (100 input + 5 previously decoded + 1 current token)
    expected = 64 * 2 * 32 * 106 * 128 * 2
    assert kvcache_size == expected, f"Expected {expected}, got {kvcache_size}"


def test_mem_simulator_batch():
    """Test memory simulator with multiple tasks"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    mem_sim = ModelMemSimulator(model_config)
    
    # Create a mock prefill event with multiple tasks
    req1 = MockRequest(input_len=50, output_len=10)
    req2 = MockRequest(input_len=100, output_len=20)
    task1 = MockTask(req1)
    task2 = MockTask(req2)
    tasks = MockTasks([task1, task2])
    event = MockEvent(EventType.PRF, tasks)
    
    # Calculate KVCache
    kvcache_size = mem_sim.kvcache(event)
    
    # Expected: sum of KVCache for both tasks
    expected1 = 64 * 2 * 32 * 50 * 128 * 2
    expected2 = 64 * 2 * 32 * 100 * 128 * 2
    expected = expected1 + expected2
    assert kvcache_size == expected, f"Expected {expected}, got {kvcache_size}"


def test_perf_simulator_prefill():
    """Test performance simulator for prefill events"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    device_config = DeviceConfig(
        id=0,
        type=DeviceType.GPU,
        memory_capacity=16384,
        memory_bandwidth=900,  # GB/s
        compute_flops=1e12  # 1 TFLOPS
    )
    
    perf_sim = ModelPerfSimulator(model_config, device_config)
    
    # Create a mock prefill event
    req = MockRequest(input_len=100, output_len=10)
    task = MockTask(req)
    tasks = MockTasks([task])
    event = MockEvent(EventType.PRF, tasks)
    
    # Calculate time
    time = perf_sim.time(event)
    
    # Should return a positive time in seconds
    assert time > 0, "Time should be positive"
    assert time < 1000, "Time should be reasonable (less than 1000 seconds)"


def test_perf_simulator_decode():
    """Test performance simulator for decode events"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    device_config = DeviceConfig(
        id=0,
        type=DeviceType.GPU,
        memory_capacity=16384,
        memory_bandwidth=900,
        compute_flops=1e12
    )
    
    perf_sim = ModelPerfSimulator(model_config, device_config)
    
    # Create a mock decode event
    req = MockRequest(input_len=100, output_len=10)
    task = MockTask(req, decode_iters=0)
    tasks = MockTasks([task])
    event = MockEvent(EventType.DCD, tasks)
    
    # Calculate time
    time = perf_sim.time(event)
    
    # Should return a positive time
    assert time > 0, "Time should be positive"
    
    # Decode should typically be faster than prefill for same seq_len
    # (though this depends on the specific implementation)


def test_perf_simulator_decode_vs_prefill():
    """Test that decode is typically faster than prefill per token"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    device_config = DeviceConfig(
        id=0,
        type=DeviceType.GPU,
        memory_capacity=16384,
        memory_bandwidth=900,
        compute_flops=1e12
    )
    
    perf_sim = ModelPerfSimulator(model_config, device_config)
    
    # Prefill event with 100 tokens
    req_prf = MockRequest(input_len=100, output_len=10)
    task_prf = MockTask(req_prf)
    tasks_prf = MockTasks([task_prf])
    event_prf = MockEvent(EventType.PRF, tasks_prf)
    time_prf = perf_sim.time(event_prf)
    
    # Decode event (1 token with cache of 100)
    req_dec = MockRequest(input_len=100, output_len=10)
    task_dec = MockTask(req_dec, decode_iters=0)
    tasks_dec = MockTasks([task_dec])
    event_dec = MockEvent(EventType.DCD, tasks_dec)
    time_dec = perf_sim.time(event_dec)
    
    # Decode should be much faster than prefill
    assert time_dec < time_prf, "Decode should be faster than prefill"


def test_compute_flops_and_memory():
    """Test that compute FLOPs and memory are calculated"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    device_config = DeviceConfig(
        id=0,
        type=DeviceType.GPU,
        memory_capacity=16384,
        memory_bandwidth=900,
        compute_flops=1e12
    )
    
    perf_sim = ModelPerfSimulator(model_config, device_config)
    
    # Create a prefill event
    req = MockRequest(input_len=100, output_len=10)
    task = MockTask(req)
    tasks = MockTasks([task])
    event = MockEvent(EventType.PRF, tasks)
    
    # Get FLOPs and memory
    flops, memory = perf_sim._compute_prefill_volume(event)
    
    # Both should be positive
    assert flops > 0, "FLOPs should be positive"
    assert memory > 0, "Memory should be positive"
    
    # FLOPs should be substantial for 100 tokens
    assert flops > 1e9, "FLOPs should be in billions for this workload"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
