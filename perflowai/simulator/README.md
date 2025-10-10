# PerFlow-AI Simulator Module

This directory contains the simulator framework for modeling inference and training performance in PerFlow-AI.

## Directory Structure

```
simulator/
├── simulator.py              # Base Simulator class
├── infer/                    # Inference simulator
│   ├── infer_simulator.py    # Main inference simulation logic
│   └── __init__.py
├── model/                    # Model-level simulators
│   ├── mem_simulator.py      # Memory and KVCache simulation
│   ├── perf_simulator.py     # Performance (FLOPs/time) simulation
│   └── __init__.py
├── oprt/                     # Operator-level simulators
│   ├── oprt_simulator.py     # Basic operators (Linear, Attention, etc.)
│   └── __init__.py
├── pipeline/                 # Pipeline parallel simulators
│   ├── pp_simulator.py       # Pipeline parallel simulation
│   └── __init__.py
├── dptp/                     # Data/Tensor parallel simulators
│   ├── dptp_simulator.py     # DP/TP simulation
│   └── __init__.py
└── comm/                     # Communication simulators
    ├── comm_simulator.py     # Network communication simulation
    └── __init__.py
```

## Key Components

### 1. Inference Simulator (`infer/`)

Simulates LLM inference with:
- Request scheduling (FIFO, priority-based)
- Prefill and decode phases
- KVCache management
- Performance metrics (TTFT, TPOT, TPS, Throughput)

**Usage:**
```python
from perflowai.simulator import InferSimulator
from perflowai.parallel import InferGraph
from perflowai.core import Request, ModelConfig, DeviceConfig

# Generate requests
requests = [Request(req_id=i, input_len=100, output_len=20, start_time=i*10.0) 
            for i in range(10)]

# Create graph
graph = InferGraph(ndevs=1)
graph.generate_nodes(requests)

# Run simulation
simulator = InferSimulator(graph, requests)
trace = simulator.simulate(scheduler, model_config, device_config)
metrics = simulator.get_metrics()
```

### 2. Operator-Level Simulator (`oprt/`)

Fine-grained operator simulation providing accurate:
- **FLOPs calculation**: Compute volume for each operation
- **Memory calculation**: Activation memory footprint
- **KVCache calculation**: Key-Value cache size

**Operators:**
- `LinearOperator`: Matrix multiplication layers
- `AttentionOperator`: Multi-head attention with KV cache
- `LayerNormOperator`: Normalization layers
- `EmbeddingOperator`: Token embeddings
- `FFNOperator`: Feed-forward networks
- `TransformerLayerOperator`: Complete transformer layer

**Usage:**
```python
from perflowai.simulator.oprt import TransformerLayerOperator
from perflowai.core import ModelConfig

model_config = ModelConfig(
    num_layers=64,
    hidden_size=4096,
    ffn_dim=16384,
    num_heads=32,
    head_dim=128,
    dtype_bytes=2
)

layer = TransformerLayerOperator(model_config)
flops = layer.compute_flops(batch_size=1, seq_len=100, is_prefill=True)
kvcache = layer.compute_kvcache(batch_size=1, seq_len=100)
```

### 3. Memory Simulator (`model/mem_simulator.py`)

Tracks memory usage during inference:
- KVCache accumulation
- Activation memory
- Memory footprint over time

**Usage:**
```python
from perflowai.simulator.model import ModelMemSimulator

mem_sim = ModelMemSimulator(model_config)
kvcache_bytes = mem_sim.kvcache(event)  # Calculate KVCache for an event
```

### 4. Performance Simulator (`model/perf_simulator.py`)

Estimates execution time based on:
- Computation (FLOPs) time
- Memory bandwidth time
- Returns the bottleneck (max of both)

**Usage:**
```python
from perflowai.simulator.model import ModelPerfSimulator

perf_sim = ModelPerfSimulator(model_config, device_config)
execution_time = perf_sim.time(event)  # Get event execution time in seconds
```

### 5. Pipeline Parallel Simulator (`pipeline/`)

Simulates pipeline parallel training with:
- Different pipeline schedules (GPipe, PipeDream, 1F1B, ZeroBubble)
- Bubble analysis
- Memory footprint tracking

**Usage:**
```python
from perflowai.simulator import PPSimulator
from perflowai.parallel.pipeline_parallel import GPipeGraph

graph = GPipeGraph(nstages=4, nmicrobatches=10, nchunks=1)
graph.build_graph()

simulator = PPSimulator(PipeType.GPipe, graph)
trace = simulator.run()
```

## Key Features

### Accurate Calculations

The simulator uses operator-level calculations for precision:

1. **KVCache**: 
   - Formula: `2 (K,V) × num_heads × seq_len × head_dim × dtype_bytes` per layer
   - Example: 100 tokens, 64 layers → 100 MB

2. **FLOPs**:
   - Prefill: O(seq_len²) for attention
   - Decode: O(seq_len) for attention with cache
   - Includes QKV projections, attention matmuls, FFN

3. **Time Estimation**:
   - Considers both compute and memory bandwidth
   - Uses efficiency factors (60% compute, 80% memory)

### Metrics

Inference simulator provides:
- **TTFT** (Time To First Token): Latency until first token
- **TPOT** (Time Per Output Token): Average token generation time
- **TPS** (Tokens Per Second): Throughput metric
- **Throughput**: Overall system throughput

## Testing

Run the test suite:

```bash
# Test operators
pytest tests/units_tests/oprt_simulator_test.py -v

# Test memory and performance simulators
pytest tests/units_tests/mem_perf_simulator_test.py -v

# Test all simulators
pytest tests/units_tests/ -v
```

## Examples

See `examples/deepseekv3_inference/dpsk_infer.py` for a complete inference simulation example.

```bash
cd examples/deepseekv3_inference
python dpsk_infer.py
```

## Recent Improvements (2025 S2)

1. **Operator-level simulator**: Fine-grained calculations from operators → layers → model
2. **Fixed KVCache tracking**: Corrected prefill/decode logic
3. **Fixed FLOPs calculations**: Proper attention and FFN formulas
4. **Fixed memory bandwidth**: Correct unit conversions
5. **Comprehensive tests**: 15+ unit tests covering all components

## Documentation

For detailed documentation, see:
- [Operator-Level Simulator Documentation](../docs/simulator_operator_level.md)
- [API Reference](../docs/api_reference.md) (if available)

## Contributing

When adding new operators:
1. Inherit from `Operator` base class
2. Implement `compute_flops()`, `compute_memory()`, and optionally `compute_kvcache()`
3. Add unit tests in `tests/units_tests/oprt_simulator_test.py`
4. Update documentation

## Future Work

Planned enhancements:
- [ ] MoE (Mixture of Experts) operator support
- [ ] Sparse attention patterns
- [ ] Quantization effects (INT8, FP8)
- [ ] Multi-GPU communication modeling
- [ ] Flash Attention optimizations
- [ ] Speculative decoding simulation
