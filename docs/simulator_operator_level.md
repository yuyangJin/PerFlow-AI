# Operator-Level Simulator Documentation

## Overview

The PerFlow-AI simulator now includes a fine-grained operator-level simulation framework that enables accurate calculation of:
- **KVCache size**: Memory required to store Key and Value tensors in attention layers
- **Memory footprint**: Activation memory required during forward/backward passes
- **Computation volume (FLOPs)**: Floating-point operations for performance estimation

## Architecture

The simulator follows a hierarchical design:

```
Operator Level → Layer Level → Model Level
     ↓               ↓              ↓
  compute_*()   aggregate ops   use in simulator
```

### Operator Classes

Located in `perflowai/simulator/oprt/oprt_simulator.py`:

#### 1. `LinearOperator`
Represents a linear transformation: `Y = X @ W + b`

**Methods:**
- `compute_flops(batch_size, seq_len)`: Computes `2 * batch_size * seq_len * in_features * out_features`
- `compute_memory(batch_size, seq_len)`: Computes input + output activation memory

**Example:**
```python
from perflowai.simulator.oprt import LinearOperator

linear = LinearOperator(in_features=4096, out_features=4096, dtype_bytes=2)
flops = linear.compute_flops(batch_size=1, seq_len=100)  # Returns FLOPs
memory = linear.compute_memory(batch_size=1, seq_len=100)  # Returns bytes
```

#### 2. `AttentionOperator`
Represents multi-head attention with QKV projections and scaled dot-product attention.

**Methods:**
- `compute_flops(batch_size, seq_len, is_prefill)`: Computes attention FLOPs
  - Prefill: Process full sequence (O(seq_len²))
  - Decode: Process 1 token with KV cache (O(seq_len))
- `compute_memory(batch_size, seq_len, is_prefill)`: Activation memory
- `compute_kvcache(batch_size, seq_len)`: KVCache size in bytes

**KVCache Formula:**
```
KVCache = 2 (K,V) × batch_size × num_heads × seq_len × head_dim × dtype_bytes
```

**Example:**
```python
from perflowai.simulator.oprt import AttentionOperator

attn = AttentionOperator(hidden_size=4096, num_heads=32, head_dim=128, dtype_bytes=2)
# Prefill
flops_prf = attn.compute_flops(1, 100, is_prefill=True)
kvcache = attn.compute_kvcache(1, 100)  # Returns 1,638,400 bytes (1.56 MB)

# Decode
flops_dec = attn.compute_flops(1, 100, is_prefill=False)  # Much faster
```

#### 3. `LayerNormOperator`
Represents layer normalization.

**Methods:**
- `compute_flops(batch_size, seq_len)`: ~5 ops per element (mean, var, norm, scale, shift)
- `compute_memory(batch_size, seq_len)`: Input + output memory

#### 4. `EmbeddingOperator`
Represents token embedding lookup.

**Methods:**
- `compute_flops(batch_size, seq_len)`: Minimal (lookup operation)
- `compute_memory(batch_size, seq_len)`: Output embedding memory

#### 5. `FFNOperator`
Represents Feed-Forward Network with two linear layers and activation.

**Methods:**
- `compute_flops(batch_size, seq_len)`: Up projection + activation + down projection
- `compute_memory(batch_size, seq_len)`: Intermediate activation memory

**Example:**
```python
from perflowai.simulator.oprt import FFNOperator

ffn = FFNOperator(hidden_size=4096, ffn_dim=16384, dtype_bytes=2)
flops = ffn.compute_flops(1, 100)
memory = ffn.compute_memory(1, 100)
```

#### 6. `TransformerLayerOperator`
Aggregates a complete transformer layer: Attention + FFN + LayerNorms

**Methods:**
- `compute_flops(batch_size, seq_len, is_prefill)`: Sum of all components
- `compute_memory(batch_size, seq_len, is_prefill)`: Sum of all components
- `compute_kvcache(batch_size, seq_len)`: Attention KVCache only

**Example:**
```python
from perflowai.simulator.oprt import TransformerLayerOperator
from perflowai.core import ModelConfig

model_config = ModelConfig(
    num_layers=64,
    hidden_size=4096,
    ffn_dim=16384,
    hidden_dim=256,
    num_heads=32,
    head_dim=128,
    dtype_bytes=2
)

layer = TransformerLayerOperator(model_config)
layer_flops = layer.compute_flops(1, 100, is_prefill=True)
layer_kvcache = layer.compute_kvcache(1, 100)  # Per layer

# For full model:
model_flops = layer_flops * model_config.num_layers
model_kvcache = layer_kvcache * model_config.num_layers  # 100 MB for 64 layers
```

## Model-Level Simulators

### `ModelMemSimulator`

Located in `perflowai/simulator/model/mem_simulator.py`

Computes KVCache for inference events using operator-level calculations.

**Key Features:**
- **Prefill**: Computes KVCache for all input tokens
- **Decode**: Tracks cumulative KVCache (input + decoded tokens)
- Handles multiple tasks in batch

**Example:**
```python
from perflowai.simulator.model import ModelMemSimulator
from perflowai.core import ModelConfig

model_config = ModelConfig(...)
mem_sim = ModelMemSimulator(model_config)

# During simulation, for each event:
kvcache_bytes = mem_sim.kvcache(event)
```

**Key Fixes:**
1. Removed incorrect `+1` in prefill calculation
2. Fixed decode to track `input_len + decode_iters + 1` (current cumulative length)
3. Uses operator-level `compute_kvcache()` for accurate calculations

### `ModelPerfSimulator`

Located in `perflowai/simulator/model/perf_simulator.py`

Computes execution time based on FLOPs and memory bandwidth.

**Key Features:**
- Computes both compute-bound and memory-bound time
- Returns the maximum (bottleneck)
- Efficiency factors: 60% compute, 80% memory bandwidth

**Formula:**
```python
compute_time = total_flops / (device_flops * 0.6)
memory_time = total_mem_bytes / (device_bandwidth_GB_s * 0.8 * 1e9)
execution_time = max(compute_time, memory_time)
```

**Example:**
```python
from perflowai.simulator.model import ModelPerfSimulator
from perflowai.core import ModelConfig, DeviceConfig, DeviceType

model_config = ModelConfig(...)
device_config = DeviceConfig(
    id=0,
    type=DeviceType.GPU,
    memory_capacity=16384,  # MB
    memory_bandwidth=900,   # GB/s
    compute_flops=1e12      # 1 TFLOPS
)

perf_sim = ModelPerfSimulator(model_config, device_config)
execution_time = perf_sim.time(event)  # Returns time in seconds
```

**Key Fixes:**
1. Removed incorrect `+1` in prefill
2. Fixed attention FLOPs formula (proper QKV projection + attention matmuls)
3. Fixed FFN to use `hidden_size` not `hidden_dim`
4. Fixed decode to properly track `decode_iters`
5. Corrected memory bandwidth unit conversion (GB/s → bytes/s)

## Usage in Inference Simulator

The inference simulator (`perflowai/simulator/infer/infer_simulator.py`) uses these components:

```python
from perflowai.simulator import InferSimulator
from perflowai.core import ModelConfig, DeviceConfig

# Setup
model_config = ModelConfig(...)
device_config = DeviceConfig(...)

# Run simulation
simulator = InferSimulator(graph, requests)
trace = simulator.simulate(scheduler, model_config, device_config)

# Get metrics
metrics = simulator.get_metrics()
print(metrics['TTFT'])  # Time To First Token
print(metrics['TPS'])   # Tokens Per Second
```

## Testing

Comprehensive tests are provided in:
- `tests/units_tests/oprt_simulator_test.py`: Tests for all operators
- `tests/units_tests/mem_perf_simulator_test.py`: Tests for model-level simulators

Run tests:
```bash
pytest tests/units_tests/oprt_simulator_test.py -v
pytest tests/units_tests/mem_perf_simulator_test.py -v
```

## Common Calculations

### KVCache for Different Scenarios

**Single request (input_len=100):**
```
Per layer: 2 × 32 × 100 × 128 × 2 = 1,638,400 bytes (1.56 MB)
64 layers: 104,857,600 bytes (100 MB)
```

**After decode iteration 5:**
```
seq_len = input_len + decode_iters + 1 = 100 + 5 + 1 = 106
Per layer: 2 × 32 × 106 × 128 × 2 = 1,736,704 bytes
64 layers: 111,149,056 bytes (106 MB)
```

### FLOPs for Different Phases

**Prefill (seq_len=100):**
- Scales as O(seq_len²) for attention
- ~40 GFLOPs per layer for this config
- ~2.5 TFLOPs for 64 layers

**Decode (cache_len=100, generate 1 token):**
- Scales as O(seq_len) for attention
- ~27 GFLOPs per layer
- ~1.7 TFLOPs for 64 layers

## Performance Considerations

1. **Prefill is compute-intensive**: O(seq_len²) attention dominates
2. **Decode is memory-intensive**: Reading KV cache becomes bottleneck
3. **Batch size**: Linear impact on FLOPs and memory
4. **Sequence length**: Quadratic impact on prefill, linear on decode

## Future Enhancements

Potential improvements:
1. Add MoE (Mixture of Experts) operators
2. Support for sparse attention patterns
3. Quantization effects (INT8, FP8)
4. Multi-GPU communication operators
5. Flash Attention optimizations
