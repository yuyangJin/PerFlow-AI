# PADoC End-to-End Example Documentation

This document provides comprehensive documentation for the PADoC end-to-end workflow example that demonstrates the complete capabilities of Performance Analytics Directly on Compressed trace.

## Overview

The `examples/padoc_end_to_end.py` script demonstrates the complete PADoC workflow in a single, comprehensive example:

1. **MST Extraction from torch.fx** - Build Model Structure Tree from symbolic trace
2. **Trace Collection with TorchProfiler** - Profile model execution and collect trace
3. **Trace-to-MST Mapping** - Map collected events to MST nodes
4. **Trace Compression** - Compress events using linear prediction + delta encoding
5. **Performance Analysis** - Analyze compressed trace without full decompression

## Running the Example

```bash
python examples/padoc_end_to_end.py
```

## Workflow Steps

### Step 1: Extract MST from torch.fx

**Purpose**: Build a hierarchical Model Structure Tree representing the model's operations.

**Process**:
1. Create a simple neural network model (FFN with residual connection)
2. Trace the model using `torch.fx.symbolic_trace()`
3. Build MST from the traced graph using `ModelStructureTree.from_torch_fx_graph()`
4. Generate visualization with Graphviz

**Key Code**:
```python
traced_model = fx.symbolic_trace(model)
mst = ModelStructureTree.from_torch_fx_graph(traced_model.graph, 'SimpleFFN')
mst.visualize_graphviz('output', format='pdf')
```

**Output**:
- MST with hierarchical node structure
- PDF visualization showing model operations as a tree
- Each node represents a model operation (input, module, function, output)

### Step 2: Collect Trace with TorchProfiler

**Purpose**: Profile model execution and collect detailed trace data.

**Process**:
1. Create sample input data
2. Warm up the model with one forward pass
3. Profile execution using `torch.profiler.profile()`
4. Use `record_function` for better structure tracking
5. Export trace to JSON format

**Key Code**:
```python
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True,
) as prof:
    with record_function("model_forward"):
        output = model(input_data)

prof.export_chrome_trace('trace.json')
```

**Output**:
- JSON trace file with detailed event information
- Event categories: cpu_op, python_function, user_annotation
- Each event includes: name, timestamp, duration, category, args

### Step 3: Map Trace to MST

**Purpose**: Connect trace events to MST nodes for hierarchical analysis.

**Process**:
1. Read trace using `TorchProfilerTraceReader`
2. Convert raw events to internal `Trace` format
3. Map events to MST nodes using `MSTMapper.map_trace_to_mst()`
4. Build hierarchy using call stack information
5. Display mapped events per MST node

**Key Code**:
```python
reader = TorchProfilerTraceReader(trace_path)
trace = Trace(1)  # Single device

# Map events to MST
mst = MSTMapper.map_trace_to_mst(trace, mst, trace_json)
```

**Output**:
- MST with trace events mapped to nodes
- Event distribution across operations
- Hierarchical view of execution

### Step 4: Compress Events on MST

**Purpose**: Compress trace data using linear prediction and delta encoding.

**Process**:
1. Initialize `TraceCompressor`
2. Apply compression with 'intra' strategy (sequential prediction)
3. Calculate compression statistics
4. Verify lossless compression by decompressing
5. Test random access to compressed events

**Key Code**:
```python
compressor = TraceCompressor()
compressed = compressor.compress_trace(trace, strategy='intra')

# Get statistics
stats = compressor.get_compression_stats(trace, compressed)

# Random access (O(1))
decompressor = TraceDecompressor()
event = decompressor.get_event_at_index(compressed, dev_id=0, index=100)
```

**Compression Algorithm**:
1. **Linear Prediction**: Predict timestamp/duration from previous events
2. **Delta Calculation**: Store only `delta = actual - predicted`
3. **Zigzag Encoding**: Map signed integers to unsigned
4. **Variable-Length Encoding**: LEB128-style for small deltas

**Output**:
- Compressed trace data
- Compression statistics (ratio, space savings)
- Verification of lossless property
- O(1) random access demonstration

### Step 5: Analyze Compressed Trace

**Purpose**: Perform performance analysis directly on compressed data.

**Process**:
1. **Bubble Analysis**: Identify idle time and bubbles
2. **Overlap Analysis**: Measure computation-communication overlap
3. **Imbalance Analysis**: Detect load imbalance across devices

**Key Code**:
```python
# Bubble analysis
bubble_result = BubbleAnalyzer.analyze_compressed(compressed)

# Overlap analysis
overlap_result = OverlapAnalyzer.analyze(trace)

# Imbalance analysis
imbalance_result = ImbalanceAnalyzer.analyze(trace)
```

**Output**:
- Bubble ratio and total bubble time
- Overlap ratio and overlapped time
- Imbalance ratio and load distribution
- Analysis performed without full decompression

## Key Benefits Demonstrated

### 1. Hierarchical Model Representation
- MST provides structured view of model operations
- Automatic extraction from torch.fx
- Visual representation with Graphviz

### 2. Accurate Trace Mapping
- Events mapped to specific model operations
- Call stack-based hierarchical mapping
- Support for Python id/parent id hierarchy

### 3. Efficient Compression
- Linear prediction exploits predictable patterns
- Lossless compression with delta encoding
- Variable-length encoding for small deltas

### 4. Fast Random Access
- O(1) access to any event in compressed format
- No need for full decompression
- Efficient for targeted analysis

### 5. Direct Analysis
- Analyze compressed traces without decompression
- Bubble, overlap, and imbalance detection
- Space-efficient performance analytics

## Generated Files

The example generates the following files:

1. **`/tmp/padoc_e2e_mst.pdf`**
   - MST visualization in PDF format
   - Shows hierarchical model structure
   - Nodes colored by type (Nature paper style)

2. **`/tmp/padoc_e2e_trace.json`**
   - TorchProfiler trace in JSON format
   - Contains all profiled events
   - Compatible with Chrome tracing

## Example Output

```
======================================================================
PADoC End-to-End Example: Complete Workflow
======================================================================

This example demonstrates:
1. MST extraction from torch.fx
2. Trace collection with TorchProfiler
3. Trace-to-MST mapping
4. Trace compression on MST
5. Performance analysis on compressed trace

======================================================================
STEP 1: Extract MST from torch.fx
======================================================================

1.1 Tracing model with torch.fx...
   ✓ Model traced successfully

1.2 Building MST from torch.fx graph...
   ✓ MST built with 9 nodes

1.3 MST Structure:
root (root)
  SimpleFFN (model) [depth: 1]
    x (input) [depth: 2]
    fc1 (module) [depth: 2]
    relu (module) [depth: 2]
    fc2 (module) [depth: 2]
    add (function) [depth: 2]
    norm (module) [depth: 2]
    output (output) [depth: 2]

[... continues with all 5 steps ...]

======================================================================
✓ End-to-End Example Complete!
======================================================================
```

## Model Architecture

The example uses a simple Feed-Forward Network with residual connection:

```python
class SimpleFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.norm = nn.LayerNorm(256)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm(x + residual)
        return x
```

This architecture demonstrates:
- Linear layers (matrix multiplication)
- Activation functions (ReLU)
- Normalization (LayerNorm)
- Residual connections

## Trace Statistics

Typical trace statistics from the example:

- **Total events**: ~300 events
- **Event categories**:
  - `cpu_op`: CPU operations (tensor ops)
  - `python_function`: Python function calls
  - `user_annotation`: User-defined annotations
  
- **Mapped events**: ~50 events (limited for demo)
- **MST nodes**: ~9 original + ~30 dynamically created

## Performance Metrics

The example demonstrates three types of analysis:

### 1. Bubble Analysis
- Identifies idle time between operations
- Measures bubble ratio (idle_time / total_time)
- Useful for pipeline optimization

### 2. Overlap Analysis
- Measures computation-communication overlap
- Calculates overlap ratio
- Helps optimize async execution

### 3. Imbalance Analysis
- Detects load imbalance across devices
- Calculates imbalance ratio
- Useful for distributed training

## Extending the Example

You can extend this example for more complex scenarios:

### Multi-Device Training
```python
# Create multi-device trace
trace = Trace(ndevs=4)  # 4 devices

# Collect traces from each device
for dev_id in range(4):
    device_trace = collect_device_trace(dev_id)
    for event in device_trace:
        trace.add_event(dev_id, event)
```

### Multiple Compression Strategies
```python
# Try different strategies
for strategy in ['intra', 'inter_mb', 'inter_iter', 'inter_proc']:
    compressed = compressor.compress_trace(trace, strategy=strategy)
    stats = compressor.get_compression_stats(trace, compressed)
    print(f"{strategy}: {stats['compression_ratio']:.2f}x")
```

### Advanced Analysis
```python
# Custom analysis on compressed trace
def custom_analyzer(compressed):
    decompressor = TraceDecompressor()
    
    # Analyze specific patterns
    for i in range(len(compressed['events'][0])):
        event = decompressor.get_event_at_index(compressed, 0, i)
        # Perform analysis
        pass
```

## Dependencies

The example requires:

- `torch` - PyTorch framework
- `torch.profiler` - TorchProfiler for trace collection
- `torch.fx` - FX symbolic tracing
- `graphviz` - Visualization library (system package + Python)
- `perflowai.padoc` - PADoC implementation

Install with:
```bash
pip install torch numpy scipy matplotlib drawsvg graphviz
sudo apt-get install graphviz  # or brew install graphviz on macOS
pip install -e .
```

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution**: Install PyTorch: `pip install torch`

### Issue: "Graphviz executables not found"
**Solution**: Install system package: `sudo apt-get install graphviz`

### Issue: "Model cannot be traced"
**Solution**: Some PyTorch modules (like MultiheadAttention) cannot be traced directly. Use simpler models or wrap complex modules.

### Issue: "Compression ratio < 1"
**Solution**: Small traces may have overhead. Use larger traces with more predictable patterns for better compression.

## Related Examples

- `examples/mst_torchfx_demo.py` - Focused on torch.fx and visualization
- `examples/comprehensive_demo.py` - All PADoC features separately
- `examples/padoc_demo.py` - Original PADoC demonstration

## Conclusion

This end-to-end example demonstrates the complete PADoC workflow, showing how all components work together to provide efficient performance analytics on compressed traces. It showcases the key benefits of PADoC:

- **Automatic model analysis** with torch.fx
- **Accurate trace mapping** with call stacks
- **Efficient compression** with linear prediction
- **Fast random access** with O(1) complexity
- **Direct analysis** without decompression

The example can be easily extended for more complex scenarios, making it a valuable starting point for using PADoC in real-world applications.
