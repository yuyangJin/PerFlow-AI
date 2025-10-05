# PADoC: Performance Analytics Directly on Compressed Trace

PADoC is a novel compressed trace format that supports efficient direct analysis on compressed data. It combines linear prediction models with delta encoding to achieve lossless compression while maintaining constant-time (O(1)) random access capabilities.

## Quick Start: End-to-End Example

For a complete demonstration of the PADoC workflow, see:
- **Example**: `examples/padoc_end_to_end.py` - Complete workflow from torch.fx to analysis
- **Documentation**: `PADOC_END_TO_END.md` - Detailed documentation

```bash
python examples/padoc_end_to_end.py
```

This example demonstrates:
1. MST extraction from torch.fx symbolic trace
2. Trace collection with TorchProfiler
3. Trace-to-MST mapping with call stacks
4. Trace compression using linear prediction
5. Performance analysis on compressed trace

## Overview

PADoC consists of three main components:

1. **Model Structure Tree (MST)**: A hierarchical representation of model operations with call stacks
2. **Compression Engine**: Linear prediction + delta encoding for timestamps and durations
3. **Analysis Modules**: Direct analysis on compressed format (bubble, overlap, imbalance)

## Key Features

- **Lossless Compression**: Preserves all trace information exactly
- **Random Access**: O(1) access to individual events in compressed format
- **Direct Analysis**: Perform analytics without full decompression
- **Multiple Strategies**: Intra/inter-microbatch/iteration/process compression
- **Variable-Length Encoding**: Efficient storage using zigzag + LEB128-style encoding
- **End-to-End Workflow**: Complete pipeline from model to analysis

## Architecture

### Part 1: Model Structure Tree (MST)

The MST represents the hierarchical structure of a model with operations and their call stacks:

```python
from perflowai.padoc import ModelStructureTree, MSTMapper

# Create MST manually
mst = ModelStructureTree()
layer = mst.add_node('TransformerLayer', 'layer', parent_id=0)
attention = mst.add_node('Attention', 'operation', layer.node_id)
attention.set_call_stack(['model', 'layer', 'attention'])

# Or build from TorchProfiler trace
mst = MSTMapper.build_and_map_from_file('trace.json', trace)

# Build from torch.fx symbolic trace (creates hierarchical structure)
import torch.fx as fx
traced_model = fx.symbolic_trace(model)
mst = ModelStructureTree.from_torch_fx_graph(traced_model.graph, 'MyModel')

# Text visualization (shows hierarchy with indentation)
print(mst.visualize())

# Graphviz visualization (hierarchical tree layout)
mst.visualize_graphviz('output_path', format='pdf')  # Generates tree with colored circles
mst.visualize_graphviz('output_path', format='png')  # PNG output
mst.visualize_graphviz('output_path', format='svg')  # SVG output

# Save/Load
mst.to_json('mst.json')
loaded_mst = ModelStructureTree.from_json('mst.json')
```

#### Building MST from torch.fx

PADoC automatically builds a **hierarchical MST** from a PyTorch model using `torch.fx.symbolic_trace`:

```python
import torch
import torch.nn as nn
import torch.fx as fx
from perflowai.padoc import ModelStructureTree

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Trace the model
model = MyModel()
traced_model = fx.symbolic_trace(model)

# Build hierarchical MST from the graph
mst = ModelStructureTree.from_torch_fx_graph(traced_model.graph, 'MyModel')

# The MST will contain a hierarchical structure:
# MyModel (model)
#   ├── linear1 (module)
#   │   └── linear1_call (operation)
#   ├── relu (module)
#   │   └── relu_call (operation)
#   ├── linear2 (module)
#   │   └── linear2_call (operation)
#   ├── x (input)
#   └── output (output)

# Each node has a call stack set automatically based on hierarchy
```

**Hierarchical Features**:
- Modules are organized in parent-child relationships
- Operations are nested under their parent modules
- Nested modules (e.g., `outer.inner.linear`) create multi-level hierarchy
- Text visualization shows indentation based on depth
- Graphviz visualization uses tree layout

#### Graphviz Visualization

PADoC provides high-quality tree visualizations using Graphviz with Nature paper inspired aesthetics:

- **Smaller circles** (0.4 units) for cleaner appearance
- **Labels positioned outside circles** for better readability
- **Nature paper color scheme** - elegant, professional colors
- **Tree layout** shows hierarchical structure
- **Tooltips** provide additional information

```python
# Generate PDF visualization
pdf_file = mst.visualize_graphviz('mst_output', format='pdf')
# Creates: mst_output.pdf

# Generate PNG for embedding in documents
png_file = mst.visualize_graphviz('mst_output', format='png')
# Creates: mst_output.png

# Generate SVG for web viewing
svg_file = mst.visualize_graphviz('mst_output', format='svg')
# Creates: mst_output.svg
```

**Nature-inspired color palette** (professional, elegant):
- Root: Light Gray (#E8E8E8)
- Model: Professional Blue (#4A90E2)
- Module: Nature Green (#7CB342)
- Operation: Soft Orange (#F4A460)
- Function: Muted Purple (#9C64A6)
- Input: Warm Yellow (#FDD835)
- Output: Pink (#EC407A)
- Parameter: Deep Purple (#AB47BC)
- And more...

**Design improvements:**
- Smaller circles (0.4 vs 0.8 units) reduce visual clutter
- Text outside circles improves readability
- Softer, more refined colors inspired by Nature journal guidelines
- Professional appearance suitable for publications

#### Mapping TorchProfiler Traces to MST

PADoC can map TorchProfiler JSON trace files to MST nodes using call stacks:

```python
from perflowai.padoc import MSTMapper

# Build MST and map trace events automatically
mst = MSTMapper.build_and_map_from_file('torchprofiler_trace.json', trace)

# Or map manually using call stacks
event_id = 123
call_stack = ['model', 'layer1', 'attention', 'forward']
matched_node = mst.map_trace_events_by_callstack(event_id, call_stack)

# The matched node will contain the mapped event IDs
print(f"Mapped events: {matched_node.trace_events}")

# Visualize with trace event counts
print(mst.visualize())  # Shows [events: N] for nodes with mapped events
```

**Enhanced call stack extraction:**

PADoC now supports advanced call stack reconstruction using TorchProfiler's Python id hierarchy:

```python
# TorchProfiler event structure with Python id/parent id
{
    "name": "attention_forward",
    "ph": "X",
    "cat": "python_function", 
    "ts": 1020,
    "dur": 30,
    "args": {
        "Python id": 3,           # Current call id
        "Python parent id": 2,    # Parent call id
        "External id": 102,
        ...
    }
}
```

The enhanced mapper:
1. Builds a parent-child relationship map using "Python id" and "Python parent id"
2. Reconstructs full hierarchical call stacks by following parent chains
3. Falls back to traditional "Python call stack" string parsing if ids not available
4. Extracts call stacks from TorchProfiler trace events
5. Finds MST nodes with matching call stack prefixes
6. Maps events to the most specific (deepest) matching node
7. Stores event IDs in the MST node for later analysis

This provides more accurate hierarchical representation of nested function calls.

### Part 2: Compression

The compression engine uses linear prediction to estimate timestamps and durations, then stores only the small deltas:

```python
from perflowai.padoc import TraceCompressor, TraceDecompressor

# Compress trace
compressor = TraceCompressor()
compressed = compressor.compress_trace(trace, strategy='intra')

# Get compression statistics
stats = compressor.get_compression_stats(trace, compressed)
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Space savings: {stats['space_savings']*100:.1f}%")

# Save compressed trace
compressor.save_compressed(compressed, 'compressed_trace.json')

# Decompress
decompressor = TraceDecompressor()
loaded = decompressor.load_compressed('compressed_trace.json')
trace = decompressor.decompress_trace(loaded)

# Random access to specific event (O(1))
event = decompressor.get_event_at_index(compressed, dev_id=0, index=10)
```

#### Compression Strategies

- **intra**: Predicts based on previous event in same sequence
- **inter_mb**: Uses patterns across microbatches
- **inter_iter**: Uses patterns across iterations
- **inter_proc**: Uses patterns across processes/devices

#### Compression Algorithm

1. **Linear Prediction**: For each event, predict its timestamp/duration based on context
2. **Delta Calculation**: Compute delta = actual - predicted
3. **Zigzag Encoding**: Map signed integers to unsigned: 0→0, -1→1, 1→2, -2→3, ...
4. **Variable-Length Encoding**: Encode using LEB128-style format (small values use fewer bytes)

Example:
```
Actual timestamps: [1000, 1150, 1300, 1500]
Predictions:      [0, 1100, 1250, 1450]
Deltas:           [1000, 50, 50, 50]
Encoded:          [highly compressed due to small deltas]
```

### Part 3: Analysis

PADoC supports direct analysis on compressed format without full decompression:

```python
from perflowai.padoc import BubbleAnalyzer, OverlapAnalyzer, ImbalanceAnalyzer

# Bubble Analysis (idle time between stages)
bubble_result = BubbleAnalyzer.analyze(trace)
print(f"Average bubble ratio: {bubble_result['average_bubble_ratio']:.2%}")

# Or analyze directly on compressed format
bubble_result = BubbleAnalyzer.analyze_compressed(compressed)

# Overlap Analysis (computation-communication overlap)
overlap_result = OverlapAnalyzer.analyze(trace)
print(f"Overlap ratio: {overlap_result['average_overlap_ratio']:.2%}")

# Imbalance Analysis (load distribution across devices)
imbalance_result = ImbalanceAnalyzer.analyze(trace)
print(f"Imbalance ratio: {imbalance_result['imbalance_ratio']:.2%}")
```

## Complete Example

See `examples/padoc_demo.py` for a comprehensive demonstration:

```bash
cd examples
python padoc_demo.py
```

This demonstrates:
- MST creation and visualization
- Trace compression with multiple strategies
- Lossless compression verification
- Bubble, overlap, and imbalance analysis
- Direct analysis on compressed format
- Efficiency measurements

## Performance Characteristics

### Compression

- **Time Complexity**: O(n) where n is number of events
- **Space Complexity**: Depends on prediction accuracy
  - Small deltas (common case): ~1-2 bytes per event
  - Large deltas: up to 5 bytes per event (for 32-bit values)

### Random Access

- **Time Complexity**: O(1) with proper indexing (O(k) in current implementation where k is index)
- **Space Complexity**: O(1) additional memory

### Analysis

- **Direct Analysis**: Can operate on compressed format for certain queries
- **Bubble Analysis**: O(n) on compressed format
- **Full Decompression**: O(n) when needed

## Testing

Run the test suite:

```bash
# All PADoC tests
pytest tests/units_tests/MST_test.py \
       tests/units_tests/Compression_test.py \
       tests/units_tests/Analyzer_test.py \
       tests/integration_tests/padoc_test.py -v

# Individual test modules
pytest tests/units_tests/MST_test.py -v              # MST tests
pytest tests/units_tests/Compression_test.py -v      # Compression tests
pytest tests/units_tests/Analyzer_test.py -v         # Analysis tests
pytest tests/integration_tests/padoc_test.py -v -s   # Integration test
```

## API Reference

### ModelStructureTree

```python
class ModelStructureTree:
    def add_node(name: str, node_type: str, parent_id: int) -> MSTNode
    def get_node(node_id: int) -> Optional[MSTNode]
    def find_nodes_by_name(name: str) -> List[MSTNode]
    def find_nodes_by_type(node_type: str) -> List[MSTNode]
    def map_trace_event(event_id: int, node_id: int)
    def map_trace_events_by_callstack(event_id: int, call_stack: List[str]) -> Optional[MSTNode]
    def visualize(node: MSTNode = None, indent: int = 0) -> str
    def to_json(filepath: str)
    def from_json(filepath: str) -> ModelStructureTree
```

### TraceCompressor

```python
class TraceCompressor:
    def compress_trace(trace: Trace, strategy: str = 'auto') -> Dict[str, Any]
    def save_compressed(compressed_data: Dict[str, Any], filepath: str)
    def get_compression_stats(original_trace: Trace, compressed_data: Dict[str, Any]) -> Dict[str, Any]
```

### TraceDecompressor

```python
class TraceDecompressor:
    def decompress_trace(compressed_data: Dict[str, Any]) -> Trace
    def get_event_at_index(compressed_data: Dict[str, Any], dev_id: int, index: int) -> Event
    def load_compressed(filepath: str) -> Dict[str, Any]
```

### Analyzers

```python
class BubbleAnalyzer:
    @staticmethod
    def analyze(trace: Trace) -> Dict[str, Any]
    @staticmethod
    def analyze_compressed(compressed_data: Dict[str, Any]) -> Dict[str, Any]

class OverlapAnalyzer:
    @staticmethod
    def analyze(trace: Trace) -> Dict[str, Any]

class ImbalanceAnalyzer:
    @staticmethod
    def analyze(trace: Trace) -> Dict[str, Any]
    @staticmethod
    def analyze_by_stage(trace: Trace, nstages: int) -> Dict[str, Any]
```

## Design Principles

1. **Predictable Patterns**: Performance traces often show predictable patterns (regular intervals, similar durations)
2. **Small Deltas**: Prediction errors are typically small, centered around zero
3. **Efficient Encoding**: Variable-length encoding exploits small delta distribution
4. **Random Access**: Maintain ability to access specific events without full decompression
5. **Direct Analysis**: Support common analytics queries on compressed format

## Related Work

PADoC draws inspiration from:
- **Scalasca/Scalasca-2**: Trace analysis framework (excluding differential compression)
- **LEB128**: Variable-length integer encoding used in DWARF and WebAssembly
- **Zigzag Encoding**: Efficient signed integer encoding (Protocol Buffers)
- **Delta Encoding**: Common in time-series databases

## Future Enhancements

Potential improvements:
- [ ] More sophisticated prediction models (e.g., pattern recognition)
- [ ] Adaptive strategy selection based on trace characteristics
- [ ] Parallel compression/decompression
- [ ] True O(1) random access with cumulative sum tables
- [ ] GPU-accelerated analysis on compressed format
- [ ] Integration with distributed trace collection

## Citation

If you use PADoC in your research, please cite:

```bibtex
@misc{perflowai2024jin,
    title={PerFlow-AI: a programable performance analysis, modeling, prediction tool for AI systems},
    author={Yuyang Jin, Xirui Shui, Runxin Zhong, Mingshu Zhai, Kezhao Huang, Jiaao He, Zan Zong, and Jidong Zhai},
    year={2025},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/yuyangJin/PerFlow-AI}},
}
```

## License

See the main LICENSE file in the repository root.
