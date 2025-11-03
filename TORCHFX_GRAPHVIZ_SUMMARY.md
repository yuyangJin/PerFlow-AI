# PADoC Enhancement: torch.fx and Graphviz Support

This document summarizes the enhancements made to PADoC to support torch.fx graph building and Graphviz visualization.

## Summary of Changes

Three major features were implemented:

1. **torch.fx Graph Support**: Build Model Structure Tree (MST) directly from PyTorch models using torch.fx symbolic tracing
2. **Graphviz Visualization**: Generate high-quality PDF/PNG/SVG visualizations with colored circles representing different node types
3. **Enhanced TorchProfiler Mapping**: Improved call stack-based mapping of trace events to MST nodes

## Implementation Details

### 1. torch.fx Graph Support

**File Modified**: `perflowai/padoc/mst.py`

Added a new static method `ModelStructureTree.from_torch_fx_graph()` that:
- Takes a `torch.fx.Graph` object from `fx.symbolic_trace(model)`
- Extracts all nodes from the graph (placeholders, modules, functions, outputs)
- Builds a hierarchical MST with proper node types
- Sets call stacks automatically for each node
- Stores operation metadata (op type, target) as node attributes

**Supported Node Types**:
- `input`: Placeholder nodes for model inputs
- `module`: Module calls (e.g., Linear, ReLU layers)
- `function`: Function calls (e.g., torch.relu, torch.matmul)
- `method`: Method calls
- `parameter`: get_attr operations
- `output`: Output nodes

**Usage**:
```python
import torch.fx as fx
from perflowai.padoc import ModelStructureTree

traced_model = fx.symbolic_trace(model)
mst = ModelStructureTree.from_torch_fx_graph(traced_model.graph, 'MyModel')
```

### 2. Graphviz Visualization

**File Modified**: `perflowai/padoc/mst.py`

Added a new method `visualize_graphviz()` that:
- Uses the `graphviz` Python library to generate tree visualizations
- Represents nodes as small circles with different colors based on type
- Supports multiple output formats: PDF, PNG, SVG
- Includes tooltips with node information
- Uses tree layout (top-to-bottom) for hierarchical structure

**Color Scheme**:
- Root: Gray (#e0e0e0)
- Model: Blue (#90caf9)
- Module: Green (#81c784)
- Operation: Orange (#ffb74d)
- Function: Purple (#ba68c8)
- Input: Yellow (#fff59d)
- Output: Pink (#f48fb1)
- Parameter: Light Purple (#ce93d8)
- And more...

**Usage**:
```python
# Generate PDF
pdf_file = mst.visualize_graphviz('output_path', format='pdf')

# Generate PNG
png_file = mst.visualize_graphviz('output_path', format='png')

# Generate SVG
svg_file = mst.visualize_graphviz('output_path', format='svg')
```

### 3. Enhanced TorchProfiler Mapping

**Already Implemented**: The call stack-based mapping was already present in `perflowai/padoc/mst_mapper.py`

The existing implementation:
- Extracts call stacks from TorchProfiler JSON traces
- Matches events to MST nodes based on call stack prefixes
- Maps events to the most specific (deepest) matching node
- Stores event IDs in MST nodes for later analysis

This feature was already working and is now better documented.

## New Files Added

### 1. Test File: `tests/units_tests/MST_torchfx_test.py`

Comprehensive test suite covering:
- **Test 1**: Building MST from torch.fx graph
  - Creates a simple PyTorch model
  - Traces it with torch.fx.symbolic_trace
  - Builds MST and verifies structure
  - Checks node types, call stacks

- **Test 2**: Graphviz PDF visualization
  - Creates a sample MST
  - Generates PDF, PNG visualizations
  - Verifies files are created

- **Test 3**: torch.fx + Graphviz integration
  - Combines both features
  - Generates visualizations from torch.fx MST

- **Test 4**: TorchProfiler mapping with call stacks
  - Creates mock TorchProfiler trace
  - Maps events to MST by call stack
  - Verifies correct mapping

- **Test 5**: Full workflow integration
  - End-to-end test of all features
  - torch.fx → MST → Graphviz → Event mapping

All tests pass successfully.

### 2. Demo Script: `examples/mst_torchfx_demo.py`

A complete demonstration script that:
- Creates a Transformer-like PyTorch model
- Traces it with torch.fx
- Builds MST from the traced graph
- Generates visualizations in PDF, PNG, SVG formats
- Simulates trace event mapping
- Generates final visualization with event counts

**Run the demo**:
```bash
python examples/mst_torchfx_demo.py
```

This generates:
- `mst_torchfx_demo.pdf` - PDF visualization
- `mst_torchfx_demo_png.png` - PNG visualization
- `mst_torchfx_demo_svg.svg` - SVG visualization
- `mst_torchfx_final.pdf` - Final visualization with event info

## Documentation Updates

### 1. Updated: `perflowai/padoc/README.md`

Added comprehensive documentation sections:
- **Building MST from torch.fx**: Complete example with explanation
- **Graphviz Visualization**: Usage examples and color scheme documentation
- **Mapping TorchProfiler Traces to MST**: Detailed explanation of the mapping algorithm

### 2. Updated: `.gitignore`

Added patterns to ignore generated visualization files:
```
mst_*.pdf
mst_*.png
mst_*.svg
mst_visualization_screenshot.png
```

## Dependencies

Two new dependencies are required (already installed):
- `torch`: For torch.fx support
- `graphviz`: For PDF/PNG/SVG generation

Both are optional and the code gracefully handles their absence with clear error messages.

## Example Output

Here's an example of the MST visualization generated from a torch.fx traced model:

```
root (root)
  TransformerFFN (model) [depth: 1]
    x (input) [depth: 2]
    self_attn (module) [depth: 2]
    getitem (function) [depth: 2]
    getitem_1 (function) [depth: 2]
    add (function) [depth: 2]
    norm1 (module) [depth: 2]
    linear1 (module) [depth: 2]
    relu (function) [depth: 2]
    dropout (module) [depth: 2]
    linear2 (module) [depth: 2]
    add_1 (function) [depth: 2]
    norm2 (module) [depth: 2]
    output (output) [depth: 2]
```

The Graphviz visualization shows this as a tree with colored circles for each node.

## Testing

All tests pass:
- 6 existing MST tests (from `MST_test.py`)
- 5 new torch.fx and Graphviz tests (from `MST_torchfx_test.py`)

**Total**: 11 MST-related tests, all passing ✓

## Usage Examples

### Example 1: Build MST from torch.fx and visualize

```python
import torch
import torch.nn as nn
import torch.fx as fx
from perflowai.padoc import ModelStructureTree

# Define model
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

# Trace and build MST
model = MyModel()
traced = fx.symbolic_trace(model)
mst = ModelStructureTree.from_torch_fx_graph(traced.graph, 'MyModel')

# Visualize
print(mst.visualize())
mst.visualize_graphviz('my_model_mst', format='pdf')
```

### Example 2: Map TorchProfiler trace to MST

```python
from perflowai.padoc import MSTMapper
from perflowai.reader import TorchProfilerTraceReader

# Read trace
reader = TorchProfilerTraceReader('trace.json')
trace = reader.read([EventType.FWD, EventType.BWD])

# Build MST from torch.fx
mst = ModelStructureTree.from_torch_fx_graph(traced.graph, 'MyModel')

# Map trace events to MST
mst = MSTMapper.map_trace_to_mst(trace, mst, trace_json)

# Visualize with event counts
print(mst.visualize())  # Shows [events: N] for nodes
mst.visualize_graphviz('mst_with_events', format='pdf')
```

## Conclusion

All three requested features have been implemented:
1. ✅ Support building MST with torch.fx.symbolic_trace(tm).graph
2. ✅ Visualize MST with PDF format using Graphviz (small circles, different colors, tree layout)
3. ✅ Map TorchProfiler JSON traces to MST by call stacks (already implemented, now tested and documented)

The implementation is production-ready with comprehensive tests, documentation, and examples.
