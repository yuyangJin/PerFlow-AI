# Hierarchical MST from torch.fx - Implementation Guide

This document describes the hierarchical Model Structure Tree (MST) generation from torch.fx graphs.

## Overview

The MST now properly reflects the hierarchical structure of PyTorch models when built from torch.fx graphs. Instead of creating a flat structure with all nodes as children of the model node, it builds a proper parent-child hierarchy based on module relationships.

## Key Improvements

### 1. Module Hierarchy

**Before** (Flat Structure):
```
root (root)
  Model (model)
    fc1 (module)
    relu (module)
    fc2 (module)
    norm (module)
    x (input)
    output (output)
```

**After** (Hierarchical Structure):
```
root (root)
  Model (model)
    fc1 (module)
      fc1_call (operation)
    relu (module)
      relu_call (operation)
    fc2 (module)
      fc2_call (operation)
    norm (module)
      norm_call (operation)
    x (input)
    add (function)
    output (output)
```

### 2. Nested Module Support

For models with nested modules (e.g., `outer.inner.linear`):

```
root (root)
  Model (model)
    outer (module)
      inner (module)
        linear (module)
          linear_call (operation)
```

The hierarchy properly reflects the module nesting structure.

## Implementation Details

### Two-Pass Algorithm

The implementation uses a two-pass algorithm:

**Pass 1: Create Module Hierarchy**
- Identifies all `call_module` operations
- Parses module paths (e.g., `outer.inner.linear`)
- Creates nested module nodes
- Builds parent-child relationships

**Pass 2: Add Operations**
- Creates nodes for all other operations
- Places them under appropriate parent modules
- Sets call stacks based on hierarchy
- Adds attributes and metadata

### Code Example

```python
import torch
import torch.nn as nn
import torch.fx as fx
from perflowai.padoc import ModelStructureTree

# Define a model with nested structure
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Trace and build hierarchical MST
model = MyModel()
traced = fx.symbolic_trace(model)
mst = ModelStructureTree.from_torch_fx_graph(traced.graph, 'MyModel')

# View hierarchical structure
print(mst.visualize())

# Generate hierarchical visualization
mst.visualize_graphviz('mst_output', format='pdf')
```

## Visualization Features

### Text Visualization

The text visualization shows the hierarchy with:
- **Indentation**: Nested levels are indented
- **Depth indicator**: `[depth: N]` shows call stack depth
- **Node type**: Shows the type of each node
- **Event count**: Shows mapped events if applicable

Example output:
```
root (root)
  SimpleFFN (model) [depth: 1]
    fc1 (module) [depth: 2]
      fc1_call (operation) [depth: 3]
    relu (module) [depth: 2]
      relu_call (operation) [depth: 3]
    x (input) [depth: 2]
```

### Graphviz Visualization

The Graphviz visualization:
- **Tree layout**: Top-to-bottom hierarchy
- **Colored circles**: Different colors for different node types
- **Edges**: Show parent-child relationships
- **Labels**: Node names outside circles
- **Professional style**: Nature paper inspired colors

## Benefits

### 1. Better Model Understanding
- Clear visualization of model architecture
- Easy identification of module relationships
- Depth-based organization

### 2. Improved Trace Mapping
- Events can be mapped to specific modules
- Hierarchical call stacks more accurate
- Better performance analysis granularity

### 3. Scalability
- Large models organized into manageable sections
- Nested modules properly represented
- Easier navigation of complex architectures

### 4. Analysis Benefits
- Analyze performance at module level
- Compare nested components
- Identify bottlenecks in specific layers

## Examples

### Example 1: Simple Feed-Forward Network

```python
class SimpleFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Creates hierarchy:
# SimpleFFN
#   ├── fc1
#   │   └── fc1_call
#   ├── relu
#   │   └── relu_call
#   └── fc2
#       └── fc2_call
```

### Example 2: Nested Modules

```python
class InnerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        
    def forward(self, x):
        return self.linear(x)

class OuterModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = InnerModule()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.inner(x)
        return self.relu(x)

# Creates hierarchy:
# Model
#   └── outer
#       ├── inner
#       │   └── linear
#       │       └── linear_call
#       └── relu
#           └── relu_call
```

## API Reference

### ModelStructureTree.from_torch_fx_graph()

```python
@staticmethod
def from_torch_fx_graph(graph, model_name: str = 'model') -> 'ModelStructureTree':
    """
    Build hierarchical MST from a torch.fx graph.
    
    Creates a proper hierarchy based on module structure:
    - Module calls create parent nodes with their operations as children
    - Sequential operations are grouped under their respective modules
    - Input/output nodes are top-level
    
    Args:
        graph: torch.fx.Graph object from symbolic_trace
        model_name: Name of the model (default: 'model')
        
    Returns:
        ModelStructureTree with hierarchical structure
    """
```

### Key Properties

- **Module nodes**: Created for each unique module path
- **Operation nodes**: Nested under their parent modules
- **Call stacks**: Automatically set based on hierarchy
- **Attributes**: Include `op` type and `target` information

## Testing

The hierarchical MST is tested in:
- `tests/units_tests/MST_torchfx_test.py` - All tests updated
- `examples/padoc_end_to_end.py` - End-to-end workflow

All tests pass with the new hierarchical structure:
```bash
python tests/units_tests/MST_torchfx_test.py
# ✓ All MST torch.fx and Graphviz tests passed!
```

## Migration Notes

### From Flat to Hierarchical

If you were using the flat structure, the main changes are:

1. **Node access**: Modules now have children operations
2. **Call stacks**: Reflect the full hierarchy depth
3. **Visualization**: Shows nested structure

### Backward Compatibility

The API remains the same:
- `ModelStructureTree.from_torch_fx_graph()` - Same signature
- `mst.visualize()` - Same method, better output
- `mst.visualize_graphviz()` - Same method, hierarchical layout

## Performance

The hierarchical generation has minimal overhead:
- **Time complexity**: O(n) where n = number of nodes
- **Space complexity**: O(n) for storing relationships
- **Two-pass algorithm**: Efficient module hierarchy building

## Future Enhancements

Possible future improvements:
- **Interactive visualization**: Expand/collapse modules
- **Filtering**: Show/hide specific hierarchy levels
- **Comparison**: Compare hierarchies of different models
- **Analysis**: Aggregate statistics at module level

## Conclusion

The hierarchical MST from torch.fx provides a much better representation of PyTorch model structure, making it easier to understand, analyze, and visualize complex models. Both text and graphical visualizations now properly reflect the module hierarchy, enabling more effective performance analysis and debugging.
