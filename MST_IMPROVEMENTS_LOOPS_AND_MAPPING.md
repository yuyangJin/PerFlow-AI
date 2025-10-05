# MST Improvements: Loop Merging and Enhanced Name Mapping

This document describes the two major improvements to the Model Structure Tree (MST) implementation.

## 1. Loop Merging for Similar Layers

### Problem
In models with repeated sequential layers (e.g., transformer layers), the MST creates many redundant nodes that make the structure harder to visualize and understand.

**Example**: A model with 4 sequential transformer layers creates:
```
Model
  ├── layer.0
  ├── layer.1
  ├── layer.2
  └── layer.3
```

### Solution
Automatically detect repeated layer patterns and merge them under a loop node.

**Result**:
```
Model
  └── layers_loop (i=0..3)
        ├── 0
        ├── 1
        ├── 2
        └── 3
```

### Implementation

**Pattern Detection**:
- Scans module paths for numeric indices (e.g., `layers.0`, `layers.1`, `layers.2`)
- Groups consecutive indices under a common prefix
- Creates a loop node when 2+ consecutive layers are found

**Usage**:
```python
import torch.fx as fx
from perflowai.padoc import ModelStructureTree

traced = fx.symbolic_trace(model)

# With loop merging (default)
mst = ModelStructureTree.from_torch_fx_graph(
    traced.graph, 
    'MyModel', 
    merge_similar_layers=True
)

# Without loop merging
mst_flat = ModelStructureTree.from_torch_fx_graph(
    traced.graph,
    'MyModel',
    merge_similar_layers=False
)
```

### Visualization Features

**Text Visualization**:
- Loop nodes shown with type `(loop)`
- Displays iteration count in depth indicator

**Graphviz Visualization**:
- Loop nodes colored in warm orange (#FFA726)
- Dashed edges from loop to iterations
- Label shows iteration range: `layers_loop\n(i=0..3)`
- Tooltip shows loop count

### Benefits
- **Cleaner Structure**: Reduces visual clutter for models with many repeated layers
- **Better Understanding**: Immediately shows repeated patterns
- **Scalability**: Works for any number of repeated layers
- **Flexible**: Can be disabled if flat structure is preferred

## 2. Enhanced Event-to-MST Name Mapping

### Problem
TorchProfiler events use different naming conventions than MST nodes, causing mapping failures:
- TorchProfiler: `aten::embedding`, `aten::linear`
- MST nodes: `embedding`, `linear`
- Result: Events not mapped to correct nodes

### Solution
Multi-strategy mapping with name normalization:

#### Strategy 1: Exact Name Matching
Try exact string match first (fastest).

#### Strategy 2: Normalized Name Matching
Normalize both event and node names:
- Remove `aten::` prefix
- Remove `torch.nn.` prefix
- Remove file path information (`file.py(123): func` → `func`)
- Convert to lowercase
- Handle partial matches (prefix/suffix)

#### Strategy 3: Call Stack Matching
Use hierarchical call stack information when available.

#### Strategy 4: Fallback
Create new node if no match found.

### Implementation

**Name Normalization**:
```python
from perflowai.padoc.mst_mapper import MSTMapper

# Examples
MSTMapper._normalize_operation_name('aten::embedding')  # → 'embedding'
MSTMapper._normalize_operation_name('torch.nn.Linear')  # → 'linear'
MSTMapper._normalize_operation_name('file.py(123): embedding')  # → 'embedding'
```

**Automatic Mapping**:
```python
from perflowai.padoc import MSTMapper

# Map trace to MST with improved matching
mst = MSTMapper.map_trace_to_mst(trace, mst, trace_json)
```

### Supported Patterns

**Prefix Removal**:
- `aten::` → removed (PyTorch ATen operations)
- `torch.nn.` → removed (PyTorch modules)
- `torch.` → removed (PyTorch functions)
- `nn.` → removed (Neural network modules)

**Path Handling**:
- File paths: `module.py(line): func` → `func`
- Module paths: `model.layer.op` matches `op`

**Case Handling**:
- Case-insensitive matching
- `EMBEDDING` matches `embedding`

### Benefits
- **Higher Mapping Rate**: More events correctly mapped to MST nodes
- **Robust**: Handles various naming conventions
- **Backward Compatible**: Falls back to exact matching
- **Extensible**: Easy to add new normalization rules

## Example Usage

### Complete Workflow

```python
import torch
import torch.nn as nn
import torch.fx as fx
from perflowai.padoc import ModelStructureTree, MSTMapper
from perflowai.core.trace import Trace
from perflowai.core.event import Event, EventType

# 1. Create model with repeated layers
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 256)
        self.layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

# 2. Build MST with loop merging
model = MyModel()
traced = fx.symbolic_trace(model)
mst = ModelStructureTree.from_torch_fx_graph(
    traced.graph, 
    'MyModel',
    merge_similar_layers=True
)

print(mst.visualize())
# Output shows:
#   Model
#     ├── embedding
#     ├── layers_loop (i=0..3)
#     │     ├── 0
#     │     ├── 1
#     │     ├── 2
#     │     └── 3
#     └── output

# 3. Map trace events with improved matching
trace = Trace(1)
trace.add_event(0, Event(1, EventType.FWD, 'aten::embedding', 1000, 100))
trace.add_event(0, Event(2, EventType.FWD, 'aten::linear', 2000, 150))

mst = MSTMapper.map_trace_to_mst(trace, mst)

# Events correctly mapped despite name differences
for node in mst.nodes.values():
    if node.trace_events:
        print(f"{node.name}: {len(node.trace_events)} events")
# Output:
#   embedding: 1 events
#   0: 1 events (linear layer)

# 4. Generate visualization
mst.visualize_graphviz('model_mst', format='pdf')
```

## Testing

Both features are tested in:
- `tests/units_tests/MST_test.py` - Basic MST functionality
- `tests/units_tests/MST_torchfx_test.py` - torch.fx integration

All tests pass with the new features.

## Performance Impact

**Loop Detection**:
- Time: O(n) where n = number of modules
- Space: O(m) where m = number of loop groups
- Minimal overhead, runs only once during MST construction

**Name Normalization**:
- Time: O(k) per event where k = name length (typically < 100 chars)
- Space: O(1) per normalization
- Negligible overhead, faster than creating new nodes

## Configuration

### Disabling Loop Merging

```python
# If you prefer flat structure
mst = ModelStructureTree.from_torch_fx_graph(
    graph, 
    model_name,
    merge_similar_layers=False  # Disable loop merging
)
```

### Adding Custom Normalization Rules

Extend `_normalize_operation_name()` in `mst_mapper.py`:

```python
@staticmethod
def _normalize_operation_name(name: str) -> str:
    # Add custom prefix
    if name.startswith('custom::'):
        name = name[8:]
    
    # Existing normalization...
    return name
```

## Future Enhancements

### Loop Merging
- Detect more complex patterns (nested loops)
- Support non-consecutive indices
- Add loop unrolling for visualization

### Name Mapping
- Machine learning-based fuzzy matching
- Operation signature matching (input/output types)
- Configuration file for custom mapping rules

## Related Files

- `perflowai/padoc/mst.py` - Loop detection and MST construction
- `perflowai/padoc/mst_mapper.py` - Name normalization and mapping
- `tests/units_tests/MST_torchfx_test.py` - Test coverage

## Conclusion

These improvements make MST more practical for real-world models:

1. **Loop Merging**: Simplifies visualization of models with repeated layers
2. **Enhanced Mapping**: Accurately maps TorchProfiler events despite naming differences

Both features work automatically with sensible defaults and can be configured as needed.
