# MST Visualization and TorchProfiler Parsing Improvements

This document summarizes the enhancements made to PADoC's Model Structure Tree visualization and TorchProfiler trace parsing capabilities.

## Summary of Improvements

### 1. Enhanced Graphviz Visualization

**Previous Design:**
- Circle size: 0.8 units
- Labels inside circles
- Material Design color scheme (bright, saturated colors)

**New Design:**
- Circle size: 0.4 units (50% smaller)
- Labels positioned outside circles (below)
- Nature paper inspired color palette (elegant, professional)
- Improved edge styling with softer colors

**Benefits:**
- ✅ Cleaner, less cluttered appearance
- ✅ Better readability with labels outside circles
- ✅ Professional aesthetics suitable for publications
- ✅ More refined color scheme inspired by Nature journal guidelines

**Nature-Inspired Color Palette:**
```python
{
    'root': '#E8E8E8',           # Light gray
    'model': '#4A90E2',          # Professional blue
    'module': '#7CB342',         # Nature green
    'operation': '#F4A460',      # Soft orange
    'function': '#9C64A6',       # Muted purple
    'input': '#FDD835',          # Warm yellow
    'output': '#EC407A',         # Pink
    'parameter': '#AB47BC',      # Deep purple
}
```

**Implementation Changes:**
- `width='0.4', height='0.4'` - Smaller circles
- `labelloc='b'` - Position labels below circles
- `fontcolor='#333333'` - Darker text for better contrast
- `edge color='#666666', penwidth='1.5'` - Softer edge styling

### 2. Enhanced TorchProfiler Trace Parsing

**Previous Implementation:**
- Only parsed "Python call stack" string field
- Simple string splitting by newlines
- No hierarchical relationship tracking

**New Implementation:**
- **Primary method**: Uses "Python id" and "Python parent id" fields
- Builds parent-child relationship map
- Reconstructs full hierarchical call stacks by following parent chains
- **Fallback**: Traditional "Python call stack" string parsing
- **Final fallback**: Uses event name

**TorchProfiler Event Structure Supported:**
```json
{
    "name": "attention_forward",
    "ph": "X",
    "cat": "python_function",
    "ts": 1020,
    "dur": 30,
    "args": {
        "Python id": 3,
        "Python parent id": 2,
        "External id": 102,
        "Ev Idx": 5,
        "Fwd thread id": 0,
        "Sequence number": 1,
        ...
    }
}
```

**Algorithm:**
1. **First pass**: Build mappings
   - `event_id_to_name`: Map Python id → event name
   - `event_id_to_parent`: Map Python id → parent id

2. **Second pass**: Reconstruct call stacks
   - Start from each event's Python id
   - Follow parent chain: id → parent → grandparent → ...
   - Build stack in reverse order (root to leaf)
   - Detect cycles to prevent infinite loops

3. **Fallbacks**:
   - If Python id not available: use "Python call stack" string
   - If no stack info: use event name

**Benefits:**
- ✅ More accurate hierarchical representation
- ✅ Handles complex nested function calls
- ✅ Captures parent-child relationships correctly
- ✅ Supports sibling operations at same level
- ✅ Backward compatible with old trace formats

**Example Output:**
```
Event 0 (model_forward): model_forward
Event 1 (transformer_layer): model_forward -> transformer_layer
Event 2 (attention_forward): model_forward -> transformer_layer -> attention_forward
Event 3 (matmul): model_forward -> transformer_layer -> attention_forward -> matmul
Event 4 (ffn_forward): model_forward -> transformer_layer -> ffn_forward
```

## Testing

### New Test: `tests/units_tests/MST_enhanced_parsing_test.py`

**Test 1: Enhanced Parsing with Python ID**
- Creates mock trace with Python id/parent id hierarchy
- Tests 5 events with various nesting levels
- Verifies correct parent-child relationships
- Checks sibling relationships (attention vs ffn)

**Test 2: Fallback to Traditional Parsing**
- Tests backward compatibility
- Verifies "Python call stack" string parsing
- Tests event name fallback

**Results:** ✅ All tests pass

### Existing Tests: Still Pass
- `tests/units_tests/MST_test.py` - 6 tests ✅
- `tests/units_tests/MST_torchfx_test.py` - 5 tests ✅

## Files Modified

1. **perflowai/padoc/mst.py**
   - Updated `visualize_graphviz()` method
   - Smaller circles (0.4 units)
   - Labels outside circles
   - Nature paper color palette
   - Improved edge styling

2. **perflowai/padoc/mst_mapper.py**
   - Enhanced `extract_callstack_from_torchprofiler()` method
   - Added Python id/parent id parsing
   - Added parent-child relationship tracking
   - Added cycle detection
   - Maintained backward compatibility

3. **perflowai/padoc/README.md**
   - Updated visualization documentation
   - Added color palette details
   - Added enhanced parsing explanation
   - Included TorchProfiler event structure

4. **tests/units_tests/MST_enhanced_parsing_test.py** (NEW)
   - Comprehensive test for enhanced parsing
   - Tests hierarchical reconstruction
   - Tests fallback mechanisms

## Usage Examples

### Enhanced Visualization
```python
from perflowai.padoc import ModelStructureTree
import torch.fx as fx

# Build MST
traced = fx.symbolic_trace(model)
mst = ModelStructureTree.from_torch_fx_graph(traced.graph, 'MyModel')

# Generate with new design
mst.visualize_graphviz('output', format='pdf')
# Creates: output.pdf with smaller circles, labels outside, Nature colors
```

### Enhanced Parsing
```python
from perflowai.padoc import MSTMapper

# Read TorchProfiler trace
with open('trace.json') as f:
    trace_json = json.load(f)

# Extract call stacks (automatically uses Python id if available)
callstacks = MSTMapper.extract_callstack_from_torchprofiler(trace_json)

# Build MST with hierarchical call stacks
mst = MSTMapper.build_mst_from_torchprofiler(trace_json)
```

## Visual Comparison

**Before:**
- Large circles (0.8 units)
- Labels cramped inside circles
- Bright Material Design colors
- Higher visual density

**After:**
- Small circles (0.4 units)
- Labels clearly positioned below
- Elegant Nature paper colors
- Cleaner, more professional appearance

## Backward Compatibility

✅ **Fully backward compatible**
- Old trace formats without Python id still work
- Falls back to traditional parsing automatically
- No breaking changes to API
- All existing tests pass

## Commit

**Commit hash:** 662b022

**Commit message:**
```
Improve Graphviz visualization and TorchProfiler parsing

- Smaller circles (0.4 vs 0.8) for cleaner appearance
- Labels positioned outside circles for better readability  
- Nature paper inspired color scheme (elegant, professional)
- Enhanced TorchProfiler parsing with Python id/parent id hierarchy
- Better call stack reconstruction using parent-child relationships
- Fallback to traditional parsing when ids not available
- Added comprehensive test for enhanced parsing
```

## Conclusion

These improvements make PADoC's MST visualization more professional and publication-ready, while also improving the accuracy of TorchProfiler trace parsing through better hierarchical call stack reconstruction. The changes maintain full backward compatibility while providing significant enhancements for new trace formats.
