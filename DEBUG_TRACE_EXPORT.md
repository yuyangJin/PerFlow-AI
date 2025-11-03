# Debug Trace Export Feature

## Overview

The PADoC end-to-end example now supports optional JSON export of intermediate trace formats for debugging and analysis purposes.

## Usage

To enable trace export, edit `examples/padoc_end_to_end.py` and set:

```python
def main():
    # Optional: Enable saving intermediate traces for debugging
    save_debug_traces = True  # Set to True to save debug traces
    ...
```

## Exported Files

When enabled, two JSON files are created in `/tmp/`:

### 1. Trace After MST Mapping
**File**: `/tmp/padoc_e2e_trace_mapped.json`

**Contents**:
- **Metadata**: Number of devices, total events
- **MST Nodes**: Complete MST structure with node hierarchy
- **Device Events**: All events with their MST node associations

**Example Structure**:
```json
{
  "metadata": {
    "type": "trace_after_mst_mapping",
    "num_devices": 1,
    "total_events": 50
  },
  "mst_nodes": {
    "0": {
      "name": "root",
      "type": "root",
      "parent_id": null,
      "call_stack": [],
      "num_events": 0
    },
    "1": {
      "name": "embedding",
      "type": "module",
      "parent_id": 0,
      "call_stack": ["model", "embedding"],
      "num_events": 3
    }
  },
  "devices": {
    "0": {
      "num_events": 50,
      "events": [
        {
          "id": 0,
          "name": "embedding_call",
          "type": "FWD",
          "timestamp": 1234567,
          "duration": 1000,
          "mst_node_id": 1,
          "mst_node_name": "embedding"
        }
      ]
    }
  }
}
```

**Use Cases**:
- Verify MST mapping accuracy
- Debug which events map to which MST nodes
- Analyze call stack associations
- Understand hierarchical structure

### 2. Compressed Trace
**File**: `/tmp/padoc_e2e_trace_compressed.json`

**Contents**:
- **Compression Statistics**: Original size, compressed size, ratio, savings
- **Compression Strategy**: Which compression strategy was used
- **Device Data**: Encoded data size and metadata per device

**Example Structure**:
```json
{
  "metadata": {
    "type": "compressed_trace",
    "compression_strategy": "intra",
    "compression_stats": {
      "original_size_bytes": 2400,
      "compressed_size_bytes": 850,
      "compression_ratio": 2.82,
      "space_savings_percent": 64.58
    }
  },
  "compressed_data": {
    "num_devices": 1,
    "devices": {
      "0": {
        "num_events": 50,
        "encoded_data_size_bytes": 425,
        "metadata": {
          "prediction_model": {},
          "encoding_info": "LEB128 variable-length encoding with zigzag for signed integers"
        }
      }
    }
  }
}
```

**Use Cases**:
- Analyze compression effectiveness
- Compare different compression strategies
- Debug compression issues
- Understand encoded data structure

## Implementation Details

### Helper Functions

1. **`_save_trace_to_json(trace, mst, output_path)`**
   - Serializes Trace and MST objects to JSON
   - Maps each event to its corresponding MST node
   - Exports complete MST hierarchy

2. **`_save_compressed_trace_to_json(compressed, stats, output_path)`**
   - Serializes compressed trace data to JSON
   - Includes compression statistics and metadata
   - Provides insight into encoding structure

### Integration

The functions are integrated into the workflow steps:
- Step 3: `step3_map_trace_to_mst(..., save_mapped_trace=save_debug_traces)`
- Step 4: `step4_compress_events_on_mst(..., save_compressed_trace=save_debug_traces)`

## Benefits

1. **Debugging**: Inspect intermediate trace formats to diagnose issues
2. **Validation**: Verify that MST mapping and compression work correctly
3. **Analysis**: Understand trace structure at different pipeline stages
4. **Learning**: Educational tool for understanding PADoC internals
5. **Testing**: Create test cases based on actual trace data

## Performance Impact

- **Disabled (default)**: No performance impact
- **Enabled**: Minor overhead from JSON serialization, only during save operations
- Files are written to `/tmp/` to avoid cluttering the repository

## Future Enhancements

Potential additions:
- Export to other formats (CSV, Parquet)
- Visualization tools for exported traces
- Diff tools to compare traces
- Compression ratio analysis across strategies
- Event distribution statistics
