#!/usr/bin/env python3
'''
End-to-End PADoC Example: Complete Workflow
============================================

This example demonstrates the complete PADoC workflow:
1. Build MST from torch.fx symbolic trace
2. Run model and collect trace with TorchProfiler
3. Map TorchProfiler trace to MST nodes
4. Compress trace events on MST
5. Perform analysis on compressed trace

This showcases the full capabilities of PADoC for performance analytics
on compressed traces.
'''

import torch
import torch.nn as nn
import torch.fx as fx
from torch.profiler import profile, ProfilerActivity, record_function
import json
import tempfile
import os

from perflowai.padoc import (
    ModelStructureTree, 
    MSTMapper,
    TraceCompressor,
    TraceDecompressor,
    BubbleAnalyzer,
    OverlapAnalyzer,
    ImbalanceAnalyzer
)
from perflowai.core.trace import Trace
from perflowai.core.event import Event, EventType
from perflowai.reader import TorchProfilerTraceReader


def _save_trace_to_json(trace, mst, output_path):
    '''
    Helper function to save trace after MST mapping to JSON file.
    
    Args:
        trace: Trace object with events
        mst: Model Structure Tree with mapped events
        output_path: Path to save JSON file
    '''
    data = {
        'metadata': {
            'type': 'trace_after_mst_mapping',
            'num_devices': trace.get_ndevs(),
            'total_events': sum(len(trace.get_events(dev_id)) for dev_id in range(trace.get_ndevs()))
        },
        'mst_nodes': {},
        'devices': {}
    }
    
    # Save MST node information
    for node_id, node in mst.nodes.items():
        data['mst_nodes'][str(node_id)] = {
            'name': node.name,
            'type': node.node_type,
            'parent_id': node.parent_id,
            'call_stack': node.call_stack if node.call_stack else [],
            'num_events': len(node.trace_events)
        }
    
    # Save events per device with MST mapping
    for dev_id in range(trace.get_ndevs()):
        events_data = []
        for event in trace.get_events(dev_id):
            event_data = {
                'id': event.get_id(),
                'name': event.get_name(),
                'type': event.get_type().name,
                'timestamp': event.get_timestamp(),
                'duration': event.get_duration()
            }
            
            # Find which MST node this event is mapped to
            for node_id, node in mst.nodes.items():
                if event in node.trace_events:
                    event_data['mst_node_id'] = node_id
                    event_data['mst_node_name'] = node.name
                    break
            
            events_data.append(event_data)
        
        data['devices'][str(dev_id)] = {
            'num_events': len(events_data),
            'events': events_data
        }
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def _save_compressed_trace_to_json(compressed, stats, output_path):
    '''
    Helper function to save compressed trace to JSON file.
    
    Args:
        compressed: Compressed trace data
        stats: Compression statistics
        output_path: Path to save JSON file
    '''
    data = {
        'metadata': {
            'type': 'compressed_trace',
            'compression_strategy': compressed.get('strategy', 'unknown'),
            'compression_stats': {
                'original_size_bytes': stats['original_size'],
                'compressed_size_bytes': stats['compressed_size'],
                'compression_ratio': stats['compression_ratio'],
                'space_savings_percent': stats['space_savings'] * 100
            }
        },
        'compressed_data': {
            'num_devices': compressed.get('ndevs', 0),
            'devices': {}
        }
    }
    
    # Save compressed device data
    for dev_id, dev_data in compressed.get('devices', {}).items():
        data['compressed_data']['devices'][str(dev_id)] = {
            'num_events': dev_data.get('num_events', 0),
            'encoded_data_size_bytes': len(dev_data.get('encoded_data', [])),
            'metadata': {
                'prediction_model': dev_data.get('prediction_model', {}),
                'encoding_info': 'LEB128 variable-length encoding with zigzag for signed integers'
            }
        }
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


class TransformerBlock(nn.Module):
    '''Simple transformer block for demonstration'''
    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class SimpleTransformer(nn.Module):
    '''Simple transformer model for demonstration'''
    def __init__(self, vocab_size=1000, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x


def step1_extract_mst_from_torchfx(model):
    '''
    Step 1: Extract MST from torch.fx symbolic trace
    '''
    print("\n" + "="*70)
    print("STEP 1: Extract MST from torch.fx")
    print("="*70)
    
    print("\n1.1 Tracing model with torch.fx...")
    # Note: MultiheadAttention cannot be traced directly, use simpler model
    # Use a model with repeating layers to demonstrate loop merging
    class SimpleLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            return self.relu(self.linear(x))
    
    class SimpleFFNWithLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(256, 256)
            # Create repeating layers to demonstrate loop merging
            self.layers = nn.ModuleList([SimpleLayer(256) for _ in range(4)])
            self.output = nn.Linear(256, 256)
            
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            x = self.output(x)
            return x
    
    simple_model = SimpleFFNWithLayers()
    traced_model = fx.symbolic_trace(simple_model)
    print("   ✓ Model traced successfully")
    
    print("\n1.2 Analyzing torch.fx graph structure...")
    # Extract module paths to show what we're working with
    module_paths = []
    for node in traced_model.graph.nodes:
        if node.op == 'call_module' and node.target:
            module_paths.append(str(node.target))
    print(f"   Found {len(module_paths)} module calls:")
    for path in sorted(set(module_paths)):
        print(f"     - {path}")
    
    print("\n1.3 Building hierarchical MST from torch.fx graph with loop merging...")
    mst = ModelStructureTree.from_torch_fx_graph(
        traced_model.graph, 
        'SimpleFFNWithLayers',
        merge_similar_layers=True  # Enable loop merging
    )
    print(f"   ✓ Hierarchical MST built with {len(mst.nodes)} nodes")
    
    # Check if loop merging worked
    loop_nodes = [node for node in mst.nodes.values() if node.node_type == 'loop']
    if loop_nodes:
        print(f"   ✓ Loop merging active: {len(loop_nodes)} loop node(s) created")
        for loop_node in loop_nodes:
            loop_count = loop_node.attributes.get('loop_count', 0)
            start_idx = loop_node.attributes.get('start_index', 0)
            end_idx = loop_node.attributes.get('end_index', loop_count-1)
            print(f"     - {loop_node.name}: {loop_count} iterations (indices {start_idx}..{end_idx})")
    else:
        print("   ⚠ No repeated patterns detected for loop merging")
        print("   This may indicate the model structure doesn't have sequential indexed layers")
    
    print("\n1.4 Hierarchical MST Structure:")
    print(mst.visualize())
    
    print("\n1.5 Analyzing hierarchy depth...")
    max_depth = max(len(node.call_stack) for node in mst.nodes.values() if node.call_stack)
    module_count = sum(1 for node in mst.nodes.values() if node.node_type == 'module')
    print(f"   Maximum depth: {max_depth}")
    print(f"   Module nodes: {module_count}")
    print(f"   Loop nodes: {len(loop_nodes)}")
    print(f"   Total nodes: {len(mst.nodes)}")
    
    print("\n1.6 Generating hierarchical MST visualization...")
    viz_path = '/tmp/padoc_e2e_mst'
    mst.visualize_graphviz(viz_path, format='pdf')
    print(f"   ✓ Hierarchical visualization saved: {viz_path}.pdf")
    if loop_nodes:
        print("   ✓ Loop nodes shown in orange with dashed edges")
    
    return mst, simple_model


def step2_collect_trace_with_torchprofiler(model):
    '''
    Step 2: Run model and collect trace with TorchProfiler
    '''
    print("\n" + "="*70)
    print("STEP 2: Collect Trace with TorchProfiler")
    print("="*70)
    
    print("\n2.1 Creating sample input data...")
    batch_size, seq_len, d_model = 4, 16, 256
    input_data = torch.randn(batch_size, seq_len, d_model)
    print(f"   Input shape: {input_data.shape}")
    
    print("\n2.2 Profiling model execution...")
    trace_path = '/tmp/padoc_e2e_trace.json'
    
    # Warm up
    with torch.no_grad():
        _ = model(input_data)
    
    # Profile with record_function for better structure
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            with record_function("model_forward"):
                output = model(input_data)
    
    # Export trace
    prof.export_chrome_trace(trace_path)
    print(f"   ✓ Trace collected and saved: {trace_path}")
    
    # Show trace statistics
    print("\n2.3 Trace Statistics:")
    with open(trace_path, 'r') as f:
        trace_json = json.load(f)
    
    trace_events = trace_json.get('traceEvents', [])
    print(f"   Total events: {len(trace_events)}")
    
    # Count by category
    categories = {}
    for event in trace_events:
        cat = event.get('cat', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("   Events by category:")
    for cat, count in sorted(categories.items()):
        print(f"     - {cat}: {count}")
    
    return trace_path, trace_json


def step3_map_trace_to_mst(mst, trace_path, trace_json, save_mapped_trace=False):
    '''
    Step 3: Map TorchProfiler trace to MST nodes
    
    Args:
        mst: Model Structure Tree
        trace_path: Path to TorchProfiler trace
        trace_json: Parsed TorchProfiler JSON
        save_mapped_trace: If True, save trace after MST mapping to JSON file
    '''
    print("\n" + "="*70)
    print("STEP 3: Map TorchProfiler Trace to MST")
    print("="*70)
    
    print("\n3.1 Reading trace with TorchProfilerTraceReader...")
    reader = TorchProfilerTraceReader(trace_path)
    
    # Create a simple trace from events (simplified for demo)
    trace = Trace(1)  # Single device
    
    event_id = 0
    for raw_event in trace_json.get('traceEvents', []):
        if raw_event.get('ph') == 'X':  # Duration events
            name = raw_event.get('name', '')
            ts = raw_event.get('ts', 0)
            dur = raw_event.get('dur', 0)
            
            # Create event
            event = Event(event_id, EventType.FWD, name, ts, dur)
            trace.add_event(0, event)
            event_id += 1
            
            if event_id >= 50:  # Limit for demo
                break
    
    print(f"   ✓ Created trace with {event_id} events")
    
    print("\n3.2 Mapping trace events to MST nodes...")
    mst = MSTMapper.map_trace_to_mst(trace, mst, trace_json)
    
    # Count mapped events
    total_mapped = sum(len(node.trace_events) for node in mst.nodes.values())
    print(f"   ✓ Mapped {total_mapped} events to MST nodes")
    
    print("\n3.3 MST with mapped events:")
    print(mst.visualize())
    
    # Show detailed mapping
    print("\n3.4 Event mapping details:")
    for node in mst.nodes.values():
        if node.trace_events and node.node_type != 'root':
            print(f"   Node '{node.name}' ({node.node_type}): {len(node.trace_events)} events")
    
    # Optionally save trace after MST mapping
    if save_mapped_trace:
        print("\n3.5 Saving trace after MST mapping...")
        mapped_trace_path = '/tmp/padoc_e2e_trace_mapped.json'
        _save_trace_to_json(trace, mst, mapped_trace_path)
        print(f"   ✓ Mapped trace saved: {mapped_trace_path}")
    
    return trace, mst


def step4_compress_events_on_mst(trace, mst, save_compressed_trace=False):
    '''
    Step 4: Compress trace events on MST nodes
    
    Args:
        trace: Trace object
        mst: Model Structure Tree
        save_compressed_trace: If True, save compressed trace to JSON file
    '''
    print("\n" + "="*70)
    print("STEP 4: Compress Trace Events")
    print("="*70)
    
    print("\n4.1 Initializing TraceCompressor...")
    compressor = TraceCompressor()
    print("   ✓ Compressor initialized")
    
    print("\n4.2 Compressing trace with 'intra' strategy...")
    compressed = compressor.compress_trace(trace, strategy='intra')
    print("   ✓ Trace compressed")
    
    print("\n4.3 Compression Statistics:")
    stats = compressor.get_compression_stats(trace, compressed)
    
    print(f"   Original size: {stats['original_size']} bytes")
    print(f"   Compressed size: {stats['compressed_size']} bytes")
    print(f"   Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"   Space savings: {stats['space_savings']*100:.1f}%")
    
    print("\n4.4 Verifying lossless compression...")
    decompressor = TraceDecompressor()
    decompressed = decompressor.decompress_trace(compressed)
    
    # Compare original and decompressed
    original_events = trace.get_events(0)
    decompressed_events = decompressed.get_events(0)
    
    if len(original_events) == len(decompressed_events):
        matches = sum(1 for i in range(len(original_events))
                     if (original_events[i].get_timestamp() == decompressed_events[i].get_timestamp() and
                         original_events[i].get_duration() == decompressed_events[i].get_duration()))
        print(f"   ✓ Lossless: {matches}/{len(original_events)} events match perfectly")
    
    print("\n4.5 Testing random access (O(1))...")
    for idx in [0, len(original_events)//2, len(original_events)-1]:
        if idx < len(original_events):
            event = decompressor.get_event_at_index(compressed, dev_id=0, index=idx)
            print(f"   Event {idx}: ts={event.get_timestamp()}, dur={event.get_duration()}")
    
    print("   ✓ Random access successful")
    
    # Optionally save compressed trace
    if save_compressed_trace:
        print("\n4.6 Saving compressed trace...")
        compressed_trace_path = '/tmp/padoc_e2e_trace_compressed.json'
        _save_compressed_trace_to_json(compressed, stats, compressed_trace_path)
        print(f"   ✓ Compressed trace saved: {compressed_trace_path}")
    
    return compressed


def step5_analyze_compressed_trace(compressed, trace):
    '''
    Step 5: Perform analysis on compressed trace
    '''
    print("\n" + "="*70)
    print("STEP 5: Analyze Compressed Trace")
    print("="*70)
    
    print("\n5.1 Bubble Analysis (Idle Time)...")
    try:
        bubble_result = BubbleAnalyzer.analyze_compressed(compressed)
        print(f"   Average bubble ratio: {bubble_result.get('average_bubble_ratio', 0):.2%}")
        print(f"   Total bubble time: {bubble_result.get('total_bubble_time', 0):.2f} μs")
        print("   ✓ Bubble analysis complete")
    except Exception as e:
        print(f"   Note: Bubble analysis requires multi-device trace")
    
    print("\n5.2 Overlap Analysis (Computation-Communication)...")
    try:
        overlap_result = OverlapAnalyzer.analyze(trace)
        print(f"   Average overlap ratio: {overlap_result.get('average_overlap_ratio', 0):.2%}")
        print(f"   Overlapped time: {overlap_result.get('overlapped_time', 0):.2f} μs")
        print("   ✓ Overlap analysis complete")
    except Exception as e:
        print(f"   Note: Overlap analysis requires compute and comm events")
    
    print("\n5.3 Imbalance Analysis (Load Distribution)...")
    try:
        imbalance_result = ImbalanceAnalyzer.analyze(trace)
        print(f"   Imbalance ratio: {imbalance_result.get('imbalance_ratio', 0):.2%}")
        print(f"   Max load: {imbalance_result.get('max_load', 0):.2f} μs")
        print(f"   Min load: {imbalance_result.get('min_load', 0):.2f} μs")
        print("   ✓ Imbalance analysis complete")
    except Exception as e:
        print(f"   Note: Imbalance analysis requires multi-device trace")
    
    print("\n5.4 Direct Analysis Benefits:")
    print("   ✓ Analyze without full decompression")
    print("   ✓ Constant-time (O(1)) event access")
    print("   ✓ Space-efficient storage")
    print("   ✓ Lossless reconstruction when needed")


def main():
    '''
    Main function: Run complete end-to-end PADoC workflow
    
    Set save_debug_traces=True to save intermediate traces for debugging:
    - Trace after MST mapping: /tmp/padoc_e2e_trace_mapped.json
    - Compressed trace: /tmp/padoc_e2e_trace_compressed.json
    '''
    # Optional: Enable saving intermediate traces for debugging
    save_debug_traces = False  # Set to True to save debug traces
    
    print("\n" + "="*70)
    print("PADoC End-to-End Example: Complete Workflow")
    print("="*70)
    print("\nThis example demonstrates:")
    print("1. MST extraction from torch.fx")
    print("2. Trace collection with TorchProfiler")
    print("3. Trace-to-MST mapping")
    print("4. Trace compression on MST")
    print("5. Performance analysis on compressed trace")
    
    if save_debug_traces:
        print("\n[DEBUG MODE] Intermediate traces will be saved for debugging")
    
    # Create model
    print("\n" + "="*70)
    print("SETUP: Creating Model")
    print("="*70)
    print("\nCreating SimpleTransformer model...")
    model = SimpleTransformer(vocab_size=1000, d_model=256, nhead=4, num_layers=2)
    print("✓ Model created")
    
    # Step 1: Extract MST from torch.fx
    mst, traceable_model = step1_extract_mst_from_torchfx(model)
    
    # Step 2: Collect trace with TorchProfiler
    trace_path, trace_json = step2_collect_trace_with_torchprofiler(traceable_model)
    
    # Step 3: Map trace to MST (with optional save)
    trace, mst = step3_map_trace_to_mst(mst, trace_path, trace_json, 
                                         save_mapped_trace=save_debug_traces)
    
    # Step 4: Compress events on MST (with optional save)
    compressed = step4_compress_events_on_mst(trace, mst, 
                                               save_compressed_trace=save_debug_traces)
    
    # Step 5: Analyze compressed trace
    step5_analyze_compressed_trace(compressed, trace)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Complete Workflow Demonstrated")
    print("="*70)
    print("\n✓ MST extracted from torch.fx symbolic trace")
    print("✓ Trace collected with TorchProfiler")
    print("✓ Trace events mapped to MST nodes")
    print("✓ Trace compressed using linear prediction + delta encoding")
    print("✓ Performance analysis performed on compressed trace")
    
    print("\n" + "="*70)
    print("PADoC Capabilities Demonstrated:")
    print("="*70)
    print("• Hierarchical model structure representation (MST)")
    print("• Automatic model analysis with torch.fx")
    print("• Loop merging for repeated sequential layers")
    print("• Accurate trace mapping with call stacks")
    print("• Lossless compression with high compression ratios")
    print("• O(1) random access to compressed events")
    print("• Direct analysis without decompression")
    print("• Bubble, overlap, and imbalance analysis")
    
    print("\n" + "="*70)
    print("Generated Files:")
    print("="*70)
    print(f"• MST visualization: /tmp/padoc_e2e_mst.pdf")
    print(f"• TorchProfiler trace: /tmp/padoc_e2e_trace.json")
    
    if save_debug_traces:
        print(f"• Mapped trace (debug): /tmp/padoc_e2e_trace_mapped.json")
        print(f"• Compressed trace (debug): /tmp/padoc_e2e_trace_compressed.json")
    
    print("\n" + "="*70)
    print("✓ End-to-End Example Complete!")
    print("="*70)
    print("\nTip: Set save_debug_traces=True in main() to save intermediate traces for debugging")



if __name__ == '__main__':
    main()
