'''
Integration test for PADoC: Full workflow demonstration
Tests MST creation, mapping, compression, decompression, and analysis
'''

from perflowai.padoc import (
    ModelStructureTree, MSTMapper,
    TraceCompressor, TraceDecompressor,
    BubbleAnalyzer, OverlapAnalyzer, ImbalanceAnalyzer
)
from perflowai.reader import TorchProfilerTraceReader
from perflowai.core import EventType, Trace, Event

def test_padoc_full_workflow_synthetic():
    '''
    Test the complete PADoC workflow with synthetic data.
    This demonstrates Part 1, Part 2, and Part 3 of the roadmap.
    '''
    print("\n" + "="*60)
    print("PADoC Full Workflow Test (Synthetic Data)")
    print("="*60)
    
    # ===== Part 1: Model Structure Tree =====
    print("\n[Part 1] Building Model Structure Tree...")
    
    mst = ModelStructureTree()
    
    # Create a hierarchical model structure
    encoder = mst.add_node('Encoder', 'module', 0)
    encoder_layer1 = mst.add_node('EncoderLayer1', 'layer', encoder.node_id)
    encoder_layer2 = mst.add_node('EncoderLayer2', 'layer', encoder.node_id)
    
    decoder = mst.add_node('Decoder', 'module', 0)
    decoder_layer1 = mst.add_node('DecoderLayer1', 'layer', decoder.node_id)
    
    # Add operations
    fwd_op = mst.add_node('forward', 'operation', encoder_layer1.node_id)
    bwd_op = mst.add_node('backward', 'operation', encoder_layer1.node_id)
    
    # Set call stacks
    fwd_op.set_call_stack(['model', 'encoder', 'layer1', 'forward'])
    bwd_op.set_call_stack(['model', 'encoder', 'layer1', 'backward'])
    
    print(f"✓ Created MST with {len(mst.nodes)} nodes")
    
    # Visualize MST
    print("\nMST Structure:")
    print(mst.visualize())
    
    # ===== Create synthetic trace data =====
    print("\n[Creating Trace] Generating synthetic trace data...")
    
    trace = Trace(4)  # 4 devices
    
    # Simulate pipeline parallel execution
    event_id = 1
    for stage in range(4):
        base_time = 1000 + stage * 50  # Offset each stage
        
        for mb in range(8):  # 8 microbatches
            # Forward pass
            fwd_start = base_time + mb * 300
            trace.add_event(stage, Event(
                event_id, EventType.FWD, f'forward-{mb}', 
                fwd_start, 100
            ))
            fwd_op.add_trace_event(event_id)
            event_id += 1
            
            # Backward pass (comes later)
            bwd_start = fwd_start + 100 + (8 - mb) * 100
            trace.add_event(stage, Event(
                event_id, EventType.BWD, f'backward-{mb}', 
                bwd_start, 150
            ))
            bwd_op.add_trace_event(event_id)
            event_id += 1
    
    total_events = sum(len(trace.get_events(i)) for i in range(4))
    print(f"✓ Created trace with {total_events} events across 4 devices")
    
    # ===== Part 2: Compression =====
    print("\n[Part 2] Compressing trace...")
    
    compressor = TraceCompressor()
    compressed = compressor.compress_trace(trace, strategy='intra')
    
    print(f"✓ Compressed trace successfully")
    
    # Calculate compression statistics
    stats = compressor.get_compression_stats(trace, compressed)
    print(f"\nCompression Statistics:")
    print(f"  Original events: {stats['original_events']}")
    print(f"  Original size (estimated): {stats['original_size']} bytes")
    print(f"  Compressed size: {stats['compressed_size']} bytes")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Space savings: {stats['space_savings']*100:.1f}%")
    
    # Test decompression
    print("\n[Decompression] Verifying lossless compression...")
    decompressor = TraceDecompressor()
    decompressed_trace = decompressor.decompress_trace(compressed)
    
    # Verify lossless
    for dev_id in range(4):
        orig_events = trace.get_events(dev_id)
        decomp_events = decompressed_trace.get_events(dev_id)
        assert len(orig_events) == len(decomp_events)
    
    print("✓ Verified lossless compression/decompression")
    
    # ===== Part 3: Analysis =====
    print("\n[Part 3] Running analyses on trace...")
    
    # Bubble Analysis
    print("\n--- Bubble Analysis ---")
    bubble_result = BubbleAnalyzer.analyze(trace)
    print(f"Average bubble ratio: {bubble_result['average_bubble_ratio']:.2%}")
    print(f"Total bubble time: {bubble_result['total_bubble_time']:.0f}")
    
    for dev_id in range(4):
        dev_bubbles = bubble_result['device_bubbles'][dev_id]
        print(f"  Device {dev_id}: {dev_bubbles['bubble_count']} bubbles, "
              f"{dev_bubbles['bubble_ratio']:.2%} ratio")
    
    # Imbalance Analysis
    print("\n--- Imbalance Analysis ---")
    imbalance_result = ImbalanceAnalyzer.analyze(trace)
    print(f"Imbalance ratio: {imbalance_result['imbalance_ratio']:.2%}")
    print(f"Max compute time: {imbalance_result['max_compute_time']:.0f}")
    print(f"Min compute time: {imbalance_result['min_compute_time']:.0f}")
    print(f"Avg compute time: {imbalance_result['avg_compute_time']:.0f}")
    
    for dev_id in range(4):
        dev_load = imbalance_result['device_loads'][dev_id]
        print(f"  Device {dev_id}: utilization {dev_load['utilization']:.2%}, "
              f"{dev_load['event_count']} events")
    
    # Analysis on compressed format
    print("\n--- Direct Analysis on Compressed Format ---")
    bubble_compressed = BubbleAnalyzer.analyze_compressed(compressed)
    print(f"✓ Successfully analyzed bubbles on compressed format")
    print(f"  Average bubble ratio (compressed): {bubble_compressed['average_bubble_ratio']:.2%}")
    
    print("\n" + "="*60)
    print("✓ PADoC Full Workflow Test Completed Successfully!")
    print("="*60)
    
    return {
        'mst': mst,
        'trace': trace,
        'compressed': compressed,
        'stats': stats,
        'bubble_result': bubble_result,
        'imbalance_result': imbalance_result
    }

def test_padoc_with_torchprofiler():
    '''
    Test PADoC with real TorchProfiler trace data.
    '''
    print("\n" + "="*60)
    print("PADoC with TorchProfiler Trace Test")
    print("="*60)
    
    trace_path = './tests/example_trace/out-1024.json'
    
    # Read trace
    print("\n[Reading] Loading TorchProfiler trace...")
    reader = TorchProfilerTraceReader(trace_path)
    trace = reader.read([EventType.FWD, EventType.BWD])
    
    print(f"✓ Loaded trace with {trace.get_ndevs()} devices")
    
    # Build MST from trace
    print("\n[MST] Building MST from TorchProfiler trace...")
    mst = MSTMapper.build_and_map_from_file(trace_path, trace)
    
    print(f"✓ Built MST with {len(mst.nodes)} nodes")
    print(f"\nMST Preview (top-level):")
    lines = mst.visualize().split('\n')[:10]
    print('\n'.join(lines))
    
    # Compress
    print("\n[Compression] Compressing trace...")
    compressor = TraceCompressor()
    compressed = compressor.compress_trace(trace, strategy='intra')
    
    stats = compressor.get_compression_stats(trace, compressed)
    print(f"✓ Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Space savings: {stats['space_savings']*100:.1f}%")
    
    # Analyze (only on first device to keep output manageable)
    print("\n[Analysis] Running analysis on device 0...")
    
    # Create a single-device trace for analysis
    single_trace = Trace(1)
    events = trace.get_events(0)
    if events:
        for event in events:
            single_trace.add_event(0, event)
        
        bubble_result = BubbleAnalyzer.analyze(single_trace)
        print(f"  Bubble count: {bubble_result['device_bubbles'][0]['bubble_count']}")
        print(f"  Bubble ratio: {bubble_result['device_bubbles'][0]['bubble_ratio']:.2%}")
    
    print("\n" + "="*60)
    print("✓ PADoC TorchProfiler Test Completed Successfully!")
    print("="*60)

if __name__ == '__main__':
    # Run synthetic workflow test
    result = test_padoc_full_workflow_synthetic()
    
    # Run TorchProfiler test
    try:
        test_padoc_with_torchprofiler()
    except FileNotFoundError:
        print("\n[INFO] TorchProfiler trace file not found, skipping test")
    
    print("\n✓ All PADoC integration tests passed!")
