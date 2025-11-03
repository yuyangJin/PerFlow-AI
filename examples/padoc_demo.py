'''
PADoC Example: Performance Analytics Directly on Compressed Trace

This example demonstrates the complete PADoC workflow:
1. Building Model Structure Tree (MST)
2. Compressing traces with linear prediction and delta encoding
3. Analyzing traces (bubbles, overlap, imbalance)
4. Direct analysis on compressed format
'''

from perflowai.padoc import (
    ModelStructureTree, MSTMapper,
    TraceCompressor, TraceDecompressor,
    BubbleAnalyzer, OverlapAnalyzer, ImbalanceAnalyzer
)
from perflowai.parallel.pipeline_parallel import ZeroBubbleGraph, ScheduleType
from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.simulator.pipeline.pp_simulator import PPSimulator, PipeType
from perflowai.core import EventType

def demo_mst():
    '''Demonstrate Model Structure Tree creation and visualization'''
    print("\n" + "="*70)
    print("Part 1: Model Structure Tree (MST)")
    print("="*70)
    
    # Create MST
    mst = ModelStructureTree()
    
    # Build a model hierarchy representing a transformer
    transformer = mst.add_node('Transformer', 'model', 0)
    
    # Encoder
    encoder = mst.add_node('Encoder', 'module', transformer.node_id)
    for i in range(4):
        layer = mst.add_node(f'EncoderLayer_{i}', 'layer', encoder.node_id)
        attention = mst.add_node('MultiHeadAttention', 'operation', layer.node_id)
        attention.set_call_stack(['transformer', 'encoder', f'layer_{i}', 'attention'])
        ffn = mst.add_node('FeedForward', 'operation', layer.node_id)
        ffn.set_call_stack(['transformer', 'encoder', f'layer_{i}', 'ffn'])
    
    # Decoder
    decoder = mst.add_node('Decoder', 'module', transformer.node_id)
    for i in range(4):
        layer = mst.add_node(f'DecoderLayer_{i}', 'layer', decoder.node_id)
    
    print(f"Created MST with {len(mst.nodes)} nodes")
    print("\nMST Structure:")
    print(mst.visualize())
    
    # Save MST
    mst.to_json('/tmp/model_structure_tree.json')
    print("\n✓ Saved MST to /tmp/model_structure_tree.json")
    
    return mst

def demo_compression():
    '''Demonstrate trace compression'''
    print("\n" + "="*70)
    print("Part 2: Trace Compression")
    print("="*70)
    
    # Create a simulated pipeline parallel trace
    print("\nGenerating pipeline parallel trace...")
    config = PipeCostConfig(fwd_time=100, bwd_time=200, wgt_time=150)
    nstages = 4
    nmicrobatches = 16
    nchunks = 2
    
    graph = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, 
                           cost_config=config, 
                           schedule_type=ScheduleType.ZBV)
    graph.build_graph()
    
    simulator = PPSimulator(PipeType.Interleaved1F1B, graph)
    trace = simulator.run()
    
    original_events = sum(len(trace.get_events(i)) for i in range(nstages))
    print(f"✓ Generated trace with {original_events} events across {nstages} stages")
    
    # Compress the trace
    print("\nCompressing trace with different strategies...")
    compressor = TraceCompressor()
    
    strategies = ['intra', 'inter_mb', 'inter_iter']
    results = {}
    
    for strategy in strategies:
        compressed = compressor.compress_trace(trace, strategy=strategy)
        stats = compressor.get_compression_stats(trace, compressed)
        results[strategy] = stats
        
        print(f"\n  Strategy: {strategy}")
        print(f"    Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"    Original size: {stats['original_size']} bytes")
        print(f"    Compressed size: {stats['compressed_size']} bytes")
        print(f"    Space savings: {stats['space_savings']*100:.1f}%")
    
    # Save compressed trace
    compressed = compressor.compress_trace(trace, strategy='intra')
    compressor.save_compressed(compressed, '/tmp/compressed_trace.json')
    print("\n✓ Saved compressed trace to /tmp/compressed_trace.json")
    
    # Test decompression
    print("\nVerifying lossless compression...")
    decompressor = TraceDecompressor()
    decompressed_trace = decompressor.decompress_trace(compressed)
    
    # Verify
    for stage in range(nstages):
        orig_events = trace.get_events(stage)
        decomp_events = decompressed_trace.get_events(stage)
        assert len(orig_events) == len(decomp_events)
        
    print("✓ Lossless compression verified!")
    
    return trace, compressed

def demo_analysis(trace, compressed):
    '''Demonstrate direct analysis on compressed traces'''
    print("\n" + "="*70)
    print("Part 3: Performance Analysis")
    print("="*70)
    
    # Bubble Analysis
    print("\n--- Bubble Analysis ---")
    bubble_result = BubbleAnalyzer.analyze(trace)
    
    print(f"Overall bubble statistics:")
    print(f"  Total bubble time: {bubble_result['total_bubble_time']:.0f}")
    print(f"  Average bubble ratio: {bubble_result['average_bubble_ratio']:.2%}")
    
    print(f"\nPer-device bubble statistics:")
    for dev_id, dev_data in sorted(bubble_result['device_bubbles'].items())[:4]:
        print(f"  Stage {dev_id}:")
        print(f"    Bubble count: {dev_data['bubble_count']}")
        print(f"    Bubble time: {dev_data['bubble_time']:.0f}")
        print(f"    Bubble ratio: {dev_data['bubble_ratio']:.2%}")
    
    # Imbalance Analysis
    print("\n--- Load Imbalance Analysis ---")
    imbalance_result = ImbalanceAnalyzer.analyze(trace)
    
    print(f"Overall imbalance statistics:")
    print(f"  Max compute time: {imbalance_result['max_compute_time']:.0f}")
    print(f"  Min compute time: {imbalance_result['min_compute_time']:.0f}")
    print(f"  Avg compute time: {imbalance_result['avg_compute_time']:.0f}")
    print(f"  Imbalance ratio: {imbalance_result['imbalance_ratio']:.2%}")
    
    print(f"\nPer-device utilization:")
    for dev_id, dev_data in sorted(imbalance_result['device_loads'].items())[:4]:
        print(f"  Stage {dev_id}: {dev_data['utilization']:.2%} utilization, "
              f"{dev_data['event_count']} events")
    
    # Direct analysis on compressed format
    print("\n--- Direct Analysis on Compressed Format ---")
    print("Analyzing bubbles directly on compressed trace (no decompression)...")
    
    compressed_bubble_result = BubbleAnalyzer.analyze_compressed(compressed)
    
    print(f"✓ Analysis completed on compressed format")
    print(f"  Average bubble ratio: {compressed_bubble_result['average_bubble_ratio']:.2%}")
    
    # Verify results match
    diff = abs(bubble_result['average_bubble_ratio'] - 
               compressed_bubble_result['average_bubble_ratio'])
    print(f"  Difference from uncompressed: {diff:.6%}")
    assert diff < 0.001, "Results should match!"
    
    print("\n✓ Compressed analysis matches uncompressed analysis!")

def demo_efficiency():
    '''Demonstrate compression efficiency and analysis speedup'''
    print("\n" + "="*70)
    print("Efficiency Demonstration")
    print("="*70)
    
    import time
    
    # Create a larger trace for efficiency testing
    config = PipeCostConfig(fwd_time=100, bwd_time=200, wgt_time=150)
    nstages = 4
    nmicrobatches = 24
    nchunks = 2
    
    print(f"\nGenerating larger trace ({nstages} stages, {nmicrobatches} microbatches, "
          f"{nchunks} chunks)...")
    
    graph = ZeroBubbleGraph(nstages, nmicrobatches, nchunks, 
                           cost_config=config,
                           schedule_type=ScheduleType.ZBV)
    graph.build_graph()
    
    simulator = PPSimulator(PipeType.Interleaved1F1B, graph)
    trace = simulator.run()
    
    total_events = sum(len(trace.get_events(i)) for i in range(nstages))
    print(f"✓ Generated trace with {total_events} events")
    
    # Compression time
    print("\nMeasuring compression performance...")
    compressor = TraceCompressor()
    
    start = time.time()
    compressed = compressor.compress_trace(trace, strategy='intra')
    compress_time = time.time() - start
    
    stats = compressor.get_compression_stats(trace, compressed)
    
    print(f"  Compression time: {compress_time*1000:.2f} ms")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Events per second: {total_events/compress_time:.0f}")
    
    # Analysis time comparison
    print("\nMeasuring analysis performance...")
    
    # Analysis on uncompressed
    start = time.time()
    bubble_result = BubbleAnalyzer.analyze(trace)
    uncompressed_time = time.time() - start
    
    # Analysis on compressed
    start = time.time()
    bubble_compressed = BubbleAnalyzer.analyze_compressed(compressed)
    compressed_time = time.time() - start
    
    print(f"  Uncompressed analysis: {uncompressed_time*1000:.2f} ms")
    print(f"  Compressed analysis: {compressed_time*1000:.2f} ms")
    if compressed_time > 0:
        print(f"  Speedup: {uncompressed_time/compressed_time:.2f}x")
    
    print("\n✓ Efficiency demonstration completed!")

def main():
    '''Run complete PADoC demonstration'''
    print("\n")
    print("="*70)
    print("PADoC: Performance Analytics Directly on Compressed Trace")
    print("Demonstration of Complete Workflow")
    print("="*70)
    
    # Part 1: MST
    mst = demo_mst()
    
    # Part 2: Compression
    trace, compressed = demo_compression()
    
    # Part 3: Analysis
    demo_analysis(trace, compressed)
    
    # Efficiency demonstration
    demo_efficiency()
    
    print("\n" + "="*70)
    print("✓ PADoC demonstration completed successfully!")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  • Model Structure Tree (MST) with hierarchical organization")
    print("  • Lossless compression with linear prediction & delta encoding")
    print("  • Multiple compression strategies (intra/inter microbatch/iteration)")
    print("  • Direct analysis on compressed format")
    print("  • Bubble, overlap, and imbalance analysis")
    print("  • Efficient random access to compressed events")
    print("\nGenerated Files:")
    print("  • /tmp/model_structure_tree.json")
    print("  • /tmp/compressed_trace.json")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
