'''
Test Trace Compression and Decompression
'''

from perflowai.padoc import TraceCompressor, TraceDecompressor
from perflowai.core import Trace, Event, EventType

def test_delta_encoding():
    '''Test variable-length delta encoding'''
    from perflowai.padoc.compressor import DeltaEncoder
    
    # Test small positive values
    encoded = DeltaEncoder.encode_delta(5)
    decoded, consumed = DeltaEncoder.decode_delta(encoded, 0)
    assert decoded == 5
    assert consumed == len(encoded)
    
    # Test small negative values
    encoded = DeltaEncoder.encode_delta(-5)
    decoded, consumed = DeltaEncoder.decode_delta(encoded, 0)
    assert decoded == -5
    
    # Test zero
    encoded = DeltaEncoder.encode_delta(0)
    decoded, consumed = DeltaEncoder.decode_delta(encoded, 0)
    assert decoded == 0
    
    # Test large values
    encoded = DeltaEncoder.encode_delta(1000000)
    decoded, consumed = DeltaEncoder.decode_delta(encoded, 0)
    assert decoded == 1000000
    
    print("✓ Delta encoding test passed")

def test_delta_list_encoding():
    '''Test encoding and decoding lists of deltas'''
    from perflowai.padoc.compressor import DeltaEncoder
    
    deltas = [0, 5, -3, 10, -100, 1000, 0, 1]
    encoded = DeltaEncoder.encode_delta_list(deltas)
    decoded = DeltaEncoder.decode_delta_list(encoded, len(deltas))
    
    assert decoded == deltas
    print(f"✓ Delta list encoding test passed (compressed {len(deltas)} deltas to {len(encoded)} bytes)")

def test_trace_compression():
    '''Test basic trace compression'''
    # Create a simple trace
    trace = Trace(2)
    
    # Add events to device 0
    trace.add_event(0, Event(1, EventType.FWD, 'forward_step-0', 1000, 100))
    trace.add_event(0, Event(2, EventType.FWD, 'forward_step-1', 1150, 100))
    trace.add_event(0, Event(3, EventType.BWD, 'backward_step-0', 1300, 200))
    
    # Add events to device 1
    trace.add_event(1, Event(4, EventType.FWD, 'forward_step-0', 1100, 100))
    trace.add_event(1, Event(5, EventType.FWD, 'forward_step-1', 1250, 100))
    
    # Compress
    compressor = TraceCompressor()
    compressed = compressor.compress_trace(trace, strategy='intra')
    
    assert 'metadata' in compressed
    assert compressed['metadata']['ndevs'] == 2
    assert compressed['metadata']['original_event_count'] == 5
    assert '0' in compressed['devices']
    assert '1' in compressed['devices']
    
    print("✓ Trace compression test passed")
    
    # Check compression stats
    stats = compressor.get_compression_stats(trace, compressed)
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Space savings: {stats['space_savings']*100:.1f}%")
    
    return trace, compressed

def test_trace_decompression():
    '''Test trace decompression'''
    # Create and compress a trace
    trace = Trace(1)
    trace.add_event(0, Event(1, EventType.FWD, 'forward_step-0', 1000, 100))
    trace.add_event(0, Event(2, EventType.FWD, 'forward_step-1', 1200, 150))
    trace.add_event(0, Event(3, EventType.BWD, 'backward_step-0', 1400, 200))
    
    compressor = TraceCompressor()
    compressed = compressor.compress_trace(trace, strategy='intra')
    
    # Decompress
    decompressor = TraceDecompressor()
    decompressed_trace = decompressor.decompress_trace(compressed)
    
    # Verify
    assert decompressed_trace.get_ndevs() == trace.get_ndevs()
    
    original_events = trace.get_events(0)
    decompressed_events = decompressed_trace.get_events(0)
    
    assert len(decompressed_events) == len(original_events)
    
    for orig, decomp in zip(original_events, decompressed_events):
        assert orig.get_id() == decomp.get_id()
        assert orig.get_name() == decomp.get_name()
        assert abs(orig.get_timestamp() - decomp.get_timestamp()) < 1  # Allow for rounding
        assert abs(orig.get_duration() - decomp.get_duration()) < 1
    
    print("✓ Trace decompression test passed")

def test_random_access():
    '''Test O(1) random access to compressed events'''
    trace = Trace(1)
    trace.add_event(0, Event(1, EventType.FWD, 'event-0', 100, 10))
    trace.add_event(0, Event(2, EventType.FWD, 'event-1', 120, 10))
    trace.add_event(0, Event(3, EventType.FWD, 'event-2', 140, 10))
    trace.add_event(0, Event(4, EventType.FWD, 'event-3', 160, 10))
    
    compressor = TraceCompressor()
    compressed = compressor.compress_trace(trace, strategy='intra')
    
    decompressor = TraceDecompressor()
    
    # Access specific event without full decompression
    event_2 = decompressor.get_event_at_index(compressed, 0, 2)
    
    assert event_2.get_id() == 3
    assert event_2.get_name() == 'event-2'
    
    print("✓ Random access test passed")

def test_compression_save_load():
    '''Test saving and loading compressed traces'''
    import tempfile
    import os
    
    trace = Trace(1)
    trace.add_event(0, Event(1, EventType.FWD, 'forward', 1000, 100))
    trace.add_event(0, Event(2, EventType.BWD, 'backward', 1200, 200))
    
    compressor = TraceCompressor()
    compressed = compressor.compress_trace(trace)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save
        compressor.save_compressed(compressed, temp_path)
        
        # Load
        decompressor = TraceDecompressor()
        loaded = decompressor.load_compressed(temp_path)
        
        # Decompress and verify
        decompressed = decompressor.decompress_trace(loaded)
        events = decompressed.get_events(0)
        
        assert len(events) == 2
        assert events[0].get_name() == 'forward'
        assert events[1].get_name() == 'backward'
        
        print("✓ Compression save/load test passed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    test_delta_encoding()
    test_delta_list_encoding()
    test_trace_compression()
    test_trace_decompression()
    test_random_access()
    test_compression_save_load()
    print("\n✓ All compression tests passed!")
