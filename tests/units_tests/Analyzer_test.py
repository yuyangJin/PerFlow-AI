'''
Test analyzers for bubble, overlap, and imbalance analysis
'''

from perflowai.padoc import BubbleAnalyzer, OverlapAnalyzer, ImbalanceAnalyzer
from perflowai.padoc import TraceCompressor
from perflowai.core import Trace, Event, EventType

def test_bubble_analysis():
    '''Test bubble analysis on a trace'''
    trace = Trace(2)
    
    # Device 0: events with bubbles
    trace.add_event(0, Event(1, EventType.FWD, 'fwd-0', 100, 50))
    trace.add_event(0, Event(2, EventType.FWD, 'fwd-1', 200, 50))  # 50 time unit bubble
    trace.add_event(0, Event(3, EventType.FWD, 'fwd-2', 300, 50))  # 50 time unit bubble
    
    # Device 1: events with no bubbles
    trace.add_event(1, Event(4, EventType.FWD, 'fwd-0', 100, 50))
    trace.add_event(1, Event(5, EventType.FWD, 'fwd-1', 150, 50))
    trace.add_event(1, Event(6, EventType.FWD, 'fwd-2', 200, 50))
    
    # Analyze
    result = BubbleAnalyzer.analyze(trace)
    
    assert 'device_bubbles' in result
    assert 0 in result['device_bubbles']
    assert 1 in result['device_bubbles']
    
    # Device 0 should have bubbles
    dev0_bubbles = result['device_bubbles'][0]
    assert dev0_bubbles['bubble_count'] == 2
    assert dev0_bubbles['bubble_time'] == 100  # 50 + 50
    
    # Device 1 should have no bubbles
    dev1_bubbles = result['device_bubbles'][1]
    assert dev1_bubbles['bubble_count'] == 0
    assert dev1_bubbles['bubble_time'] == 0
    
    print("✓ Bubble analysis test passed")
    print(f"  Device 0 bubble ratio: {dev0_bubbles['bubble_ratio']:.2%}")
    print(f"  Device 1 bubble ratio: {dev1_bubbles['bubble_ratio']:.2%}")

def test_bubble_analysis_compressed():
    '''Test bubble analysis on compressed trace'''
    trace = Trace(1)
    trace.add_event(0, Event(1, EventType.FWD, 'fwd-0', 100, 50))
    trace.add_event(0, Event(2, EventType.FWD, 'fwd-1', 200, 50))  # 50 unit bubble
    
    # Compress
    compressor = TraceCompressor()
    compressed = compressor.compress_trace(trace)
    
    # Analyze compressed
    result = BubbleAnalyzer.analyze_compressed(compressed)
    
    assert result['device_bubbles'][0]['bubble_count'] == 1
    assert result['device_bubbles'][0]['bubble_time'] == 50
    
    print("✓ Bubble analysis on compressed trace test passed")

def test_overlap_analysis():
    '''Test computation-communication overlap analysis'''
    trace = Trace(1)
    
    # Computation events
    trace.add_event(0, Event(1, EventType.FWD, 'fwd-0', 100, 100))
    trace.add_event(0, Event(2, EventType.BWD, 'bwd-0', 200, 100))
    
    # Communication events that overlap with computation
    trace.add_event(0, Event(3, EventType.COMM, 'comm-0', 150, 100))  # Overlaps with fwd
    trace.add_event(0, Event(4, EventType.COMM, 'comm-1', 250, 50))   # Overlaps with bwd
    
    result = OverlapAnalyzer.analyze(trace)
    
    assert 'device_overlaps' in result
    assert 0 in result['device_overlaps']
    
    dev0_overlap = result['device_overlaps'][0]
    assert dev0_overlap['comp_time'] == 200  # 100 + 100
    assert dev0_overlap['comm_time'] == 150  # 100 + 50
    
    # Overlap: fwd overlaps 50 units with comm-0, bwd overlaps 50 units with comm-1
    # But we count overlap for each comm event, so it's actually:
    # comm-0: overlaps 50 with fwd, comm-1: overlaps 50 with bwd, plus comm-0 also overlaps 50 with bwd
    # Total: 50 (fwd+comm-0) + 50 (bwd+comm-0) + 50 (bwd+comm-1) = 150
    assert dev0_overlap['overlap_time'] == 150
    
    print("✓ Overlap analysis test passed")
    print(f"  Overlap ratio: {dev0_overlap['overlap_ratio']:.2%}")

def test_imbalance_analysis():
    '''Test load imbalance analysis'''
    trace = Trace(3)
    
    # Device 0: 100 units of work
    trace.add_event(0, Event(1, EventType.FWD, 'fwd-0', 100, 50))
    trace.add_event(0, Event(2, EventType.BWD, 'bwd-0', 150, 50))
    
    # Device 1: 200 units of work (imbalanced - more work)
    trace.add_event(1, Event(3, EventType.FWD, 'fwd-0', 100, 100))
    trace.add_event(1, Event(4, EventType.BWD, 'bwd-0', 200, 100))
    
    # Device 2: 150 units of work
    trace.add_event(2, Event(5, EventType.FWD, 'fwd-0', 100, 75))
    trace.add_event(2, Event(6, EventType.BWD, 'bwd-0', 175, 75))
    
    result = ImbalanceAnalyzer.analyze(trace)
    
    assert 'device_loads' in result
    assert result['max_compute_time'] == 200
    assert result['min_compute_time'] == 100
    assert result['avg_compute_time'] == 150
    
    # Imbalance ratio: (max - avg) / avg = (200 - 150) / 150 = 0.333...
    assert abs(result['imbalance_ratio'] - 0.333) < 0.01
    
    print("✓ Imbalance analysis test passed")
    print(f"  Max compute time: {result['max_compute_time']}")
    print(f"  Min compute time: {result['min_compute_time']}")
    print(f"  Avg compute time: {result['avg_compute_time']}")
    print(f"  Imbalance ratio: {result['imbalance_ratio']:.2%}")

def test_imbalance_by_utilization():
    '''Test imbalance analysis including utilization'''
    trace = Trace(2)
    
    # Device 0: High utilization (no bubbles)
    trace.add_event(0, Event(1, EventType.FWD, 'fwd-0', 100, 50))
    trace.add_event(0, Event(2, EventType.FWD, 'fwd-1', 150, 50))
    
    # Device 1: Low utilization (with bubbles)
    trace.add_event(1, Event(3, EventType.FWD, 'fwd-0', 100, 50))
    trace.add_event(1, Event(4, EventType.FWD, 'fwd-1', 200, 50))  # 50 unit bubble
    
    result = ImbalanceAnalyzer.analyze(trace)
    
    dev0_util = result['device_loads'][0]['utilization']
    dev1_util = result['device_loads'][1]['utilization']
    
    assert dev0_util == 1.0  # 100% utilization
    assert dev1_util < dev0_util  # Lower utilization due to bubble
    
    print("✓ Utilization-based imbalance analysis test passed")
    print(f"  Device 0 utilization: {dev0_util:.2%}")
    print(f"  Device 1 utilization: {dev1_util:.2%}")

def test_comprehensive_analysis():
    '''Test all analyzers on the same trace'''
    trace = Trace(2)
    
    # Create a realistic pipeline trace
    # Device 0 (Stage 0)
    trace.add_event(0, Event(1, EventType.FWD, 'fwd-0', 100, 50))
    trace.add_event(0, Event(2, EventType.FWD, 'fwd-1', 200, 50))
    trace.add_event(0, Event(3, EventType.BWD, 'bwd-0', 300, 100))
    
    # Device 1 (Stage 1)
    trace.add_event(1, Event(4, EventType.FWD, 'fwd-0', 150, 50))
    trace.add_event(1, Event(5, EventType.FWD, 'fwd-1', 250, 50))
    trace.add_event(1, Event(6, EventType.BWD, 'bwd-0', 350, 100))
    
    # Run all analyses
    bubble_result = BubbleAnalyzer.analyze(trace)
    overlap_result = OverlapAnalyzer.analyze(trace)
    imbalance_result = ImbalanceAnalyzer.analyze(trace)
    
    print("✓ Comprehensive analysis test passed")
    print("\nAnalysis Results:")
    print(f"  Average bubble ratio: {bubble_result['average_bubble_ratio']:.2%}")
    print(f"  Imbalance ratio: {imbalance_result['imbalance_ratio']:.2%}")
    print(f"  Device count: {trace.get_ndevs()}")

if __name__ == '__main__':
    test_bubble_analysis()
    test_bubble_analysis_compressed()
    test_overlap_analysis()
    test_imbalance_analysis()
    test_imbalance_by_utilization()
    test_comprehensive_analysis()
    print("\n✓ All analyzer tests passed!")
