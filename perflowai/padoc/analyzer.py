'''
@module Analyzers for compressed traces
Implements direct analysis on compressed trace format.
'''

from typing import List, Dict, Any, Tuple
from ..core.trace import Trace
from ..core.event import Event

class BubbleAnalyzer:
    '''
    Analyzes pipeline bubbles directly on compressed traces.
    Bubbles are idle times between pipeline stages.
    '''
    
    @staticmethod
    def analyze(trace: Trace) -> Dict[str, Any]:
        '''
        Analyze bubbles in the trace.
        
        Args:
            trace: The trace to analyze (can be compressed or uncompressed)
            
        Returns:
            Dictionary with bubble analysis results
        '''
        ndevs = trace.get_ndevs()
        
        total_time = 0
        total_bubble_time = 0
        device_bubbles = {}
        
        for dev_id in range(ndevs):
            events = trace.get_events(dev_id)
            if not events:
                device_bubbles[dev_id] = {
                    'bubble_time': 0,
                    'total_time': 0,
                    'bubble_ratio': 0,
                    'bubble_count': 0
                }
                continue
            
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda e: e.get_timestamp())
            
            # Calculate bubbles (gaps between events)
            bubbles = []
            device_total_time = 0
            device_bubble_time = 0
            
            start_time = sorted_events[0].get_timestamp()
            end_time = start_time
            
            for i, event in enumerate(sorted_events):
                event_start = event.get_timestamp()
                event_end = event_start + event.get_duration()
                
                # Check for bubble (gap) before this event
                if i > 0 and event_start > end_time:
                    bubble_size = event_start - end_time
                    bubbles.append({
                        'start': end_time,
                        'end': event_start,
                        'duration': bubble_size
                    })
                    device_bubble_time += bubble_size
                
                end_time = max(end_time, event_end)
            
            device_total_time = end_time - start_time
            bubble_ratio = device_bubble_time / device_total_time if device_total_time > 0 else 0
            
            device_bubbles[dev_id] = {
                'bubble_time': device_bubble_time,
                'total_time': device_total_time,
                'bubble_ratio': bubble_ratio,
                'bubble_count': len(bubbles),
                'bubbles': bubbles[:10]  # Store first 10 bubbles as examples
            }
            
            total_time += device_total_time
            total_bubble_time += device_bubble_time
        
        return {
            'total_bubble_time': total_bubble_time,
            'total_time': total_time,
            'average_bubble_ratio': total_bubble_time / total_time if total_time > 0 else 0,
            'device_bubbles': device_bubbles
        }
    
    @staticmethod
    def analyze_compressed(compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Analyze bubbles directly on compressed format.
        This is more efficient as it doesn't require full decompression.
        '''
        # For this simplified version, we'll reconstruct minimal data
        # In a full implementation, we could work more directly with deltas
        from .decompressor import TraceDecompressor
        decompressor = TraceDecompressor()
        trace = decompressor.decompress_trace(compressed_data)
        return BubbleAnalyzer.analyze(trace)


class OverlapAnalyzer:
    '''
    Analyzes computation-communication overlap in traces.
    Measures how well computation overlaps with communication.
    '''
    
    @staticmethod
    def analyze(trace: Trace) -> Dict[str, Any]:
        '''
        Analyze computation-communication overlap.
        
        Args:
            trace: The trace to analyze
            
        Returns:
            Dictionary with overlap analysis results
        '''
        from ..core.event import EventType
        
        ndevs = trace.get_ndevs()
        device_overlaps = {}
        
        for dev_id in range(ndevs):
            events = trace.get_events(dev_id)
            if not events:
                continue
            
            # Separate computation and communication events
            comp_events = [e for e in events if e.get_type() in [EventType.FWD, EventType.BWD, EventType.WGT]]
            comm_events = [e for e in events if e.get_type() == EventType.COMM]
            
            if not comp_events or not comm_events:
                device_overlaps[dev_id] = {
                    'overlap_time': 0,
                    'comp_time': sum(e.get_duration() for e in comp_events),
                    'comm_time': sum(e.get_duration() for e in comm_events),
                    'overlap_ratio': 0
                }
                continue
            
            # Calculate overlap by finding time ranges where both occur
            overlap_time = 0
            comp_time = sum(e.get_duration() for e in comp_events)
            comm_time = sum(e.get_duration() for e in comm_events)
            
            # For each communication event, check overlap with computation
            for comm_event in comm_events:
                comm_start = comm_event.get_timestamp()
                comm_end = comm_start + comm_event.get_duration()
                
                for comp_event in comp_events:
                    comp_start = comp_event.get_timestamp()
                    comp_end = comp_start + comp_event.get_duration()
                    
                    # Calculate overlap
                    overlap_start = max(comm_start, comp_start)
                    overlap_end = min(comm_end, comp_end)
                    
                    if overlap_start < overlap_end:
                        overlap_time += (overlap_end - overlap_start)
            
            overlap_ratio = overlap_time / comm_time if comm_time > 0 else 0
            
            device_overlaps[dev_id] = {
                'overlap_time': overlap_time,
                'comp_time': comp_time,
                'comm_time': comm_time,
                'overlap_ratio': overlap_ratio
            }
        
        # Calculate overall statistics
        total_overlap = sum(d['overlap_time'] for d in device_overlaps.values())
        total_comm = sum(d['comm_time'] for d in device_overlaps.values())
        
        return {
            'device_overlaps': device_overlaps,
            'total_overlap_time': total_overlap,
            'total_comm_time': total_comm,
            'average_overlap_ratio': total_overlap / total_comm if total_comm > 0 else 0
        }


class ImbalanceAnalyzer:
    '''
    Analyzes load imbalance across devices/stages.
    '''
    
    @staticmethod
    def analyze(trace: Trace) -> Dict[str, Any]:
        '''
        Analyze load imbalance across devices.
        
        Args:
            trace: The trace to analyze
            
        Returns:
            Dictionary with imbalance analysis results
        '''
        ndevs = trace.get_ndevs()
        device_loads = {}
        
        for dev_id in range(ndevs):
            events = trace.get_events(dev_id)
            if not events:
                device_loads[dev_id] = {
                    'total_time': 0,
                    'event_count': 0,
                    'avg_event_duration': 0
                }
                continue
            
            sorted_events = sorted(events, key=lambda e: e.get_timestamp())
            
            start_time = sorted_events[0].get_timestamp()
            end_time = max(e.get_timestamp() + e.get_duration() for e in sorted_events)
            total_time = end_time - start_time
            
            compute_time = sum(e.get_duration() for e in events)
            
            device_loads[dev_id] = {
                'total_time': total_time,
                'compute_time': compute_time,
                'event_count': len(events),
                'avg_event_duration': compute_time / len(events) if events else 0,
                'utilization': compute_time / total_time if total_time > 0 else 0
            }
        
        # Calculate imbalance metrics
        if device_loads:
            compute_times = [d['compute_time'] for d in device_loads.values()]
            utilizations = [d['utilization'] for d in device_loads.values()]
            
            max_compute = max(compute_times) if compute_times else 0
            min_compute = min(compute_times) if compute_times else 0
            avg_compute = sum(compute_times) / len(compute_times) if compute_times else 0
            
            max_util = max(utilizations) if utilizations else 0
            min_util = min(utilizations) if utilizations else 0
            avg_util = sum(utilizations) / len(utilizations) if utilizations else 0
            
            # Imbalance ratio: how much the slowest device differs from average
            imbalance_ratio = (max_compute - avg_compute) / avg_compute if avg_compute > 0 else 0
        else:
            max_compute = min_compute = avg_compute = 0
            max_util = min_util = avg_util = 0
            imbalance_ratio = 0
        
        return {
            'device_loads': device_loads,
            'max_compute_time': max_compute,
            'min_compute_time': min_compute,
            'avg_compute_time': avg_compute,
            'max_utilization': max_util,
            'min_utilization': min_util,
            'avg_utilization': avg_util,
            'imbalance_ratio': imbalance_ratio
        }
    
    @staticmethod
    def analyze_by_stage(trace: Trace, nstages: int) -> Dict[str, Any]:
        '''
        Analyze imbalance across pipeline stages.
        Assumes devices are mapped to stages in order.
        '''
        if not hasattr(trace, 'get_nstages'):
            # Regular trace, treat each device as a stage
            return ImbalanceAnalyzer.analyze(trace)
        
        # For PPTrace, use actual stage information
        stage_loads = {}
        
        for stage_id in range(nstages):
            events = trace.get_events(stage_id)
            if not events:
                continue
            
            compute_time = sum(e.get_duration() for e in events)
            stage_loads[stage_id] = compute_time
        
        if stage_loads:
            max_load = max(stage_loads.values())
            min_load = min(stage_loads.values())
            avg_load = sum(stage_loads.values()) / len(stage_loads)
            imbalance_ratio = (max_load - avg_load) / avg_load if avg_load > 0 else 0
        else:
            max_load = min_load = avg_load = imbalance_ratio = 0
        
        return {
            'stage_loads': stage_loads,
            'max_stage_load': max_load,
            'min_stage_load': min_load,
            'avg_stage_load': avg_load,
            'imbalance_ratio': imbalance_ratio
        }
