'''
@module Trace Compressor
Implements lossless compression of traces using linear prediction and delta encoding.
'''

from typing import List, Dict, Optional, Tuple, Any
import json
import struct
from ..core.trace import Trace
from ..core.event import Event

class LinearPredictor:
    '''
    Linear predictor for timestamps and durations.
    Predicts values based on previous events.
    '''
    
    @staticmethod
    def predict_timestamp_intra_microbatch(events: List[Event], index: int) -> float:
        '''
        Predict timestamp for an event within a microbatch.
        Uses the previous event's end time as prediction.
        '''
        if index == 0:
            return 0  # No prediction for first event
        
        prev_event = events[index - 1]
        return prev_event.get_timestamp() + prev_event.get_duration()
    
    @staticmethod
    def predict_timestamp_inter_microbatch(current_mb_events: List[Event], 
                                          prev_mb_events: List[Event], 
                                          index: int) -> float:
        '''
        Predict timestamp using corresponding event from previous microbatch.
        '''
        if not prev_mb_events or index >= len(prev_mb_events):
            return 0
        
        # Find time delta from previous microbatch
        if index == 0:
            if len(current_mb_events) > 0:
                return prev_mb_events[0].get_timestamp()
        else:
            # Use the time offset from previous microbatch
            prev_event = prev_mb_events[index]
            prev_start = prev_mb_events[0].get_timestamp()
            offset = prev_event.get_timestamp() - prev_start
            return current_mb_events[0].get_timestamp() + offset
        
        return 0
    
    @staticmethod
    def predict_duration_inter_microbatch(prev_mb_events: List[Event], 
                                         index: int) -> float:
        '''
        Predict duration using corresponding event from previous microbatch.
        '''
        if not prev_mb_events or index >= len(prev_mb_events):
            return 0
        
        return prev_mb_events[index].get_duration()
    
    @staticmethod
    def predict_duration_inter_iteration(current_iter_events: List[Event],
                                        prev_iter_events: List[Event],
                                        index: int) -> float:
        '''
        Predict duration using corresponding event from previous iteration.
        '''
        if not prev_iter_events or index >= len(prev_iter_events):
            return 0
        
        return prev_iter_events[index].get_duration()


class DeltaEncoder:
    '''
    Encodes deltas using variable-length encoding (similar to LEB128).
    Small values use fewer bytes.
    '''
    
    @staticmethod
    def encode_delta(delta: int) -> bytes:
        '''
        Encode a signed integer delta using variable-length encoding.
        Encodes in zigzag format: 0->0, -1->1, 1->2, -2->3, 2->4, ...
        '''
        # Zigzag encoding: map signed to unsigned
        zigzag = (delta << 1) ^ (delta >> 63) if delta < 0 else (delta << 1)
        
        # Variable-length encoding
        result = []
        while zigzag > 0x7F:
            result.append((zigzag & 0x7F) | 0x80)
            zigzag >>= 7
        result.append(zigzag & 0x7F)
        
        return bytes(result)
    
    @staticmethod
    def decode_delta(data: bytes, offset: int) -> Tuple[int, int]:
        '''
        Decode a variable-length encoded delta.
        
        Returns:
            (decoded_value, bytes_consumed)
        '''
        zigzag = 0
        shift = 0
        bytes_read = 0
        
        while offset + bytes_read < len(data):
            byte = data[offset + bytes_read]
            bytes_read += 1
            
            zigzag |= (byte & 0x7F) << shift
            shift += 7
            
            if not (byte & 0x80):
                break
        
        # Zigzag decoding: map unsigned back to signed
        delta = (zigzag >> 1) ^ (-(zigzag & 1))
        
        return delta, bytes_read
    
    @staticmethod
    def encode_delta_list(deltas: List[int]) -> bytes:
        '''Encode a list of deltas'''
        result = []
        for delta in deltas:
            result.extend(DeltaEncoder.encode_delta(delta))
        return bytes(result)
    
    @staticmethod
    def decode_delta_list(data: bytes, count: int) -> List[int]:
        '''Decode a list of deltas'''
        deltas = []
        offset = 0
        
        for _ in range(count):
            delta, consumed = DeltaEncoder.decode_delta(data, offset)
            deltas.append(delta)
            offset += consumed
            
        return deltas


class TraceCompressor:
    '''
    Compresses trace data using linear prediction and delta encoding.
    Supports multiple compression strategies:
    - Intra-microbatch: compress events within a microbatch
    - Inter-microbatch: compress using patterns across microbatches
    - Inter-iteration: compress using patterns across iterations
    - Inter-process: compress using patterns across processes/devices
    '''
    
    def __init__(self):
        self.predictor = LinearPredictor()
        self.encoder = DeltaEncoder()
        
    def compress_trace(self, trace: Trace, strategy: str = 'auto') -> Dict[str, Any]:
        '''
        Compress a trace using the specified strategy.
        
        Args:
            trace: The trace to compress
            strategy: Compression strategy ('intra', 'inter_mb', 'inter_iter', 'inter_proc', 'auto')
            
        Returns:
            Compressed trace data as a dictionary
        '''
        compressed = {
            'metadata': {
                'ndevs': trace.get_ndevs(),
                'strategy': strategy,
                'original_event_count': sum(len(trace.get_events(i)) 
                                          for i in range(trace.get_ndevs()))
            },
            'devices': {}
        }
        
        # Compress each device's events
        for dev_id in range(trace.get_ndevs()):
            events = trace.get_events(dev_id)
            if not events:
                continue
                
            if strategy == 'intra' or strategy == 'auto':
                compressed['devices'][str(dev_id)] = self._compress_intra_microbatch(events)
            elif strategy == 'inter_mb':
                compressed['devices'][str(dev_id)] = self._compress_inter_microbatch(events)
            elif strategy == 'inter_iter':
                compressed['devices'][str(dev_id)] = self._compress_inter_iteration(events)
            elif strategy == 'inter_proc':
                # For inter-process, we need events from other devices
                all_events = [trace.get_events(i) for i in range(trace.get_ndevs())]
                compressed['devices'][str(dev_id)] = self._compress_inter_process(events, all_events, dev_id)
                
        return compressed
    
    def _compress_intra_microbatch(self, events: List[Event]) -> Dict[str, Any]:
        '''
        Compress events using intra-microbatch prediction.
        Predicts based on previous event in the same sequence.
        '''
        if not events:
            return {'timestamps': b'', 'durations': b'', 'event_data': []}
        
        timestamp_deltas = []
        duration_deltas = []
        event_data = []
        
        for i, event in enumerate(events):
            # Predict timestamp
            predicted_ts = self.predictor.predict_timestamp_intra_microbatch(events, i)
            actual_ts = event.get_timestamp()
            ts_delta = int(actual_ts - predicted_ts)
            timestamp_deltas.append(ts_delta)
            
            # For duration, we predict 0 for first event, previous duration for others
            if i == 0:
                predicted_dur = 0
            else:
                predicted_dur = events[i-1].get_duration()
            actual_dur = event.get_duration()
            dur_delta = int(actual_dur - predicted_dur)
            duration_deltas.append(dur_delta)
            
            # Store event metadata
            event_data.append({
                'id': event.get_id(),
                'type': event.get_type().name if hasattr(event.get_type(), 'name') else str(event.get_type()),
                'name': event.get_name()
            })
        
        return {
            'count': len(events),
            'timestamps': self.encoder.encode_delta_list(timestamp_deltas),
            'durations': self.encoder.encode_delta_list(duration_deltas),
            'event_data': event_data
        }
    
    def _compress_inter_microbatch(self, events: List[Event]) -> Dict[str, Any]:
        '''
        Compress events using inter-microbatch prediction.
        Groups events by microbatch and uses previous microbatch for prediction.
        '''
        # This is a simplified version - in practice, would need to identify microbatch boundaries
        return self._compress_intra_microbatch(events)
    
    def _compress_inter_iteration(self, events: List[Event]) -> Dict[str, Any]:
        '''
        Compress events using inter-iteration prediction.
        Uses patterns from previous iteration.
        '''
        # This is a simplified version - in practice, would need to identify iteration boundaries
        return self._compress_intra_microbatch(events)
    
    def _compress_inter_process(self, events: List[Event], 
                                all_events: List[List[Event]], 
                                dev_id: int) -> Dict[str, Any]:
        '''
        Compress events using inter-process prediction.
        Uses patterns from other processes/devices.
        '''
        # This is a simplified version - in practice, would use events from other devices
        return self._compress_intra_microbatch(events)
    
    def save_compressed(self, compressed_data: Dict[str, Any], filepath: str):
        '''Save compressed trace to file'''
        # Convert bytes to base64 for JSON serialization
        import base64
        
        def encode_bytes(obj):
            if isinstance(obj, bytes):
                return {
                    '_type': 'bytes',
                    'data': base64.b64encode(obj).decode('utf-8')
                }
            elif isinstance(obj, dict):
                return {k: encode_bytes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [encode_bytes(item) for item in obj]
            return obj
        
        encoded_data = encode_bytes(compressed_data)
        
        with open(filepath, 'w') as f:
            json.dump(encoded_data, f, indent=2)
    
    def get_compression_stats(self, original_trace: Trace, 
                             compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Calculate compression statistics.
        
        Returns:
            Dictionary with compression metrics
        '''
        # Calculate original size (rough estimate)
        original_events = sum(len(original_trace.get_events(i)) 
                            for i in range(original_trace.get_ndevs()))
        # Each event: 8 bytes timestamp + 8 bytes duration + overhead
        original_size = original_events * 32  # Rough estimate
        
        # Calculate compressed size
        compressed_size = 0
        for dev_data in compressed_data['devices'].values():
            compressed_size += len(dev_data.get('timestamps', b''))
            compressed_size += len(dev_data.get('durations', b''))
            # Add metadata overhead (rough estimate)
            compressed_size += len(str(dev_data.get('event_data', [])))
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'space_savings': 1 - (compressed_size / original_size) if original_size > 0 else 0,
            'original_events': original_events
        }
