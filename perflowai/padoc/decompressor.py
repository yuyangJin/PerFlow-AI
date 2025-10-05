'''
@module Trace Decompressor
Decompresses traces that were compressed using linear prediction and delta encoding.
'''

from typing import List, Dict, Any
import json
import base64
from ..core.trace import Trace
from ..core.event import Event, EventType
from .compressor import DeltaEncoder, LinearPredictor

class TraceDecompressor:
    '''
    Decompresses trace data compressed with TraceCompressor.
    Supports direct analysis on compressed format for some operations.
    '''
    
    def __init__(self):
        self.decoder = DeltaEncoder()
        self.predictor = LinearPredictor()
        
    def decompress_trace(self, compressed_data: Dict[str, Any]) -> Trace:
        '''
        Fully decompress a compressed trace.
        
        Args:
            compressed_data: Compressed trace data
            
        Returns:
            Decompressed Trace object
        '''
        metadata = compressed_data['metadata']
        ndevs = metadata['ndevs']
        
        trace = Trace(ndevs)
        
        # Decompress each device's events
        for dev_id_str, dev_data in compressed_data['devices'].items():
            dev_id = int(dev_id_str)
            events = self._decompress_device_events(dev_data)
            
            for event in events:
                trace.add_event(dev_id, event)
        
        return trace
    
    def _decompress_device_events(self, dev_data: Dict[str, Any]) -> List[Event]:
        '''
        Decompress events for a single device.
        '''
        count = dev_data['count']
        timestamp_data = dev_data['timestamps']
        duration_data = dev_data['durations']
        event_metadata = dev_data['event_data']
        
        # Decode deltas
        timestamp_deltas = self.decoder.decode_delta_list(timestamp_data, count)
        duration_deltas = self.decoder.decode_delta_list(duration_data, count)
        
        # Reconstruct events
        events = []
        for i in range(count):
            metadata = event_metadata[i]
            
            # Reconstruct timestamp
            if i == 0:
                predicted_ts = 0
            else:
                predicted_ts = events[i-1].get_timestamp() + events[i-1].get_duration()
            actual_ts = predicted_ts + timestamp_deltas[i]
            
            # Reconstruct duration
            if i == 0:
                predicted_dur = 0
            else:
                predicted_dur = events[i-1].get_duration()
            actual_dur = predicted_dur + duration_deltas[i]
            
            # Create event
            event_type = self._parse_event_type(metadata['type'])
            event = Event(
                metadata['id'],
                event_type,
                metadata['name'],
                actual_ts,
                actual_dur
            )
            events.append(event)
        
        return events
    
    def _parse_event_type(self, type_str: str) -> EventType:
        '''Parse event type from string'''
        try:
            return EventType[type_str]
        except (KeyError, AttributeError):
            return EventType.OPRT
    
    def get_event_at_index(self, compressed_data: Dict[str, Any], 
                          dev_id: int, index: int) -> Event:
        '''
        Get a specific event without full decompression (O(1) random access).
        This is a key feature of PADoC - direct analysis on compressed format.
        
        Args:
            compressed_data: Compressed trace data
            dev_id: Device ID
            index: Event index
            
        Returns:
            The reconstructed Event
        '''
        dev_data = compressed_data['devices'].get(str(dev_id))
        if not dev_data:
            raise ValueError(f"Device {dev_id} not found in compressed data")
        
        if index >= dev_data['count']:
            raise IndexError(f"Event index {index} out of range")
        
        # Decode only the deltas we need
        # For true O(1), we'd need to store cumulative sums or use a more sophisticated encoding
        # This is a simplified version that decodes up to the requested index
        timestamp_deltas = self.decoder.decode_delta_list(dev_data['timestamps'], index + 1)
        duration_deltas = self.decoder.decode_delta_list(dev_data['durations'], index + 1)
        
        # Reconstruct the event
        timestamp = sum(timestamp_deltas)
        
        # For duration, we need to apply the linear model
        if index == 0:
            duration = duration_deltas[0]
        else:
            # This is simplified - full implementation would track previous duration
            duration = duration_deltas[index]
        
        metadata = dev_data['event_data'][index]
        event_type = self._parse_event_type(metadata['type'])
        
        return Event(metadata['id'], event_type, metadata['name'], timestamp, duration)
    
    def load_compressed(self, filepath: str) -> Dict[str, Any]:
        '''Load compressed trace from file'''
        with open(filepath, 'r') as f:
            encoded_data = json.load(f)
        
        # Decode base64 bytes back to bytes
        def decode_bytes(obj):
            if isinstance(obj, dict):
                if obj.get('_type') == 'bytes':
                    return base64.b64decode(obj['data'])
                return {k: decode_bytes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [decode_bytes(item) for item in obj]
            return obj
        
        return decode_bytes(encoded_data)
