'''
@module torchprofiler trace reader
'''

from .reader import TraceReader
from ..core import Event, EventType, Trace

from typing import List

import json

'''
@class TorchProfilerTraceReader
A trace reader for TorchProfiler.
'''

class TorchProfilerTraceReader(TraceReader):
    def __init__(self, trace_path: str):
        super().__init__('Torch Profiler Trace Reader', trace_path)


    def read(self, event_types: List[EventType]) -> Trace:

        # Read the trace file
        with open(self.m_trace_path) as f:
            trace_json = json.load(f)
        
        # Get the basic information
        ndevs = int(trace_json['distributedInfo']['world_size'])

        # Convert event types to a torch profiler event type set
        event_name_to_type_map = dict()
        for event_type in event_types:
            if event_type == EventType.FWD:
                event_name_to_type_map['forward_step'] = EventType.FWD
            elif event_type == EventType.BWD:
                event_name_to_type_map['backward_step'] = EventType.BWD


        # Read in from trace json
        trace = Trace(ndevs)
        raw_traces = trace_json["traceEvents"]

        id = 0
        for raw_event in raw_traces:
            # Count event id 
            id += 1

            # Check if the event name starts with strings in event_name_to_type_map's keys.
            
            # If yes, return the event type 
            name = raw_event['name']
            type = None

            for event_name, event_type in event_name_to_type_map.items():
                if name.startswith(event_name):
                    type = event_type
                    break

            # If no, handle the next event
            if type == None:
                continue 

            # Get related info of event 
            dev_id = int(raw_event['tid'])
            start_ts = raw_event['ts']
            duration = raw_event['dur']

            # Add the event into the trace
            
            event = Event(id, type, name, start_ts, duration)
            trace.add_event(dev_id, event)

        self.m_outputs.append(trace)

        return trace
