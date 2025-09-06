'''
@module torchprofiler trace reader
'''

from .reader import TraceReader
from ..core import Event, EventType, Trace

from typing import List

from glob import glob
import json
import threading
from threading import Lock

'''
@class TorchProfilerTraceReader
A trace reader for TorchProfiler.
'''

class TorchProfilerTraceReader(TraceReader):
    def __init__(self, trace_path: str):
        super().__init__('Torch Profiler Trace Reader', trace_path)
        self.m_metadata = None

    def get_events(self, events):
        ret = []
        id = 0
        for e in events:
            # Count event id
            id += 1 

            # Get related info of event 
            start_ts = e['ts']
            duration = e['dur']
            name = e['name']
            type = None

            event = Event(id, type, name, start_ts, duration)

            ret.append(event)

        return ret

    def read_trace(self, fn, rank, thread_results, lock):
        try:
            # Process file safely
            with open(fn, 'r') as f:
                trace = json.load(f)
                print(f"Loaded trace file {fn} with {len(trace['traceEvents'])} events")
            events = self.get_events(trace["traceEvents"])
            # Use lock to safely store events
            with lock:
                thread_results[rank] = events
            print(f'Imported {len(thread_results[rank])} events from {fn}')
        except Exception as e:
            print(f"Error processing file {fn}: {e}")
            thread_results[rank] = []

    def parallel_read(self, start = None, end = None, stride = None) -> Trace:
        
        # Thread-safe structures
        thread_results = {}
        lock = Lock()
        threads = []

        for rank in range(start, end, stride):
            
            # Generate file name
            fns = glob(self.m_trace_path + f"/profile_{rank}.json")

            # Check if file exists or not
            if len(fns) != 1:
                print(f"Error: Expected 1 file for rank {rank}, found {len(fns)}")
                continue
            
            fn = fns[0]

            # Thread related code
            thread = threading.Thread(
                target=self.read_trace, 
                args=(fn, rank, thread_results, lock)
            )

            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Start rank contributes the metadata
        fns = glob(self.m_trace_path + f"/profile_{start}.json")
        with open(fns[0], 'r') as f:
            rank_start_trace = json.load(f)
        self.m_metadata = rank_start_trace
        self.m_metadata["traceEvents"] = []


        # Get the basic information
        ndevs = int(self.m_metadata['distributedInfo']['world_size'])

        trace = Trace(ndevs)

        # Combine results from all threads
        for rank in range(start, end, stride):
            events = thread_results.get(rank, [])
            trace.add_events(rank, events)

        return trace

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
