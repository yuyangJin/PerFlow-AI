'''
@module trace reader
'''

from ..workflow import FlowNode
from ..core import Event, EventType, Trace

from typing import List

'''
@class TraceReader
A trace reader.
'''
class TraceReader(FlowNode):
    def __init__(self, trace_path: str):
        super().__init__()
        self.m_trace_path = trace_path

    def read(self, event_types: List[EventType]) -> Trace:
        '''
        Read the trace.
        To be implemented by subclasses.
        '''
        pass
        return None

    def write(self, flows: List[FlowNode]):
        '''
        Write the trace.
        Not implemented yet.
        '''
        pass

    def run(self):
        '''
        Run the trace reader by calling read().
        Gets configuration from inputs if available, puts trace in outputs.
        '''
        # Get event types from inputs if provided
        event_types = []
        if self.m_inputs.size() > 0:
            input_data = list(self.m_inputs.get_data())
            # Check if first input is a list of event types
            if input_data and isinstance(input_data[0], list):
                event_types = input_data[0]
        
        result = self.read(event_types)
        if result is not None:
            self.m_outputs.add_data(result)
        return result