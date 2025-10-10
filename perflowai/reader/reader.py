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
        '''
        result = self.read([])
        if result is not None:
            self.m_outputs.add_data(result)
        return result