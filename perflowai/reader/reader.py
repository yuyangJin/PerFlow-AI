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
    def __init__(self, trace_reader_str, trace_path: str, id=0):
        super().__init__(trace_reader_str, id, [], [])
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

    def run(self, *args, **kwargs):
        '''
        Run the trace reader by calling read().
        '''
        return self.read(*args, **kwargs)