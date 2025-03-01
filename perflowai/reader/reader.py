'''

'''
from ..workflow.flow import FlowNode
from ..core.event import Event, EventType
from ..core.trace import Trace

from typing import List

'''
@class TraceReader
A trace reader.
'''
class TraceReader(FlowNode):
    def __init__(self, trace_reader_str, trace_path: str):
        super().__init__(trace_reader_str, 0, [], [])
        self.m_trace_path = trace_path

    def read(self, event_types: List[EventType]) -> Trace:
        '''
        To be implemented.
        '''
        pass
        return None

    def write(self, flows: List[FlowNode]):
        '''
        Not implemented yet.
        '''
        pass