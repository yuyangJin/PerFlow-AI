'''
@module nsys trace reader
'''

from .reader import TraceReader
from ..core import EventType
from typing import List

'''
@class NsysTraceReader
'''
class NsysTraceReader(TraceReader):
    def __init__(self, trace_path: str):
        super().__init__(trace_path)

    def read(self, event_types: List[EventType] = []):
        '''
        Read the nsys trace.
        To be implemented.
        '''
        pass

    def get_trace(self):
        '''
        Get the trace.
        To be implemented.
        '''
        pass