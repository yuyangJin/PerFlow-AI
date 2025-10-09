'''
@module nsys trace reader
'''

from .reader import TraceReader

'''
@class NsysTraceReader
'''
class NsysTraceReader(TraceReader):
    def __init__(self, trace_path: str):
        super().__init__('NsysTraceReader', trace_path, 0)

    def read(self, *args, **kwargs):
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