from .reader import TraceReader

'''
@class TorchProfilerTraceReader
A trace reader for TorchProfiler.
'''

class TorchProfilerTraceReader(TraceReader):
    def __init__(self, trace_path: str):
        super().__init__(trace_path)

    def read(self):
        '''
        To be implemented.
        '''
        pass

    def get_trace(self):
        '''
        To be implemented.
        '''
        pass