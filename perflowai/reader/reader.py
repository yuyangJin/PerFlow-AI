'''

'''
from perflowai.workflow.flow import FlowNode

from typing import List

'''
@class TraceReader
A trace reader.
'''
class TraceReader(FlowNode):
    def __init__(self, trace_path: str):
        self.trace_path = trace_path

    def read(self) -> List[FlowNode]:
        '''
        To be implemented.
        '''
        pass
        return []

    def write(self, flows: List[FlowNode]):
        '''
        Not implemented yet.
        '''
        pass