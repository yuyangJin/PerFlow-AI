'''
@module pipeline parallel
'''

from perflowai.core.trace import Trace

'''
@class PPGraph
A PPGraph is a pipeline trace.
An event is a node in the PPGraph, and it must be FwdBwdEvent.
'''
class PPGraph(Trace):
    def __init__(self, nstages, nmicrobatches, nchunks, n):
        # Basic information for pipeline parallelism
        self.m_nstages = nstages
        self.m_nmicrobatches = nmicrobatches
        self.m_nchunks = nchunks
        
        self.m_edges = dict()

    def get_event_id(self):
        '''
        Calculate the node id based on the stage, microbatch, and chunk
        '''
        
        pass

    def generate_nodes(self):
        pass

    def add_edge(self, src, dst):
        pass

    def get_nodes(self):
        return self.m_nodes

    def get_edges(self):
        return self.m_edges

    def check(self):
        for src in self.m_edges:
            for dst in self.m_edges[src]:
                assert dst in self.m_edges, f"node {dst} not found"

    def run(self, *args, **kwargs):
        for node in self.m_nodes:
            node.run(*args, **kwargs)