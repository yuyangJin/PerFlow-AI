from perflowai.parallel.pipeline_parallel.ppgraph import PPGraph

'''
@class ZeroBubbleGraph
A ZeroBubbleGraph is a pipeline trace for ZeroBubble
'''
class ZeroBubbleGraph(PPGraph):
    def __init__(self, nstages, nmicrobatches, nchunks):
        super().__init__(nstages, nmicrobatches, nchunks)
    
    def buid_graph(self):
        '''
        Build the graph for ZeroBubble
        To be implemented
        '''
        pass