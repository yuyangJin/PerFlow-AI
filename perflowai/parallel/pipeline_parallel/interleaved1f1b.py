from perflowai.parallel.pipeline_parallel.ppgraph import PPGraph

'''
@class Interleaved1F1BGraph
An Interleaved1F1BGraph is a pipeline trace for Interleaved 1F1B.
'''
class Interleaved1F1BGraph(PPGraph):
    def __init__(self, nstages, nmicrobatches, nchunks):
        super().__init__(nstages, nmicrobatches, nchunks)
    
    def buid_graph(self):
        '''
        Build the graph for Interleaved 1F1B 
        To be implemented
        '''
        pass