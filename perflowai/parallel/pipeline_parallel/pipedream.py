from perflowai.parallel.pipeline_parallel.ppgraph import PPGraph

'''
@class PipeDreamGraph
A PipeDreamGraph is a pipeline trace for PipeDream.
'''
class PipeDreamGraph(PPGraph):
    def __init__(self, nstages, nmicrobatches, nchunks):
        super().__init__(nstages, nmicrobatches, nchunks)
    
    def build_graph(self):
        '''
        Build the graph for PipeDream 
        To be implemented
        '''
        pass