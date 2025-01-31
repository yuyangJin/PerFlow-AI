from perflowai.parallel.pipeline_parallel.ppgraph import PPGraph

'''
@class GPipeGraph
A GPipeGraph is a pipeline trace for GPipe.
'''
class GPipeGraph(PPGraph):
    def __init__(self, nstages, nmicrobatches, nchunks):
        super().__init__(nstages, nmicrobatches, nchunks)
    
    def build_graph(self):
        '''
        Build the graph for GPipe 
        To be implemented
        '''
        pass