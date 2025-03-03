'''
@module PipeDream
'''

from .ppgraph import PPGraph
from ...core import EventType

'''
@class PipeDreamGraph
A PipeDreamGraph is a pipeline trace for PipeDream.
'''
class PipeDreamGraph(PPGraph):
    def __init__(self, nstages, nmicrobatches, nchunks, cost_config=None, eager=False):
        assert nchunks == 1, 'Not support for GPipe - the number of chunks is larger than 1!'
        super().__init__(nstages, nmicrobatches, nchunks, cost_config)
        self.eager = eager
    
    def __compute_num_warmup_micro_batches(self, stage):
        if self.eager:
            return min(2 * (self.m_nstages - stage) - 1, self.m_nmicrobatches)
        else:
            return min(self.m_nstages - stage, self.m_nmicrobatches)

    '''
    Build the graph for PipeDream 
    '''
    def build_graph(self):
        '''
        Add nodes/events
        '''
        self.generate_nodes()

        '''
        Add data dependence / edges
        '''
        for stage in range(self.m_nstages):
            # Warmup fwd
            n_warmup_fwd = self.__compute_num_warmup_micro_batches(stage)
            for mb in range(n_warmup_fwd):
                for chk in range(self.m_nchunks):
                    src_id = self.get_event_id(EventType.FWD, stage, mb, chk)

                    # The last microbatch
                    if mb == n_warmup_fwd - 1 and chk == self.m_nchunks - 1: 
                        dest1_id = self.get_event_id(EventType.BWD, stage, 0, chk)
                    # Normal case
                    else:                                                    
                        dest1_id = self.get_event_id(EventType.FWD, stage, mb+1, chk)
                    
                    self.add_edge(src_id, dest1_id)

                    # Not the last stage
                    if stage != self.m_nstages - 1:                         
                        dest2_id = self.get_event_id(EventType.FWD, stage+1, mb, chk)
                        self.add_edge(src_id, dest2_id) 


            
            # Fwd & bwd 1f1b
            for mb in range(self.m_nmicrobatches):
                for chk in range(self.m_nchunks):
                    
                    # 1 bwd

                    # Inter-stage dependence, no need for the last stage
                    if stage != self.m_nstages - 1: 
                        src_id = self.get_event_id(EventType.BWD, stage+1, mb, chk)
                        dest_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                        self.add_edge(src_id, dest_id)
                    
                    # Intra-stage fwd-bwd dependence, no need for the last microbatch
                    ## Steady 1F1B part
                    if n_warmup_fwd + mb - 1 < self.m_nmicrobatches - 1: 
                        src_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                        dest_id = self.get_event_id(EventType.FWD, stage, n_warmup_fwd + mb, chk)
                        self.add_edge(src_id, dest_id)
                    
                    ## Tail part with no fwd, only 
                    else:
                        if mb != self.m_nmicrobatches - 1:
                            src_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                            dest_id = self.get_event_id(EventType.BWD, stage, mb+1, chk)
                            self.add_edge(src_id, dest_id)



                    # 1 fwd

                    if n_warmup_fwd + mb - 1 < self.m_nmicrobatches - 1: 
                        # Not the last microbatch
                        if mb != self.m_nmicrobatches - 1: 
                            src1_id = self.get_event_id(EventType.FWD, stage, n_warmup_fwd + mb, chk)
                            dest1_id = self.get_event_id(EventType.BWD, stage, mb + 1, chk)
                            self.add_edge(src1_id, dest1_id)
