from .ppgraph import PPGraph
from ...core.event import EventType

'''
@class GPipeGraph
A GPipeGraph is a pipeline trace for GPipe.
'''
class GPipeGraph(PPGraph):
    def __init__(self, nstages, nmicrobatches, nchunks, cost_config=None):
        assert nchunks == 1, 'Not support for GPipe - the number of chunks is larger than 1!'
        super().__init__(nstages, nmicrobatches, nchunks, cost_config)

    '''
    Build the graph for GPipe 
    '''
    def build_graph(self):

        '''
        Add nodes/events
        '''
        self.generate_nodes()


        '''
        Add data dependence / edges
        '''
        # fwd
        for stage in range(self.m_nstages):
            for mb in range(self.m_nmicrobatches):
                for chk in range(self.m_nchunks):
                    src_id = self.get_event_id(EventType.FWD, stage, mb, chk)
                    if mb == self.m_nmicrobatches - 1: # the last microbatch
                        dest1_id = self.get_event_id(EventType.BWD, stage, 0, chk)
                    else:                              # normal 
                        dest1_id = self.get_event_id(EventType.FWD, stage, mb+1, chk)
                    
                    self.add_edge(src_id, dest1_id)
                    
                    if stage != self.m_nstages - 1: # not the last stage
                        dest2_id = self.get_event_id(EventType.FWD, stage+1, mb, chk)
                        self.add_edge(src_id, dest2_id)
                    
                    # inter-chunk dependence
                    if chk != 0 and stage == 0:
                        src3_id = self.get_event_id(EventType.FWD, self.m_nstages - 1, mb, chk - 1)
                        dest3_id = src_id
                        self.add_edge(src3_id, dest3_id)

        # bwd
        for stage in range(self.m_nstages):
            for mb in range(self.m_nmicrobatches):
                for chk in range(self.m_nchunks):
                    if mb != self.m_nmicrobatches - 1: # not the last microbatch
                        src1_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                        dest1_id = self.get_event_id(EventType.BWD, stage, mb+1, chk)
                        self.add_edge(src1_id, dest1_id)


                    if stage != self.m_nstages - 1: # not the last stage
                        src2_id = self.get_event_id(EventType.BWD, stage+1, mb, chk)
                        dest2_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                        self.add_edge(src2_id, dest2_id)

                    # inter-chunk dependence
                    if chk != 0 :
                        # inter-stage 
                        if stage == self.m_nstages - 1:
                            src3_id = self.get_event_id(EventType.BWD, 0, mb, chk - 1)
                            dest3_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                            self.add_edge(src3_id, dest3_id)
                        # intra-stage
                        if mb == 0:
                            src3_id = self.get_event_id(EventType.BWD, stage, self.m_nmicrobatches - 1, chk - 1)
                            dest3_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                            self.add_edge(src3_id, dest3_id)