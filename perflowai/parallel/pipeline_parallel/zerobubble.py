from perflowai.parallel.pipeline_parallel.ppgraph import PPGraph
from ...core.event import EventType, FwdBwdEvent, NoneTimestamp

'''
@class ZeroBubbleGraph
A ZeroBubbleGraph is a pipeline trace for ZeroBubble
'''
class ZeroBubbleGraph(PPGraph):
    def __init__(self, nstages, nmicrobatches, nchunks, cost_config=None):
        super().__init__(nstages, nmicrobatches, nchunks, cost_config)
        self.event_types = [EventType.FWD, EventType.BWD, EventType.WGT]
    

    def generate_nodes(self):
        if self.m_cost_config == None:
            raise ValueError("Cost config should be setup at initialization.")

        event_types = self.get_event_types()
        n_stages = self.get_nstages()
        n_microbatches = self.get_nmicrobatches()
        n_chunks = self.get_nchunks()
        for type in event_types:
            for stage in range(n_stages):
                for mb in range(n_microbatches):
                    for chk in range(n_chunks):
                        if type == EventType.FWD:
                            self.add_node(type, stage, mb, self.m_cost_config.fwd_time, chk)
                        elif type == EventType.BWD:
                            self.add_node(type, stage, mb, self.m_cost_config.bwd_time, chk)
                        elif type == EventType.WGT:
                            self.add_node(type, stage, mb, self.m_cost_config.wgt_time, chk)
                        else:
                            assert False

    '''
        Build the graph for ZeroBubble
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
                    if mb != self.m_nmicrobatches - 1: # fwdxfwd#1: When the chk and stage are the same, execute in the order of mb. fwd(stage, 0, chk) -> ... -> fwd(stage, self.m_nmicrobatches - 1, chk)
                        src1_id = self.get_event_id(EventType.FWD, stage, mb, chk)
                        dest1_id = self.get_event_id(EventType.FWD, stage, mb+1, chk)
                        self.add_edge(src1_id, dest1_id)

                    if stage != self.m_nstages - 1: # fwdxfwd#2: When the chk and mb are the same, execute in the order of stage. fwd(0, mb, chk) -> ... -> fwd(self.m_nstages - 1, mb, chk)
                        src2_id = self.get_event_id(EventType.FWD, stage, mb, chk)
                        dest2_id = self.get_event_id(EventType.FWD, stage+1, mb, chk)
                        self.add_edge(src2_id, dest2_id)

                    #with above two rules, within the same chk, execute fwd(0, 0, chk) firstly, execute fwd(self.m_nstages - 1, self.m_nmicrobatches - 1, chk) lastly
                    
                    # inter-chunk dependence
                    if chk != 0:
                        if stage == 0:
                            src3_id = self.get_event_id(EventType.FWD, self.m_nstages - 1, mb, chk - 1)
                            dest3_id = self.get_event_id(EventType.FWD, 0, mb, chk)
                            self.add_edge(src3_id, dest3_id)
                    
        # bwd
        for stage in range(self.m_nstages):
            for mb in range(self.m_nmicrobatches):
                for chk in range(self.m_nchunks):
                    if mb != self.m_nmicrobatches - 1: # bwdxbwd#1: When the chk and stage are the same, execute in the order of mb. bwd(stage, 0, chk) -> ... -> bwd(stage, self.m_nmicrobatches - 1, chk)
                        src1_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                        dest1_id = self.get_event_id(EventType.BWD, stage, mb+1, chk)
                        self.add_edge(src1_id, dest1_id)

                    if stage != self.m_nstages - 1: # bwdxbwd#2: When the chk and mb are the same, execute in the order of stage. bwd(self.m_nstages - 1, mb, chk) -> ... -> bwd(0, mb, chk)
                        src2_id = self.get_event_id(EventType.BWD, stage+1, mb, chk)
                        dest2_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                        self.add_edge(src2_id, dest2_id)
                    
                    #with above two rules, within the same chk, execute bwd(self.m_nstages - 1, 0, chk) firstly, execute bwd(0, self.m_nmicrobatches - 1, chk) lastly
                    
                    if stage == self.m_nstages - 1: # fwdxbwd#1: describe the boundary between fwd and bwd, for each mb completing fwd before starting bwd. fwd(self.m_nstages - 1, mb, chk) -> bwd(self.m_nstages - 1, mb, chk)
                        src3_id = self.get_event_id(EventType.FWD, self.m_nstages - 1, mb, chk)
                        dest3_id = self.get_event_id(EventType.BWD, self.m_nstages - 1, mb, chk)
                        self.add_edge(src3_id, dest3_id)

                    # inter-chunk dependence
                    if chk != 0 :
                        if stage == self.m_nstages - 1:
                            src3_id = self.get_event_id(EventType.BWD, 0, mb, chk - 1)
                            dest3_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                            self.add_edge(src3_id, dest3_id)

        #wgt
        for stage in range(self.m_nstages):
            for mb in range(self.m_nmicrobatches):
                for chk in range(self.m_nchunks):
                    if mb != self.m_nmicrobatches - 1: # wgtxwgt#1: When the chk and stage are the same, execute in the order of mb. wgt(stage, 0, chk) -> wgt(stage, self.m_nmicrobatches - 1, chk)
                        src1_id = self.get_event_id(EventType.WGT, stage, mb, chk)
                        dest1_id = self.get_event_id(EventType.WGT, stage, mb+1, chk)
                        self.add_edge(src1_id, dest1_id)

                    # bwdxwgt#1: for each stage&mb completing bwd before starting wgt. bwd(stage, mb, chk) -> wgt(stage, mb, chk)
                    src2_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                    dest2_id = self.get_event_id(EventType.WGT, stage, mb, chk)
                    self.add_edge(src2_id, dest2_id)