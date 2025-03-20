'''
@module Interleaved 1F1B
'''

from .ppgraph import PPGraph
from ...core import EventType, FwdBwdEvent, OffReLoadEvent, NoneTimestamp, ResourceType

'''
@class Interleaved1F1BGraph
An Interleaved1F1BGraph is a pipeline trace for Interleaved 1F1B.
'''
class Interleaved1F1BGraph(PPGraph):
    def __init__(self, nstages, nmicrobatches, nchunks, cost_config=None, offload_config=None, recompute_config=None):
        super().__init__(nstages, nmicrobatches, nchunks, cost_config = cost_config, offload_config = offload_config, recompute_config = recompute_config)

    '''
    mb, chk -> minibatch id
    '''
    def __compute_minib_id(self, stage, mb, chk):
        # n_stage-sized mbb
        if (mb < (self.m_nmicrobatches // self.m_nstages) * self.m_nstages) or (self.m_nmicrobatches % self.m_nstages == 0):
            return (mb // self.m_nstages) * (self.m_nstages * self.m_nchunks) + (self.m_nstages) * chk + (mb % self.m_nstages)
        # last m_nchunks mbbs (non-n_stage-sized mbb)
        else:
            step = self.m_nmicrobatches % self.m_nstages
            return (self.m_nmicrobatches // self.m_nstages) * self.m_nstages * self.m_nchunks + (step * chk) + (mb % self.m_nstages)
    '''
    minibatch id -> mb, chk 
    '''
    def __compute_mb_and_chk(self, minib_id):
        last_mbb_start_minib_id = (self.m_nmicrobatches // self.m_nstages) * (self.m_nstages * self.m_nchunks)
        # n_stage-sized mbb
        if minib_id < last_mbb_start_minib_id or (self.m_nmicrobatches % self.m_nstages == 0):
            mb = ((minib_id // self.m_nstages) // self.m_nchunks) * self.m_nstages + minib_id % self.m_nstages
            chk = (minib_id // self.m_nstages) % self.m_nchunks
        # last m_nchunks mbbs (non-n_stage-sized mbb)
        else:
            mbb_start = (self.m_nmicrobatches // self.m_nstages) * self.m_nstages 
            step = self.m_nmicrobatches % self.m_nstages
            mb = mbb_start + (minib_id - last_mbb_start_minib_id) % step 
            chk = (minib_id - last_mbb_start_minib_id) // step 
        return mb, chk



    '''
    Build the graph for Interleaved 1F1B 
    '''
    def build_graph(self):

        ###############################################
        #  --mbb-- mbb is minibatch block -- continuous minibatches with the same chk
        # |     mb|   chk
        # |     | |    |
        # F0-0 F1-0 F0-1 F1-1 F2-0 F3-0 F2-1 B0-0 F3-1 B1-0      B0-1      B1-1 B2-0 B3-0 B2-1 B3-1
        #      F0-1 F1-0 F0-1 F1-1 F2-0 B0-0 F3-0 B1-0 F2-1 B0-1 F3-1 B1-1 B2-0 B3-0 B2-1 B3-1
        ###############################################
    
        '''
        Add nodes/events
        '''
        self.generate_nodes()

        '''
        Add data dependence / edges
        There are two id:
            1. get_event_id(...) returns graph id, for adding dependence between nodes
            2. __compute_minib_id(...) returns minibatch id, for identify the next node (mb, chk)
        '''
        for stage in range(self.m_nstages):
            n_warmup_minibs = (self.m_nstages * (self.m_nchunks - 1) + (self.m_nstages - stage) * 2 - 2)
            for mb in range(self.m_nmicrobatches):
                for chk in range(self.m_nchunks):

                    cur_fwd_graph_id = self.get_event_id(EventType.FWD, stage, mb, chk)
                    cur_fwd_minib_id = self.__compute_minib_id(stage, mb, chk)

                    ### Warmup F

                    if cur_fwd_minib_id < n_warmup_minibs:
                        # Intra-stage F dependence
                        next_fwd_mb, next_fwd_chk = self.__compute_mb_and_chk(cur_fwd_minib_id+1)
                        src_id = cur_fwd_graph_id
                        dest_id = self.get_event_id(EventType.FWD, stage, next_fwd_mb, next_fwd_chk)
                        if 0 <= next_fwd_mb < self.m_nmicrobatches and 0 <= next_fwd_chk < self.m_nchunks:
                            self.add_edge(src_id, dest_id)
                        # print_dep(0, stage, mb, chk, 0, stage, next_fwd_mb, next_fwd_chk)

                    # Inter-stage F dependence (the last stage do not need)
                    if stage != self.m_nstages - 1: 
                        src_id = cur_fwd_graph_id
                        dest_id = self.get_event_id(EventType.FWD, stage+1, mb, chk)
                        self.add_edge(src_id, dest_id)
                        # print_dep(0, stage, mb, chk, 0, stage+1, mb, chk)
                    
                    ### Steady 1F1B 
                    
                    cur_bwd_graph_id = self.get_event_id(EventType.BWD, stage, mb, self.m_nchunks - chk - 1)
                    cur_bwd_minib_id = self.__compute_minib_id(stage, mb, chk)
                    
                    if cur_bwd_minib_id < (self.m_nmicrobatches * self.m_nchunks - n_warmup_minibs):

                        # Previous F -> current B
                        fwd1_minb_id = cur_bwd_minib_id + n_warmup_minibs
                        fwd1_mb, fwd1_chk = self.__compute_mb_and_chk(fwd1_minb_id)

                        src_id = self.get_event_id(EventType.FWD, stage, fwd1_mb, fwd1_chk)
                        dest_id = cur_bwd_graph_id
                        self.add_edge(src_id, dest_id)
                        # print_dep(0, stage, fwd1_mb, fwd1_chk, 1, stage, mb, chk)

                        # Current B -> next F
                        if cur_bwd_minib_id != self.m_nmicrobatches * self.m_nchunks - n_warmup_minibs - 1:
                            
                            src_id = cur_bwd_graph_id
                            fwd2_mb, fwd2_chk = self.__compute_mb_and_chk(fwd1_minb_id+1)
                            dest_id = self.get_event_id(EventType.FWD, stage, fwd2_mb, fwd2_chk)
                            self.add_edge(src_id, dest_id)
                            # print_dep(1, stage, mb, chk, 0, stage,fwd2_mb, fwd2_chk)
                        
                        # Current B -> next B (last 1F1B, no next F) 
                        else:
                            # Make sure it is not the last B (last B has no next B)
                            if cur_bwd_minib_id < self.m_nmicrobatches * self.m_nchunks - 1:
                                print('last 1F1B')
                                src_id = cur_bwd_graph_id
                                bwd_mb, bwd_chk = self.__compute_mb_and_chk(cur_bwd_minib_id+1)
                                dest_id = self.get_event_id(EventType.BWD, stage, bwd_mb, self.m_nchunks - bwd_chk - 1)
                                self.add_edge(src_id, dest_id)
                                # print_dep(1, stage, mb, chk, 1, stage, bwd_mb, bwd_chk)

                    ### Tails with no F, only B
                    else:
                        if cur_bwd_minib_id < self.m_nmicrobatches * self.m_nchunks - 1:

                            src_id = cur_bwd_graph_id
                            bwd_mb, bwd_chk = self.__compute_mb_and_chk(cur_bwd_minib_id+1)
                            dest_id = self.get_event_id(EventType.BWD, stage, bwd_mb, self.m_nchunks - bwd_chk - 1)
                            self.add_edge(src_id, dest_id)
                            # print_dep(1, stage, mb, chk, 1, stage, bwd_mb, bwd_chk)
                    
                    ### Inter-stage B dependence for every B nodes (the first stage do not need)
                    if stage != 0: 
                        src_id = cur_bwd_graph_id
                        dest_id = self.get_event_id(EventType.BWD, stage-1, mb, self.m_nchunks - chk - 1)
                        self.add_edge(src_id, dest_id)
                        # print_dep(0, stage, mb, chk, 0, stage-1, mb, chk)

        '''
            OffReloading
        '''
        if self.m_offload:
            #get basic information
            n_stages = self.get_nstages()
            n_microbatches = self.get_nmicrobatches()
            n_chunks = self.get_nchunks()
            events = self.get_nodes()
            for stage in range(n_stages):
                for mb in range(n_microbatches):
                    for chk in range(n_chunks):
                        #get related fwd & bwd event
                        fwd_event_id = self.get_event_id(EventType.FWD, stage, mb, chk)
                        fwd_event = events[fwd_event_id]
                        bwd_event_id = self.get_event_id(EventType.BWD, stage, mb, chk)
                        bwd_event = events[bwd_event_id]
                        #for stage==nstages - 1 & chunk == nchunks - 1, fwd and bwd excute closely, so there is no need to offload
                        if (not stage == n_stages - 1) or (not chk == n_chunks - 1):
                            '''
                                Offload
                            '''
                            offl_event_id = self.get_event_id(EventType.OFFL, stage, mb, chk)
                            self.add_edge(fwd_event_id, offl_event_id)
                            '''
                                Reload
                            '''
                            rel_event_id = self.get_event_id(EventType.REL, stage, mb, chk)
                            self.add_edge(offl_event_id, rel_event_id)
                            self.add_edge(rel_event_id, bwd_event_id)

                            n_warmup_minibs = (self.m_nstages * (self.m_nchunks - 1) + (self.m_nstages - stage) * 2 - 2)
                            cur_bwd_minib_id = self.__compute_minib_id(stage, mb, n_chunks - chk - 1)
                            if cur_bwd_minib_id < (self.m_nmicrobatches * self.m_nchunks - n_warmup_minibs):
                                pre2_fwd_minib_id = cur_bwd_minib_id + n_warmup_minibs - 1
                                pre2_fwd_mb, pre2_fwd_chk = self.__compute_mb_and_chk(pre2_fwd_minib_id)
                                pre2_fwd_event_id = self.get_event_id(EventType.FWD, stage, pre2_fwd_mb, pre2_fwd_chk)
                                if 0 <= pre2_fwd_mb < self.m_nmicrobatches and 0 <= pre2_fwd_chk < self.m_nchunks:
                                    self.add_edge(pre2_fwd_event_id, rel_event_id)
                            else:
                                pre2_bwd_minib_id = cur_bwd_minib_id - 2
                                pre2_bwd_mb, pre2_bwd_chk = self.__compute_mb_and_chk(pre2_bwd_minib_id)
                                pre2_bwd_event_id = self.get_event_id(EventType.BWD, stage, pre2_bwd_mb, self.m_nchunks - pre2_bwd_chk - 1)
                                if 0 <= pre2_bwd_mb < self.m_nmicrobatches and 0 <= pre2_bwd_chk < self.m_nchunks:
                                    self.add_edge(pre2_bwd_event_id, rel_event_id)