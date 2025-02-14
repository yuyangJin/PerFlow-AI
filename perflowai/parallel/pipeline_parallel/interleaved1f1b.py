from .ppgraph import PPGraph
from ...core.event import EventType

'''
@class Interleaved1F1BGraph
An Interleaved1F1BGraph is a pipeline trace for Interleaved 1F1B.
'''
class Interleaved1F1BGraph(PPGraph):
    def __init__(self, nstages, nmicrobatches, nchunks, cost_config=None):
        super().__init__(nstages, nmicrobatches, nchunks, cost_config)

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

                        self.add_edge(src_id, dest_id)
                        # print_dep(0, stage, mb, chk, 0, stage, next_fwd_mb, next_fwd_chk)

                        # Inter-stage F dependence (the last stage do not need)
                        if stage != self.m_nstages - 1: 
                            src_id = cur_fwd_graph_id
                            dest_id = self.get_event_id(EventType.FWD, stage+1, mb, chk)
                            self.add_edge(src_id, dest_id)
                            # print_dep(0, stage, mb, chk, 0, stage+1, mb, chk)
                    
                    ### Steady 1F1B 
                    
                    cur_bwd_graph_id = self.get_event_id(EventType.BWD, stage, mb, chk)
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
                                dest_id = self.get_event_id(EventType.BWD, stage, bwd_mb, bwd_chk)
                                self.add_edge(src_id, dest_id)
                                # print_dep(1, stage, mb, chk, 1, stage, bwd_mb, bwd_chk)

                    ### Tails with no F, only B
                    else:
                        if cur_bwd_minib_id < self.m_nmicrobatches * self.m_nchunks - 1:

                            src_id = cur_bwd_graph_id
                            bwd_mb, bwd_chk = self.__compute_mb_and_chk(cur_bwd_minib_id+1)
                            dest_id = self.get_event_id(EventType.BWD, stage, bwd_mb, bwd_chk)
                            self.add_edge(src_id, dest_id)
                            # print_dep(1, stage, mb, chk, 1, stage, bwd_mb, bwd_chk)
                    
                    ### Inter-stage B dependence for every B nodes (the first stage do not need)
                    if stage != 0: 
                        src_id = cur_bwd_graph_id
                        dest_id = self.get_event_id(EventType.BWD, stage-1, mb, chk)
                        self.add_edge(src_id, dest_id)
                        # print_dep(0, stage, mb, chk, 0, stage-1, mb, chk)

