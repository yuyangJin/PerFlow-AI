'''
@module pipeline parallel graph
'''

from ...core import EventType, FwdBwdEvent, NoneTimestamp, NoneMem, Trace

from dataclasses import dataclass

'''
PipeCostConfig
'''
@dataclass
class PipeCostConfig:
    fwd_time: int = 1
    bwd_time: int = 1
    wgt_time: int = 1
    comm_time: int = 1
    fwd_mem: float = 2.0
    bwd_mem: float = -2.0
    wgt_mem: float = 0.0

    def __post_init__(self):
        # int for balanced partition, list for imbalanced partition
        assert isinstance(self.fwd_time, (int, list)) 
        assert isinstance(self.bwd_time, (int, list)) 
        assert isinstance(self.wgt_time, (int, list)) 
        assert isinstance(self.comm_time, (int, list))

        # At least, the format and shape of fwd_time and bwd_time should be the same
        assert type(self.fwd_time) == type(self.bwd_time)

        if isinstance(self.fwd_time, list):
            assert len(self.fwd_time) == len(self.bwd_time)
        
        # Check memory foorprint

        assert isinstance(self.fwd_mem, (float, list)) 
        assert isinstance(self.bwd_mem, (float, list)) 
        assert isinstance(self.wgt_mem, (float, list)) 

        assert type(self.fwd_mem) == type(self.bwd_mem)

        if isinstance(self.fwd_mem, float):
            assert self.fwd_mem + self.bwd_mem + self.wgt_mem == 0

        if isinstance(self.fwd_mem, list):
            assert len(self.fwd_mem) == len(self.bwd_mem)

'''
@class PipeOffloadConfig
'''
@dataclass
class PipeOffloadConfig:
    offload_ratio: float = 0.0

    def __post_init__(self):
        assert isinstance(self.offload_ratio, (float, list))
        if isinstance(self.offload_ratio, float):
            assert self.offload_ratio >= 0.0 and  self.offload_ratio <= 1.0


'''
@class PipeRecomputeConfig
'''
@dataclass
class PipeRecomputeConfig:
    recompute_mask: list = None

    def __post_init__(self):
        assert isinstance(self.recompute_mask, list)
        if isinstance(self.recompute_mask, list):
            e_len = 0
            for i in range(len(self.recompute_mask)):
                e = self.recompute_mask[i]
                assert isinstance(e, (int, list))
                if isinstance(e, int):
                    assert e in [0,1]
                # Must be list
                else: 
                    # Check the length of each list element
                    if i == 0:
                        e_len = len(e)
                    else:
                        assert e_len == len(e)
                    for ee in e:
                        assert isinstance(ee, int) # Must be int
                        assert ee in [0,1]

'''
@class PPGraph
A PPGraph is a pipeline trace.
An event is a node in the PPGraph, and it must be FwdBwdEvent.
'''
class PPGraph(Trace):
    def __init__(self, nstages, nmicrobatches, nchunks, cost_config=None, offload_config=None):
        ''' 
        Basic information for pipeline parallelism 
        '''
        self.m_nstages = nstages
        self.m_nmicrobatches = nmicrobatches
        self.m_nchunks = nchunks
        self.m_cost_config = cost_config
        
        # Check whether the cost config (type: PipeCostConfig) is valid
        # Check fwd time only is sufficient, because the lengths of fwd, bwd, and wgt time have been check
        if cost_config != None and isinstance(cost_config.fwd_time, list):
            assert nstages == len(cost_config.fwd_time)
        
        if cost_config != None:
            if isinstance(cost_config.fwd_time, int):
                assert isinstance(cost_config.bwd_time, int)
            elif isinstance(cost_config.fwd_time, list):
                assert isinstance(cost_config.bwd_time, list)
                assert self.m_nstages == len(cost_config.fwd_time) == len(cost_config.bwd_time)
                
            if isinstance(cost_config.fwd_mem, list):
                assert len(cost_config.fwd_time) == len(cost_config.fwd_mem) == len(cost_config.bwd_mem)

        # Offloading
        if offload_config != None:
            self.m_offload == True
        else:
            self.m_offload = False
        self.m_offload_config = offload_config


        '''
        int m_nnodes
        the number of nodes
        '''
        self.m_nnodes = 0

        '''
        Dict<int, FwdBwdEvent> m_nodes
        The first key is event_id, the value is the corresponding event.
        '''
        self.m_nodes = dict()

        '''
        Dict<int, List<int>> m_out_edges/m_in_edges
        m_out_edges: The first key is src id, the value is the list of dst id.
        m_in_edges: The first key is dst id, the value is the list of src id.
        '''
        self.m_in_edges = dict()
        self.m_out_edges = dict()

        '''
        '''
        self.event_types = [EventType.FWD, EventType.BWD]

    def get_event_id(self, event_type, stage_id, microbatch_id, chunk_id = 0):
        '''
        Calculate the node id based on the stage, microbatch, and chunk
        '''
        return (event_type, stage_id, microbatch_id, chunk_id)

    def get_event_types(self):
        return self.event_types

    def get_event_name(self, event_type, microbatch_id, chunk_id):
        if event_type == EventType.FWD:
            event_type_str = 'F'
        elif event_type == EventType.BWD:
            event_type_str = 'B'
        elif event_type == EventType.WGT:
            event_type_str = 'W'
        else:
            raise TypeError("Event type must be FWD/BWD/WGT")
        return event_type_str + ':' + str(microbatch_id) + '-' + str(chunk_id)

    def get_nstages(self):
        return self.m_nstages

    def get_nchunks(self):
        return self.m_nchunks

    def get_nmicrobatches(self):
        return self.m_nmicrobatches

    def add_node(self, event_type, stage_id, microbatch_id, duration, chunk_id = 0, mem = 0):
        '''
        Get event id
        '''
        event_id = self.get_event_id(event_type, stage_id, microbatch_id, chunk_id)

        '''
        Get event num
        '''
        self.m_nnodes = self.m_nnodes + 1
        event_num = self.m_nnodes
        
        '''
        Get event name
        '''
        event_name = self.get_event_name(event_type, microbatch_id, chunk_id)
        
        '''
        Create a new fwd/bwd event
        '''
        fwdbwd_event = FwdBwdEvent(id = event_num, 
                            type = event_type, 
                            name = event_name, 
                            timestamp = NoneTimestamp, 
                            duration = duration, 
                            stage_id = stage_id, 
                            microbatch_id = microbatch_id, 
                            chunk_id = chunk_id,
                            mem = mem)
        
        '''
        Insert the created event into the node list
        '''
        self.m_nodes[event_id] = fwdbwd_event

        return id

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
                            # Duration
                            if isinstance(self.m_cost_config.fwd_time, int): #  global level: every event share the same duration and memory
                                duration = self.m_cost_config.fwd_time
                            elif isinstance(self.m_cost_config.fwd_time[stage], int): # stage level: every event in the same stage share the same duration and memory
                                duration = self.m_cost_config.fwd_time[stage]
                            elif isinstance(self.m_cost_config.fwd_time[stage][chk], int): # chunk level: every event in the same chunk share the same duration and memory
                                duration = self.m_cost_config.fwd_time[stage][chk]
                            # Memory
                            if isinstance(self.m_cost_config.fwd_mem, float):
                                mem = self.m_cost_config.fwd_mem
                            elif isinstance(self.m_cost_config.fwd_mem[stage], float):
                                mem = self.m_cost_config.fwd_mem[stage]
                            elif isinstance(self.m_cost_config.fwd_mem[stage][chk], float):
                                mem = self.m_cost_config.fwd_mem[stage][chk]
                        elif type == EventType.BWD:
                            # Duration
                            if isinstance(self.m_cost_config.bwd_time, int):
                                duration = self.m_cost_config.bwd_time
                            elif isinstance(self.m_cost_config.bwd_time[stage], int):
                                duration = self.m_cost_config.bwd_time[stage]
                            elif isinstance(self.m_cost_config.bwd_time[stage][chk], int):
                                duration = self.m_cost_config.bwd_time[stage][chk]
                            # Memory
                            if isinstance(self.m_cost_config.bwd_mem, float):
                                mem = self.m_cost_config.bwd_mem
                            elif isinstance(self.m_cost_config.bwd_mem[stage], float):
                                mem = self.m_cost_config.bwd_mem[stage]
                            elif isinstance(self.m_cost_config.bwd_mem[stage][chk], float):
                                mem = self.m_cost_config.bwd_mem[stage][chk]
                        elif type == EventType.WGT:
                            continue
                        else:
                            assert False
                        
                        self.add_node(event_type = type, 
                                      stage_id = stage, 
                                      microbatch_id = mb, 
                                      duration = duration, 
                                      chunk_id = chk, 
                                      mem = mem)

    def add_edge(self, src_id, dst_id):
        '''
        Add out edges
        '''
        if src_id not in self.m_out_edges.keys():
            self.m_out_edges[src_id] = set()
        self.m_out_edges[src_id].add(dst_id)

        '''
        Add in edges
        '''
        if dst_id not in self.m_in_edges.keys():
            self.m_in_edges[dst_id] = set()
        self.m_in_edges[dst_id].add(src_id)

    def get_nodes(self):
        return self.m_nodes

    def get_in_edges(self):
        return self.m_in_edges

    def get_out_edges(self):
        return self.m_out_edges

    def check(self):
        for src_id in self.m_out_edges:
            for dst_id in self.m_out_edges[src_id]:
                assert dst_id in self.m_nodes.keys(), f"node {dst_id} not found"
        for dst_id in self.m_in_edges:
            for src_id in self.m_in_edges[dst_id]:
                assert src_id in self.m_nodes.keys(), f"node {src_id} not found"

    def output(self):
        print(self.m_nodes)