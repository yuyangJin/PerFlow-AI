'''
module trace converter
'''

from ..workflow import FlowNode
from ..core.trace import PPTrace
from ..parallel.pipeline_parallel import PPGraph, GPipeGraph, PipeDreamGraph, Interleaved1F1BGraph, ZeroBubbleGraph
from ..simulator.pp_simulator import PipeType


'''
@class TraceConverter
Convert trace to ppgraph
'''
class TraceConverter(FlowNode):
    def __init__(self, input_trace, from_type = None, to_type = None):
        super().__init()
        self.m_input_trace = input_graph
        self.m_from_type = from_type
        self.m_to_type = to_type

    '''
    @func convert from a type to another type
    TODO: only support PPTrace to corresponding Pipeline Graph
    '''
    def Convert(self):

        '''
        0. FIRSTLY, BUILD A NEW PPGraph
        '''

        # Get basic infos of the input PPTrace
        n_stages = self.m_input_trace.get_nstages()
        n_microbatches = self.m_input_trace.get_nmicrobatches()
        n_chunks = self.m_input_trace.get_nchunks()

        g = None

        # Only support PPTrace to corresponding Pipeline Graph
        if isinstance(self.m_input_trace, PPTrace) and self.m_from_type == None:

            if self.m_to_type == PipeType.GPipe:
                g = GPipeGraph(n_stages, n_microbatches, n_chunks, 
                            cost_config = PipeCostConfig(
                                fwd_time = 1, 
                                bwd_time = 1,
                                wgt_time = 1
                            )) # The cost of FWD/BWD/WGT will be reset 
            elif self.m_to_type == PipeType.PipeDream:
                g = PipeDreamGraph(n_stages, n_microbatches, n_chunks, 
                            cost_config = PipeCostConfig(
                                fwd_time = 1, 
                                bwd_time = 1,
                                wgt_time = 1
                            )) # The cost of FWD/BWD/WGT will be reset 
            elif self.m_to_type == PipeType.Interleaved1F1B:
                g = Interleaved1F1BGraph(n_stages, n_microbatches, n_chunks, 
                            cost_config = PipeCostConfig(
                                fwd_time = 1, 
                                bwd_time = 1,
                                wgt_time = 1
                            )) # The cost of FWD/BWD/WGT will be reset 
            elif self.m_to_type == PipeType.ZeroBubble:
                g = ZeroBubbleGraph(n_stages, n_microbatches, n_chunks, 
                            cost_config = PipeCostConfig(
                                fwd_time = 1, 
                                bwd_time = 1,
                                wgt_time = 1
                            )) # The cost of FWD/BWD/WGT will be reset 
        '''
        0. COPY THE DURATION OF EACH EVENT IN THE INPUT PPTrace TO THE CORRESPONDING GRAPH IN THE OUTPUT PPGraph
        '''

        # Only support the situation: n_devs == n_stages
        n_devs = self.m_input_trace.get_ndevs()
        assert n_devs == n_stages

        # Traverse all stages
        for stage_id in range(n_stages):
            events = self.m_input_trace.get_events(stage_id)
            # Traverse all events
            for event in events:
                # Get the corresponding event in PPGraph of the event in the input PPTrace 
                microbatch_id = event.get_microbatch_id()
                chunk_id = event.get_chunk_id()
                corresponding_event = g.get_event(event_type = event.get_type(), , microbatch_id, chunk_id,
                                          stage_id = stage_id,
                                          microbatch_id = microbatch_id,
                                          chunk_id = chunk_id
                                          duration,
                                            )
                # Reset the duration/cost of corresponding event
                duration = event.get_duration()
                corresponding_event.set_duration(duration)


        return g

    def run(self):
        self.m_outputs = [self.Convert()]