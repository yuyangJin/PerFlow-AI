from perflowai.parallel.pipeline_parallel.ppgraph import PPGraph
from perflowai.parallel.pipeline_parallel.gpipe import GPipeGraph
from perflowai.parallel.pipeline_parallel.pipedream import PipeDreamGraph
from perflowai.parallel.pipeline_parallel.interleaved1f1b import Interleaved1F1BGraph
from perflowai.parallel.pipeline_parallel.zerobubble import ZeroBubbleGraph


class Filter():
    def __init__(self):
        pass
    
    def filter_pp(self, graph, stage_list):
        assert len(stage_list) > 0
        assert max(stage_list) < graph.m_nstages
        assert min(stage_list) >= 0

        g = PPGraph(graph.m_nstages, graph.m_nmicrobatches, graph.m_nchunks)
        g.event_types = graph.event_types
        if hasattr(graph, 'schedule_type'):
            g.schedule_type = graph.schedule_type

        events = graph.get_nodes()

        for stage in range(graph.m_nstages):
            if stage in stage_list:
                for type in graph.event_types:
                    for mb in range(graph.m_nmicrobatches):
                        for chk in range(graph.m_nchunks):
                            src_id = graph.get_event_id(type, stage, mb, chk)

                            g.m_nodes[src_id] = events[src_id]

                            for dst_id in graph.m_out_edges[src_id]:
                                dst_event = events[dst_id]
                                if dst_event.m_stage_id in stage_list:
                                    g.add_edge(src_id, dst_id)
        return g
    
    def filter(self, graph, stage_list):

        if isinstance(graph, (PPGraph, GPipeGraph, PipeDreamGraph, Interleaved1F1BGraph, ZeroBubbleGraph)):
            return self.filter_pp(graph, stage_list)
        else:
            raise TypeError("TP&DP not support now")