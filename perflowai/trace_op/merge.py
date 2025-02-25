from perflowai.parallel.pipeline_parallel.ppgraph import PPGraph

class Merge():
    def __init__(self):
        pass
    
    def merge_pp(self, graph_list, edge_list):
        nstages = graph_list[0].m_nstages
        nmicrobatches = graph_list[0].m_nmicrobatches
        nchunks = graph_list[0].m_nchunks
        event_types = graph_list[0].event_types
        schedule_type = None
        if hasattr(graph_list[0], 'schedule_type'):
            g.schedule_type = graph_list[0].schedule_type

        for graph in graph_list:
            assert nstages == graph.m_nstages
            assert nmicrobatches == graph.m_nmicrobatches
            assert nchunks == graph.m_nchunks
            assert event_types == graph.event_types
            if not schedule_type == None:
                assert schedule_type == graph.schedule_type
        
        g = PPGraph(nstages, nmicrobatches, nchunks)
        g.event_types = event_types
        if not schedule_type == None:
            g.schedule_type = schedule_type
        
        for graph in graph_list:
            for eventid, event in graph.get_nodes().items():
                assert eventid not in g.m_nodes.keys()
                g.m_nodes[eventid] = event
                if eventid in graph.m_in_edges:
                    g.m_in_edges[eventid] = graph.m_in_edges[eventid]
                if eventid in graph.m_out_edges:
                    g.m_out_edges[eventid] = graph.m_out_edges[eventid]

        for edge in edge_list:
            g.add_edge(edge[0], edge[1])
        
        return g


    def merge(self, graph_list, edge_list = []):
        assert(len(graph_list) > 0)
        graph_type = type(graph_list[0])
        for graph in graph_list:
            assert(type(graph) == graph_type)
        
        if graph_type == PPGraph:
            return self.merge_pp(graph_list, edge_list)
        else:
            raise TypeError("TP&DP not support now")