'''
@module trace merger
'''

from ..parallel.pipeline_parallel import PPGraph
from ..workflow import FlowNode

import copy

# derived from FlowNode.
'''
@class Merge
A graph Merger
'''
class Merge(FlowNode):
    def __init__(self):
        pass
    
    def merge_pp(self, graph_list, edge_list):
        '''
        0. SETUP VARIABLES
        '''
        # get basic information
        nstages = graph_list[0].m_nstages
        nmicrobatches = graph_list[0].m_nmicrobatches
        nchunks = graph_list[0].m_nchunks
        event_types = copy.deepcopy(graph_list[0].event_types)
        # schedule_type only exist in zbgraph, should use hasattr check
        schedule_type = None
        if hasattr(graph_list[0], 'schedule_type'):
            schedule_type = copy.deepcopy(graph_list[0].schedule_type)

        '''
            1 CHECK INPUT
        '''
        # Ensure that the parameters of all input graphs are consistent.
        for graph in graph_list:
            assert nstages == graph.m_nstages
            assert nmicrobatches == graph.m_nmicrobatches
            assert nchunks == graph.m_nchunks
            assert event_types == graph.event_types
            if not schedule_type == None:
                assert schedule_type == graph.schedule_type
        
        '''
            2 BUILD EMPTY SUBGRAPH
        '''
        # Create a graph with both nodes and edges being empty, inheriting the parameters of the original graph.
        g = PPGraph(nstages, nmicrobatches, nchunks)
        g.event_types = event_types
        if not schedule_type == None:
            g.schedule_type = schedule_type
        
        '''
            3 MERGE NODES AND EDGES
        '''
        # Iterate through the graphs in graph._list
        for graph in graph_list:
            # Iterate through the nodes in graph, also get its id
            for eventid, event in graph.get_nodes().items():
                '''
                    3.1 MERGE NODES
                '''
                # Ensure that there are no duplicate nodes in the graphs to be merged
                assert eventid not in g.m_nodes.keys()

                g.m_nodes[eventid] = copy.deepcopy(event)

                '''
                    3.2 MERGE EDGES
                '''
                #If the node corresponding to eventid has in_edges, copy them to the new graph.
                if eventid in graph.m_in_edges:
                    g.m_in_edges[eventid] = copy.deepcopy(graph.m_in_edges[eventid])
                #If the node corresponding to eventid has out_edges, copy them to the new graph.
                if eventid in graph.m_out_edges:
                    g.m_out_edges[eventid] = copy.deepcopy(graph.m_out_edges[eventid])

        '''
            4 ADD EDGES PROVIDED BY USERS
        '''
        for edge in edge_list:
            g.add_edge(edge[0], edge[1])
        
        return g


    def merge(self, graph_list, edge_list = []):
        # Check the input. At least one graph must be provided, and the type of all graphs must be the same.
        assert(len(graph_list) > 0)
        graph_type = type(graph_list[0])
        for graph in graph_list:
            assert(type(graph) == graph_type)
        
        # Classify and process based on the input type.
        if graph_type == PPGraph:
            return self.merge_pp(graph_list, edge_list)
        else:
            raise TypeError("TP&DP not support now")