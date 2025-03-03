'''
@module trace filter
'''

from ..parallel.pipeline_parallel import PPGraph, GPipeGraph, PipeDreamGraph, Interleaved1F1BGraph, ZeroBubbleGraph
from ..workflow import FlowNode

import copy

# derived from FlowNode.
'''
@class Filter
A graph Filter
'''
class Filter(FlowNode):
    def __init__(self):
        pass
    
    '''
        This function is used to filter out a subgraph from a given graph.
        For nodes, it retains only those nodes whose stage_id is in the stage_list.
        For edges, it retains only those edges where both endpoints have a stage_id that is in the stage_list.
        For other parameters, keep same with the original graph.
    '''
    def filter_pp(self, graph, stage_list):
        '''
            0. CHECK INPUT
        '''
        # Check the input.  At least one stage must be provided, and the values of all stages must within 0 ~ graph.m_nstages-1
        assert len(stage_list) > 0
        assert max(stage_list) < graph.m_nstages
        assert min(stage_list) >= 0
        
        '''
            1 BUILD EMPTY SUBGRAPH
        '''
        # Create a graph with both nodes and edges being empty, inheriting the parameters of the original graph.
        g = PPGraph(graph.m_nstages, graph.m_nmicrobatches, graph.m_nchunks)
        g.event_types = copy.deepcopy(graph.event_types)
        if hasattr(graph, 'schedule_type'):
            g.schedule_type = copy.deepcopy(graph.schedule_type)

        '''
            2 FILTER NODES AND EDGES
        '''
        # get nodes from the original graph. can only be accessed.
        events = graph.get_nodes()

        for stage in range(graph.m_nstages):
            # Extract the nodes whose stage attribute is in the stage_list.
            if stage in stage_list: 
                # Iterate through all the nodes for this particular stage in the following three lines of code.
                for type in graph.event_types:
                    for mb in range(graph.m_nmicrobatches):
                        for chk in range(graph.m_nchunks):
                            '''
                                2.1 FILTER NODES
                            '''
                            # Retrieve the corresponding ID based on the type, stage, mb, and chk attributes.
                            src_id = graph.get_event_id(type, stage, mb, chk)

                            # Based on the ID, migrate the nodes from the original graph to the subgraph.
                            g.m_nodes[src_id] = copy.deepcopy(events[src_id])

                            '''
                                2.2 FILTER EDGES
                            '''
                            # If the node has edges in the original graph, then it is necessary to consider whether the stage of the corresponding node is in the stage_list.
                            if src_id in graph.m_out_edges.keys():
                                for dst_id in graph.m_out_edges[src_id]:
                                    dst_event = events[dst_id]
                                    if dst_event.m_stage_id in stage_list:
                                        g.add_edge(src_id, dst_id)
        return g
    
    def filter(self, graph, stage_list):
        # Classify and process based on the input type.
        if isinstance(graph, (PPGraph, GPipeGraph, PipeDreamGraph, Interleaved1F1BGraph, ZeroBubbleGraph)):
            return self.filter_pp(graph, stage_list)
        else:
            raise TypeError("TP&DP not support now")