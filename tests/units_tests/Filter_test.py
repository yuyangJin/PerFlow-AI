'''
test Filter class
'''
import pytest

from perflowai.parallel.pipeline_parallel import PipeCostConfig, GPipeGraph
from perflowai.core import NoneTimestamp, EventType
from perflowai.trace_op import Filter

def test_Filter():
    '''
        0 BUILD A GRAPH
    '''
    graph = GPipeGraph(4, 10, 1, cost_config = PipeCostConfig(
        fwd_time = 1000,
        bwd_time = 2000,
        wgt_time = 3000,
    ))
    graph.build_graph()

    '''
        1 FILTER STAGES [2,3] FROM ORIGINAL GRAPH AS SUBGRAPH
    '''
    fil = Filter()
    target_stage_list = [2,3]
    subgraph = fil.filter(graph, target_stage_list)

    '''
        2 CHECK NODES IN SUBGRAPH
    '''
    # For all nodes in the original graph whose stageid in target_stage_list, they must exist in the subgraph.
    for eventid, event in graph.get_nodes().items():
        if event.m_stage_id in target_stage_list:
            assert event == subgraph.get_nodes()[eventid]
    # All nodes in the subgraph must exist in the original graph, and their stageid must be in the target_stage_list.
    for eventid, event in subgraph.get_nodes().items():
        assert event == graph.get_nodes()[eventid]
        assert event.m_stage_id in target_stage_list
    
    '''
        3 CHECK EDGES IN SUBGRAPH
    '''
    # For all edges in the original graph where the stageid of both endpoints belongs to target_stage_list, they must exist in the subgraph.
    for src_id in graph.m_out_edges:
        src_event = graph.get_nodes()[src_id]
        if src_event.m_stage_id in target_stage_list:
            for dst_id in graph.m_out_edges[src_id]:
                dst_event = graph.get_nodes()[dst_id]
                if dst_event.m_stage_id in target_stage_list:
                    assert dst_id in subgraph.m_out_edges[src_id]
    for src_id in graph.m_in_edges:
        src_event = graph.get_nodes()[src_id]
        if src_event.m_stage_id in target_stage_list:       
            for dst_id in graph.m_in_edges[src_id]:
                dst_event = graph.get_nodes()[dst_id]
                if dst_event.m_stage_id in target_stage_list:
                    assert dst_id in subgraph.m_in_edges[src_id]

    # All edges in the subgraph must exist in the original graph
    for src_id in subgraph.m_out_edges:
        for dst_id in graph.m_out_edges[src_id]:
            assert dst_id in graph.m_out_edges[src_id]
    for src_id in subgraph.m_in_edges:
        for dst_id in subgraph.m_in_edges[src_id]:
            assert dst_id in graph.m_in_edges[src_id]
