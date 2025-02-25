'''
test Filter class
'''
import pytest

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.gpipe import GPipeGraph
from perflowai.core.event import NoneTimestamp, EventType
from perflowai.trace_op.filter import Filter

def test_Filter():
    graph = GPipeGraph(4, 10, 1, cost_config = PipeCostConfig(
        fwd_time = 1000,
        bwd_time = 2000,
        wgt_time = 3000,
    ))
    graph.build_graph()

    fil = Filter()
    target_stage_list = [2,3]
    subgraph = fil.filter(graph, target_stage_list)

    #check nodes
    for eventid, event in graph.get_nodes().items():
        if event.m_stage_id in target_stage_list:
            assert event == subgraph.get_nodes()[eventid]

    for eventid, event in subgraph.get_nodes().items():
        assert event == graph.get_nodes()[eventid]
        assert event.m_stage_id in target_stage_list
    
    #check edge
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

    for src_id in subgraph.m_out_edges:
        for dst_id in graph.m_out_edges[src_id]:
            assert dst_id in graph.m_out_edges[src_id]
    for src_id in subgraph.m_in_edges:
        for dst_id in subgraph.m_in_edges[src_id]:
            assert dst_id in graph.m_in_edges[src_id]
