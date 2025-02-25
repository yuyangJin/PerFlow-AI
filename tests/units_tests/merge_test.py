'''
test Filter class
'''
import pytest

from perflowai.parallel.pipeline_parallel.ppgraph import PipeCostConfig
from perflowai.parallel.pipeline_parallel.gpipe import GPipeGraph
from perflowai.core.event import NoneTimestamp, EventType
from perflowai.trace_op.filter import Filter
from perflowai.trace_op.merge import Merge

def test_Filter():
    def check_subgraph(graph, subgraph, target_stage_list):
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


    graph = GPipeGraph(4, 10, 1, cost_config = PipeCostConfig(
        fwd_time = 1000,
        bwd_time = 2000,
        wgt_time = 3000,
    ))
    graph.build_graph()

    fil = Filter()
    subgraph0 = fil.filter(graph, [0, 1])
    subgraph1 = fil.filter(graph, [2, 3])

    mer = Merge()
    mgraph = mer.merge([subgraph0,subgraph1], []) #merge graph

    check_subgraph(mgraph, subgraph0, [0, 1])
    check_subgraph(mgraph, subgraph1, [2, 3])

    res_edge_list = []
    for src_id in graph.m_out_edges:
        if not src_id in mgraph.m_out_edges.keys():
            for dst_id in graph.m_out_edges[src_id]:
                res_edge_list.append([src_id, dst_id])
        else:
            for dst_id in graph.m_out_edges[src_id]:
                if not dst_id in mgraph.m_out_edges[src_id]:
                    res_edge_list.append([src_id, dst_id])
    mgraph = mer.merge([mgraph], res_edge_list) #add edge

    check_subgraph(mgraph, graph, [0, 1, 2, 3])

