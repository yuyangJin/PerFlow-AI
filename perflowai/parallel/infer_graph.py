'''
@module inference graph builder
'''


from ..core import Tasks, EventType, Trace, Request, RequestEvent, ScheduleEvent, PrefillEvent, DecodeEvent
from typing import Dict, List, Optional

schedule_id = 0

'''
@class InferGraph
A graph representing the inference process of a model.
It contains nodes (tasks) and edges (dependencies between tasks).
'''
class InferGraph(Trace):
    def __init__(self, ndevs: int):
        '''
        Initialize the graph with empty nodes and edges
        '''
        super().__init__(ndevs)

        '''
        Dict<int, Event> m_nodes
        The first key is event_id, the value is the corresponding event.
        It is a dynamic dictionary, event nodes are added to it during the inference process.
        '''
        self.m_nodes = dict()

        '''
        Dict<int, List<int>> m_out_edges/m_in_edges
        m_out_edges: The first key is src id, the value is the list of dst id.
        m_in_edges: The first key is dst id, the value is the list of src id.
        It is a dynamic dictionary, event edges are added to it during the inference process.
        '''
        self.m_in_edges = dict()
        self.m_out_edges = dict()      


        self.event_types = [EventType.REQ, EventType.PRF, EventType.DCD]  

    def get_event_id(self, event_type, stage_id, request_id):
        '''
        Calculate the node id based on the event_type, stage and request id
        '''
        return (event_type, stage_id, request_id)

    def add_request_node(self, req: Request):
        # id = self.get_event_id(EventType.REQ, 0, req.req_id)
        event_name = self.get_event_name(EventType.REQ, req)
        event = RequestEvent(
                    type = EventType.REQ, 
                    name = event_name, 
                    timestamp = req.start_time, 
                    duration = 0,
                    stage_id = 0, 
                    request = req)
        # self.add_cpu_event(event)
        id = event.get_id()
        self.m_nodes[id] = event
        return id

    def add_schedule_node(self, timestamp: int):
        # id = self.get_event_id(EventType.SCHD, 0, tasks.id)
        event_name = self.get_event_name(EventType.SCHD, 0) #FIXME: 0 is a fake number
        event = ScheduleEvent(
                    type = EventType.SCHD, 
                    name = event_name, 
                    timestamp = timestamp, 
                    duration = 0)
        # self.add_cpu_event(event)
        id = event.get_id()
        self.m_nodes[id] = event
        return id

    def add_prefill_node(self, tasks: Tasks, dev_id: int):
        # id = self.get_event_id(EventType.PRF, dev_id, tasks.id)
        event_name = self.get_event_name(EventType.PRF, tasks)
        event = PrefillEvent(
                    type = EventType.PRF, 
                    name = event_name, 
                    timestamp = 0, 
                    duration = 0,
                    dev_id = dev_id, 
                    input_len = tasks.max_input_len,
                    tasks = tasks)
        # self.add_event(dev_id, event)
        id = event.get_id()
        self.m_nodes[id] = event
        return id

    def add_decode_node(self, tasks: Tasks, dev_id: int):
        # id = self.get_event_id(EventType.DCD, dev_id, tasks.id)
        event_name = self.get_event_name(EventType.DCD, tasks)
        event = DecodeEvent(
                    type = EventType.DCD, 
                    name = event_name, 
                    timestamp = 0, 
                    duration = 0,
                    dev_id = dev_id,
                    tasks = tasks)
        # self.add_event(dev_id, event)
        id = event.get_id()
        self.m_nodes[id] = event
        return id

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

    def get_event_name(self, event_type, req = None) -> str:
        if isinstance(req, Request):
            if event_type == EventType.REQ:
                return "R:" + str(req.req_id)
        elif isinstance(req, Tasks):
            if event_type == EventType.PRF:
                return "P:" + str(req.id)
            elif event_type == EventType.DCD:
                return "D:" + str(req.id)
        elif isinstance(req, int):
            if event_type == EventType.SCHD:
                return "S:" + str(req)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    def generate_nodes(self, reqs: List[Request]) -> None:
        # Build the graph based on the requests, 
        for req in reqs:
            self.add_request_node(req)

    def check(self):
        for src_id in self.m_out_edges:
            for dst_id in self.m_out_edges[src_id]:
                assert dst_id in self.m_nodes.keys(), f"node {dst_id} not found"
        for dst_id in self.m_in_edges:
            for src_id in self.m_in_edges[dst_id]:
                assert src_id in self.m_nodes.keys(), f"node {src_id} not found"

    def output(self):
        print(self.m_nodes)