'''
@module inference simulator
'''
from ..simulator import Simulator
from ..model import ModelMemSimulator
from ...core import ModelConfig
from ...core import Tasks, Task, TaskType, Trace, Scheduler, TaskPool, ResourceType, EventType, NoneTimestamp
from collections import deque
from typing import List, Dict, Any
import copy

class InferSimulator(Simulator):
    '''
    @class InferSimulator
    A simulator for the inference process of a model.
    It simulates the inference process of a model by using the provided graph and request.
    # It also provides methods to get the status of the simulator and to get the time taken for each step.
    '''

    def __init__(self, graph, request):
        self.m_graph = graph
        self.m_ndevs = graph.get_ndevs()
        self.m_request = request
        self.m_task_pool = TaskPool()
        self.m_resource_types = [ResourceType.GPU, ResourceType.CPU]

    def get_task_pool(self):
        '''
        Get the task pool
        '''
        return self.m_task_pool

    def _get_pending_request_event_ids(self, req_nodes, current_timestamp):
        '''
        Get the pending request events
        '''

        pending_request_event_ids = []
        for event_id, event in req_nodes.items():
            if event.get_type() == EventType.REQ and event.get_timestamp() <= current_timestamp:
                pending_request_event_ids.append(event_id)

        return pending_request_event_ids

    def _process_requeset_events(self, req_nodes, pending_request_event_ids):
        '''
        Process the pending request events
        '''

        for event_id in pending_request_event_ids:
            event = req_nodes[event_id]
            if event.get_type() == EventType.REQ:
                # Add the request event to the trace
                # trace.add_cpu_event(event)
                event.set_duration(1)
                # Add the prefill of the request to the task pool
                self.m_task_pool.add(Task.from_request(event.get_request()))
        
            req_nodes.pop(event_id)            

    # def _get_next_request_event(self):
    
    def simulate(self, scheduler: Scheduler, model_config: ModelConfig = None):
        '''
        Run the inference simulator
        '''

        req_nodes = copy.deepcopy(self.m_graph.get_nodes())

        nodes = self.m_graph.get_nodes()
        print(nodes)

        if model_config is not None:
            self.m_memsim = ModelMemSimulator(model_config)

        '''
        0. SETUP VARIABLES
        '''

        # Get in edges
        # in_data_deps/out_data_deps can only be accessed,
        # in_data_deps_tmp can be modified for updating current status
        in_data_deps = self.m_graph.get_in_edges()
        out_data_deps = self.m_graph.get_out_edges()
        in_data_deps_tmp = copy.deepcopy(in_data_deps)

        # A queue for events that are READY to be executed 
        ready_event_ids = deque()

        # A list for event that is completed
        completed_event_ids = dict()

        # Store the current timestamp of each resource on each stage
        # current_cpu_timestamps = 0 # replace with current_timestamps[0][ResourceType.CPU]
        current_cpu_timestamp = 0
        current_timestamps = [dict() for _ in range(self.m_ndevs)]
        # Initialize
        for dev_id in range(self.m_ndevs):
            current_stage_timestamps = current_timestamps[dev_id]
            for rsc in self.m_resource_types:
                current_stage_timestamps[rsc] = 0

        # Store the event execution trace
        trace = Trace(self.m_ndevs)


        '''
        1. INITIALIZATION
        '''
        # Get the ready events -- request nodes which timestampe are 0
        
        while len(self.m_task_pool.pool) == 0:
            # Get the pending request events
            pending_request_event_ids = self._get_pending_request_event_ids(req_nodes, current_timestamps[0][ResourceType.CPU])

            # Deal with the pending request events
            self._process_requeset_events(req_nodes, pending_request_event_ids)
            
            current_timestamps[0][ResourceType.CPU] += 1

        # Update timestamp
        current_cpu_timestamp = current_timestamps[0][ResourceType.CPU]

        # Create a schedule event node as the first one
        sched_event_id = self.m_graph.add_schedule_node(0)

        # Add the schedule event to the ready events
        ready_event_ids.append(sched_event_id)

        # TasksSimulator(model_config, hardware_config, )

        '''
        2. SIMULATION
        '''
        while not (len(ready_event_ids) == 0 and len(req_nodes) == 0 and self.m_task_pool.is_empty()):  
            print ("===============================================")

            '''
            Naive verison: could be faster by directly searching for the next event
            '''
            if len(ready_event_ids) == 0:
                print('here we check new request')
                # Get the pending request events
                pending_request_event_ids = self._get_pending_request_event_ids(req_nodes, current_cpu_timestamp)

                # If the task pool has tasks, them deal with the pending request, and add a schedule node
                if not self.m_task_pool.is_empty() or len(pending_request_event_ids) != 0:
                    # Deal with the pending request events
                    self._process_requeset_events(req_nodes, pending_request_event_ids)

                    # Create a schedule event node as the first one
                    sched_event_id = self.m_graph.add_schedule_node(0)

                    # Add the schedule event to the ready events
                    ready_event_ids.append(sched_event_id)

                    continue

                # If there exist pending request events
                elif len(pending_request_event_ids) == 0:
                    # Update current timestamps
                    current_timestamps[0][ResourceType.CPU] += 1

                    # Update timestamp
                    current_cpu_timestamp = current_timestamps[0][ResourceType.CPU]

                    continue

                # else:
                #     # Deal with the pending request events
                #     self._process_requeset_events(req_nodes, pending_request_event_ids)

                #     # Create a schedule event node as the first one
                #     sched_event_id = self.m_graph.add_schedule_node(0)

                #     # Add the schedule event to the ready events
                #     ready_event_ids.append(sched_event_id)


            '''
            2.0 GET A READY EVENT TO EXECUTE/SIMULATE
            '''

            # Choose the first one
            current_event_id = ready_event_ids[0]

            # Get the corresponding node by id
            current_event  = nodes[current_event_id]


            '''
            2.3.1 CALCULATE THE START TIMESTAMP
            '''

            stage = current_event.get_dev_id()
            resource = current_event.get_resource_type()

            start_timestamp = 0
            if current_event_id in in_data_deps.keys():
                # The start timestamp should be after the latest compeletion time of events with in edge dependence
                dep_stimestamps = list(map(lambda x : completed_event_ids[x].get_timestamp() + completed_event_ids[x].get_duration(), in_data_deps[current_event_id]))
                # if len(dep_stimestamps) == 0:
                    # start_timestamp = 0
                # else:
                start_timestamp = max(dep_stimestamps)
                print('dep_stimestamps:', dep_stimestamps)
                # The start timestamp should also be after the latest compeletion time of executed events on current stage/device
                start_timestamp = max(start_timestamp, current_timestamps[stage][resource])
                # Update timestamp
                current_cpu_timestamp = current_timestamps[stage][resource]

            else: # For schedule nodes with no in deps
                current_timestamps[stage][resource] = current_cpu_timestamp

            


            '''
            2.2 EXECUTE THE CHOSEN EVENT
            '''

            '''
            2.2.1 FOR SCHEDULE EVENT NODES, CHECK THE PENDING REQUEST EVENTS, CREATE PREFILL/DECODE EVENTS AND THE NEXT SCHEDULE EVENT
            '''

            # Check if the current event is a schedule event
            if current_event.get_type() == EventType.SCHD:

                current_event.set_duration(1)
                current_event.set_mem(0)

                # Get the pending request events
                pending_request_event_ids = self._get_pending_request_event_ids(req_nodes, current_cpu_timestamp)

                # Deal with the pending request events
                self._process_requeset_events(req_nodes, pending_request_event_ids)

                # # Set the start timestamp of the current event
                # current_event.set_timestamp(start_timestamp)
                
                # Schedule the task
                task_type, task_lists = scheduler.schedule(self.m_task_pool, self.m_ndevs)
                # print(f"Task type: {task_type}, task lists: {task_lists}")

                # # Record the completed event on the trace
                # completed_event_ids[current_event_id] = current_event
                # trace.add_cpu_event(event)

                # Add new prefill/decode events to the ready events
                if task_type != TaskType.EMPTY:
                    # Create a new schedule event node
                    sched_event_id = self.m_graph.add_schedule_node(NoneTimestamp)

                    for dev_id in range(len(task_lists)):
                        task_list = task_lists[dev_id]

                        # Create a new event for the tasks
                        if task_type == TaskType.PREFILL:    
                            event_id = self.m_graph.add_prefill_node(task_list, dev_id)
                            # Add the event to the ready events
                            # ready_event_ids.append(event_id)
                        elif task_type == TaskType.DECODE:
                            event_id = self.m_graph.add_decode_node(task_list, dev_id)
                            # Add the event to the ready events
                            # ready_event_ids.append(event_id)
                        else:
                            assert False, f"Unknown task type: {task_type}"

                        # Add dependence edge
                        self.m_graph.add_edge(current_event_id, event_id) # current schedule event -> prefill/decode event
                        self.m_graph.add_edge(event_id, sched_event_id) # prefill/decode event -> next schedule event

                        # TODO: check reduendency
                        if event_id not in in_data_deps_tmp.keys():
                            in_data_deps_tmp[event_id] = set()
                        in_data_deps_tmp[event_id].add(current_event_id)

                        if sched_event_id not in in_data_deps_tmp.keys():
                            in_data_deps_tmp[sched_event_id] = set()
                        in_data_deps_tmp[sched_event_id].add(event_id)





            # Check if the current event is a prefill/decode event
            elif current_event.get_type() == EventType.PRF or current_event.get_type() == EventType.DCD:
                '''
                2.2.2 FOR PREFILL/DECODE EVENT NODES, EXECUTE THE TASKS, ADD NEW TASKS TO THE TASK POOL
                '''
                

                # Get the duration of the event
                duration = 20

                current_event.set_duration(duration)

                if model_config is not None:
                    kvcache_size = self.m_memsim.kvcache(current_event)
                    print("[kvcache_size] ", kvcache_size)
                    current_event.set_mem(kvcache_size)

                '''
                2.2.3 EXECUTE PREFILL/DECODE EVENTS, PROCESS THE TASKS, GENERATE NEW TASKS TO THE TASKPOOL
                '''
                if current_event.get_type() == EventType.PRF:
                    self.m_task_pool.adds(Tasks.from_prefill_tasks(current_event.get_tasks(), complete_time))
                elif current_event.get_type() == EventType.DCD:
                    self.m_task_pool.adds(Tasks.from_decode_tasks(current_event.get_tasks(), complete_time))


            else:
                assert False, f"Unknown event type: {current_event.get_type()}"



            complete_time = start_timestamp + current_event.get_duration()


            '''
            2.3 UPDATE THE EVENT AND TRACE
            '''
            # Update the timestamp of current event
            current_event.set_timestamp(start_timestamp) 

            # Update the list of completed events, and the timestamps of each stages
            completed_event_ids[current_event_id] = current_event
            current_timestamps[stage][resource] = complete_time
            current_cpu_timestamp = complete_time

            # Update the output trace
            trace.add_event(stage, current_event)


            '''
            2.4 UPDATE THE IN EDGES (DATA DEPENDENCE)
            '''
            # if current_event_id in out_data_deps.keys():
            #     for dest_event_id in out_data_deps[current_event_id]:
            #         in_data_deps_tmp[dest_event_id].remove(current_event_id)
            #         if 0 == len(in_data_deps_tmp[dest_event_id]) and dest_event_id not in completed_event_ids.keys():
            #             ready_event_ids.append(dest_event_id)  

            if current_event_id in out_data_deps.keys():
                for dest_event_id in out_data_deps[current_event_id]:
                    in_data_deps_tmp[dest_event_id].remove(current_event_id)
                    if 0 == len(in_data_deps_tmp[dest_event_id]):
                        ready_event_ids.append(dest_event_id)  

            ready_event_ids.remove(current_event_id)  














            # '''
            # 2.1 DEAL WITH ALL READY REQ EVENTS
            # '''
            # to_remove = []

            # for event_id in ready_event_ids:
            #     event = nodes[event_id]
            #     if event.get_type() == EventType.REQ:
            #         print(f"Executing {event.get_name()} on CPU...")
            #         # Add the request event to the trace
            #         trace.add_cpu_event(event)
            #         # Add the prefill of the request to the task pool
            #         self.m_task_pool.add(Task.from_request(event.get_request()))

            #         to_remove.append(event_id)
                    
            #         # Update cpu timestamp
            #         current_cpu_timestamps += event.get_duration()

            #         completed_event_ids[event_id] = event

            # print('to_remove:', to_remove)

            # # Remove the request event from the ready events
            # for event_id in to_remove:
            #     ready_event_ids.remove(event_id)
                
            
            # print('ready_event_ids:', ready_event_ids)
            
            # if len(ready_event_ids) != 0:

            #     '''
            #     2.2 GET A READY EVENT TO EXECUTE/SIMULATE
            #     '''
            #     # Choose the first one
            #     current_event_id = ready_event_ids[0]

            #     # Get the corresponding node by id
            #     current_event  = nodes[current_event_id]
                
            #     '''
            #     2.3 EXECUTE THE CHOSEN EVENT
            #     '''
                
            #     print(f"Executing {current_event.get_name()} (ID: {current_event.get_id()}) on stage {current_event.get_dev_id()}...")  
                
            #     stage = current_event.get_dev_id()
            #     resource = current_event.get_resource_type()

            #     '''
            #     2.3.1 CALCULATE THE DURATION, START AND END TIMESTAMP
            #     '''
            #     # Get the duration of the event
            #     duration = 20

            #     current_event.set_duration(duration)

            #     start_timestamp = 0
            #     if current_event_id in in_data_deps.keys():
            #         # The start timestamp should be after the latest compeletion time of events with in edge dependence
            #         start_timestamp = max(list(map(lambda x : completed_event_ids[x].get_timestamp() + completed_event_ids[x].get_duration(), in_data_deps[current_event_id])))
            #         # The start timestamp should also be after the latest compeletion time of executed events on current stage/device
            #         start_timestamp = max(start_timestamp, current_timestamps[stage][resource])
                
            #     complete_time = start_timestamp + current_event.get_duration()


            #     '''
            #     2.3.2 UPDATE THE EVENT AND TRACE
            #     '''
            #     # Update the timestamp of current event
            #     current_event.set_timestamp(start_timestamp) 

            #     # Update the list of completed events, and the timestamps of each stages
            #     completed_event_ids[current_event_id] = current_event
            #     current_timestamps[stage][resource] = complete_time

            #     # Update the output trace
            #     trace.add_event(stage,current_event)


            #     '''
            #     2.3.3 UPDATE THE IN EDGES (DATA DEPENDENCE)
            #     '''
            #     in_data_deps_tmp = copy.deepcopy(in_data_deps)
            #     if current_event_id in out_data_deps.keys():
            #         for dest_event_id in out_data_deps[current_event_id]:
            #             in_data_deps_tmp[dest_event_id].remove(current_event_id)
            #             if 0 == len(in_data_deps_tmp[dest_event_id]) and dest_event_id not in completed_event_ids.keys():
            #                 ready_event_ids.append(dest_event_id)  

            #     ready_event_ids.remove(current_event_id)  

            #     '''
            #     2.3.4 THE END OF PREFILL/DECODE EVENTS
            #     '''
            #     if current_event.get_type() == EventType.PRF:
            #         # Add new task to the task pool
            #         self.m_task_pool.adds(Tasks.from_prefill_tasks(current_event.get_tasks(), complete_time))
            #     elif current_event.get_type() == EventType.DCD:
            #         # Add new task to the task pool
            #         self.m_task_pool.adds(Tasks.from_decode_tasks(current_event.get_tasks(), complete_time))
                    
            # '''
            # 2.4 SCHEDULER & NEW EVENTS
            # '''

            # # Create a scheduler event node as src of dependence edge
            # sched_event_id = self.m_graph.add_schedule_node(current_cpu_timestamps)
            
            # # Schedule the task
            # task_type, task_lists = scheduler.schedule(self.m_task_pool, self.m_ndevs)
            # print(f"Task type: {task_type}, task lists: {task_lists}")

            # completed_event_ids[sched_event_id] = nodes[sched_event_id]
            # trace.add_cpu_event(event)
            # if task_type != TaskType.EMPTY:
            #     # Create new events for the scheduled tasks
            #     for dev_id in range(len(task_lists)):
            #         task_list = task_lists[dev_id]
            #         if task_type == TaskType.PREFILL:
            #             # Create a new event for the tasks
            #             event_id = self.m_graph.add_prefill_node(task_list, dev_id)
            #             # Add the event to the ready events
            #             ready_event_ids.append(event_id)
            #             # Add edge
            #             self.m_graph.add_edge(sched_event_id, event_id)
            #         elif task_type == TaskType.DECODE:
            #             # Create a new event for the tasks
            #             event_id = self.m_graph.add_decode_node(task_list, dev_id)
            #             # Add the event to the ready events
            #             ready_event_ids.append(event_id)
            #             # Add edge
            #             self.m_graph.add_edge(sched_event_id, event_id)

            # '''
            # 2.5 ADD NEW REQS
            # '''
            # print('ready_event_ids at the iter end:', ready_event_ids)
            print('[Start Timestamp]', start_timestamp)
            print('[Complete Timestamp]', complete_time)
            print('[Timestamp] (only CPU)', current_cpu_timestamp)
            print('completed_event_ids:', completed_event_ids)
            # for event_id, event in nodes.items():
            #     if event.get_type() == EventType.REQ and event.get_timestamp() <= current_cpu_timestamps:
            #         if event_id not in completed_event_ids.keys(): # not visited
            #             ready_event_ids.append(event_id)

            print('req_nodes:', req_nodes)
            print('[Task Pool]:', self.m_task_pool.pool)
            print('ready_event_ids at the iter end:', ready_event_ids)

        self.set_outputs([trace])

        return trace

