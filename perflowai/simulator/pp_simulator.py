'''
@package perflowai.simulator.pp_simulator
'''
from ..core.trace import PPTrace
from .simulator import Simulator

from enum import Enum
from collections import deque
import copy

class PipeType(Enum):
    GPipe = 0
    PipeDream = 1
    Interleaved1F1B = 2
    ZeroBubble = 3
    Custom = 4

class PipePartitionType(Enum):
    Balanced = 0
    Custom = 1


'''
@class PipeConfig
A pipeline parallel configuration.
'''
class PipeConfig:
    def __init__(self):
        pass

'''
@class PipePartitionConfig
A pipeline partition configuration.
'''
class PipePartitionConfig:

    '''
    @func __init__
    The input partition plan is a list, each element number is 
    the ratio of partitioned layers on each stage. For example,
    A balanced 4-stage partition plan is [0.25, 0.25, 0.25, 0.25]
    '''
    def __init__(self, n_stages, partition_plan):
        self.m_nstages = n_stages
        self.m_partition_plan = partition_plan
        self.__check()

    '''
    @func __check
    Check whether the partition plan is valid or not.
    '''
    def __check(self):
        if self.m_nstages != len(self.m_partition_plan):
            raise AssertionError("The partition plan is not valid. The length of partition plan shoule be equal to the number of stages!")
        total = sum(self.m_partition_plan)
        if total != 1:
            raise AssertionError("The partition plan is not valid. The sum of each number in a partition plan shoule be 1.0!")

    def get_nstages(self):
        return self.m_nstages
    
    def get_partition_plan(self):
        return self.m_partition_plan
        
'''
@class PPSimulator
A pipeline simulator.
'''
class PPSimulator(Simulator):
    def __init__(self, pipetype, ppgraph):
        self.m_pipetype = pipetype
        self.m_graph = ppgraph
        self.m_nstages = ppgraph.get_nstages()
        self.m_nmicrobatches = ppgraph.get_nmicrobatches()
        self.m_nchunks = ppgraph.get_nchunks()

    def simulate(self):
        nodes = self.m_graph.get_nodes()

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

        # Store the current timestamp of each stage
        current_timestamps = [0 for _ in range(self.m_nstages)]

        # Store the event execution trace
        pptrace = PPTrace(self.m_nstages, self.m_nstages, self.m_nmicrobatches, self.m_nchunks)
        # event_execution_order = [[] for _ in range(self.m_nstages)] 

        '''
        1. INITIALIZATION
        '''
        # Get the ready events with no in (data dependence) edges
        for event_id in nodes.keys():  
            if event_id not in in_data_deps.keys(): 
                ready_event_ids.append(event_id)  

        '''
        2. SIMULATION
        '''
        while len(ready_event_ids) != 0:  
            '''
            2.1 GET A READY EVENT TO EXECUTE/SIMULATE
            '''
            # Choose the first one
            current_event_id = ready_event_ids[0]

            # # Prefer to choose an event with smallest chunk id
            # for event_id in ready_event_ids:
            #     event = nodes[event_id]
            #     if event.chunk < nodes[current_event_id].chunk:
            #         current_event_id = event_id
            #         break

            # Get the corresponding node by id
            current_event  = nodes[current_event_id]

            
            '''
            2.2 EXECUTE THE CHOSEN EVENT
            '''
            
            print(f"Executing {current_event.get_name()} (ID: {current_event.get_id()}) on stage {current_event.get_stage_id()}...")  
            
            stage = current_event.get_stage_id()

            '''
            2.2.1 CALCULATE THE START AND END TIMESTAMP
            '''
            start_timestamp = 0
            if current_event_id in in_data_deps.keys():
                # The start timestamp should be after the latest compeletion time of events with in edge dependence
                start_timestamp = max(list(map(lambda x : completed_event_ids[x].get_timestamp() + completed_event_ids[x].get_duration(), in_data_deps[current_event_id])))
                # The start timestamp should also be after the latest compeletion time of executed events on current stage/device
                start_timestamp = max(start_timestamp, current_timestamps[stage])
            
            complete_time = start_timestamp + current_event.get_duration()


            '''
            2.2.2 UPDATE THE EVENT AND TRACE
            '''
            # Update the timestamp of current event
            current_event.set_timestamp(start_timestamp) 

            # Update the list of completed events, and the timestamps of each stages
            completed_event_ids[current_event_id] = current_event
            current_timestamps[stage] = complete_time

            # Update the output trace
            pptrace.add_event(stage,current_event)


            '''
            2.2.3 UPDATE THE IN EDGES (DATA DEPENDENCE)
            '''
            if current_event_id in out_data_deps.keys():
                for dest_event_id in out_data_deps[current_event_id]:
                    in_data_deps_tmp[dest_event_id].remove(current_event_id)
                    if 0 == len(in_data_deps_tmp[dest_event_id]):
                        ready_event_ids.append(dest_event_id)  

            ready_event_ids.remove(current_event_id)  
        
        self.set_outputs([pptrace])

        return pptrace


    def get_trace(self):
        pass

    def get_perf(self):
        pass

    def get_config(self):
        pass

    def get_partition_config(self):
        pass

    def set_config(self):
        pass

    def set_partition_config(self):
        pass

    def set_pipe_type(self):
        pass

    def set_pipe_partition_type(self):
        pass

    def get_pipe_type(self):
        pass

    def get_pipe_partition_type(self):
        pass 

    '''
    @method run
    Run the pipeline simulator.
    '''
    def run(self):
        return self.simulate()