'''
@module Trace
'''
from .event import Event


'''
@class Trace
A trace is a collection of events.
'''

class Trace:
    def __init__(self, ndevs):
        self.m_ndevs = ndevs

        self.m_cpu_events = list()

        self.m_events = dict()
        for i in range(ndevs):
            self.m_events[i] = list()

    def add_event(self, dev_id, event):
        self.m_events[dev_id].append(event)
    
    def add_cpu_event(self, event):
        self.m_cpu_events.append(event)
        
    def get_cpu_events(self):
        return self.m_cpu_events

    def get_events(self, dev_id):
        return self.m_events[dev_id]

    def get_ndevs(self):
        return self.m_ndevs

    def output(self):
        for i in range(self.m_ndevs):
            print(self.m_events[i])
    
    def get_memory_foorprint(self):

        memory_usages = dict()

        # Process each stage separately  
        for dev_id in range(self.get_ndevs()):
            events = self.get_events(dev_id)

            # Dictionary to hold memory changes at specific time points  
            time_memory_changes = {}  
            
            for event in events:  
                # Calculate start and end times  
                start_time = event.get_timestamp()
                end_time = event.get_timestamp() + event.get_duration()
                
                # Update memory changes at the start time  
                if start_time not in time_memory_changes:  
                    time_memory_changes[start_time] = 0  
                time_memory_changes[start_time] += 0 
                
                # Update memory changes at the end time  
                if end_time not in time_memory_changes:  
                    time_memory_changes[end_time] = 0  
                time_memory_changes[end_time] += event.get_mem()  # Memory is released after duration 

            # Sort time points  
            sorted_times = sorted(time_memory_changes.keys())  


            # Calculate memory usage over time  
            memory_usage = []  
            current_memory = 0  
            
            for time in sorted_times:    
                memory_usage.append((time, current_memory))  
                current_memory += time_memory_changes[time]

            memory_usages[dev_id] = memory_usage

            
        return memory_usages

class PPTrace(Trace):
    def __init__(self, ndevs, nstages, nmicrobatches, nchunks):
        super().__init__(ndevs)
        self.m_nstages = nstages
        self.m_nmicrobatches = nmicrobatches
        self.m_nchunks = nchunks
    
    def get_nstages(self):
        return self.m_nstages
    
    def get_nmicrobatches(self):
        return self.m_nmicrobatches

    def get_nchunks(self):
        return self.m_nchunks

    def get_memory_foorprint(self):

        memory_usages = dict()

        # Process each stage separately  
        for stage_id in range(self.get_nstages()):
            events = self.get_events(stage_id)

            # Dictionary to hold memory changes at specific time points  
            time_memory_changes = {}  
            
            for event in events:  
                # Calculate start and end times  
                start_time = event.get_timestamp()
                end_time = event.get_timestamp() + event.get_duration()
                
                # Update memory changes at the start time  
                if start_time not in time_memory_changes:  
                    time_memory_changes[start_time] = 0  
                time_memory_changes[start_time] += 0 
                
                # Update memory changes at the end time  
                if end_time not in time_memory_changes:  
                    time_memory_changes[end_time] = 0  
                time_memory_changes[end_time] += event.get_mem()  # Memory is released after duration 

            # Sort time points  
            sorted_times = sorted(time_memory_changes.keys())  


            # Calculate memory usage over time  
            memory_usage = []  
            current_memory = 0  
            
            for time in sorted_times:    
                memory_usage.append((time, current_memory))  
                current_memory += time_memory_changes[time]

            memory_usages[stage_id] = memory_usage

            
        return memory_usages
