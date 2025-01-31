'''
@module Trace
'''
from perflowai.core.event import Event


'''
@class Trace
A trace is a collection of events.
'''

class Trace:
    def __init__(self, ndevs):
        self.m_ndevs = ndevs

        self.m_events = dict()
        for i in range(ndevs):
            self.m_events[i] = list()

    def add_event(self, dev_id, event):
        self.m_events[dev_id].append(event)
    
    def get_events(self, dev_id):
        return self.m_events[dev_id]

    def get_ndevs(self):
        return self.m_ndevs

