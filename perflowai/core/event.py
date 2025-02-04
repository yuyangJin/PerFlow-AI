'''
@module Event
'''

from enum import Enum

class EventType(Enum):
    OPRT = 0
    COMM = 1
    FWD = 2
    BWD = 3
    WGT = 4

'''
@class Event
An event is a basic unit in the trace.
'''
class Event:
    def __init__(self, id, type, name, timestamp, duration):
        if not isinstance(type, EventType):
            raise ValueError("Type must be an instance of EventType")
        self.m_id = id
        self.m_type = type
        self.m_name = name
        self.m_timestamp = timestamp
        self.m_duration = duration
    
    def get_id(self):
        return self.m_id
    
    def get_name(self):
        return self.m_name

    def get_type(self):
        return self.m_type

    def get_timestamp(self):
        return self.m_timestamp

    def get_duration(self):
        return self.m_duration

    def set_duration(self, duration):
        if not isinstance(duration, (int, float)):
            raise TypeError("Duration must be a number")
        self.m_duration = duration


'''
@class OprtEvent
An operation event.
'''
class OprtEvent(Event):
    def __init__(self, id, type, name, timestamp, duration, 
                ):
        super().__init__(id, type, name, timestamp, duration)


'''
@class CommEvent
A communication event.
'''
class CommEvent(Event):
    def __init__(self, id, type, name, timestamp, duration, 
                src_dev_id, dst_dev_id):
        super().__init__(id, type, name, timestamp, duration)
        self.m_src_dev_id = src_dev_id
        self.m_dst_dev_id = dst_dev_id

    def get_src_dev_id(self):
        '''
        To be implemented
        '''
        pass

    def get_dst_dev_id(self):
        '''
        To be implemented
        '''
        pass


'''
@class FwdBwdEvent
A forward-backward event.
'''
class FwdBwdEvent(Event):
    def __init__(self, id, type, name, timestamp, duration, 
                stage_id, microbatch_id, chunk_id):
        super().__init__(id, type, name, timestamp, duration)
        self.m_stage_id = stage_id
        self.m_microbatch_id = microbatch_id
        self.m_chunk_id = chunk_id

    def get_stage_id(self):
        '''
        To be implemented
        '''
        pass

    def get_microbatch_id(self):
        '''
        To be implemented
        '''
        pass

    def get_chunk_id(self):
        '''
        To be implemented
        '''
        pass
    
