'''
@module Event
'''

from enum import Enum

NoneTimestamp = -1

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

    def set_timestamp(self, timestamp):
        if not isinstance(timestamp, (int, float)):
            raise TypeError("Timestamp must be a number")
        self.m_timestamp = timestamp

    def get_duration(self):
        return self.m_duration

    def set_duration(self, duration):
        if not isinstance(duration, (int, float)):
            raise TypeError("Duration must be a number")
        self.m_duration = duration

    def get_info_str(self):
        return '<Event ' + str(self.m_id) + ':' + self.m_name + '>'

    def __str__(self):
        return self.get_info_str()

    def __repr__(self):
        return self.get_info_str()
    
    def __eq__(self, other):
        if type(self) == type(other):
            return (self.m_id == other.m_id and
                    self.m_type == other.m_type and
                    self.m_name == other.m_name and
                    self.m_timestamp == other.m_timestamp and
                    self.m_duration == other.m_duration)
        return false


    def __eq__(self, other):
        if type(self) == type(other):
            return (self.m_id == other.m_id and
                    self.m_type == other.m_type and
                    self.m_name == other.m_name and
                    self.m_timestamp == other.m_timestamp and
                    self.m_duration == other.m_duration)
        return false


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
        if not (type == EventType.FWD or type == EventType.BWD or type == EventType.WGT):
            raise ValueError("FwdBwdEvent's Type must be FWD or BWD or WGT")
        self.m_stage_id = stage_id
        self.m_microbatch_id = microbatch_id
        self.m_chunk_id = chunk_id

    def get_stage_id(self):
        return self.m_stage_id

    def get_microbatch_id(self):
        return self.m_microbatch_id

    def get_chunk_id(self):
        return self.m_chunk_id