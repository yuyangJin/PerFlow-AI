'''
@module Event
'''

from enum import Enum

NoneTimestamp = -1
NoneMem = -1

class EventType(Enum):
    OPRT = 0
    COMM = 1
    FWD = 2
    BWD = 3
    WGT = 4
    OFFL = 5
    REL = 6
    REQ = 7
    SCHD = 8
    PRF = 9
    DCD = 10

class ResourceType(Enum):
    NONE = -1
    G2C_PCIE = 0
    C2G_PCIE = 1
    G2C_NVLK = 2
    C2G_NVLK = 3
    G2G_NVLK = 4
    G2G_DRCT = 5
    GPU = 6
    CPU = 7

'''
@class Event
An event is a basic unit in the trace.
'''
class Event:
    def __init__(self, id, type, name, timestamp, duration, mem = NoneMem, resource_type = ResourceType.NONE):
        if not isinstance(type, EventType):
            raise ValueError("Type must be an instance of EventType")
        self.m_id = id
        self.m_type = type
        self.m_name = name
        self.m_timestamp = timestamp
        self.m_duration = duration
        self.m_mem = mem
        self.m_resource_type = resource_type
    
    def get_id(self):
        return self.m_id
    
    def get_name(self):
        return self.m_name

    def get_type(self):
        return self.m_type

    def get_timestamp(self):
        return self.m_timestamp

    def get_resource_type(self):
        return self.m_resource_type

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

    def get_mem(self):
        return self.m_mem

    def set_mem(self, mem):
        if not isinstance(mem, (int, float)):
            raise TypeError("memory must be a number")
        self.m_mem = mem

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
                    self.m_duration == other.m_duration and
                    self.m_mem == other.m_mem)
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


'''========================================================================'''
'''                             Training Events                            '''
'''========================================================================'''

'''
@class FwdBwdEvent
A forward-backward event.
'''
class FwdBwdEvent(Event):
    def __init__(self, id, type, name, timestamp, duration, 
                stage_id, microbatch_id, chunk_id, mem = NoneMem, resource_type = ResourceType.GPU, recompute_mask = None):
        super().__init__(id, type, name, timestamp, duration, mem = mem, resource_type = resource_type)
        if not (type == EventType.FWD or type == EventType.BWD or type == EventType.WGT):
            raise ValueError("FwdBwdEvent's Type must be FWD or BWD or WGT")
        self.m_stage_id = stage_id
        self.m_microbatch_id = microbatch_id
        self.m_chunk_id = chunk_id
        self.m_recompute_mask = recompute_mask

    def get_stage_id(self):
        return self.m_stage_id

    def get_microbatch_id(self):
        return self.m_microbatch_id

    def get_chunk_id(self):
        return self.m_chunk_id
    
    '''
    @func recompute()
    recompute or not recompute
    '''
    def recompute(self):
        if self.m_recompute_mask == None:
            return False
        else:
            return True
    
    '''
    @func get_recompute_mask()
    Recompute mask is a list of 0 or 1, representing the the compute flag of each layer.
    Thus, the number of the list should be equal to the number of layers. 
    '''
    def get_recompute_mask(self):
        return self.m_recompute_mask

'''
@class OffReLoadEvent
A offloading-reloading event.
'''
class OffReLoadEvent(Event):
    def __init__(self, id, type, name, timestamp, duration, 
                load_ratio, resource_type,
                stage_id, microbatch_id, chunk_id, mem = NoneMem):
        super().__init__(id, type, name, timestamp, duration, mem, resource_type = resource_type)

        if not (type == EventType.OFFL or type == EventType.REL):
            raise ValueError("OffReLoadEvent's Type must be OFFL or REL")
        if not isinstance(resource_type, ResourceType):
            raise ValueError("resource_type must be an instance of ResourceType")
        
        self.m_stage_id = stage_id
        self.m_microbatch_id = microbatch_id
        self.m_chunk_id = chunk_id

        self.m_load_ratio = 0.0
        self.m_base_duration = duration
        self.m_base_mem = mem
        self.set_load_ratio(load_ratio)
    
    def calc_load_duration_mem(self, load_ratio):
        if self.m_type == EventType.OFFL:
            mem = -1 * self.m_base_mem * load_ratio
        else:
            mem = self.m_base_mem * load_ratio
        duration = self.m_base_duration * load_ratio #TODO: Calculate the time required based on the volume and speed of data transmission.
        return duration, mem
    
    def set_load_ratio(self, load_ratio, duration = None, mem = None):
        if not self.m_load_ratio == load_ratio:
            self.m_load_ratio = load_ratio
            if duration == None or mem == None: #TODO: Refine the methods of calculating duration and mem.
                duration, mem = self.calc_load_duration_mem(load_ratio)
            self.set_duration(duration)
            self.set_mem(mem)

    def get_load_ratio(self):
        return self.m_load_ratio
        
    def get_stage_id(self):
        return self.m_stage_id

    def get_microbatch_id(self):
        return self.m_microbatch_id

    def get_chunk_id(self):
        return self.m_chunk_id


'''========================================================================'''
'''                             Inference Events                           '''
'''========================================================================'''

infer_event_id = 0

'''
@class RequestEvent
A request event.
'''
class RequestEvent(Event):
    def __init__(self, type, name, timestamp, duration,
                stage_id, request, mem = NoneMem):
        global infer_event_id
        super().__init__(infer_event_id, type, name, timestamp, duration, mem, resource_type = ResourceType.CPU)
        infer_event_id += 1
        if not (type == EventType.REQ):
            raise ValueError("RequestEvent's Type must be REQ")
        self.m_stage_id = stage_id
        self.m_request = request

    def get_stage_id(self):
        return self.m_stage_id

    def get_request(self):
        return self.m_request

    def generate_task(self):
        '''
        Generate a task from the request event.
        '''
        task = Task.from_request(self.m_request)
        return task

'''
@class ScheduleEvent
A schedule event.
'''
class ScheduleEvent(Event):
    def __init__(self, type, name, timestamp, duration, dev_id = 0, mem = NoneMem):
        global infer_event_id
        super().__init__(infer_event_id, type, name, timestamp, duration, mem, resource_type = ResourceType.CPU)
        infer_event_id += 1
        if not (type == EventType.SCHD):
            raise ValueError("ScheduleEvent's Type must be SCHD")
        self.m_dev_id = dev_id

    def get_dev_id(self):
        return self.m_dev_id
        
'''
@class PrefillEvent
A prefill event.
'''
class PrefillEvent(Event):
    def __init__(self, type, name, timestamp, duration,
                dev_id, input_len, tasks, mem = NoneMem):
        global infer_event_id
        super().__init__(infer_event_id, type, name, timestamp, duration, mem, resource_type = ResourceType.GPU)
        infer_event_id += 1
        if not (type == EventType.PRF):
            raise ValueError("PrefillEvent's Type must be PRF")
        self.m_dev_id = dev_id
        self.m_input_len = input_len
        self.m_tasks = tasks
        

    def get_dev_id(self):
        return self.m_dev_id

    def get_input_len(self):
        return self.m_input_len

    def get_tasks(self):
        return self.m_tasks

'''
@class DecodeEvent
A decode event.
'''
class DecodeEvent(Event):
    def __init__(self, type, name, timestamp, duration,
                dev_id, tasks, mem = NoneMem):
        global infer_event_id
        super().__init__(infer_event_id, type, name, timestamp, duration, mem, resource_type = ResourceType.GPU)
        infer_event_id += 1
        if not (type == EventType.DCD):
            raise ValueError("DecodeEvent's Type must be DCD")
        self.m_dev_id = dev_id
        self.m_tasks = tasks

    def get_dev_id(self):
        return self.m_dev_id
    
    def get_tasks(self):
        return self.m_tasks