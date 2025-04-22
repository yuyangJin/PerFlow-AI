from .device import Device
from typing import List, Dict
from .request import Request, TaskPool, Task, TaskType, Tasks

class Scheduler:
    '''
    output -> (TaskType, List[Tasks])
    '''
    def schedule(self, pool: TaskPool, devices: List[Device]): 
        raise NotImplementedError
