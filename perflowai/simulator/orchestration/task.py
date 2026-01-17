from typing import List, Union

from perflowai import FlowNode


class Task:
    def __init__(self,
                 id: str,
                 workload: FlowNode,
                 run_after: List[str] = None
                 ):
        if run_after is None:
            run_after = []

        self.id = id
        self.workload = workload
        self.run_after = run_after


class Assignment:
    def __init__(self,
                 task_id: str,
                 device_id: Union[int, str],
                 ):
        self.task_id = task_id
        self.device_id = device_id
