'''
@module Request
'''

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

task_id = 0

tasks_id = 0

@dataclass
class Request:
    req_id: int
    input_len: int
    output_len: int
    start_time: float
    prefill_time: Optional[float] = None
    decode_time: List[float] = field(default_factory=list)

class TaskType(Enum):
    PREFILL = 0
    DECODE = 1
    REMOVE = 2
    EMPTY = 3

'''
@class Task
A task from a request during the inference
'''
@dataclass
class Task:
    id: int
    req: Request
    task_type: TaskType

    # which dp instance / device the cache of this task belongs to
    cache_owner: int = 0

    # tokens: Optional[List[int]] = None
    prefix_length: int = 0
    # response: List[int] = field(default_factory=list)
    decode_iters: int = 0

    # waiting: bool = False
    # handle: Optional[Work] = None

    def __post_init__(self):
        # Initialize any additional attributes or perform validation here
        if self.task_type not in TaskType:
            raise ValueError(f"Invalid task type: {self.task_type}")
        if self.req is None:
            raise ValueError("Request cannot be None")
        if self.req.input_len <= 0:
            raise ValueError("Input length must be greater than 0")
    
    def __str__(self):
        return f"Task(id={self.id}, req_id={self.req.req_id}, task_type={self.task_type})"
    
    def __repr__(self):
        return f"Task(id={self.id}, req_id={self.req.req_id}, task_type={self.task_type})"

    @classmethod
    def from_request(cls, req: Request):
        global task_id
        task = cls(
            id=task_id,
            req=req,
            task_type=TaskType.PREFILL,
        )
        task_id += 1
        return task
    
    @classmethod
    def from_prefill_task(cls, task):
        global task_id
        task = cls(
            id=task_id,
            req=task.req,
            task_type=TaskType.DECODE,
            decode_iters=1,
        )
        task_id += 1
        return task

    @classmethod
    def from_decode_task(cls, task):
        global task_id
        print("task.decode_iters:", task.decode_iters)
        task = cls(
            id=task_id,
            req=task.req,
            task_type=TaskType.DECODE,
            decode_iters=min(task.req.output_len, task.decode_iters + 1),
        )
        task_id += 1
        return task

class Tasks:
    def __init__(self):
        global tasks_id
        self.id = tasks_id
        tasks_id += 1
        self.tasks = []
        self.max_input_len = 0

    def add(self, task: Task):
        self.tasks.append(task)
        if task.req.input_len > self.max_input_len:
            self.max_input_len = task.req.input_len

    def remove(self, task: Task):
        self.tasks.remove(task)

    def get(self):
        return self.tasks
    
    def get_num_task(self) -> int:
        return len(self.tasks)

    def get_max_input_len(self) -> int:
        return self.max_input_len

    def is_empty(self) -> bool:
        return len(self.tasks) == 0
    
    @classmethod
    def from_prefill_tasks(cls, tasks, complete_time: float):
        task_list = cls()
        for task in tasks.get():
            if task.task_type == TaskType.PREFILL:
                task.req.prefill_time = complete_time
                task_list.add(Task.from_prefill_task(task))
        return task_list

    @classmethod
    def from_decode_tasks(cls, tasks, complete_time: float):
        task_list = cls()
        for task in tasks.get():
            if task.task_type == TaskType.DECODE:
                task.req.decode_time.append(complete_time)
                # The end of the decode task
                if task.decode_iters >= task.req.output_len:
                    continue
                task_list.add(Task.from_decode_task(task))
        return task_list

class TaskPool:
    pool: Dict[int, Task] = {}

    def add(self, task: Task):
        assert task.id not in self.pool, "Task already exists in pool"
        self.pool[task.id] = task
    
    def adds(self, tasks: Tasks):
        for task in tasks.get():
            self.add(task)

    # def get_task(self) -> Optional[Task]:
    #     if self.pool:
    #         return self.pool.pop()
    #     return None
    
    def get_prefill_task(self, max_num_task, ndevs) -> Optional[List[Tasks]]:
        prefill_task_ids = list(filter(
            lambda x: self.pool[x].task_type == TaskType.PREFILL
                        # and not self.pool[x].waiting
                        ,
                self.pool.keys(),
        ))
        prefill_task_ids = list(sorted(
            prefill_task_ids,
            key=lambda x: self.pool[x].req.start_time,
            reverse=False,
        ))
        prefill_task_ids = list(prefill_task_ids)[
            : max_num_task * ndevs
        ]

        # self.max_num_tasks_per_dp * self.dp_size
        # Assgin to different devices
        if len(prefill_task_ids) > 0:
            task_lists = [Tasks() for _ in range(ndevs)]
            for i, task_id in enumerate(prefill_task_ids):
                task = self.pool[task_id]
                task.cache_owner = i % ndevs
                task_lists[i % ndevs].add(task)

            # make sure tasks do not exceed max_num_tasks_per_dp
            for i in range(ndevs):
                tasks = task_lists[i].get()
                if len(tasks) > max_num_task:
                    tasks = tasks[: max_num_task]
                # Remove the fetched prefill tasks from the pool
                for task in tasks:
                    print('task.id in pop:', task.id)
                    self.pool.pop(task.id)

            print(f"prefill_task_ids: {prefill_task_ids}")
            print('pool after get_prefill_task:', self.pool)
            return task_lists
        
        return None

    def get_decode_task(self, max_num_task, ndevs) -> Optional[List[Tasks]]:
        decode_task_ids = list(filter(
            lambda x: self.pool[x].task_type == TaskType.DECODE
                        # and not self.pool[x].waiting
                        ,
                self.pool.keys(),
        ))
        decode_task_ids = list(decode_task_ids)[
            : max_num_task * ndevs
        ]
        if len(decode_task_ids) > 0:
            # For decode tasks, we need to make sure they are sent to their cache owner
            task_lists = [Tasks() for _ in range(ndevs)]
            for task_id in decode_task_ids:
                task = self.pool[task_id]
                task_lists[task.cache_owner].add(task)

            # make sure tasks do not exceed max_num_task
            for i in range(ndevs):
                tasks = task_lists[i].get()
                if len(tasks) > max_num_task:
                    tasks = tasks[: max_num_task]
                # Remove the fetched decode tasks from the pool
                for task in tasks:
                    self.pool.pop(task.id)

            print('pool after get_decode_task:', self.pool)

            return task_lists
        return None

    def is_empty(self) -> bool:
        return len(self.pool) == 0
    
    def remove(self, id: int):
        assert id in self.pool.keys(), "Task not found in pool"
        # task = self.pool[id]
        self.pool.pop(id)