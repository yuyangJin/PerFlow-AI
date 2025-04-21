'''
test Task class
'''
from perflowai.core import Request, Task, TaskType

def test_Task():
    # Create a mock request
    req = Request(
        req_id=1,
        input_len=10,
        output_len=100,
        start_time=0,
    )

    # Create a task from the request
    task = Task.from_request(req)

    # Check if the task is created correctly
    assert task.id == 1
    assert task.task_type == TaskType.PREFILL
    assert task.req == req

    # Create a task directly
    task2 = Task(
        id=2,
        req=req,
        task_type=TaskType.DECODE,
    )

    # Check if the task is created correctly
    assert task2.id == 2
    assert task2.task_type == TaskType.DECODE
    assert task2.req == req