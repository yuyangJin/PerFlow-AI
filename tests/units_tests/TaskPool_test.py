'''
test TaskTool class
'''

from perflowai.core import Request, TaskPool, Task, TaskType
import pytest


def test_TaskPool():
    # Create a mock request
    req = Request(
        req_id=1,
        input_len=10,
        output_len=100,
        start_time=0,
    )

    # Create a task from the request
    task = Task.from_request(req)

    # Create a TaskPool instance
    pool = TaskPool()

    # Add the task to the pool
    pool.add(task)

    # Check if the task is added correctly
    assert len(pool.pool) == 1
    print(pool.pool)
    assert pool.pool[0] == task

    # # Get a task from the pool
    # task = pool.get_task()
    # assert task == req
    pool.remove(0)

    # Check if the pool is empty after getting the task
    assert len(pool.pool) == 0

    # Add multiple tasks to the pool
    for i in range(2, 6):
        task = Task(
            id=i,
            req=req,
            task_type=TaskType.PREFILL,
        )
        pool.add(task)

    # Check if the tasks are added correctly
    assert len(pool.pool) == 4

    # Get a prefill task from the pool
    prefill_tasks = pool.get_prefill_task(1,2)
    assert len(prefill_tasks) == 2
    assert len(prefill_tasks[0].get()) == 1

    # Check if the tasks are removed from the pool
    assert len(pool.pool) == 2

    # Remove a prefill task from the pool
    with pytest.raises(AssertionError, match='Task not found in pool'):
        pool.remove(0)

    pool.remove(4)
    assert len(pool.pool) == 1

    pool.remove(5)
    assert pool.is_empty() == True