import onnx
from onnx import helper, TensorProto
import pytest

from perflowai.core.device import DeviceConfig, DeviceType, NodeConfig
from perflowai.simulator.orchestration.device_info import NodeInstance
from perflowai.simulator.orchestration.scheduler import Scheduler

def _create_node_instance():
    # host
    host = DeviceConfig(id=0, type=DeviceType.CPU, memory_capacity=16384, memory_bandwidth=50.0, compute_flops=1e12)
    # gpu
    gpu = DeviceConfig(id=1, type=DeviceType.GPU, memory_capacity=8192, memory_bandwidth=900.0, compute_flops=100e12)
    node_config = NodeConfig(host=host, gpu_devices=[gpu])
    return NodeInstance(node_config)

def test_matmul_schedule():
    # Create ONNX graph for C = A * B
    # Shapes: A[1024, 1024], B[1024, 1024] -> C[1024, 1024]

    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1024, 1024])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1024, 1024])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1024, 1024])

    matmul_node = helper.make_node(
        'MatMul',
        inputs=['A', 'B'],
        outputs=['C'],
        name='MatMul_0'
    )

    graph_def = helper.make_graph(
        [matmul_node],
        'test_matmul_graph',
        [A, B],
        [C]
    )

    model_def = helper.make_model(graph_def, producer_name='test_scheduler')

    # Check model
    onnx.checker.check_model(model_def)

    # Scheduler
    node_instance = _create_node_instance()
    scheduler = Scheduler(model_def, [node_instance])

    result = scheduler.schedule()

    # Assertions
    tasks = result.tasks
    # We expect: Malloc C, Compute MatMul, Free A, Free B
    assert len(tasks) >= 4

    task_ids = [t.id for t in tasks]
    print(f"Tasks: {task_ids}")

    # Check for Malloc C
    # The malloc id is defined as f"malloc_{output_name}" in Scheduler
    assert "malloc_C" in task_ids

    # Check for MatMul
    assert "MatMul_0" in task_ids

    # Check for Free A and Free B
    # The free id is defined as f"free_{input_name}_{idx}" OR just f"free_{input_name}"?
    # In code: free_id = f"free_{input_name}_{idx}"
    # name argument to FreeSimulator is f"free_{input_name}"
    # Check task IDs:
    # idx is 0 for the first node.
    assert "free_A_0" in task_ids
    assert "free_B_0" in task_ids

    # Validate dependencies

    matmul_task = next(t for t in tasks if "MatMul_0" == t.id)
    malloc_c_task = next(t for t in tasks if "malloc_C" == t.id)

    # Compute should separate Malloc from usage?
    # In Scheduler: compute_task depends on run_after, which includes malloc_id
    assert malloc_c_task.id in matmul_task.run_after

    # Free depends on Compute
    free_a_task = next(t for t in tasks if "free_A_0" == t.id)
    assert matmul_task.id in free_a_task.run_after

    free_b_task = next(t for t in tasks if "free_B_0" == t.id)
    assert matmul_task.id in free_b_task.run_after

