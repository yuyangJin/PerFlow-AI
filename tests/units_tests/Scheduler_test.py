import onnx
from onnx import helper, TensorProto
import pytest

from perflowai.core.device import DeviceConfig, DeviceType, NodeConfig
from perflowai.simulator.orchestration.device_info import NodeInstance
from perflowai.simulator.orchestration.scheduler import Scheduler, default_simulator_factory
from perflowai.simulator.kernel.kernel_simulator import GEMMKernelSimulator, Parameter

class GEMMFactory:
    def __init__(self, model_proto: onnx.ModelProto):
        self.model = model_proto
        # Build shape map
        self.shapes = {}
        for container in [self.model.graph.input, self.model.graph.output, self.model.graph.value_info]:
            for vi in container:
                if vi.type.HasField('tensor_type'):
                    shape = []
                    for d in vi.type.tensor_type.shape.dim:
                        if d.HasField('dim_value'):
                            shape.append(d.dim_value)
                        else:
                            shape.append(1) # Default to 1 for unknown
                    self.shapes[vi.name] = tuple(shape)

    def __call__(self, node: onnx.NodeProto, idx: int, device_config: DeviceConfig):
        if node.op_type == "MatMul":
            # Assume A and B are inputs
            name_a = node.input[0]
            name_b = node.input[1]
            name_c = node.output[0]

            shape_a = self.shapes.get(name_a, (1,1))
            shape_b = self.shapes.get(name_b, (1,1))
            shape_c = self.shapes.get(name_c, (1,1))


            # Create Parameters
            # Note: dtype hardcoded to float32 for simplicity or we can parse from model
            # But GEMMKernelSimulator might default or take string

            param_a = Parameter(name_a, "float32", shape_a)
            param_b = Parameter(name_b, "float32", shape_b)
            param_c = Parameter(name_c, "float32", shape_c)

            name = node.name if node.name else f"MatMul_{idx}"
            return GEMMKernelSimulator(
                device_config,
                a=param_a,
                b=param_b,
                c=param_c,
                name=name,
                id=name
            )

        return default_simulator_factory(node, idx, device_config)

def _create_node_instance():

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

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model_def)

    # Factory
    factory = GEMMFactory(inferred_model)

    # Scheduler
    node_instance = _create_node_instance()
    scheduler = Scheduler(inferred_model, [node_instance], simulator_factory=factory)

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

    # --- New Simulation Test ---
    print("\nRunning Estimate Makespan...")
    makespan = result.estimate_makespan()
    print(f"Makespan: {makespan.time_s} s")
    print(f"Max Memory: {makespan.max_memory_usage_bytes}")

    assert makespan.time_s > 0
    # C is 1024*1024 float (4 bytes) = 4MB
    # Malloc C happens.
    # GPU 1 is selected (Scheduler selects GPU if available).
    # Memory usage should be non-zero.

    # The Scheduler uses device 1 (GPU) because it is present in node_instance.
    # We should check usage on GPU id.
    # The ID is defined in _create_node_instance as int 1, but DeviceConfig usually stores ID.
    # DeviceInstance.id returns config.id.

    # Scheduler logic:
    # if node.gpu_devices: self.device = node.gpu_devices[0]
    # self.device.id is 1.

    dev_id = 1
    assert makespan.max_memory_usage_bytes[dev_id] > 0
