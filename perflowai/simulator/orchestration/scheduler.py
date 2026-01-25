from typing import List, Dict, Callable, Optional

import onnx
from onnx import shape_inference
import collections

from perflowai import NodeInstance, OrchestrationResult, DeviceConfig
from perflowai.simulator.orchestration.task import Task, Assignment
from perflowai.workflow import FlowNode
from perflowai.simulator.orchestration.op import MallocSimulator, FreeSimulator
from perflowai.simulator.kernel.kernel_simulator import Parameter


class OnnxFlowNode(FlowNode):
    """Wrapper for ONNX node to adapt to FlowNode interface."""
    def __init__(self, node: onnx.NodeProto, idx: int, device_config: Optional[DeviceConfig] = None):
        # Generate a name/ID if not present
        name = node.name if node.name else f"{node.op_type}_{idx}"
        # Ensure inputs/outputs are lists
        inputs = list(node.input)
        outputs = list(node.output)
        super().__init__(name, name, inputs, outputs)
        self.node = node
        self.device_config = device_config


SimulatorFactory = Callable[[onnx.NodeProto, int, DeviceConfig], FlowNode]


def default_simulator_factory(node: onnx.NodeProto, idx: int, device_config: DeviceConfig) -> FlowNode:
    return OnnxFlowNode(node, idx, device_config)


class Scheduler:
    def __init__(self,
                 graph: onnx.ModelProto,
                 nodes: List[NodeInstance],
                 simulator_factory: Optional[SimulatorFactory] = None):
        if len(nodes) < 1:
            raise ValueError("At least one node is required for scheduling.")

        if len(nodes) > 1:
            raise NotImplementedError("Multi-node scheduling not supported yet.")

        node = nodes[0]
        if len(node.gpu_devices) > 1:
            raise NotImplementedError("Multi-GPU scheduling not supported yet.")

        self.graph = shape_inference.infer_shapes(graph)
        self.node_instance = node

        # Determine device: Use GPU 0 if available, else Host
        if node.gpu_devices:
            self.device = node.gpu_devices[0]
        else:
            self.device = node.host

        self.device_config = self.device.config

        self.simulator_factory = simulator_factory or default_simulator_factory

    def schedule(self) -> OrchestrationResult:
        tasks: List[Task] = []
        assignments: List[Assignment] = []
        device_id = self.device.id

        # 1. Analyze Tensor Lifetimes (Reference Counting)
        tensor_ref_count = collections.defaultdict(int)

        # Count consumers (inputs of nodes)
        for node in self.graph.graph.node:
            for input_name in node.input:
                tensor_ref_count[input_name] += 1

        # Graph outputs should not be freed
        for output in self.graph.graph.output:
            tensor_ref_count[output.name] += 1

        # 2. Calculate sizes
        tensor_sizes = self._get_all_tensor_sizes()

        # 3. Generate Tasks
        # Track producers to set dependencies
        tensor_producers: Dict[str, str] = {}

        # --- Handle Graph Inputs ---
        for input_proto in self.graph.graph.input:
            input_name = input_proto.name
            size = tensor_sizes.get(input_name, 0)
            malloc_id = f"malloc_{input_name}"

            # Assume byte type for unknown/simplified inputs
            param = Parameter(input_name, "uint8", (size,), None)
            malloc_op = MallocSimulator(self.device_config, size, param, name=malloc_id)

            malloc_task = Task(malloc_id, malloc_op, run_after=[])
            tasks.append(malloc_task)
            assignments.append(Assignment(malloc_id, device_id))

            # Treat this malloc task as the producer of the input tensor
            tensor_producers[input_name] = malloc_id

        for idx, node in enumerate(self.graph.graph.node):
            workload = self.simulator_factory(node, idx, self.device_config)
            task_id = workload.id

            # --- Dependency Analysis ---
            run_after = []
            for input_name in node.input:
                if input_name in tensor_producers:
                    producer = tensor_producers[input_name]
                    if producer not in run_after:
                        run_after.append(producer)

            # --- Malloc Insertion (Outputs) ---
            # Create Malloc tasks for outputs of this node
            for output_name in node.output:
                size = tensor_sizes.get(output_name, 0)
                malloc_id = f"malloc_{output_name}" # Unique enough if names unique

                # Simple Malloc Simulator
                param = Parameter(output_name, "float32", (size,), None)
                malloc_op = MallocSimulator(self.device_config, size, param, name=malloc_id)

                # Malloc Dependency Strategy:
                # Ideally, Malloc runs just before Compute.
                # We can make Malloc depend on the same things as Compute?
                # Or make Malloc depend on nothing (alloc ASAP)?
                # Or depend on Previous Node (Serialization)?
                # Let's make Malloc depend on the producers of the INPUTS of this node?
                # This ensures we don't malloc too early before data is ready to be consumed?
                # Actually, standard allocator just allocates when called.
                # In graph, let's treat Malloc as prerequisite for Compute.
                # Compute depends on Malloc.

                malloc_task = Task(malloc_id, malloc_op, run_after=[])
                tasks.append(malloc_task)
                assignments.append(Assignment(malloc_id, device_id))

                run_after.append(malloc_id)

                # Register this task as producer
                tensor_producers[output_name] = task_id

            # --- Compute Task ---
            compute_task = Task(task_id, workload, run_after=run_after)
            tasks.append(compute_task)
            assignments.append(Assignment(task_id, device_id))

            # --- Free Insertion (Inputs) ---
            # Decrement refcounts for inputs. If 0, insert Free.
            for input_name in node.input:
                if input_name in tensor_ref_count:
                    tensor_ref_count[input_name] -= 1
                    if tensor_ref_count[input_name] <= 0:
                        # Insert Free
                        free_id = f"free_{input_name}_{idx}"
                        size = tensor_sizes.get(input_name, 0)

                        free_op = FreeSimulator(
                            self.device_config,
                            size,
                            Parameter(input_name, "uint8", (size,)),
                            name=f"free_{input_name}"
                        )

                        # Free depends on this Compute task
                        free_task = Task(free_id, free_op, run_after=[task_id])
                        tasks.append(free_task)
                        assignments.append(Assignment(free_id, device_id))

        return OrchestrationResult([self.node_instance], tasks, assignments)

    def _get_all_tensor_sizes(self) -> Dict[str, int]:
        sizes = {}
        # ValueInfo
        for vi in self.graph.graph.value_info:
             sizes[vi.name] = self._calculate_size(vi)
        # Inputs
        for inp in self.graph.graph.input:
             sizes[inp.name] = self._calculate_size(inp)
        # Outputs
        for out in self.graph.graph.output:
             sizes[out.name] = self._calculate_size(out)
        return sizes

    def _calculate_size(self, value_info) -> int:
        type_proto = value_info.type
        if not type_proto.HasField('tensor_type'):
            return 0
        tensor_type = type_proto.tensor_type

        # Calculate Input Size
        elem_type = tensor_type.elem_type
        # Mapping ONNX TensorProto.DataType to bytes
        bytes_per_elem = 4 # default float32
        if elem_type == 1: # FLOAT
            bytes_per_elem = 4
        elif elem_type == 10: # FLOAT16
            bytes_per_elem = 2
        elif elem_type == 7: # INT64
            bytes_per_elem = 8
        elif elem_type == 6: # INT32
            bytes_per_elem = 4
        # Add other types as needed

        num_elem = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                num_elem *= dim.dim_value
            elif dim.HasField('dim_param'):
                # Dynamic dimension, treat as 1 or heuristic
                # For simulation, maybe warn?
                pass
        return num_elem * bytes_per_elem

    def _register_device(self, device_instance, node_id):
        pass
