"""
Combined example: multi-node multi-GPU GPT-3 inference simulation.
Models a single Transformer layer forward pass.
"""

from perflowai import DeviceConfig, DeviceType, NodeConfig, NodeInstance
from perflowai.simulator.kernel.kernel_simulator import Parameter, GEMMKernelSimulator, AttentionKernelSimulator
from perflowai.simulator.orchestration.op import MallocSimulator
from perflowai.simulator.orchestration.orchestration import OrchestrationResult, Task, Assignment
from perflowai.simulator.comm import AllReduceNetworkSimulator, GroupTopology


def flatten_param(p: Parameter) -> Parameter:
    """Returns a view of the parameter as rank-2 (flattening leading dims), sharing the same name/id."""
    if p.shape and len(p.shape) == 3:
        b, s, d = p.shape
        # Intentionally keep the same name so the memory manager treats it as the same buffer.
        return Parameter(name=p.name, dtype=p.dtype, shape=(b * s, d))
    return p


def create_8node_cluster():
    """Create 8 nodes, each with 8 A100-like GPUs."""
    nodes = []
    all_gpu_ids = []

    # Define A100-like GPU specs
    base_gpu_cfg = DeviceConfig(
        id="template", # Will be overwritten
        type=DeviceType.GPU,
        memory_capacity=80_000,  # 80 GB
        memory_bandwidth=2000.0,  # 2 TB/s
        compute_flops=312e12,  # 312 TFLOPS (BF16)
        intra_node_bandwidth=600.0, # 600 GB/s NVLink
        inter_node_bandwidth=50.0, # 400 Gbps / 8 bits = 50 GB/s
    )
    # Host CPU
    base_cpu_cfg = DeviceConfig(
        id="template_cpu",
        type=DeviceType.CPU,
        memory_capacity=512_000,
        memory_bandwidth=200.0,
        compute_flops=5e12,
    )

    for node_idx in range(8):
        node_gpu_configs = []
        for gpu_idx in range(8):
            # Unique ID for each GPU: Node0_GPU0, Node0_GPU1, etc.
            gid = f"Node{node_idx}_GPU{gpu_idx}"
            g_cfg = DeviceConfig(
                id=gid,
                type=base_gpu_cfg.type,
                memory_capacity=base_gpu_cfg.memory_capacity,
                memory_bandwidth=base_gpu_cfg.memory_bandwidth,
                compute_flops=base_gpu_cfg.compute_flops,
                intra_node_bandwidth=base_gpu_cfg.intra_node_bandwidth,
                inter_node_bandwidth=base_gpu_cfg.inter_node_bandwidth,
            )
            node_gpu_configs.append(g_cfg)
            all_gpu_ids.append(gid)

        # Node Host
        cpu_cfg = DeviceConfig(
            id=f"Node{node_idx}_CPU",
            type=base_cpu_cfg.type,
            memory_capacity=base_cpu_cfg.memory_capacity,
            memory_bandwidth=base_cpu_cfg.memory_bandwidth,
            compute_flops=base_cpu_cfg.compute_flops
        )

        node = NodeInstance(
            nodeConfig=NodeConfig(host=cpu_cfg, gpu_devices=node_gpu_configs)
        )
        nodes.append(node)

    return nodes, all_gpu_ids

def create_tasks_for_gpu(
    global_rank: int,
    gpu_id: str,
    node_id: str,
    num_layers: int,
    gpu_cfg: DeviceConfig,
    tasks_list: list,
    assignments_list: list
):
    """Generates tasks for one GPU (TP=8 scenario)."""

    # GPT-3 Hyperparams
    batch_size = 8
    seq_len = 2048
    d_model = 12288
    n_heads = 96

    # Tensor Parallelism (TP=8)
    tp_size = 8
    # Split heads / d_model
    n_heads_per_rank = n_heads // tp_size
    d_model_per_rank = d_model // tp_size # Not strictly used for all weights, depends on splitting

    # For simplicity, we just divide the large dims by 8 for weight shapes
    # Column Parallel: (d_model, d_model/tp)
    # Row Parallel: (d_model/tp, d_model)

    # Shapes
    input_shape = (batch_size, seq_len, d_model) # Inputs are usually broadcast or split along sequence in some implementations, but let's assume replicated input for TP simple case

    # Attention Q,K,V (Column Parallel)
    w_qkv_shape = (d_model, d_model // tp_size)

    # Output Proj (Row Parallel)
    w_o_shape = (d_model // tp_size, d_model)

    # MLP Up (Column Parallel)
    w_up_shape = (d_model, (4 * d_model) // tp_size)

    # MLP Down (Row Parallel)
    w_down_shape = ((4 * d_model) // tp_size, d_model)

    # Define Parameters (unique naming per rank to avoid confusion in visualizer logs)
    # Suffix with _R{rank}
    suffix = f"_R{global_rank}"

    def p(name, shape):
        return Parameter(name=f"{name}{suffix}", dtype="float16", shape=shape)

    # Reusable Helper
    def add(workload, run_after=None):
        if run_after is None: run_after = []
        t = Task(id=workload.id, workload=workload, run_after=run_after)
        tasks_list.append(t)
        assignments_list.append(Assignment(t.id, gpu_id))
        return t.id

    # 1. Allocations (Weights) - Created once
    # We create params for all layers upfront.

    # To save code space, let's just generate for the loop
    # We track dependencies: layer i depends on layer i-1

    previous_step_id = [] # Initially empty or depend on input load

    # Initial Input Alloc
    input_full = p("Input", input_shape)
    t_load_input = add(MallocSimulator(gpu_cfg, input_full.get_size(), input_full, name=f"Malloc_In{suffix}"))
    previous_step_id = [t_load_input]

    # TP Group Topology (ranks 0-7, 8-15, etc.)
    # Within perflowai, AllReduce identifies peers effectively.
    # We specify group size 8.
    topo = GroupTopology(num_nodes=1, ranks_per_node=tp_size)

    for layer in range(num_layers):
        l_prefix = f"L{layer}{suffix}"

        # --- Params for this layer ---
        w_q = p(f"L{layer}_WQ", w_qkv_shape)
        w_k = p(f"L{layer}_WK", w_qkv_shape)
        w_v = p(f"L{layer}_WV", w_qkv_shape)
        w_o = p(f"L{layer}_WO", w_o_shape)
        w_up = p(f"L{layer}_WUp", w_up_shape)
        w_down = p(f"L{layer}_WDown", w_down_shape)

        layer_weights = [w_q, w_k, w_v, w_o, w_up, w_down]

        # Malloc Weights (Parallel with prev comp, but let's just chain for simplicity)
        weight_malloc_ids = []
        for w in layer_weights:
            t_m = add(MallocSimulator(gpu_cfg, w.get_size(), w, name=f"Malloc_{w.name}"), run_after=[])
            weight_malloc_ids.append(t_m)

        # Depend on input from prev layer AND weight loading
        layer_start_deps = previous_step_id + weight_malloc_ids

        # --- Attention ---
        # QKV Proj
        # Input (replicated) x Weight (split) -> Output (split)
        # Activation shapes
        q_local = p(f"L{layer}_Q", (batch_size, seq_len, d_model // tp_size))
        k_local = p(f"L{layer}_K", (batch_size, seq_len, d_model // tp_size))
        v_local = p(f"L{layer}_V", (batch_size, seq_len, d_model // tp_size))

        # Malloc Intermediates Q, K, V
        t_mq = add(MallocSimulator(gpu_cfg, q_local.get_size(), q_local, name=f"Malloc_{q_local.name}"), run_after=[])
        t_mk = add(MallocSimulator(gpu_cfg, k_local.get_size(), k_local, name=f"Malloc_{k_local.name}"), run_after=[])
        t_mv = add(MallocSimulator(gpu_cfg, v_local.get_size(), v_local, name=f"Malloc_{v_local.name}"), run_after=[])

        # Flatten for GEMM
        in_flat = flatten_param(input_full)
        q_flat = flatten_param(q_local)
        k_flat = flatten_param(k_local)
        v_flat = flatten_param(v_local)

        t_q = add(GEMMKernelSimulator(gpu_cfg, in_flat, w_q, q_flat, name=f"{l_prefix}_GEMMQ", id=f"{l_prefix}_GEMMQ"), run_after=layer_start_deps + [t_mq])
        t_k = add(GEMMKernelSimulator(gpu_cfg, in_flat, w_k, k_flat, name=f"{l_prefix}_GEMMK", id=f"{l_prefix}_GEMMK"), run_after=layer_start_deps + [t_mk])
        t_v = add(GEMMKernelSimulator(gpu_cfg, in_flat, w_v, v_flat, name=f"{l_prefix}_GEMMV", id=f"{l_prefix}_GEMMV"), run_after=layer_start_deps + [t_mv])

        # Attention Score
        attn_out_local = p(f"L{layer}_AttnOut", (batch_size, seq_len, d_model // tp_size))
        t_m_attn = add(MallocSimulator(gpu_cfg, attn_out_local.get_size(), attn_out_local, name=f"Malloc_{attn_out_local.name}"), run_after=[])

        t_attn = add(AttentionKernelSimulator(
            gpu_cfg, q_local, k_local, v_local, attn_out_local,
            num_heads=n_heads_per_rank,
            name=f"{l_prefix}_Attn", id=f"{l_prefix}_Attn"
        ), run_after=[t_q, t_k, t_v, t_m_attn])

        # Output Proj
        # Input (split) x Weight (split cols of rows) -> Partial Sum (replicated shape but partial values)
        # Wait, Row Parallel: Input is split (last dim), Weight is split (first dim). Result is (Batch, Seq, d_model) PARTIAL SUM.
        attn_out_flat = flatten_param(attn_out_local)
        partial_out = p(f"L{layer}_PartOut", (batch_size, seq_len, d_model))
        partial_out_flat = flatten_param(partial_out)

        t_m_pout = add(MallocSimulator(gpu_cfg, partial_out.get_size(), partial_out, name=f"Malloc_{partial_out.name}"), run_after=[])

        t_o = add(GEMMKernelSimulator(gpu_cfg, attn_out_flat, w_o, partial_out_flat, name=f"{l_prefix}_GEMMO", id=f"{l_prefix}_GEMMO"), run_after=[t_attn, t_m_pout])

        # All Reduce (Sum Partial Results)
        # Output is "Full" (Simulated) - In real code it might be in-place on partial_out.
        # We'll assume in-place here so we don't need another Malloc, but logically it's the same buffer.
        t_ar_attn = add(AllReduceNetworkSimulator(
            gpu_cfg, partial_out, topology=topo, name=f"{l_prefix}_AR_Attn", id=f"{l_prefix}_AR_Attn"
        ), run_after=[t_o])

        # --- MLP ---
        # Up Proj (Column Parallel)

        mlp_h_local = p(f"L{layer}_MLPH", (batch_size, seq_len, (4*d_model)//tp_size))
        mlp_h_flat = flatten_param(mlp_h_local)

        t_m_mlph = add(MallocSimulator(gpu_cfg, mlp_h_local.get_size(), mlp_h_local, name=f"Malloc_{mlp_h_local.name}"), run_after=[])

        t_up = add(GEMMKernelSimulator(gpu_cfg, flatten_param(partial_out), w_up, mlp_h_flat, name=f"{l_prefix}_GEMMUp", id=f"{l_prefix}_GEMMUp"), run_after=[t_ar_attn, t_m_mlph])

        # Down Proj (Row Parallel)
        mlp_out_partial = p(f"L{layer}_MLPOutP", (batch_size, seq_len, d_model))

        t_m_mlpo = add(MallocSimulator(gpu_cfg, mlp_out_partial.get_size(), mlp_out_partial, name=f"Malloc_{mlp_out_partial.name}"), run_after=[])

        t_down = add(GEMMKernelSimulator(gpu_cfg, mlp_h_flat, w_down, flatten_param(mlp_out_partial), name=f"{l_prefix}_GEMMDown", id=f"{l_prefix}_GEMMDown"), run_after=[t_up, t_m_mlpo])

        # All Reduce MLP
        t_ar_mlp = add(AllReduceNetworkSimulator(
            gpu_cfg, mlp_out_partial, topology=topo, name=f"{l_prefix}_AR_MLP", id=f"{l_prefix}_AR_MLP"
        ), run_after=[t_down])

        # Update previous_step for next layer
        previous_step_id = [t_ar_mlp]

        # Cleanup layer intermediates (optional, skip for simplicity as user asked for simple)

    return

def run_gpt3_inference():
    print("Setting up GPT-3 (175B) on 8 Nodes (64 GPUs)...")

    # 1. Create Hardware
    nodes, all_gpu_ids = create_8node_cluster()
    print(f"Created {len(nodes)} nodes with total {len(all_gpu_ids)} GPUs.")

    tasks = []
    assignments = []

    # 2. Assign Tasks for each GPU (Implicit DP=8 parallel replicas)
    # We iterate 64 times for the 64 GPUs.
    # We can model standard GPT-3 layers.
    # Since modeling 96 layers for 64 GPUs is heavy, we simulate a few layers.
    num_layers = 4
    print(f"Generating tasks for {num_layers} layers per GPU (TP=8)...")

    # Access device configs from nodes
    # Map ID -> Config
    gpu_configs = {}
    for node in nodes:
        for d_inst in node.gpu_devices:
            gpu_configs[d_inst.id] = d_inst.config

    for i, gpu_id in enumerate(all_gpu_ids):
        # Determine TP rank (0-7)
        tp_rank = i % 8
        node_id = f"Node{i // 8}"

        create_tasks_for_gpu(
            global_rank=i,
            gpu_id=gpu_id,
            node_id=node_id,
            num_layers=num_layers,
            gpu_cfg=gpu_configs[gpu_id],
            tasks_list=tasks,
            assignments_list=assignments
        )

    print(f"Total Tasks Generated: {len(tasks)}")

    # 3. Run Simulator
    res = OrchestrationResult(
        nodes=nodes,
        tasks=tasks,
        assignments=assignments
    )

    print("Simulating...")
    estimate = res.estimate_makespan_s()

    print("\nSimulation Results (TP=8, DP=8):")
    print(f"Makespan ({num_layers} layers): {estimate.time_s * 1000:.4f} ms")

    # Check Memory on GPU0
    peak_mem_gb = estimate.max_memory_usage_bytes[all_gpu_ids[0]] / 1e9
    print(f"Peak Memory Usage (GPU0): {peak_mem_gb:.4f} GB")


if __name__ == "__main__":
    run_gpt3_inference()
