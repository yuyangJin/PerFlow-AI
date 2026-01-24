"""
Combined example: single-node 2-GPU GPT-3 inference simulation with Tensor Parallelism.
Models a single Transformer layer forward pass split across 2 GPUs (TP=2).
"""
from examples.simulator.util import flatten_param
from perflowai import DeviceConfig, DeviceType, NodeConfig, NodeInstance, FlowNode
from perflowai.simulator.kernel.kernel_simulator import Parameter, GEMMKernelSimulator
from perflowai.simulator.comm.comm_simulator import AllReduceNetworkSimulator, GroupTopology
from perflowai.simulator.orchestration.orchestration import OrchestrationResult, Task, Assignment
from perflowai.simulator.orchestration.op import MallocSimulator, FreeSimulator


def run_gpt3_2gpu_inference():
    print("Setting up GPT-3 (175B) 2-GPU Tensor Parallel Inference Simulation...")

    # 1. Define Hardware (A100-like with NVLink)
    # NVLink 600GB/s total BW -> ~300GB/s per direction per link?
    # We set intra_node_bandwidth to 600.0 GB/s for high speed interconnect.
    intra_bw = 600.0

    gpu_cfg0 = DeviceConfig(
        id="GPU0", type=DeviceType.GPU, memory_capacity=80_000, memory_bandwidth=2000.0, compute_flops=312e12,
        intra_node_bandwidth=intra_bw
    )
    gpu_cfg1 = DeviceConfig(
        id="GPU1", type=DeviceType.GPU, memory_capacity=80_000, memory_bandwidth=2000.0, compute_flops=312e12,
        intra_node_bandwidth=intra_bw
    )

    cpu_cfg = DeviceConfig(
        id="CPU0", type=DeviceType.CPU, memory_capacity=512_000, memory_bandwidth=200.0, compute_flops=5e12
    )

    node = NodeInstance(nodeConfig=NodeConfig(host=cpu_cfg, gpu_devices=[gpu_cfg0, gpu_cfg1]))

    # 2. Model Config (GPT-3 175B)
    batch_size = 8
    seq_len = 2048
    d_model = 12288
    n_heads = 96

    # Tensor Parallelism Config
    tp_size = 2
    assert d_model % tp_size == 0
    assert n_heads % tp_size == 0

    d_model_per_rank = d_model // tp_size
    n_heads_per_rank = n_heads // tp_size

    print(f"Global Config: Batch={batch_size}, SeqLen={seq_len}, ModelDim={d_model}, Heads={n_heads}")
    print(f"TP Config (Rank size): ModelDim_Part={d_model_per_rank}, Heads_Part={n_heads_per_rank}")

    # 3. Helpers
    tasks = []
    assignments = []

    def add_task(workload: FlowNode, gpu_id: str, run_after=None) -> str:
        if run_after is None: run_after = []
        # Task ID: combine workload name and GPU ID.
        # Note: workload.id is also unique (e.g. gemm_qkv_GPU0), but we use name_gpu_id for Task ID too.
        t_id = f"{workload.m_name}_{gpu_id}"

        tasks.append(Task(id=t_id, workload=workload, run_after=run_after))
        assignments.append(Assignment(t_id, gpu_id))
        return t_id


    # --- Builder / Factory Pattern Helpers ---
    def create_gemm_simulator(gpu_id: str, device_config, a, b, c, name):
        return GEMMKernelSimulator(
            device_config, a=a, b=b, c=c, name=name, id=f"{name}_{gpu_id}"
        )

    def create_allreduce_simulator(gpu_id: str, device_config, x, topology, name):
        return AllReduceNetworkSimulator(
            device_config, x=x, topology=topology, name=name, id=f"{name}_{gpu_id}"
        )

    # 4. Topology for Comm
    tp_topology = GroupTopology(num_nodes=1, ranks_per_node=2)

    # 5. Build Layer: QKV (Col) -> Attn -> O (Row) -> AllReduce

    # Shapes
    # X: (B, S, H)
    input_shape = (batch_size, seq_len, d_model)
    # W_qkv (Column Parallel): (H, 3 * H/P)
    w_qkv_shape_tp = (d_model, 3 * d_model_per_rank)
    # Output of QKV: (B, S, 3 * H/P)
    qkv_out_shape_tp = (batch_size, seq_len, 3 * d_model_per_rank)

    # Attn Out Local: (B, S, H/P)
    attn_out_shape_tp = (batch_size, seq_len, d_model_per_rank)

    # W_o (Row Parallel): (H/P, H)
    w_o_shape_tp = (d_model_per_rank, d_model)
    # Output of O: (B, S, H) -- Partial Sum
    o_out_shape_partial = (batch_size, seq_len, d_model)

    # --- GPU 0 Parameters ---
    p_x_0 = Parameter(name="X", dtype="float16", shape=input_shape)
    p_wqkv_0 = Parameter(name="W_qkv", dtype="float16", shape=w_qkv_shape_tp)
    p_qkv_out_0 = Parameter(name="QKV_out", dtype="float16", shape=qkv_out_shape_tp)

    p_attn_out_0 = Parameter(name="Attn_out", dtype="float16", shape=attn_out_shape_tp)
    p_wo_0 = Parameter(name="W_o", dtype="float16", shape=w_o_shape_tp)
    p_o_out_0 = Parameter(name="O_out_partial", dtype="float16", shape=o_out_shape_partial)

    # --- GPU 1 Parameters ---
    p_x_1 = Parameter(name="X", dtype="float16", shape=input_shape)
    p_wqkv_1 = Parameter(name="W_qkv", dtype="float16", shape=w_qkv_shape_tp)
    p_qkv_out_1 = Parameter(name="QKV_out", dtype="float16", shape=qkv_out_shape_tp)

    p_attn_out_1 = Parameter(name="Attn_out", dtype="float16", shape=attn_out_shape_tp)
    p_wo_1 = Parameter(name="W_o", dtype="float16", shape=w_o_shape_tp)
    p_o_out_1 = Parameter(name="O_out_partial", dtype="float16", shape=o_out_shape_partial)

    # --- Step 0: Memory Allocation ---
    # GPU0
    gpu0_params = [p_x_0, p_wqkv_0, p_qkv_out_0, p_attn_out_0, p_wo_0, p_o_out_0]
    alloc_tasks_0 = []
    for p in gpu0_params:
        m = MallocSimulator(gpu_cfg0, size=p.get_size(), allocated=p, name=f"Alloc_{p.name}_GPU0")
        alloc_tasks_0.append(add_task(m, "GPU0"))

    # GPU1
    gpu1_params = [p_x_1, p_wqkv_1, p_qkv_out_1, p_attn_out_1, p_wo_1, p_o_out_1]
    alloc_tasks_1 = []
    for p in gpu1_params:
        m = MallocSimulator(gpu_cfg1, size=p.get_size(), allocated=p, name=f"Alloc_{p.name}_GPU1")
        alloc_tasks_1.append(add_task(m, "GPU1"))

    # --- Step 1: QKV Projection (Column Parallel) ---
    # GPU0
    gemm_qkv_0 = create_gemm_simulator(
        "GPU0", gpu_cfg0, a=flatten_param(p_x_0), b=p_wqkv_0, c=flatten_param(p_qkv_out_0), name="gemm_qkv"
    )
    t_qkv_0 = add_task(gemm_qkv_0, "GPU0", run_after=alloc_tasks_0)

    # GPU1
    gemm_qkv_1 = create_gemm_simulator(
        "GPU1", gpu_cfg1, a=flatten_param(p_x_1), b=p_wqkv_1, c=flatten_param(p_qkv_out_1), name="gemm_qkv"
    )
    t_qkv_1 = add_task(gemm_qkv_1, "GPU1", run_after=alloc_tasks_1)

    # --- Step 2: Output Projection (Row Parallel) ---
    # GPU0
    gemm_o_0 = create_gemm_simulator(
        "GPU0", gpu_cfg0, a=flatten_param(p_attn_out_0), b=p_wo_0, c=flatten_param(p_o_out_0), name="gemm_o"
    )
    t_o_0 = add_task(gemm_o_0, "GPU0", run_after=[t_qkv_0])

    # GPU1
    gemm_o_1 = create_gemm_simulator(
        "GPU1", gpu_cfg1, a=flatten_param(p_attn_out_1), b=p_wo_1, c=flatten_param(p_o_out_1), name="gemm_o"
    )
    t_o_1 = add_task(gemm_o_1, "GPU1", run_after=[t_qkv_1])


    # --- Step 3: AllReduce (Sum) ---
    # Crucial Logic: The AllReduce tasks on ANY rank should ideally assume
    # that the data is ready on ALL participating ranks.
    # So we make both AllReduce tasks depend on both computations.
    # This simulates a BARRIER.

    barrier_deps = [t_o_0, t_o_1]

    # Comm GPU0
    ar_0 = create_allreduce_simulator(
        "GPU0", device_config=gpu_cfg0, x=p_o_out_0, topology=tp_topology, name="ar_o"
    )
    t_ar_0 = add_task(ar_0, "GPU0", run_after=barrier_deps)

    # Comm GPU1
    ar_1 = create_allreduce_simulator(
        "GPU1", device_config=gpu_cfg1, x=p_o_out_1, topology=tp_topology, name="ar_o"
    )
    t_ar_1 = add_task(ar_1, "GPU1", run_after=barrier_deps)

    # --- Step 4: Cleanup (Free Memory) ---
    for p in gpu0_params:
        f = FreeSimulator(gpu_cfg0, size=p.get_size(), allocated=p, name=f"Free_{p.name}_GPU0")
        add_task(f, "GPU0", run_after=[t_ar_0])

    for p in gpu1_params:
        f = FreeSimulator(gpu_cfg1, size=p.get_size(), allocated=p, name=f"Free_{p.name}_GPU1")
        add_task(f, "GPU1", run_after=[t_ar_1])

    # --- Simulation ---
    res = OrchestrationResult(
        nodes=[node],
        tasks=tasks,
        assignments=assignments
    )

    estimate = res.estimate_makespan()
    print(f"\nEstimated Latency (Partial Layer): {estimate.time_s * 1000:.4f} ms")

    peak_mem_gb0 = estimate.max_memory_usage_bytes["GPU0"] / 1e9
    peak_mem_gb1 = estimate.max_memory_usage_bytes["GPU1"] / 1e9
    print(f"Peak Memory Usage GPU0: {peak_mem_gb0:.4f} GB")
    print(f"Peak Memory Usage GPU1: {peak_mem_gb1:.4f} GB")


if __name__ == "__main__":
    run_gpt3_2gpu_inference()

