"""
This is an example of using NodeFlow simulator to estimate a running time and memory usage for a workflow.
If you need to try a Pipeline Parallelism simulator, please check out `PPGraph` related examples.

Combined example: single-node single-GPU GPT-3 inference simulation.
Models a single Transformer layer forward pass.
"""
from examples.simulator.workflow.util import flatten_param
from perflowai import DeviceConfig, DeviceType, NodeConfig, NodeInstance, FlowNode
from perflowai.simulator.kernel.kernel_simulator import Parameter, GEMMKernelSimulator, AttentionKernelSimulator
from perflowai.simulator.orchestration.op import MallocSimulator, FreeSimulator
from perflowai.simulator.orchestration.orchestration import OrchestrationResult, Task, Assignment


def run_gpt3_inference():
    print("Setting up GPT-3 (175B) Single Layer Inference Simulation...")

    # 1. Define Hardware (A100-like)
    gpu_id = "GPU0"
    gpu_cfg = DeviceConfig(
        id=gpu_id,
        type=DeviceType.GPU,
        memory_capacity=80_000,  # 80 GB
        memory_bandwidth=2000.0,  # 2 TB/s
        compute_flops=312e12,  # 312 TFLOPS (BF16)
    )
    # Host CPU (Simulated but mostly just as controller here)
    cpu_cfg = DeviceConfig(
        id="CPU0",
        type=DeviceType.CPU,
        memory_capacity=512_000,
        memory_bandwidth=200.0,
        compute_flops=5e12,
    )
    node = NodeInstance(nodeConfig=NodeConfig(host=cpu_cfg, gpu_devices=[gpu_cfg]))

    # 2. Define Model Hyperparams (GPT-3 175B)
    # Using a small batch size for inference
    batch_size = 8
    seq_len = 2048
    d_model = 12288
    n_heads = 96
    d_head = d_model // n_heads  # 128

    print(f"Config: Batch={batch_size}, SeqLen={seq_len}, ModelDim={d_model}, Heads={n_heads}")

    # Shapes
    input_shape = (batch_size, seq_len, d_model)
    weight_shape = (d_model, d_model)
    mlp_up_shape = (d_model, 4 * d_model)
    mlp_down_shape = (4 * d_model, d_model)

    # 3. Define Parameters
    # Data type float16 (2 bytes)

    # Inputs
    x = Parameter(name="input", dtype="float16", shape=input_shape)

    # Attention Weights
    w_q = Parameter(name="W_Q", dtype="float16", shape=weight_shape)
    w_k = Parameter(name="W_K", dtype="float16", shape=weight_shape)
    w_v = Parameter(name="W_V", dtype="float16", shape=weight_shape)
    w_o = Parameter(name="W_O", dtype="float16", shape=weight_shape)

    # MLP Weights
    w_up = Parameter(name="W_up", dtype="float16", shape=mlp_up_shape)
    w_down = Parameter(name="W_down", dtype="float16", shape=mlp_down_shape)

    # Intermediate Activations
    q_out = Parameter(name="Q", dtype="float16", shape=input_shape)
    k_out = Parameter(name="K", dtype="float16", shape=input_shape)
    v_out = Parameter(name="V", dtype="float16", shape=input_shape)
    attn_out = Parameter(name="Attn_Out", dtype="float16", shape=input_shape)
    o_out = Parameter(name="O_Out", dtype="float16", shape=input_shape)

    # MLP Activations
    # MLP hidden expansion: (B, S, 4*D)
    mlp_h = Parameter(name="MLP_H", dtype="float16", shape=(batch_size, seq_len, 4 * d_model))
    mlp_out = Parameter(name="MLP_Out", dtype="float16", shape=input_shape)

    # 4. Define Tasks
    tasks = []
    assignments = []

    def add_task(workload: FlowNode, run_after=None):
        if run_after is None:
            run_after = []
        t_id = workload.id
        tasks.append(Task(id=t_id, workload=workload, run_after=run_after))
        assignments.append(Assignment(t_id, gpu_id))
        return t_id

    # 4.1 Memory Allocation (Simulate loading weights and allocating buffers)
    # For simplicity, we allocate everything upfront or in topological order.
    # Cost model: We ignore allocation time (usually negligible or hidden),
    # but we care about memory capacity checks.

    # Weights & Inputs (Pre-loaded)
    init_vars = [x, w_q, w_k, w_v, w_o, w_up, w_down]
    init_loaders = []
    for var in init_vars:
        m = MallocSimulator(gpu_cfg, size=var.get_size(), allocated=var, name=f"Alloc_{var.name}")
        init_loaders.append(add_task(m))

    # Intermediates
    inter_vars = [q_out, k_out, v_out, attn_out, o_out, mlp_h, mlp_out]
    inter_allocs = []
    for var in inter_vars:
        m = MallocSimulator(gpu_cfg, size=var.get_size(), allocated=var, name=f"Alloc_{var.name}")
        # Allocation can happen just before compute. For simple linear flow,
        # let's just make them depend on init.
        inter_allocs.append(add_task(m, run_after=init_loaders))

    base_deps = init_loaders + inter_allocs

    # 4.2 Computation Tasks

    # Rank-2 views for GEMM
    x_flat = flatten_param(x)
    q_flat = flatten_param(q_out)
    k_flat = flatten_param(k_out)
    v_flat = flatten_param(v_out)

    # 1. Q, K, V Projections
    gemm_q = GEMMKernelSimulator(gpu_cfg, a=x_flat, b=w_q, c=q_flat, name="GEMM_Q", id="GEMM_Q")
    t_q = add_task(gemm_q, run_after=base_deps)

    gemm_k = GEMMKernelSimulator(gpu_cfg, a=x_flat, b=w_k, c=k_flat, name="GEMM_K", id="GEMM_K")
    t_k = add_task(gemm_k, run_after=base_deps)

    gemm_v = GEMMKernelSimulator(gpu_cfg, a=x_flat, b=w_v, c=v_flat, name="GEMM_V", id="GEMM_V")
    t_v = add_task(gemm_v, run_after=base_deps)

    # 2. Attention
    # Uses rank-3 params: q_out, k_out, v_out, attn_out
    attn_op = AttentionKernelSimulator(
        gpu_cfg,
        q=q_out, k=k_out, v=v_out, o=attn_out,
        num_heads=n_heads,
        name="Attention_Op",
        id="Attention_Op",
    )
    t_attn = add_task(attn_op, run_after=[t_q, t_k, t_v])

    # 3. Output Projection
    # Input: attn_out (flat), Weight: w_o, Output: o_out (flat)
    attn_out_flat = flatten_param(attn_out)
    o_out_flat = flatten_param(o_out)

    gemm_o = GEMMKernelSimulator(gpu_cfg, a=attn_out_flat, b=w_o, c=o_out_flat, name="GEMM_O", id="GEMM_O")
    t_o = add_task(gemm_o, run_after=[t_attn])

    # 4. MLP Up
    # Input: o_out (flat), Weight: w_up, Output: mlp_h (flat)
    # Note: Skipping Add & RMSNorm before MLP for FLOPs simplicity (negligible).
    # Realistically, o_out is added to x, then normed.
    mlp_h_flat = flatten_param(mlp_h)

    gemm_up = GEMMKernelSimulator(gpu_cfg, a=o_out_flat, b=w_up, c=mlp_h_flat, name="GEMM_Up", id="GEMM_Up")
    t_up = add_task(gemm_up, run_after=[t_o])

    # 5. MLP Down
    # Input: mlp_h (flat), Weight: w_down, Output: mlp_out (flat)
    # Skipping GeLU activation cost (elementwise, negligible compared to GEMM).
    mlp_out_flat = flatten_param(mlp_out)

    gemm_down = GEMMKernelSimulator(gpu_cfg, a=mlp_h_flat, b=w_down, c=mlp_out_flat, name="GEMM_Down", id="GEMM_Down")
    t_down = add_task(gemm_down, run_after=[t_up])

    # 4.3 Cleanup (Free Memory)
    # Freeing intermediates
    # Wait until last use.
    cleanup_tasks = []

    # Free input x after GEMM_Q/K/V? No, we need it for residual.
    # But we didn't model residual add explicitly.
    # Let's free everything at the end for simplicity.
    all_vars = init_vars + inter_vars
    for var in all_vars:
        f = FreeSimulator(gpu_cfg, size=var.get_size(), allocated=var, name=f"Free_{var.name}")
        cleanup_tasks.append(add_task(f, run_after=[t_down]))

    # 5. Run Simulator
    res = OrchestrationResult(
        nodes=[node],
        tasks=tasks,
        assignments=assignments
    )

    estimate = res.estimate_makespan()

    print("\nSimulation Results:")
    print(f"Total Latency (1 Layer): {estimate.time_s * 1000:.4f} ms")

    peak_mem_gb = estimate.max_memory_usage_bytes[gpu_id] / 1e9
    print(f"Peak Memory Usage: {peak_mem_gb:.4f} GB")


if __name__ == "__main__":
    run_gpt3_inference()
