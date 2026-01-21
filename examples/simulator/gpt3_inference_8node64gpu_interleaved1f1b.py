"""
Combined example: multi-node multi-GPU GPT-3 inference simulation.
Models a single Transformer layer forward pass.
"""

from perflowai import DeviceConfig, DeviceType, NodeConfig, NodeInstance
from perflowai.simulator.comm import AllReduceNetworkSimulator
from perflowai.simulator.comm.comm_simulator import GroupTopology, P2PSimulator
from perflowai.simulator.kernel.kernel_simulator import Parameter, GEMMKernelSimulator, AttentionKernelSimulator
from perflowai.simulator.orchestration.op import MallocSimulator
from perflowai.simulator.orchestration.orchestration import OrchestrationResult, Task, Assignment


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
        id="template",  # Will be overwritten
        type=DeviceType.GPU,
        memory_capacity=80_000,  # 80 GB
        memory_bandwidth=2000.0,  # 2 TB/s
        compute_flops=312e12,  # 312 TFLOPS (BF16)
        intra_node_bandwidth=600.0,  # 600 GB/s NVLink
        inter_node_bandwidth=50.0,  # 400 Gbps / 8 bits = 50 GB/s
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

    # ----------------------------------------------------------------------------------
    #  Implement 1F1B Pipeline Parallelism Logic
    # ----------------------------------------------------------------------------------

    # Pipeline Config
    num_stages = 8  # 1 Stage per Node
    tp_size = 8  # 8 GPUs per Stage
    num_microbatches = 16
    layers_per_stage = num_layers

    # Global Task Registry: matrix[stage][microbatch][type] -> TaskID
    # Type: 'fwd_comp', 'bwd_comp', 'send_fwd', 'recv_fwd', 'send_bwd', 'recv_bwd'
    # Actually we just track the LAST task of the computation chain to link dependencies.
    registry = {}

    # Helper to create Fwd/Bwd computation chain for a set of layers
    def create_computation_chain(
            stage_idx, mb_idx, c_type,
            base_run_after_ids
    ):
        """
        Creates all tasks for (Stage, MB, Type).
        Returns the ID of the LAST task in the chain.
        """
        # We need to iterate over all TP ranks in this stage
        # But we only return the task IDs for a representative rank (e.g. Rank 0) to use for
        # inter-stage dependency logic (assuming all ranks synchronize via AllReduce).
        # OR we return a list of last tasks for all ranks?
        # Actually, P2P usually happens on one rank or all ranks.
        # Megatron usually send from all ranks orgather -> send -> scatter.
        # Let's assume P2P All-to-All or just Parallel P2P (Rank i sends to Rank i next stage).

        last_task_ids = []

        for rank_offset in range(tp_size):
            global_rank = stage_idx * tp_size + rank_offset
            gpu_id = all_gpu_ids[global_rank]
            gpu_cfg = gpu_configs[gpu_id]

            # Suffix for uniqueness
            suffix = f"_S{stage_idx}_R{rank_offset}_M{mb_idx}_{c_type}"

            # Param Helper
            def p(name, shape):
                return Parameter(name=f"{name}{suffix}", dtype="float16", shape=shape)

            def add(workload, run_after=None):
                if run_after is None: run_after = []
                t = Task(id=workload.id, workload=workload, run_after=run_after)
                tasks.append(t)
                assignments.append(Assignment(t.id, gpu_id))
                return t.id

            # Link to the previous step provided by caller
            # For rank_offset, we might depend on specific previous tasks on this GPU if provided
            # But here `base_run_after_ids` are likely Comm tasks or previous Compute tasks.
            # We filter for this GPU's dependencies if we were meticulous,
            # but simplified: depend on all provided `base_run_after_ids`. (Or filtering)

            current_deps = list(base_run_after_ids)

            # --- Model Definition (Simplified for brevity) ---
            # We assume Fwd and Bwd have same cost structure (Bwd = 2x Fwd simulated by 2 passes or bigger GEMM)
            # Cost Factor: Fwd=1.0, Bwd=2.0

            cost_factor = 2 if c_type == 'bwd' else 1

            # Shapes
            batch_size = 8
            seq_len = 2048
            d_model = 12288

            # Just one big GEMM to represent the layer cost?
            # Or iterate layers. Let's iterate layers_per_stage.
            # To save simulation object count (which is huge now: 8 stages * 8 GPUs * 16 MB * 4 Layers * ~15 Ops),
            # that is ~30,000 tasks. Orchestrator might be slow but fine.
            # Let's reduce layers_per_stage to 1 for specific demo unless needed.
            # User asked for "gpt3 multi-layer transformer engine", so we keep it.

            # Initial Input (activation from prev stage)
            # We simulate it creates a new "Input" param
            input_shape = (batch_size, seq_len, d_model)

            for l in range(layers_per_stage):
                l_name = f"L{l}"

                # --- Attention ----
                # Simplified: 1 GEMM for QKV, 1 P2P-like dependency or just compute time
                # We reuse the logic from previous function but condensed.

                # 1. QKV
                # (B, S, D) * (D, D) -> (B, S, D) (Actually 3 of them)
                # Cost is dominant part.

                # Workload = GEMM(D, D, D) * batch * seq
                # We just create one KernelSimulator representing the aggregate COMPUTE of the layer
                # To reduce graph size for the 1F1B demo.
                # Justification: "Multi-layer transformer engine" was implemented in previous step.
                # Now we focus on Pipeline Sched.
                # If we want full fidelity, graph becomes huge (~50k nodes).
                # Let's do Full Fidelity for 1 Layer per Stage.

                # ... Just reusing specific parts ...

                # QKV GEMM (Combined)
                # D * 3D
                w_qkv = p(f"{l_name}_WQKV", (d_model, 3 * d_model // tp_size))

                # Faked input/output params
                act_in = p(f"{l_name}_In", input_shape)
                act_out = p(f"{l_name}_Out", input_shape)

                # Duration Scaler for Bwd
                # We can't easily scale duration of KernelSimulator without changing internal logic or using Generic one.
                # We'll validly schedule 2 GEMMs for Bwd.

                passes = 1 if c_type == 'fwd' else 2

                step_task_id = None

                for pass_i in range(passes):
                    t_gemm1 = add(GEMMKernelSimulator(
                        gpu_cfg,
                        flatten_param(act_in),
                        w_qkv,
                        flatten_param(act_out),
                        name=f"{l_name}_GEMM1_P{pass_i}",
                        id=f"{l_name}_GEMM1_P{pass_i}{suffix}"
                    ), run_after=current_deps)

                    # Attention (Self)
                    t_attn = add(AttentionKernelSimulator(
                        gpu_cfg, act_out, act_out, act_out, act_out,  # Fake params
                        num_heads=96 // 8,
                        name=f"{l_name}_Attn_P{pass_i}",
                        id=f"{l_name}_Attn_P{pass_i}{suffix}"
                    ), run_after=[t_gemm1])

                    # Output Dense + MLP (Aggregated into one big GEMM for simulation speed)
                    # 1. Proj (D*D)
                    # 2. MLP Up (D*4D)
                    # 3. MLP Down (4D*D)
                    # Total approx: 1 + 4 + 4 = 9 units of D*D matmul.
                    # QKV was 3 units.

                    w_huge = p(f"{l_name}_Huge", (d_model, 9 * d_model // tp_size))

                    t_mlp = add(GEMMKernelSimulator(
                        gpu_cfg,
                        flatten_param(act_out),
                        w_huge,
                        flatten_param(act_in),  # Recycled buffer
                        name=f"{l_name}_Rest_P{pass_i}",
                        id=f"{l_name}_Rest_P{pass_i}{suffix}"
                    ), run_after=[t_attn])

                    # All Reduce
                    topo = GroupTopology(1, tp_size)
                    t_ar = add(AllReduceNetworkSimulator(
                        gpu_cfg, act_in, topology=topo,
                        name=f"{l_name}_AR",
                        id=f"{l_name}_AR_{pass_i}{suffix}"
                    ), run_after=[t_mlp])

                    current_deps = [t_ar]
                    step_task_id = t_ar

            if step_task_id is not None:
                last_task_ids.append(step_task_id)

        return last_task_ids

    # Helper for P2P
    def create_p2p(stage_idx, mb_idx, p_type, dep_tasks):
        """
        p_type: 'send_fwd', 'recv_fwd', 'send_bwd', 'recv_bwd'
        """
        # Determine direction
        if p_type == 'send_fwd':
            src_stage, dst_stage = stage_idx, stage_idx + 1
        elif p_type == 'recv_fwd':
            src_stage, dst_stage = stage_idx - 1, stage_idx
        elif p_type == 'send_bwd':
            src_stage, dst_stage = stage_idx, stage_idx - 1
        elif p_type == 'recv_bwd':
            src_stage, dst_stage = stage_idx + 1, stage_idx
        else:
            raise ValueError

        # Create Task for each TP pair (Rank r sends to Rank r)
        # Assuming P2P happens in parallel

        task_ids = []
        for r in range(tp_size):
            src_rank_global = src_stage * tp_size + r
            dst_rank_global = dst_stage * tp_size + r

            # Where does this task run?
            # Send runs on Src, Recv runs on Dst.
            if 'send' in p_type:
                active_gpu = all_gpu_ids[src_rank_global]
                # Dep tasks should be from Src GPU
                my_deps = [dep_tasks[r]] if dep_tasks else []
            else:
                active_gpu = all_gpu_ids[dst_rank_global]
                # Dep tasks should be from Dst GPU (e.g. previous computation finished using buffer)
                # AND the paired Send task?
                # In this simulator, Recv 'run_after' Send ensures capability.
                my_deps = [dep_tasks[r]] if dep_tasks else []

            gpu_cfg = gpu_configs[active_gpu]
            suffix = f"_S{stage_idx}_R{r}_M{mb_idx}_{p_type}"

            # Activation Size
            # (B, S, D) float16
            size = 8 * 2048 * 12288 * 2

            # Is Inter-node?
            # Node ID = global // 8.
            src_node = src_rank_global // 8
            dst_node = dst_rank_global // 8
            is_inter = (src_node != dst_node)

            sim = P2PSimulator(gpu_cfg, size, is_inter, name=p_type, id=p_type + suffix)

            t = Task(id=sim.id, workload=sim, run_after=my_deps)
            tasks.append(t)
            assignments.append(Assignment(t.id, active_gpu))
            task_ids.append(t.id)

        return task_ids

    # ==============================================================================
    # 1F1B Schedule Generation
    # ==============================================================================

    # Store task IDs: task_map[stage][mb][step_type] = [list of task ids for 8 gpus]
    task_map = {}

    # Init dictionary
    for s in range(num_stages):
        task_map[s] = {}
        for m in range(num_microbatches):
            task_map[s][m] = {}

    # We enforce order by maintaining a "last_op_on_gpu" list
    last_op_on_gpu = [[] for _ in range(64)]

    def register_and_chain_execution(stage, mb, step_type, tid_list):
        task_map[stage][mb][step_type] = tid_list
        # Chain resource dependency (Execution Order)
        for r in range(tp_size):
            gid_idx = stage * tp_size + r
            # The new task must run after the previous task on this GPU
            if last_op_on_gpu[gid_idx]:
                # Modify the Task object to add dependency
                # We have to find the task in `tasks` list.
                # Since we just added it, it's likely manageable.
                # But tasks list is large. We can use a dict for fast lookup?
                # Or just set run_after at creation time.
                pass
            # Update last op
            last_op_on_gpu[gid_idx] = tid_list[r]

    # Re-impl Create helpers to accept 'last_op'
    # Actually, we can just build the schedule list first, then create tasks sequentially.

    schedule_grid = [[] for _ in range(num_stages)]

    for s in range(num_stages):
        # 1F1B Logic
        warmup = min(num_microbatches, num_stages - s - 1)
        active_mbs = num_microbatches - warmup

        # 1. Warmup Fwds
        for m in range(warmup):
            schedule_grid[s].append(('fwd', m))

        # 2. 1F1B Loop
        for m in range(active_mbs):
            wm = m + warmup  # actual mb index for fwd
            schedule_grid[s].append(('fwd', wm))

            # Accompanied Bwd is for which MB?
            # Standard: earliest pending Bwd.
            # Pending Bwd starts from 0.
            bm = m
            schedule_grid[s].append(('bwd', bm))

        # 3. Cooldown Bwds (remaining)
        # We did `active_mbs` Bwds. Total `num_microbatches`.
        # Remaining = num_microbatches - active_mbs = warmup.
        done_bwds = active_mbs
        for m in range(done_bwds, num_microbatches):
            schedule_grid[s].append(('bwd', m))

    # Now Iterate the Schedule and Create Tasks

    # Mapping to find Data Dependencies
    # data_deps[stage][mb]['fwd_done'] -> tids
    # data_deps[stage][mb]['recv_fwd_done'] -> tids
    data_deps = {}  # Nested dict

    def get_dep(s, m, key):
        if s not in data_deps: data_deps[s] = {}
        if m not in data_deps[s]: data_deps[s][m] = {}
        return data_deps[s][m].get(key, [])

    def set_dep(s, m, key, val):
        if s not in data_deps: data_deps[s] = {}
        if m not in data_deps[s]: data_deps[s][m] = {}
        data_deps[s][m][key] = val

    # Pre-allocate dummy tensors for Compute simulation on all GPUs
    # To avoid Malloc overhead in the loop and satisfy memory checks
    # Use realistic shapes for memory and flops estimation
    _d_model = 12288
    _batch = 8
    _seq = 2048

    # Activation: (B*S, H)
    _shape_act = (_batch * _seq, _d_model) # A (m, k)

    # Super-Weight: (H, N_eff)
    # Total Weights per GPU = (12 * H^2 / TP) * L
    # H * N_eff = 12 * H^2 * L / TP => N_eff = 12 * H * L / TP
    _n_eff = int(12 * _d_model * layers_per_stage / tp_size)
    _shape_w = (_d_model, _n_eff) # B (k, n)

    # Output: (B*S, N_eff)
    _shape_out = (_batch * _seq, _n_eff) # C (m, n)

    # Simulate realistic pipeline: We need to reserve "Pipeline Depth" number of Activation slots
    # because in steady state of 1F1B, there are num_stages microbatches in flight.
    _activation_slots = num_stages

    gpu_dummy_params = {}
    for gid in all_gpu_ids:
        gpu_cfg = gpu_configs[gid]
        suffix_w = f"_init_{gid}_W"

        # 1. Static Weights (Allocated only once)
        p_w = Parameter(f"Weight{suffix_w}", dtype="float16", shape=_shape_w)
        t_mw = Task(id=f"MallocW{suffix_w}", workload=MallocSimulator(gpu_cfg, p_w.get_size(), p_w, name=f"MallocW{suffix_w}"), run_after=[])
        tasks.append(t_mw)
        assignments.append(Assignment(t_mw.id, gid))

        # 2. Ring Buffer for Activations & Outputs (Allocated N slots)
        act_slots = []
        out_slots = []
        malloc_deps_slots = []

        for slot_i in range(_activation_slots):
            suffix_slot = f"_init_{gid}_Slot{slot_i}"

            p_act = Parameter(f"Act{suffix_slot}", dtype="float16", shape=_shape_act)
            p_out = Parameter(f"Out{suffix_slot}", dtype="float16", shape=_shape_out)

            t_ma = Task(id=f"MallocAct{suffix_slot}", workload=MallocSimulator(gpu_cfg, p_act.get_size(), p_act, name=f"MallocAct{suffix_slot}"), run_after=[])
            t_mo = Task(id=f"MallocOut{suffix_slot}", workload=MallocSimulator(gpu_cfg, p_out.get_size(), p_out, name=f"MallocOut{suffix_slot}"), run_after=[])

            tasks.extend([t_ma, t_mo])
            assignments.extend([Assignment(t_ma.id, gid), Assignment(t_mo.id, gid)])

            act_slots.append(p_act)
            out_slots.append(p_out)
            # Each slot depends on its own Mallocs + global Weight Malloc
            malloc_deps_slots.append([t_ma.id, t_mw.id, t_mo.id])

        # Store as (List[Act], W, List[Out], List[Deps])
        gpu_dummy_params[gid] = (act_slots, p_w, out_slots, malloc_deps_slots)

    print("Generating Tasks with 1F1B dependency...")

    # We process step-by-step across all stages?
    # No, we can process stage by stage, but data deps might prevent topological creation if we create consumers before producers?
    # Python objects creation order doesn't matter for `run_after` strings, BUT PerFlow Task object requires run_after to be List[str] or IDs?
    # The `Task` definition uses `run_after: List[Union[str, int]]`. So we can use IDs even if not created yet?
    # Wait, `run_after` resolves at runtime in Orchestrator?
    # Orchestrator checks `if dep not in tasks`. So tasks must exist in the map *eventually*.
    # So creation order doesn't strict.

    # However, to `add_task` properly, we need IDs.

    # Let's verify schedule length
    # for s in range(num_stages):
    #     print(f"Stage {s} steps: {len(schedule_grid[s])}")

    # We iterate stages.

    for s in range(num_stages):
        # GPU Execution Stream
        # Current 'previous' task on each rank
        exec_stream_last = [[] for _ in range(tp_size)]

        for step_type, mb in schedule_grid[s]:

            # -------------------------------------------------------------
            # FORWARD
            # -------------------------------------------------------------
            if step_type == 'fwd':
                # Prerequisites:
                # 1. Recv from Prev Stage (if s > 0)
                # 2. Exec Stream (Implicitly added later)

                # A. Recv
                recv_tasks = []
                if s > 0:
                    # Depends on send from s-1
                    # We might not have IDs for s-1 yet if we iterate stages 0..7
                    # But s-1 is processed.

                    # Find Send task IDs from s-1
                    sender_ids = get_dep(s - 1, mb, 'send_fwd_done')

                    # Create Recv Tasks
                    # Recv depends on Sender
                    # And Recv needs to be scheduled on GPU?
                    # Yes, Recv is an op. It also effectively blocks computation.
                    # Usually Recv and Compute can overlap, but here we run them strictly or dependent.
                    # Recv runs strictly after Sender.
                    recv_tasks = create_p2p(s, mb, 'recv_fwd', sender_ids)

                    # Chain execution stream: Recv occupies GPU (simplified) or at least must wait for slot
                    for r in range(tp_size):
                        # Task: recv_tasks[r]
                        # Add dep-on-exec-stream to THIS task object
                        # We have to access task object to append run_after?
                        # `tasks` list has them.
                        pass
                        # To simplify, we make Compute depend on Recv.
                        # And Recv depends on Sender.
                        # We don't force Recv to serialize with previous Compute on this node strictly,
                        # allowing 'arrival' to happen anytime. But Computation must wait for arrival.

                # B. Compute Fwd
                # Depends on Recv (if s>0)
                # Depends on Exec Stream (Previous step on this GPU)

                compute_deps = []
                # Per rank, we combine (Recv[r]) + (ExecStream[r])
                # We pass 'ExecStream' to helper, it resolves per rank.

                # Prepare base deps per rank
                # list of lists
                deps_per_rank = []
                for r in range(tp_size):
                    d = []
                    if recv_tasks: d.append(recv_tasks[r])
                    if exec_stream_last[r]: d.append(exec_stream_last[r])
                    deps_per_rank.append(d)

                # Create Fwd Chain
                # Note: `create_computation_chain` needs to be updated to accept per-rank run_after.
                # I'll hack it: pass the list-of-lists, handle inside.

                # Wait, I can't modify `create_computation_chain` easily signature-wise without rewriting it
                # Logic above: `current_deps = list(base_run_after_ids)` is shared.
                # I need to separate it.

                # Let's Inline the essential part or assuming `create_computation_chain` is smart.
                # I will Modify `create_computation_chain` to accept `per_rank_deps` (List[List]).

                # (Skipping modifying helper again, I will just call it inside loop or something? No)
                # I will replace the helper above with a loop here for flexibility.

                fwd_task_ids = []
                for r in range(tp_size):
                    global_rank = s * tp_size + r
                    gid = all_gpu_ids[global_rank]
                    gpu_cfg = gpu_configs[gid]

                    my_deps = (deps_per_rank[r] if deps_per_rank else [])

                    # Create single huge GEMM for Fwd
                    # Fwd Cost
                    suffix = f"_S{s}_R{r}_M{mb}_fwd"

                    # Use Ring Buffer Index
                    _activation_slots = num_stages
                    slot_idx = mb % _activation_slots
                    dummy_params_all = gpu_dummy_params[gid]

                    # Unpack
                    p_act = dummy_params_all[0][slot_idx]
                    p_w   = dummy_params_all[1]
                    p_out = dummy_params_all[2][slot_idx]
                    p_deps = dummy_params_all[3][slot_idx]

                    # Ensure we invoke mallocs first
                    gemm_deps = my_deps + p_deps

                    t = Task(
                        id=f"Comp_Fwd{suffix}",
                        workload=GEMMKernelSimulator(
                            gpu_cfg,
                            p_act, p_w, p_out,
                            name=f"ChunkFwd", id=f"ChunkFwd{suffix}",
                            # Manual workload override for precise duration?
                            # Or just big shape.
                            # 4 layers * 4 big GEMMs.
                            # Let's use shape: 16 * GEMM(B,S,D^2).
                        ),
                        run_after=gemm_deps
                    )
                    # Overwrite simulate for GEMM to just return cost?
                    # Or trust GEMMKernelSimulator.
                    # Using Standard Params:
                    # 16 * (Batch, Seq, D) x (D, D)
                    # Let's just create 4 Layer tasks.

                    # Shortcut: Create ONE Task that represents the whole chunk
                    tasks.append(t)
                    assignments.append(Assignment(t.id, gid))

                    fwd_task_ids.append(t.id)

                set_dep(s, mb, 'fwd_done', fwd_task_ids)

                # Update Exec Stream
                exec_stream_last = fwd_task_ids

                # C. Send (if not last stage)
                if s < num_stages - 1:
                    # Send depends on Fwd
                    send_ids = create_p2p(s, mb, 'send_fwd', fwd_task_ids)
                    set_dep(s, mb, 'send_fwd_done', send_ids)

            # -------------------------------------------------------------
            # BACKWARD
            # -------------------------------------------------------------
            elif step_type == 'bwd':
                # Prerequisites:
                # 1. Recv Bwd from Next Stage (if s < last)
                # 2. Exec Stream

                # A. Recv
                recv_tasks = []
                if s < num_stages - 1:
                    # Depends on send_bwd from s+1
                    # We haven't processed s+1 yet!!!
                    # PROBLEM: We are iterating stages 0..7.
                    # s+1 tasks are not created.

                    # We need IDs. We can pre-generate IDs deterministically!
                    # "Send_Bwd_S{s+1}_R{r}_M{mb}"

                    sender_ids_virtual = [f"send_bwd_S{s + 1}_R{r}_M{mb}_send_bwd" for r in range(tp_size)]

                    # Create Recv
                    recv_tasks = create_p2p(s, mb, 'recv_bwd', sender_ids_virtual)

                # B. Compute Bwd
                deps_per_rank = []
                for r in range(tp_size):
                    d = []
                    if recv_tasks: d.append(recv_tasks[r])
                    if exec_stream_last[r]: d.append(exec_stream_last[r])
                    deps_per_rank.append(d)

                bwd_task_ids = []
                for r in range(tp_size):
                    global_rank = s * tp_size + r
                    gid = all_gpu_ids[global_rank]
                    gpu_cfg = gpu_configs[gid]
                    my_deps = deps_per_rank[r]
                    suffix = f"_S{s}_R{r}_M{mb}_bwd"

                    # Bwd Task (2x Fwd)
                    # We can just put 2 GEMMs
                    _activation_slots = num_stages
                    slot_idx = mb % _activation_slots
                    dummy_params_all = gpu_dummy_params[gid]

                    # Unpack
                    p_act = dummy_params_all[0][slot_idx]
                    p_w   = dummy_params_all[1]
                    p_out = dummy_params_all[2][slot_idx]
                    p_deps = dummy_params_all[3][slot_idx]

                    gemm_deps = my_deps + p_deps

                    t = Task(
                        id=f"Comp_Bwd{suffix}",
                        workload=GEMMKernelSimulator(
                            gpu_cfg,
                            p_act, p_w, p_out,
                            name=f"ChunkBwd", id=f"ChunkBwd{suffix}"
                        ),
                        run_after=gemm_deps
                    )
                    # Hack: Force workload compute time to be higher?
                    # BaseKernelSimulator doesn't allow easy override.
                    # We chain 2 dummy tasks.
                    tasks.append(t)
                    assignments.append(Assignment(t.id, gid))

                    t2 = Task(
                        id=f"Comp_Bwd2{suffix}",
                        workload=GEMMKernelSimulator(
                            gpu_cfg,
                            p_act, p_w, p_out,
                            name=f"ChunkBwd2", id=f"ChunkBwd2{suffix}"
                        ),
                        run_after=[t.id]
                    )
                    tasks.append(t2)
                    assignments.append(Assignment(t2.id, gid))

                    bwd_task_ids.append(t2.id)

                set_dep(s, mb, 'bwd_done', bwd_task_ids)
                exec_stream_last = bwd_task_ids

                # C. Send Bwd (if s > 0)
                if s > 0:
                    send_ids = create_p2p(s, mb, 'send_bwd', bwd_task_ids)
                    # Note: These IDs must match `sender_ids_virtual` predicted by stage s-1
                    # Virtual: "send_bwd_S{s}_R{r}_M{mb}_send_bwd" (from Logic A above, adapted index)
                    # My `create_p2p` generates ID: `p_type + suffix`.
                    # suffix = f"_S{stage_idx}_R{r}_M{mb_idx}_{p_type}"
                    # ID = "send_bwd" + "_S{s}_R{r}_M{mb}_send_bwd"
                    # Yes, match seems correct.
                    pass

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
