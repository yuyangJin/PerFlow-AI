'''
Example: Gernerate deepseek graph & Visualize the simulated trace
'''

from perflowai.parallel.pipeline_parallel import PipeCostConfig, PipeOffloadConfig, Interleaved1F1BGraph
from perflowai.simulator import PPSimulator, PipeType
from perflowai.visualizer import MemoryFootprintVisualizer, TraceVisiualizer
from perflowai.core import OffReLoadEvent, EventType, ResourceType

def deepseek_training_Visualize():

    num_layer = 16 #num of layer
    num_dense = 0 #num of dense
    num_res = 0 #num of stage get less layer
    num_stage = 4 #num of stage
    num_chunks = 4 #num of chunk
    num_microbatches = 32 #num of microbatch
    num_offload = 0 #num of stage use offload
    offload_ratio = 0.136743897 
    ONLY_DISPLAY_ACT = False


    '''
        dense
    '''
    #Default
    fwd_dense, bwd_dense, mem_dense, base_dense = (1, 1, 0.0, 0.0)
    #Google 
    #fwd_dense, bwd_dense, mem_dense, base_dense = (6010, 12650, 361.0, 2130.0) #EP16, FP8
    #fwd_dense, bwd_dense, mem_dense, base_dense = (5790, 12070, 465.0, 2130.0) #EP16, BF16
    #512 log
    #fwd_dense, bwd_dense, mem_dense, base_dense = (int(6010*0.9), int(12650*0.9), 361.0, 1714.901733) #ep16 pp16
    #fwd_dense, bwd_dense, mem_dense, base_dense = (int(6010*0.9), int(12650*0.9), 361.0, 1053.783386) #ep32 pp8
    '''
        MOE
    '''
    #Default
    fwd_moe, bwd_moe, mem_moe, base_moe = (1, 1, 0.0, 0.0)
    #Google
    #fwd_moe, bwd_moe, mem_moe, base_moe = (11370, 16940, 737.0, 8175.0) #EP16, FP8
    #fwd_moe, bwd_moe, mem_moe, base_moe = (10890, 16010, 953.0, 8216.75) #EP16, BF16
    #MoE Mem
    fwd_moe, bwd_moe, mem_moe, base_moe = (9358, 14053, 732.135, 7640.95375) #PP8, EP8
    #fwd_moe, bwd_moe, mem_moe, base_moe = (12228, 16662, 732.135, 7625.95375) #PP4, EP16
    #fwd_moe, bwd_moe, mem_moe, base_moe = (17570, 22365, 732.135, 7655.95125) #PP2, EP32
    #fwd_moe, bwd_moe, mem_moe, base_moe = (9413, 14910, 676.135, 7786.46625) #PP4, EP8
    #fwd_moe, bwd_moe, mem_moe, base_moe = (11590, 16053, 676.135, 7694.96375) #PP2, EP16
    #512 log
    #fwd_moe, bwd_moe, mem_moe, base_moe = (int(12000*0.9), int(16880*0.9), 599.6611328, 5807.950073) #ep16 pp16
    #fwd_moe, bwd_moe, mem_moe, base_moe = (int(17079.897*0.9), int(22164*0.9), 599.9355469, 4352.10852) #ep32 pp16
    #fwd_moe, bwd_moe, mem_moe, base_moe = (int(9700*0.9), int(14580*0.9), 599.9355469, 3176.559753) #ep32 pp8

    '''
        Head & Tail
    '''
    #Default
    head_base, tail_base = (0.0, 0.0)
    #MoE Mem 
    head_base, tail_base = (2345.57, -4378.97) #PP8, EP8
    #head_base, tail_base = (2569.28, -4378.98) #PP4, EP16
    #head_base, tail_base = (2561.15, -4378.98) #PP2, EP32
    #head_base, tail_base = (2302.79, -4799.25) #PP4, EP8
    #head_base, tail_base = (2170.65, -4799.25) #PP2, EP16
    #512 log
    #head_base, tail_base = (1259.37793, 2777.195679) #ep16 pp16 moe
    #head_base, tail_base = (1336.459815, 2840.531372) #ep32 pp16 moe
    #head_base, tail_base = (1609.766944, 2426.972351) #ep32 pp8 moe
    #head_base, tail_base = (1287.381836, 2777.195679) #ep16 pp16 dense
    #head_base, tail_base = (1860.041992, 2426.972351) #ep32 pp8 dense


    #Assemble the configs
    fwd_time = [[fwd_dense] + [fwd_moe] * (num_chunks - 1)] * num_dense + [[fwd_moe] * num_chunks] * (num_stage - num_dense - num_res) + [[fwd_moe] * (num_chunks - 1) + [1]] * num_res
    bwd_time = [[bwd_dense] + [bwd_moe] * (num_chunks - 1)] * num_dense + [[bwd_moe] * num_chunks] * (num_stage - num_dense - num_res) + [[bwd_moe] * (num_chunks - 1) + [1]] * num_res
    fwd_mem  = [[mem_dense] + [mem_moe] * (num_chunks - 1)] * num_dense + [[mem_moe] * num_chunks] * (num_stage - num_dense - num_res) + [[mem_moe] * (num_chunks - 1) + [0]] * num_res
    bwd_mem  = [[-mem_dense] + [-mem_moe] * (num_chunks - 1)] * num_dense + [[-mem_moe] * num_chunks] * (num_stage - num_dense - num_res) + [[-mem_moe] * (num_chunks - 1) + [0]] * num_res
    base_mem = [sum([base_dense] + [base_moe] * (num_chunks - 1))] * num_dense + [sum([base_moe] * num_chunks)] * (num_stage - num_dense - num_res) + [sum([base_moe] * (num_chunks - 1))] * num_res
    base_mem[0] = base_mem[0] + head_base
    base_mem[-1] = base_mem[-1] + tail_base
    if ONLY_DISPLAY_ACT:
        base_mem = [0] * num_stage

    config = PipeCostConfig(fwd_time = fwd_time, 
                            bwd_time = bwd_time,
                            fwd_mem = fwd_mem,
                            bwd_mem = bwd_mem 
                            )
    poc = PipeOffloadConfig(offload_ratio = [offload_ratio] * num_offload + [0.0] * (num_stage - num_offload))

    g = Interleaved1F1BGraph(num_stage, num_microbatches, num_chunks, cost_config = config, offload_config = poc)
    g.build_graph()

    sim = PPSimulator(PipeType.Interleaved1F1B, g)
    trace = sim.run()

    mfp = trace.get_memory_foorprint()

    MemoryFootprintVisualizer(mfp).visualize(base_mem)

    visualizer = TraceVisiualizer(trace)
    visualizer.visualize()


deepseek_training_Visualize()
