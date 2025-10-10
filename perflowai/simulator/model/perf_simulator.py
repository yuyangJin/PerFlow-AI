'''
@module performance simulator
'''

from ..simulator import Simulator
from ...core import ModelConfig, DeviceConfig, EventType

'''
@class PerfSimulator
'''
class PerfSimulator(Simulator):
    '''
    @method __init__
    Initialize the performance simulator.
    '''
    def __init__(self):
        super().__init__()
        self.trace = None
        self.perf = None
        self.config = None

    '''
    @method simulate
    Simulate the performance.
    '''
    def simulate(self):
        pass

    '''
    @method get_trace
    Get the trace.
    '''
    def get_trace(self):
        return self.trace


class ModelPerfSimulator(PerfSimulator):
    def __init__(self, model_config: ModelConfig, device_config: DeviceConfig):
        super().__init__()
        self.m_model_config = model_config
        self.m_device_config = device_config

    def _compute_prefill_volume(self, event):
        tasks = event.get_tasks() 
        seq_len = 0
        seq_len_2 = 0
        for task in tasks.get():
            seq_len += task.req.input_len + 1 # Prompt tokens
            seq_len_2 += (task.req.input_len + 1) ** 2 # Prompt tokens

        attn_flops = 4 * seq_len_2 * self.m_model_config.hidden_size + 2 * seq_len_2 # QKV + output
        attn_mem = 4 * seq_len * self.m_model_config.hidden_size + 4 * seq_len_2

        ffn_flops = 2 * seq_len * self.m_model_config.hidden_dim * self.m_model_config.ffn_dim  # 2 linear transforms
        ffn_mem = 2 * seq_len * self.m_model_config.hidden_dim + 2 * seq_len_2
        
        # Compute the total FLOPs and memory for the prefill
        total_flops = attn_flops + ffn_flops
        total_mem = attn_mem + ffn_mem
        
        return total_flops, total_mem

    def _compute_decode_volume(self, event):
        tasks = event.get_tasks() 
        seq_len = 0
        seq_len_2 = 0
        for task in tasks.get():
            seq_len += task.req.input_len + task.decode_iters # Prompt tokens
            seq_len_2 += (task.req.input_len + task.decode_iters) ** 2 # Prompt tokens

        attn_flops = 4 * seq_len * self.m_model_config.hidden_size + 2 * seq_len # QKV + output
        attn_mem = 4 * seq_len * self.m_model_config.hidden_size + 4 * seq_len

        ffn_flops = 2 * seq_len * self.m_model_config.hidden_dim * self.m_model_config.ffn_dim  # 2 linear transforms
        ffn_mem = 2 * seq_len * self.m_model_config.hidden_dim + 2 * seq_len
        
        # Compute the total FLOPs and memory for the prefill
        total_flops = attn_flops + ffn_flops
        total_mem = attn_mem + ffn_mem
        
        return total_flops, total_mem


    '''
    @method time
    Calculate the time for the event.
    @param event: The event to calculate the time for.
    @return: The time for the event.
    '''
    def time(self, event):
        '''
        The main idea is to calculate the maximun time of computation and memory access
        '''
        compute_time = 0.0
        memory_time = 0.0

        if event.get_type() == EventType.PRF:
            # Calculate the computation volume for prefill
            total_flops, total_mem = self._compute_prefill_volume(event)

        elif event.get_type() == EventType.DCD:
            # Calculate the computation volume for decode
            total_flops, total_mem = self._compute_decode_volume(event)
        else:
            raise ValueError(f"Unsupported event type: {event}")
        
        compute_time = total_flops / (self.m_device_config.compute_flops * 0.6)
        memory_time = total_mem / (self.m_device_config.memory_bandwidth * 0.8)

        return max(compute_time, memory_time)