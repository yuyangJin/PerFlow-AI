'''
@module performance simulator
'''

from ..simulator import Simulator
from ...core import ModelConfig, DeviceConfig, EventType
from ..oprt import TransformerLayerOperator

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
    """
    Model-level performance simulator that computes FLOPs and memory access
    using fine-grained operator-level calculations
    """
    def __init__(self, model_config: ModelConfig, device_config: DeviceConfig):
        super().__init__()
        self.m_model_config = model_config
        self.m_device_config = device_config
        # Create a transformer layer operator for calculations
        self.m_layer_op = TransformerLayerOperator(model_config)

    def _compute_prefill_volume(self, event):
        """
        Compute FLOPs and memory access for prefill phase.
        
        Prefill processes all input tokens at once for each task.
        """
        tasks = event.get_tasks()
        total_flops = 0
        total_mem = 0
        
        for task in tasks.get():
            seq_len = task.req.input_len
            # Compute per-layer FLOPs and memory
            layer_flops = self.m_layer_op.compute_flops(batch_size=1, seq_len=seq_len, is_prefill=True)
            layer_mem = self.m_layer_op.compute_memory(batch_size=1, seq_len=seq_len, is_prefill=True)
            
            # Aggregate over all layers
            total_flops += layer_flops * self.m_model_config.num_layers
            total_mem += layer_mem * self.m_model_config.num_layers
        
        return total_flops, total_mem

    def _compute_decode_volume(self, event):
        """
        Compute FLOPs and memory access for decode phase.
        
        Decode generates one token per task in the batch.
        Each task attends to all previously cached tokens.
        """
        tasks = event.get_tasks()
        total_flops = 0
        total_mem = 0
        
        for task in tasks.get():
            # Current sequence length (for attention over KV cache)
            # decode_iters starts at 0, so current position is input_len + decode_iters
            kv_cache_len = task.req.input_len + task.decode_iters
            
            # Compute per-layer FLOPs and memory for decoding 1 token
            # The seq_len parameter represents KV cache length for attention
            layer_flops = self.m_layer_op.compute_flops(batch_size=1, seq_len=kv_cache_len, is_prefill=False)
            layer_mem = self.m_layer_op.compute_memory(batch_size=1, seq_len=kv_cache_len, is_prefill=False)
            
            # Aggregate over all layers
            total_flops += layer_flops * self.m_model_config.num_layers
            total_mem += layer_mem * self.m_model_config.num_layers
        
        return total_flops, total_mem


    '''
    @method time
    Calculate the time for the event.
    @param event: The event to calculate the time for.
    @return: The time for the event.
    '''
    def time(self, event):
        '''
        The main idea is to calculate the maximum time of computation and memory access.
        Time is bounded by either compute (FLOPs) or memory bandwidth.
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
        
        # Compute time (FLOPs / peak FLOPs)
        # Use 60% efficiency factor for realistic performance
        compute_time = total_flops / (self.m_device_config.compute_flops * 0.6)
        
        # Memory time (bytes / bandwidth)
        # Use 80% efficiency factor for memory bandwidth
        memory_time = total_mem / (self.m_device_config.memory_bandwidth * 0.8 * 1e9)  # Convert GB/s to bytes/s

        # Return the maximum (bottleneck)
        return max(compute_time, memory_time)