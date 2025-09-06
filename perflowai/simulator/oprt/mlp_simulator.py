'''
@module MLP simulator
'''

from .oprt_simulator import OprtSimulator
from ...core import ModelConfig, DeviceConfig, EventType



'''
@class MLPSimulator
'''

class MLPSimulator(OprtSimulator):
    '''
    @method __init__
    Initialize the MLP simulator.
    '''
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.tp_size = tp_size
        self.dtype_bytes = model_config.dtype_bytes
        self.hidden_size = model_config.hidden_size
        assert (
            model_config.intermediate_size % tp_size == 0
        ), f"MLPBenchmark: intermediate_size {model_config.intermediate_size} is not divisible by tp_size {tp_size}"
        # self.intermediate_size = model_config.intermediate_size // tp_size
        self.intermediate_size = model_config.intermediate_size

    def compute_flop(self, batch_mul_seq_len: int):
        '''
        compute = gate + up + down
            gate = 2 * hidden_size * intermediate_size * batch_size * seq_len
            up = 2 * hidden_size * intermediate_size * batch_size * seq_len
            down = 2 * intermediate_size * hidden_size * batch_size * seq_len
        '''
        self.compute = (
            6
            * self.hidden_size
            * self.intermediate_size
            * batch_mul_seq_len
        ) / self.tp_size

    def memory_size(self, batch_size: int, seq_len: int, mode: EventType):
        '''
        memory = gate + up + down
            gate = hidden_size * intermediate_size + batch_size * seq_len * hidden_size + batch_size * seq_len * intermediate_size
            up = hidden_size * intermediate_size + batch_size * seq_len * hidden_size + batch_size * seq_len * intermediate_size
            down = intermediate_size * hidden_size + batch_size * seq_len * hidden_size + batch_size * seq_len * intermediate_size

        tips:
            1. merged gate and up should have less memory access
            2. this is an approximation, the actual memory usage may vary based on the implementation and hardware
        '''
        self.memory = (
            (
                3 * self.hidden_size * self.intermediate_size
                + 3 * batch_mul_seq_len * self.hidden_size
                + 3 * batch_mul_seq_len * self.intermediate_size
            )
            * self.dtype_bytes
            / self.tp_size # @yk: is it right?
        )


'''
@class MLPDeepSeekV3Simulator
'''

class MLPDeepSeekV3Simulator(MLPSimulator):
    '''
    @method __init__
    Initialize the MLP simulator.
    '''
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.m_compute_volume = 0
        self.m_memory_volume = 0
        self.hidden_size = model_config.hidden_size
        assert (
            model_config.intermediate_size % tp_size == 0
        ), f"MLPBenchmark: intermediate_size {model_config.intermediate_size} is not divisible by tp_size {tp_size}"
        self.intermediate_size = model_config.intermediate_size

    def flop(self, batch_mul_seq_len: int):
        
        # Need to be modified
        self.m_compute_volume = (
            self.hidden_size * self.intermediate_size * 2 * batch_mul_seq_len * 2
            + batch_mul_seq_len * self.intermediate_size * 2
            + batch_mul_seq_len * self.hidden_size * self.intermediate_size * 2
        ) / self.tp_size

        return self.m_compute_volume
    
    def memory(self, batch_mul_seq_len: int):
        self.m_memory_volume = (
            self.hidden_size * self.intermediate_size * 2 / self.tp_size
            + batch_mul_seq_len * self.hidden_size
            + batch_mul_seq_len * self.intermediate_size * 2 / self.tp_size
            + batch_mul_seq_len * self.intermediate_size / self.tp_size * (3 + 2 + 4)
            + batch_mul_seq_len * self.intermediate_size / self.tp_size
            + self.hidden_size * self.intermediate_size / self.tp_size
            + batch_mul_seq_len * self.hidden_size
        )

        return self.m_memory_volume