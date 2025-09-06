'''
@module moe simulator
'''

from .oprt_simulator import OprtSimulator
from ...core import ModelConfig, DeviceConfig, EventType


'''
@class MoESimulator
'''

class DeepSeekMoESimulator(OprtSimulator):
    '''
    @method __init__
    Initialize the MOE simulator.
    '''
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.m_compute_volume = 0
        self.m_memory_volume = 0
        self.hidden_size = self.m_model.hidden_size
        self.moe_inter_dim = self.m_model.moe_inter_dim
        self.num_activated_experts = model_config.num_activated_experts
        self.num_routed_experts = model_config.num_routed_experts

    def _generate_fake_indices(self, batch_mul_seq_len, ep_size):
        x = [random.randint(0, self.num_routed_experts-1) for _ in range(batch_mul_seq_len * self.num_routed_experts)]
        local_expert_list = [i for i in x if i < self.num_routed_experts / ep_size]
        total_activate_expert_num = len(local_expert_list)
        local_expert_activate_num = len(list(set(local_expert_list)))
        return total_activate_expert_num, local_expert_activate_num

    def flop_and_memory(self, batch_mul_seq_len: int):
        # https://zhuanlan.zhihu.com/p/16445683081

        total_activate_expert_num, local_expert_activate_num = self._generate_fake_indices(batch_mul_seq_len, ep_size)
        # Need to be modified
        self.m_compute_volume = (
            6 * self.hidden_size  * self.moe_inter_dim * (total_activate_expert_num + batch_mul_seq_len)
            + 2 * (total_activate_expert_num + batch_mul_seq_len) * self.moe_inter_dim
        )

        self.m_memory_volume = (
            batch_mul_seq_len * self.hidden_size 
            + layer.gate.numel
            + batch_mul_seq_len * 256
            + 2 * batch_mul_seq_len * self.hidden_size
            + 3 * self.hidden_size * self.moe_inter_dim
            + 3 * batch_mul_seq_len * self.moe_inter_dim
            + 2 * batch_mul_seq_len * self.hidden_size
            + 3 * self.hidden_size * self.moe_inter_dim * local_expert_activate_num
            + 3 * batch_mul_seq_len * self.moe_inter_dim
        )
    
        return self.m_compute_volume, self.m_memory_volume