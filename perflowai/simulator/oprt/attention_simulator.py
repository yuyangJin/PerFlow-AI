'''
@module attention simulator
'''

from .oprt_simulator import OprtSimulator
from ...core import ModelConfig, DeviceConfig, EventType


'''
@class AttentionSimulator
'''

class AttentionSimulator(OprtSimulator):
    '''
    @method __init__
    Initialize the attention simulator.
    '''
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.hidden_size = model_config.hidden_size

        # Check if num_attention_heads is divisible by tp_size
        self.num_attention_heads = model_config.num_attention_heads
        assert (
            model_config.num_attention_heads % tp_size == 0
        ), f"AttentionBenchmark: num_attention_heads {model_config.num_attention_heads} is not divisible by tp_size {tp_size}"

        # Give num_key_value_heads a default value of num_attention_heads
        self.num_key_value_heads = getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        )

        # Check if num_key_value_heads is divisible by tp_size
        # or tp_size is divisible by num_key_value_heads
        if model_config.num_key_value_heads > tp_size:
            assert (
                model_config.num_key_value_heads % tp_size == 0
            ), f"AttentionBenchmark: num_key_value_heads {model_config.num_key_value_heads} is not divisible by tp_size {tp_size}"
        else:
            assert (
                tp_size % model_config.num_key_value_heads == 0
            ), f"AttentionBenchmark: tp_size {tp_size} is not divisible by num_key_value_heads {self.num_key_value_heads}"

        self.q_head = self.num_attention_heads // tp_size
        self.kv_head = max(1, self.num_key_value_heads // tp_size)
        self.kv_head_replicas = max(1, tp_size // self.num_key_value_heads)

        self.head_dim = self.hidden_size // self.num_attention_heads

        self.q_size = self.q_head * self.head_dim
        self.kv_size = self.kv_head * self.head_dim



    def flop(self, batch_size: int, seq_len: int, mode: EventType):
        if mode == EventType.PRF:
            t = seq_len
        else:
            t = 1
        """
        compute (for one batch):
            shapes:
                hidden_states (t, hidden_size)
                q (t, q_size)
                k (t, kv_size)
                v (t, kv_size)
            process:
                q_proj: t * hidden_size * q_size
                k_proj: t * hidden_size * kv_size
                v_proj: t * hidden_size * kv_size
                q * k: t * q_size * seq_len
                qk * v: t * seq_len * q_size
                o_proj: t * q_size * hidden_size
        """
        self.compute = (
            4
            * batch_size
            * t
            * (self.hidden_size * (self.q_size + self.kv_size) + seq_len * self.q_size)
        )

    def memory(self, batch_size: int, seq_len: int, mode: EventType):
        if mode == EventType.PRF:
            t = seq_len
        else:
            t = 1
        """
        memory:
            weights:
                q_proj: hidden_size * q_size
                k_proj: hidden_size * kv_size
                v_proj: hidden_size * kv_size
                o_proj: q_size * hidden_size
            kv_cache:
                batch_size * seq_len * kv_size
        """
        self.memory = (
            2 * self.hidden_size * (self.q_size + self.kv_size)  # weights
            + batch_size * seq_len * self.kv_size  # kv_cache
        )


    # def return_flops_and_mem(self, x, bs, is_prefill, input_length):
    #     bs_sl = x.shape[0]

    #     # qkv compute
    #     mem = (
    #         x.numel()
    #         + self.wqkv_a_weight.numel()
    #         + bs_sl * (self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim)
    #         + self.wq_b.return_flops_and_mem(x)[1]
    #         + self.wkv_b.return_flops_and_mem(x)[1]
    #         + self.wo.return_flops_and_mem(x)[1]
    #     )
    #     flops = (
    #         2
    #         * bs_sl
    #         * self.dim
    #         * (self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim)
    #         + self.wq_b.return_flops_and_mem(x)[0]
    #         + self.wkv_b.return_flops_and_mem(x)[0]
    #         + self.wo.return_flops_and_mem(x)[0]
    #     )

    #     # core attention
    #     if is_prefill:
    #         flops += (
    #             bs_sl
    #             * bs_sl
    #             / bs
    #             * self.n_local_heads
    #             * (self.qk_head_dim + self.v_head_dim)
    #         )
    #         mem += (
    #             self.qk_head_dim * bs_sl * 2 + self.v_head_dim * bs_sl
    #         ) * self.n_local_heads + x.shape[0] * self.v_head_dim * self.n_local_heads
    #     else:
    #         flops += bs * self.n_local_heads * self.qk_head_dim * (input_length + 1)
    #         flops += bs * (input_length + 1) * self.n_local_heads * self.v_head_dim
    #         mem += (
    #             bs * self.qk_head_dim * self.n_local_heads
    #             + bs
    #             * (input_length + 1)
    #             * (self.qk_head_dim + self.v_head_dim)
    #             * self.n_local_heads
    #             + x.shape[0] * self.v_head_dim * self.n_local_heads
    #         )

    #     return flops, mem
