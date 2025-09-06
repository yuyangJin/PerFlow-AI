'''
@module model 
'''

from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    name: str
    url: str

@dataclass
class ModelTestConfig(ModelConfig):
    num_layers: int = 0
    hidden_size: int = 0
    ffn_dim: int = 0
    hidden_dim: int = 0
    num_heads: int = 0
    head_dim: int = 0
    dtype_bytes: int = 0
    num_experts: int = 0


@dataclass
class LlamaConfig(ModelConfig):
    """
    Llama model configuration.
    """
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    num_key_value_heads: int
    dtype_bytes: int = 2

@dataclass
class DeepSeekConfig(ModelConfig):
    vocab_size: int
    dim: int
    inter_dim: int
    moe_inter_dim: int
    n_layers: int
    n_dense_layers: int
    n_heads: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    route_scale: float = 0.0
    score_func: str = "sigmoid"
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    dtype: str = "fp16"
    mscale: float = 1.0

    def __post_init__(self):
        self.hidden_size = self.dim
        self.intermediate_size = self.inter_dim
        self.num_attention_heads = self.n_heads
        self.num_key_value_heads = self.n_heads
        self.num_layers = self.n_layers
        self.num_dense_layers = self.n_dense_layers
        self.num_activated_experts = self.n_activated_experts
        self.num_routed_experts = self.n_routed_experts
        self.num_expert_groups = self.n_expert_groups
        self.num_shared_experts = self.n_shared_experts
        if self.dtype == "fp8":
            self.dtype_bytes = 1
        elif self.dtype == "fp16":
            self.dtype_bytes = 2
        elif self.dtype == "bf16":
            self.dtype_bytes = 2
        elif self.dtype == "fp4":
            self.dtype_bytes = 0.5



DeepSeekVersionMap = {
    "DeepSeek-V3-671B": DeepSeekConfig(
        name = "DeepSeek-V3-671B",
        url = "https://huggingface.co/deepseek-ai/DeepSeek-V3-Base",
        vocab_size = 129280,
        dim = 7168,
        inter_dim = 18432,
        moe_inter_dim = 2048,
        n_layers = 61,
        n_dense_layers = 3,
        n_heads = 128,
        n_routed_experts = 256,
        n_shared_experts = 1,
        n_activated_experts = 8,
        n_expert_groups = 8,
        n_limited_groups = 4,
        route_scale = 2.5,
        score_func = "sigmoid",
        q_lora_rank = 1536,
        kv_lora_rank = 512,
        qk_nope_head_dim = 128,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        dtype = "fp8"
    ),
    
    "DeepSeek-V3-236B": DeepSeekConfig(
        name = "DeepSeek-V3-236B",
        url = "https://huggingface.co/deepseek-ai/DeepSeek-V3-Base",
        vocab_size = 102400,
        dim = 5120,
        inter_dim = 12288,
        moe_inter_dim = 1536,
        n_layers = 60,
        n_dense_layers = 1,
        n_heads = 128,
        n_routed_experts = 160,
        n_shared_experts = 2,
        n_activated_experts = 6,
        n_expert_groups = 8,
        n_limited_groups = 3,
        route_scale = 16.0,
        q_lora_rank = 1536,
        kv_lora_rank = 512,
        qk_nope_head_dim = 128,
        qk_rope_head_dim = 64,
        v_head_dim = 128
    ),

    "DeepSeek-V3-16B": DeepSeekConfig(
        name = "DeepSeek-V3-16B",
        url = "https://huggingface.co/deepseek-ai/DeepSeek-V3-Base",
        vocab_size = 102400,
        dim = 2048,
        inter_dim = 10944,
        moe_inter_dim = 1408,
        n_layers = 27,
        n_dense_layers = 1,
        n_heads = 16,
        n_routed_experts = 64,
        n_shared_experts = 2,
        n_activated_experts = 6,
        route_scale = 1.0,
        q_lora_rank = 0,
        kv_lora_rank = 512,
        qk_nope_head_dim = 128,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        mscale = 0.707
    )

}



LlamaVersionMap = {
    "Llama-3.1-8B-Instruct": LlamaConfig(
        name="Llama-3.1-8B-Instruct",
        url="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_key_value_heads=8,
    ),
    "Llama-3.1-70B-Instruct": LlamaConfig(
        name="Llama-3.1-70B-Instruct",
        url="https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
        hidden_size=8192,
        num_attention_heads=64,
        num_hidden_layers=80,
        intermediate_size=28672,
        num_key_value_heads=8,
    ),
    "Llama-3.1-405B-Instruct": LlamaConfig(
        name="Llama-3.1-405B-Instruct",
        url="https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct",
        hidden_size=16384,
        num_attention_heads=128,
        num_hidden_layers=126,
        intermediate_size=53248,
        num_key_value_heads=8,
    ),
    "Llama-3.2-1B-Instruct": LlamaConfig(
        name="Llama-3.2-1B-Instruct",
        url="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
        hidden_size=2048,
        num_attention_heads=32,
        num_hidden_layers=16,
        intermediate_size=8192,
        num_key_value_heads=8,
    ),
    "Llama-3.2-3B-Instruct": LlamaConfig(
        name="Llama-3.2-3B-Instruct",
        url="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
        hidden_size=3072,
        num_attention_heads=24,
        num_hidden_layers=28,
        intermediate_size=8192,
        num_key_value_heads=8,
    ),
    # "Llama-3.2-11B-Vision-Instruct": ,
    # "Llama-3.2-90B-Vision-Instruct": ,
    "LLama-3.3-70B-Instruct": LlamaConfig(
        name="Llama-3.3-70B-Instruct",
        url="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
        hidden_size=8192,
        num_attention_heads=64,
        num_hidden_layers=80,
        intermediate_size=28672,
        num_key_value_heads=8,
    ),
    # "Llama-4-Scout-17B-16E-Instruct": ,
    # "Llama-4-Maverick-17B-128E-Instruct": ,
}