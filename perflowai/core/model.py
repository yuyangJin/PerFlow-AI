'''
@module model 
'''

from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    num_layers: int
    hidden_size: int
    hidden_dim: int
    ffn_dim: int
    num_heads: int
    head_dim: int
    dtype_bytes: int
    num_experts: int = 0
    moe_layers: List[int] = field(default_factory=list)