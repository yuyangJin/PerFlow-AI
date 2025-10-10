'''
@module operator simulator
This module provides fine-grained operator-level simulation for computing
KVCache, memory footprint, and computation volume (FLOPs).
'''

from abc import ABC, abstractmethod
from typing import Tuple
from ...core import ModelConfig


class Operator(ABC):
    """
    Base class for all operators in a neural network.
    Each operator can compute:
    - KVCache size (for attention-related operators)
    - Memory footprint (activation memory)
    - Computation volume (FLOPs)
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute_flops(self, *args, **kwargs) -> int:
        """Compute FLOPs for this operator"""
        pass
    
    @abstractmethod
    def compute_memory(self, *args, **kwargs) -> int:
        """Compute memory footprint (activation memory) in bytes"""
        pass
    
    def compute_kvcache(self, *args, **kwargs) -> int:
        """Compute KVCache size in bytes (only for attention operators)"""
        return 0


class LinearOperator(Operator):
    """
    Linear layer: Y = X @ W + b
    where X: [batch_size, seq_len, in_features]
          W: [in_features, out_features]
          Y: [batch_size, seq_len, out_features]
    """
    
    def __init__(self, in_features: int, out_features: int, dtype_bytes: int = 2):
        super().__init__("Linear")
        self.in_features = in_features
        self.out_features = out_features
        self.dtype_bytes = dtype_bytes
    
    def compute_flops(self, batch_size: int, seq_len: int) -> int:
        """
        FLOPs for linear layer: 2 * batch_size * seq_len * in_features * out_features
        (factor of 2 for multiply-add)
        """
        return 2 * batch_size * seq_len * self.in_features * self.out_features
    
    def compute_memory(self, batch_size: int, seq_len: int) -> int:
        """
        Memory for activations: 
        - Input: batch_size * seq_len * in_features
        - Output: batch_size * seq_len * out_features
        """
        input_mem = batch_size * seq_len * self.in_features * self.dtype_bytes
        output_mem = batch_size * seq_len * self.out_features * self.dtype_bytes
        return input_mem + output_mem


class AttentionOperator(Operator):
    """
    Multi-head attention operator
    Q, K, V projections followed by scaled dot-product attention
    """
    
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, dtype_bytes: int = 2):
        super().__init__("Attention")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes
    
    def compute_flops(self, batch_size: int, seq_len: int, is_prefill: bool = True) -> int:
        """
        FLOPs for attention:
        - Prefill: Process full sequence
          - QKV projections: 3 * 2 * batch_size * seq_len * hidden_size * hidden_size
          - Attention scores: 2 * batch_size * num_heads * seq_len * seq_len * head_dim
          - Output projection: 2 * batch_size * seq_len * hidden_size * hidden_size
        - Decode: Process single token with full KV cache
          - QKV projections: 3 * 2 * batch_size * 1 * hidden_size * hidden_size
          - Attention scores: 2 * batch_size * num_heads * 1 * seq_len * head_dim
          - Output projection: 2 * batch_size * 1 * hidden_size * hidden_size
        """
        if is_prefill:
            # QKV projections
            qkv_flops = 3 * 2 * batch_size * seq_len * self.hidden_size * self.hidden_size
            # Attention computation: Q @ K^T and attn @ V
            attn_flops = 2 * batch_size * self.num_heads * seq_len * seq_len * self.head_dim * 2
            # Output projection
            out_flops = 2 * batch_size * seq_len * self.hidden_size * self.hidden_size
        else:
            # Decode: process 1 token per batch element
            qkv_flops = 3 * 2 * batch_size * 1 * self.hidden_size * self.hidden_size
            # Attention with existing KV cache (seq_len is cache length)
            attn_flops = 2 * batch_size * self.num_heads * 1 * seq_len * self.head_dim * 2
            out_flops = 2 * batch_size * 1 * self.hidden_size * self.hidden_size
        
        return qkv_flops + attn_flops + out_flops
    
    def compute_memory(self, batch_size: int, seq_len: int, is_prefill: bool = True) -> int:
        """
        Memory for attention activations:
        - Q, K, V: 3 * batch_size * seq_len * hidden_size
        - Attention scores: batch_size * num_heads * seq_len * seq_len
        - Output: batch_size * seq_len * hidden_size
        """
        if is_prefill:
            qkv_mem = 3 * batch_size * seq_len * self.hidden_size * self.dtype_bytes
            scores_mem = batch_size * self.num_heads * seq_len * seq_len * self.dtype_bytes
            output_mem = batch_size * seq_len * self.hidden_size * self.dtype_bytes
        else:
            # Decode: only 1 token per batch
            qkv_mem = 3 * batch_size * 1 * self.hidden_size * self.dtype_bytes
            scores_mem = batch_size * self.num_heads * 1 * seq_len * self.dtype_bytes
            output_mem = batch_size * 1 * self.hidden_size * self.dtype_bytes
        
        return qkv_mem + scores_mem + output_mem
    
    def compute_kvcache(self, batch_size: int, seq_len: int) -> int:
        """
        KVCache size: 2 (K and V) * batch_size * num_heads * seq_len * head_dim
        This is the total size needed to store K and V for all heads
        """
        return 2 * batch_size * self.num_heads * seq_len * self.head_dim * self.dtype_bytes


class LayerNormOperator(Operator):
    """
    Layer normalization operator
    """
    
    def __init__(self, normalized_shape: int, dtype_bytes: int = 2):
        super().__init__("LayerNorm")
        self.normalized_shape = normalized_shape
        self.dtype_bytes = dtype_bytes
    
    def compute_flops(self, batch_size: int, seq_len: int) -> int:
        """
        FLOPs for LayerNorm: ~5 * batch_size * seq_len * normalized_shape
        (mean, variance, normalize, scale, shift)
        """
        return 5 * batch_size * seq_len * self.normalized_shape
    
    def compute_memory(self, batch_size: int, seq_len: int) -> int:
        """
        Memory for LayerNorm:
        - Input and output: 2 * batch_size * seq_len * normalized_shape
        """
        return 2 * batch_size * seq_len * self.normalized_shape * self.dtype_bytes


class EmbeddingOperator(Operator):
    """
    Embedding lookup operator
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, dtype_bytes: int = 2):
        super().__init__("Embedding")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype_bytes = dtype_bytes
    
    def compute_flops(self, batch_size: int, seq_len: int) -> int:
        """
        FLOPs for embedding: essentially lookup, minimal computation
        """
        return batch_size * seq_len * self.embedding_dim
    
    def compute_memory(self, batch_size: int, seq_len: int) -> int:
        """
        Memory for embedding output
        """
        return batch_size * seq_len * self.embedding_dim * self.dtype_bytes


class FFNOperator(Operator):
    """
    Feed-Forward Network (FFN) operator
    Typically: Linear(hidden_size -> ffn_dim) + Activation + Linear(ffn_dim -> hidden_size)
    """
    
    def __init__(self, hidden_size: int, ffn_dim: int, dtype_bytes: int = 2):
        super().__init__("FFN")
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.dtype_bytes = dtype_bytes
        
        # Component operators
        self.up_proj = LinearOperator(hidden_size, ffn_dim, dtype_bytes)
        self.down_proj = LinearOperator(ffn_dim, hidden_size, dtype_bytes)
    
    def compute_flops(self, batch_size: int, seq_len: int) -> int:
        """
        FLOPs for FFN: up projection + down projection
        """
        up_flops = self.up_proj.compute_flops(batch_size, seq_len)
        down_flops = self.down_proj.compute_flops(batch_size, seq_len)
        # Add activation FLOPs (e.g., GELU/ReLU): ~batch_size * seq_len * ffn_dim
        activation_flops = batch_size * seq_len * self.ffn_dim
        return up_flops + activation_flops + down_flops
    
    def compute_memory(self, batch_size: int, seq_len: int) -> int:
        """
        Memory for FFN:
        - Intermediate activation: batch_size * seq_len * ffn_dim
        """
        intermediate_mem = batch_size * seq_len * self.ffn_dim * self.dtype_bytes
        return intermediate_mem


class TransformerLayerOperator(Operator):
    """
    Complete transformer layer: Attention + FFN with LayerNorms
    """
    
    def __init__(self, model_config: ModelConfig):
        super().__init__("TransformerLayer")
        self.model_config = model_config
        
        # Component operators
        self.attn = AttentionOperator(
            model_config.hidden_size,
            model_config.num_heads,
            model_config.head_dim,
            model_config.dtype_bytes
        )
        self.ffn = FFNOperator(
            model_config.hidden_size,
            model_config.ffn_dim,
            model_config.dtype_bytes
        )
        self.ln1 = LayerNormOperator(model_config.hidden_size, model_config.dtype_bytes)
        self.ln2 = LayerNormOperator(model_config.hidden_size, model_config.dtype_bytes)
    
    def compute_flops(self, batch_size: int, seq_len: int, is_prefill: bool = True) -> int:
        """Total FLOPs for a transformer layer"""
        attn_flops = self.attn.compute_flops(batch_size, seq_len, is_prefill)
        ffn_flops = self.ffn.compute_flops(batch_size, seq_len)
        ln_flops = self.ln1.compute_flops(batch_size, seq_len) + self.ln2.compute_flops(batch_size, seq_len)
        return attn_flops + ffn_flops + ln_flops
    
    def compute_memory(self, batch_size: int, seq_len: int, is_prefill: bool = True) -> int:
        """Total memory for a transformer layer"""
        attn_mem = self.attn.compute_memory(batch_size, seq_len, is_prefill)
        ffn_mem = self.ffn.compute_memory(batch_size, seq_len)
        ln_mem = self.ln1.compute_memory(batch_size, seq_len) + self.ln2.compute_memory(batch_size, seq_len)
        return attn_mem + ffn_mem + ln_mem
    
    def compute_kvcache(self, batch_size: int, seq_len: int) -> int:
        """KVCache for a transformer layer"""
        return self.attn.compute_kvcache(batch_size, seq_len)