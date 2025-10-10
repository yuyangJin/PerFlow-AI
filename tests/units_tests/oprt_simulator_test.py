'''
Unit tests for operator simulator
'''

import pytest
from perflowai.simulator.oprt import (
    LinearOperator,
    AttentionOperator,
    LayerNormOperator,
    EmbeddingOperator,
    FFNOperator,
    TransformerLayerOperator
)
from perflowai.core import ModelConfig


def test_linear_operator():
    """Test LinearOperator FLOPs and memory calculations"""
    linear = LinearOperator(in_features=4096, out_features=4096, dtype_bytes=2)
    
    batch_size = 1
    seq_len = 100
    
    # FLOPs: 2 * batch_size * seq_len * in_features * out_features
    expected_flops = 2 * 1 * 100 * 4096 * 4096
    flops = linear.compute_flops(batch_size, seq_len)
    assert flops == expected_flops, f"Expected {expected_flops}, got {flops}"
    
    # Memory: (input + output) * dtype_bytes
    expected_mem = (100 * 4096 + 100 * 4096) * 2
    mem = linear.compute_memory(batch_size, seq_len)
    assert mem == expected_mem, f"Expected {expected_mem}, got {mem}"
    
    # KVCache should be 0 for linear layers
    assert linear.compute_kvcache() == 0


def test_attention_operator():
    """Test AttentionOperator calculations"""
    attn = AttentionOperator(hidden_size=4096, num_heads=32, head_dim=128, dtype_bytes=2)
    
    batch_size = 1
    seq_len = 100
    
    # Test prefill
    flops_prefill = attn.compute_flops(batch_size, seq_len, is_prefill=True)
    mem_prefill = attn.compute_memory(batch_size, seq_len, is_prefill=True)
    kvcache = attn.compute_kvcache(batch_size, seq_len)
    
    # Basic sanity checks
    assert flops_prefill > 0, "Prefill FLOPs should be positive"
    assert mem_prefill > 0, "Prefill memory should be positive"
    
    # KVCache: 2 (K, V) * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    expected_kvcache = 2 * 1 * 32 * 100 * 128 * 2
    assert kvcache == expected_kvcache, f"Expected {expected_kvcache}, got {kvcache}"
    
    # Test decode
    flops_decode = attn.compute_flops(batch_size, seq_len, is_prefill=False)
    mem_decode = attn.compute_memory(batch_size, seq_len, is_prefill=False)
    
    # Decode should have less FLOPs than prefill for same seq_len
    assert flops_decode < flops_prefill, "Decode should have fewer FLOPs than prefill"
    assert mem_decode < mem_prefill, "Decode should use less memory than prefill"


def test_layernorm_operator():
    """Test LayerNormOperator calculations"""
    ln = LayerNormOperator(normalized_shape=4096, dtype_bytes=2)
    
    batch_size = 1
    seq_len = 100
    
    # FLOPs: ~5 * batch_size * seq_len * normalized_shape
    expected_flops = 5 * 1 * 100 * 4096
    flops = ln.compute_flops(batch_size, seq_len)
    assert flops == expected_flops, f"Expected {expected_flops}, got {flops}"
    
    # Memory: 2 * batch_size * seq_len * normalized_shape * dtype_bytes
    expected_mem = 2 * 1 * 100 * 4096 * 2
    mem = ln.compute_memory(batch_size, seq_len)
    assert mem == expected_mem, f"Expected {expected_mem}, got {mem}"


def test_embedding_operator():
    """Test EmbeddingOperator calculations"""
    emb = EmbeddingOperator(vocab_size=50000, embedding_dim=4096, dtype_bytes=2)
    
    batch_size = 1
    seq_len = 100
    
    flops = emb.compute_flops(batch_size, seq_len)
    mem = emb.compute_memory(batch_size, seq_len)
    
    # Basic sanity checks
    assert flops > 0, "Embedding FLOPs should be positive"
    assert mem > 0, "Embedding memory should be positive"
    
    # Memory: batch_size * seq_len * embedding_dim * dtype_bytes
    expected_mem = 1 * 100 * 4096 * 2
    assert mem == expected_mem, f"Expected {expected_mem}, got {mem}"


def test_ffn_operator():
    """Test FFNOperator calculations"""
    ffn = FFNOperator(hidden_size=4096, ffn_dim=16384, dtype_bytes=2)
    
    batch_size = 1
    seq_len = 100
    
    flops = ffn.compute_flops(batch_size, seq_len)
    mem = ffn.compute_memory(batch_size, seq_len)
    
    # FFN should have substantial FLOPs (two linear projections)
    assert flops > 0, "FFN FLOPs should be positive"
    assert mem > 0, "FFN memory should be positive"
    
    # Memory: intermediate activation
    expected_mem = 1 * 100 * 16384 * 2
    assert mem == expected_mem, f"Expected {expected_mem}, got {mem}"


def test_transformer_layer_operator():
    """Test TransformerLayerOperator aggregation"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    layer = TransformerLayerOperator(model_config)
    
    batch_size = 1
    seq_len = 100
    
    # Test prefill
    flops_prefill = layer.compute_flops(batch_size, seq_len, is_prefill=True)
    mem_prefill = layer.compute_memory(batch_size, seq_len, is_prefill=True)
    kvcache = layer.compute_kvcache(batch_size, seq_len)
    
    # Should aggregate attention + FFN + LayerNorms
    assert flops_prefill > 0, "Layer FLOPs should be positive"
    assert mem_prefill > 0, "Layer memory should be positive"
    assert kvcache > 0, "Layer KVCache should be positive"
    
    # KVCache per layer: 2 * num_heads * seq_len * head_dim * dtype_bytes
    expected_kvcache = 2 * 32 * 100 * 128 * 2
    assert kvcache == expected_kvcache, f"Expected {expected_kvcache}, got {kvcache}"
    
    # Test decode
    flops_decode = layer.compute_flops(batch_size, seq_len, is_prefill=False)
    mem_decode = layer.compute_memory(batch_size, seq_len, is_prefill=False)
    
    # Decode should have less FLOPs/memory for attention part
    assert flops_decode < flops_prefill, "Decode should have fewer FLOPs"
    assert mem_decode < mem_prefill, "Decode should use less memory"


def test_model_level_aggregation():
    """Test that model-level calculations aggregate correctly"""
    model_config = ModelConfig(
        num_layers=64,
        hidden_size=4096,
        ffn_dim=16384,
        hidden_dim=256,
        num_heads=32,
        head_dim=128,
        dtype_bytes=2
    )
    
    layer = TransformerLayerOperator(model_config)
    
    batch_size = 1
    seq_len = 100
    num_layers = 64
    
    # Per-layer KVCache
    kvcache_per_layer = layer.compute_kvcache(batch_size, seq_len)
    
    # Total model KVCache
    total_kvcache = kvcache_per_layer * num_layers
    
    # Expected: 64 layers * 2 * 32 * 100 * 128 * 2
    expected_total = 64 * 2 * 32 * 100 * 128 * 2
    assert total_kvcache == expected_total, f"Expected {expected_total}, got {total_kvcache}"
    
    # Should be exactly 100 MB
    assert total_kvcache == 104857600, "Should be 100 MB for this config"
    assert total_kvcache / (1024**2) == 100.0, "Should be exactly 100 MB"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
