'''
test ModelConfig class
'''

from perflowai.core import ModelConfig, LlamaConfig, LlamaVersionMap, DeepSeekConfig, DeepSeekVersionMap

def test_ModelConfig():
    model_config = ModelConfig(
        name="test_model",
        url="https://example.com/model",
    )
    assert model_config.name == "test_model"
    assert model_config.url == "https://example.com/model"

def test_LlamaConfig():
    llama_config = LlamaConfig(
        name="Llama-3.1-8B-Instruct",
        url="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_key_value_heads=8,
    )
    assert llama_config.name == "Llama-3.1-8B-Instruct"
    assert llama_config.url == "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    assert llama_config.hidden_size == 4096
    assert llama_config.num_attention_heads == 32
    assert llama_config.num_hidden_layers == 32
    assert llama_config.intermediate_size == 14336
    assert llama_config.num_key_value_heads == 8
    assert llama_config.dtype_bytes == 2

    llama_405b_config = LlamaVersionMap["Llama-3.1-405B-Instruct"]
    assert llama_405b_config.name == "Llama-3.1-405B-Instruct"
    assert llama_405b_config.url == "https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct"
    assert llama_405b_config.hidden_size == 16384
    assert llama_405b_config.num_attention_heads == 128
    assert llama_405b_config.num_hidden_layers == 126
    assert llama_405b_config.intermediate_size == 53248
    assert llama_405b_config.num_key_value_heads == 8
    assert llama_405b_config.dtype_bytes == 2


def test_DeepSeekConfig():
    deepseek_config = DeepSeekConfig(
        name="DeepSeek-V3-671B-Instruct",
        url="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base",
        vocab_size= 129280,
        dim= 7168,
        inter_dim= 18432,
        moe_inter_dim= 2048,
        n_layers= 61,
        n_dense_layers= 3,
        n_heads= 128,
        n_routed_experts= 256,
        n_shared_experts= 1,
        n_activated_experts= 8,
        n_expert_groups= 8,
        n_limited_groups= 4,
        route_scale= 2.5,
        score_func= "sigmoid",
        q_lora_rank= 1536,
        kv_lora_rank= 512,
        qk_nope_head_dim= 128,
        qk_rope_head_dim= 64,
        v_head_dim= 128,
        dtype= "fp8"
    )
    assert deepseek_config.name == "DeepSeek-V3-671B-Instruct"
    assert deepseek_config.url == "https://huggingface.co/deepseek-ai/DeepSeek-V3-Base"
    assert deepseek_config.hidden_size == 7168
    assert deepseek_config.intermediate_size == 18432
    assert deepseek_config.moe_inter_dim == 2048
    assert deepseek_config.num_layers == 61
    assert deepseek_config.num_dense_layers == 3
    assert deepseek_config.num_attention_heads == 128
    assert deepseek_config.num_key_value_heads == 128
    assert deepseek_config.num_routed_experts == 256
    assert deepseek_config.num_shared_experts == 1
    assert deepseek_config.num_activated_experts == 8
    assert deepseek_config.dtype_bytes == 1