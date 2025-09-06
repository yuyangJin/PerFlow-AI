'''
@module model simulator
'''

'''
@class ModelSimulator
'''

class ModelSimulator(Simulator):
    '''
    @method __init__
    Initialize the model simulator.
    '''
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.m_compute_volume = 0
        self.m_memory_volume = 0
        self.m_model_config = ModelConfig

    def _compute_prefill_volume(self, event):
        tasks = event.get_tasks() 


    def _calculate_compute_volume(self, batch_size: int, seq_len: int):

        mlp_layer = MLPSimulator(self.m_model_config)
        attn_layer = AttentionSimulator(self.m_model_config)

        if self.mode == AttentionMode.PREFILL:
            mlp_compute_volume = mlp_layer._calculate_compute_volume(batch_size * seq_len)
        else:
            mlp_compute_volume = mlp_layer._calculate_compute_volume(batch_size)
        attn_compute_volume = attn_layer._calculate_compute_volume(batch_size, seq_len, self.mode)

        self.m_compute_volume += (mlp_compute_volume + attn_compute_volume) * self.num_hidden_layers


    def _calculate_memory_volume(self, batch_size: int, seq_len: int, mode: EventType):
    
        mlp_layer = MLPSimulator(self.m_model_config)
        attn_layer = AttentionSimulator(self.m_model_config)

        if self.mode == AttentionMode.PREFILL:
            mlp_memory_volume = mlp_layer._calculate_memory_volume(batch_size * seq_len)
        else:
            mlp_memory_volume = mlp_layer._calculate_memory_volume(batch_size)
        attn_memory_volume = attn_layer._calculate_memory_volume(batch_size, seq_len, self.mode)

        self.m_memory_volume += (mlp_memory_volume + attn_memory_volume) * self.num_hidden_layers

class DeepSeekSimulator(Simulator):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.m_compute_volume = 0
        self.m_memory_volume = 0
        self.attn_layer = DeepSeekAttentionSimulator(model_config)
        self.mlp_layer = DeepSeekMLPSimulator(model_config)
        self.moe_layer = DeepSeekMoESimulator(model_config)
        self.num_dense_layers = model_config.num_dense_layers
        self.num_layers = model_config.num_layers
        self.num_attention_heads = model_config.num_attention_heads
        self.hidden_size = model_config.hidden_size
        self.dtype_bytes = model_config.dtype_bytes
    
    def flop(self, batch_size: int, seq_len: int):


        mlp_flops = self.mlp_layer.flop(batch_size)
        attn_flops = self.attn_layer.flop(batch_size, seq_len)
        moe_flops = self.moe_layer.flop(batch_size, seq_len)

        self.m_memory_volume = self.num_dense_layers * mlp_flops \
                               + (self.num_layers - self.num_dense_layers) * moe_flops \
                               + self.num_layers * attn_flops
    
    def kvcache(self, event):
        seq_len = 0
        if event.get_type() == EventType.PRF:
            
            tasks = event.get_tasks()
            
            for task in tasks.get():
                seq_len += task.req.input_len + 1 # Prompt tokens 

        elif event.get_type() == EventType.DCD:

            # New adding element size
            tasks = event.get_tasks()
            seq_len += tasks.get_num_task() # Each task only generate one token

            # Elements should be removed after decode
            for task in tasks.get():
                if task.decode_iters == task.req.output_len - 1: # The task is finished, the KVCache of this task is no longer needed
                    seq_len -= task.req.input_len + task.req.output_len


        # Calculate element size per layer, Q, V
        kv_head = max(1, self.num_attention_heads // tp_size)
        head_dim = self.hidden_size // self.num_attention_heads

        per_layer_elements = 2 * seq_len * kv_head * head_dim

        # Calculate the total element size of the model
        total_elements = per_layer_elements * self.num_layers
        

        # Calculate the total kvcache size 
        total_bytes = total_elements * self.dtype_bytes

        return total_bytes