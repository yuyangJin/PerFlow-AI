'''
@module memory simulator
'''

# import onnx
# from onnx import helper, shape_inference

from ...workflow import FlowNode
from ...core import ModelConfig, EventType
from ..oprt import TransformerLayerOperator


'''
@class MemSimulator
'''
class MemSimulator(FlowNode):
    def __init__(self):
        pass

# class OpMemSimulator(MemSimulator):

# class ModelMemSimulator(MemSimulator):
#     def __init__(self, onnx_path, ):
#         self.m_onnx_path
#         self.m_parallel_config = 
#         self.m_model = 
#         self.m_para_mem_size = 
#         self.m_grad_mem_size = 
#         self.m_optz_mem_size = 

    # def _read_model(self):
    #     """
    #     参数:
    #         onnx_path (str): ONNX模型文件路径
    #     """
    #     self.model = onnx.load(onnx_path)
    #     self.parameters = self._parse_parameters()

    #     _model = onnx.load(self.m_model_path)
    #     self.m_model_graph = shape_inference.infer_shapes(_model).graph

    # def simulate(self):
    #     for 


class ModelMemSimulator(MemSimulator):
    """
    Model-level memory simulator that computes KVCache size
    using fine-grained operator-level calculations
    """
    def __init__(self, model_config: ModelConfig):
        self.m_model_config = model_config
        # Create a transformer layer operator for calculations
        self.m_layer_op = TransformerLayerOperator(model_config)


    def kvcache(self, event):
        """
        Calculate KVCache size for an event.
        
        KVCache stores Key and Value tensors for all attention layers:
        - For each layer: 2 (K, V) * num_heads * seq_len * head_dim
        - Total: num_layers * per_layer_kvcache
        
        During prefill: We add KVCache for all input tokens
        During decode: We add KVCache for new generated tokens and track total
        """
        total_kvcache_bytes = 0
        
        if event.get_type() == EventType.PRF:
            # Prefill: Add KVCache for all input tokens
            tasks = event.get_tasks()
            
            for task in tasks.get():
                # Each task processes input_len tokens
                seq_len = task.req.input_len
                # KVCache per layer for this task
                per_layer_kvcache = self.m_layer_op.compute_kvcache(batch_size=1, seq_len=seq_len)
                # Total for all layers
                total_kvcache_bytes += per_layer_kvcache * self.m_model_config.num_layers

        elif event.get_type() == EventType.DCD:
            # Decode: Add KVCache for newly generated tokens
            tasks = event.get_tasks()
            
            for task in tasks.get():
                # Each decode iteration generates 1 token
                # Current sequence length includes input + previously decoded tokens
                current_seq_len = task.req.input_len + task.decode_iters + 1  # +1 for current token
                # KVCache per layer for this task (cumulative)
                per_layer_kvcache = self.m_layer_op.compute_kvcache(batch_size=1, seq_len=current_seq_len)
                # Total for all layers
                total_kvcache_bytes += per_layer_kvcache * self.m_model_config.num_layers
                
                # If task is finished, we can mark it for cleanup (but not subtract here)
                # The scheduler/memory manager should handle actual memory deallocation
                
        return total_kvcache_bytes


# class PipeMemSiulator(MemSimulator):
#     def __init__(self):
#         self.m_fwd_activation = 
#         self.m_bwd_activation = 
#         self.m_



