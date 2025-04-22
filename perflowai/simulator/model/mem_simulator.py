'''
@module memory simulator
'''

# import onnx
# from onnx import helper, shape_inference

from ...workflow import FlowNode
from ...core import ModelConfig, EventType


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
    def __init__(self, model_config: ModelConfig):
        self.m_model_config = model_config


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
        per_layer_elements = 2 * seq_len * self.m_model_config.num_heads * self.m_model_config.head_dim

        # Calculate the total element size of the model
        total_elements = per_layer_elements * self.m_model_config.num_layers
        

        # Calculate the total kvcache size 
        total_bytes = total_elements * self.m_model_config.dtype_bytes

        return total_bytes



# class PipeMemSiulator(MemSimulator):
#     def __init__(self):
#         self.m_fwd_activation = 
#         self.m_bwd_activation = 
#         self.m_



