from .trace_nodes import TreeNode

unsupported_nodes = {}

class OpsCalculator:
    """Class for calculating FLOPs and Memory Access operations."""
    
    # Mapping from dtype to byte size
    DTYPE_BYTES = {
        'torch.float32': 4,
        'torch.float64': 8,
        'torch.float16': 2,
        'torch.bfloat16': 2,
        'torch.int32': 4,
        'torch.int64': 8,
        'torch.int16': 2,
        'torch.int8': 1,
        'torch.uint8': 1,
        'torch.bool': 1,
    }
    
    def _get_dtype_bytes(self, dtype_str: str) -> int:
        """Get byte size for given dtype string."""
        return self.DTYPE_BYTES.get(dtype_str, 4)  # Default 4 bytes
    
    def calculate_tree_ops(self, root_node: TreeNode) -> tuple:
        """
        Recursively calculate FLOPs and memory access for the entire tree,
        and update the actual duration of nodes.
        
        Args:
            root_node: Root node of the tree
            
        Returns:
            tuple: (total_flops, total_bytes, overhead_ratio)
        """
        total_flops, total_bytes = self._calculate_subtree_ops(root_node)
        
        # Calculate performance overhead ratio
        if root_node.elapsed and root_node.elapsed > 0:
            overhead_ratio = (root_node.elapsed - root_node.actual_dur) / root_node.elapsed * 100
            root_node.overhead_ratio = overhead_ratio
        else:
            overhead_ratio = 0
        
        return total_flops, total_bytes, overhead_ratio

    def _calculate_subtree_ops(self, node: TreeNode) -> tuple:
        """
        Recursively calculate FLOPs and memory access for a subtree.
        
        Args:
            node: Current node to process
            
        Returns:
            tuple: (subtree_flops, subtree_bytes)
        """
        subtree_flops = 0
        subtree_bytes = 0
        children_actual_dur = 0
        
        # Recursively process all child nodes
        for child in node.children:
            child_flops, child_bytes = self._calculate_subtree_ops(child)
            subtree_flops += child_flops
            subtree_bytes += child_bytes
            children_actual_dur += child.actual_dur if hasattr(child, 'actual_dur') else 0
        
        # If it's a leaf node, calculate its own operations
        if not node.children:
            node_flops, node_bytes = self.calculate_leaf_node_ops(node)
            node.actual_dur = node.elapsed  # Leaf node's actual duration is the measured elapsed time
        else:
            # Non-leaf node's operations are the sum of children's operations
            node_flops = subtree_flops
            node_bytes = subtree_bytes
            # Non-leaf node's actual duration is the sum of children's actual durations
            node.actual_dur = children_actual_dur
        
        # Update node's FLOPs and bytes
        node.flops = node_flops
        node.bytes = node_bytes
        
        return node_flops, node_bytes

    def calculate_leaf_node_ops(self, node) -> tuple:
        """Calculate FLOPs and Memory Access for a single leaf node."""
        if isinstance(node, dict):
            node_name = node['name']
            input_info = node['input_info']
            output_info = node['output_info']
        else:
            node_name = node.name
            input_info = node.input_info
            output_info = node.output_info
        
        # Dispatch calculation based on node type
        if 'Linear' in node_name or 'Dense' in node_name:
            return self._calculate_linear_ops(input_info, output_info)
        elif 'Conv' in node_name:
            return self._calculate_conv_ops(input_info, output_info)
        elif 'Norm' in node_name:
            return self._calculate_norm_ops(input_info, output_info)
        elif 'ReLU' in node_name or 'Activation' in node_name:
            return self._calculate_activation_ops(input_info, output_info)
        elif 'Pool' in node_name:
            return self._calculate_pool_ops(input_info, output_info)
        elif 'Attention' in node_name or 'MultiheadAttention' in node_name:
            return self._calculate_attention_ops(input_info, output_info)
        elif 'SiluAndMul' in node_name:
            return self._calculate_silu_and_mul_ops(input_info, output_info)
        elif 'RotaryEmbedding' in node_name:
            return self._calculate_rotary_embedding_ops(input_info, output_info)
        elif 'Vocab' in node_name and 'Embedding' in node_name:
            return self._calculate_vocab_embedding_ops(input_info, output_info)
        else:
            if node_name not in unsupported_nodes:
                unsupported_nodes[node_name] = 1
                print(f'[Warning] Unsupported node type: {node_name}')
            return 0, 0
    
    def _calculate_linear_ops(self, input_info, output_info) -> tuple:
        """Calculate FLOPs and Memory Access for Linear layer."""
        if not input_info or not output_info:
            return 0, 0
        
        # Get input tensor info
        input_tensor = input_info[0] if isinstance(input_info, list) else input_info
        input_shape = input_tensor.get('shape', [])
        input_dtype = input_tensor.get('dtype', 'torch.float32')
        
        # Get output tensor info
        output_shape = output_info.get('shape', []) if isinstance(output_info, dict) else []
        
        if len(input_shape) < 2 or len(output_shape) < 2:
            return 0, 0
        
        input_features = input_shape[-1]
        output_features = output_shape[-1]
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        
        # FLOPs: 2 * batch_size * input_features * output_features
        flops = 2 * batch_size * input_features * output_features
        
        # Memory Access (bytes): input + output + weights + bias
        dtype_bytes = self._get_dtype_bytes(input_dtype)
        memory_access = (
            batch_size * input_features * dtype_bytes +  # input
            batch_size * output_features * dtype_bytes +  # output
            input_features * output_features * dtype_bytes +  # weights
            output_features * dtype_bytes  # bias
        )
        
        return flops, memory_access
    
    def _calculate_conv_ops(self, input_info, output_info) -> tuple:
        """Calculate FLOPs and Memory Access for Conv layer."""
        if not input_info or not output_info:
            return 0, 0
        
        input_tensor = input_info[0] if isinstance(input_info, list) else input_info
        input_shape = input_tensor.get('shape', [])
        input_dtype = input_tensor.get('dtype', 'torch.float32')
        
        output_shape = output_info.get('shape', []) if isinstance(output_info, dict) else []
        
        if len(input_shape) < 4 or len(output_shape) < 4:
            return 0, 0
        
        # Assume kernel size 3x3, stride=1, padding=1
        kernel_size = 3
        groups = 1  # Default groups=1
        
        batch_size, in_channels, height, width = input_shape
        _, out_channels, out_height, out_width = output_shape
        
        # FLOPs: 2 * batch_size * out_channels * out_height * out_width * in_channels * kernel_size² / groups
        flops = 2 * batch_size * out_channels * out_height * out_width * in_channels * kernel_size * kernel_size // groups
        
        # Memory Access (bytes)
        dtype_bytes = self._get_dtype_bytes(input_dtype)
        memory_access = (
            batch_size * in_channels * height * width * dtype_bytes +  # input
            batch_size * out_channels * out_height * out_width * dtype_bytes +  # output
            out_channels * in_channels * kernel_size * kernel_size * dtype_bytes // groups +  # weights
            out_channels * dtype_bytes  # bias
        )
        
        return flops, memory_access
    
    def _calculate_norm_ops(self, input_info, output_info) -> tuple:
        """Calculate FLOPs and Memory Access for Normalization layer."""
        if not input_info:
            return 0, 0
        
        input_tensor = input_info[0] if isinstance(input_info, list) else input_info
        input_shape = input_tensor.get('shape', [])
        input_dtype = input_tensor.get('dtype', 'torch.float32')
        
        if not input_shape:
            return 0, 0
        
        # Calculate number of elements
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        # Normalization layer: FLOPs ≈ 4 * num_elements (mean, variance, normalization, scale/shift)
        flops = 4 * num_elements
        
        # Memory Access (bytes): input + output + weights + bias + statistics
        dtype_bytes = self._get_dtype_bytes(input_dtype)
        # Assume scale and shift parameters per channel
        num_channels = input_shape[1] if len(input_shape) > 1 else 1
        memory_access = (
            num_elements * dtype_bytes +  # input
            num_elements * dtype_bytes +  # output
            2 * num_channels * dtype_bytes +  # weights and bias
            2 * num_channels * dtype_bytes  # mean and variance statistics
        )
        
        return flops, memory_access
    
    def _calculate_activation_ops(self, input_info, output_info) -> tuple:
        """Calculate FLOPs and Memory Access for Activation layer."""
        if not input_info:
            return 0, 0
        
        input_tensor = input_info[0] if isinstance(input_info, list) else input_info
        input_shape = input_tensor.get('shape', [])
        input_dtype = input_tensor.get('dtype', 'torch.float32')
        
        if not input_shape:
            return 0, 0
        
        # Calculate number of elements
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        # Activation layer: FLOPs = num_elements
        flops = num_elements
        
        # Memory Access (bytes): input + output
        dtype_bytes = self._get_dtype_bytes(input_dtype)
        memory_access = 2 * num_elements * dtype_bytes
        
        return flops, memory_access
    
    def _calculate_pool_ops(self, input_info, output_info) -> tuple:
        """Calculate FLOPs and Memory Access for Pooling layer."""
        if not input_info:
            return 0, 0
        
        input_tensor = input_info[0] if isinstance(input_info, list) else input_info
        input_shape = input_tensor.get('shape', [])
        input_dtype = input_tensor.get('dtype', 'torch.float32')
        
        if not input_shape:
            return 0, 0
        
        # Calculate number of elements
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        # Pooling layer: FLOPs ≈ num_elements (one comparison per element)
        flops = num_elements
        
        # Memory Access (bytes): input + output
        dtype_bytes = self._get_dtype_bytes(input_dtype)
        memory_access = 2 * num_elements * dtype_bytes
        
        return flops, memory_access
    
    def _calculate_attention_ops(self, input_info, output_info) -> tuple:
        """Calculate FLOPs and Memory Access for Attention layer."""
        if not input_info or not isinstance(input_info, list):
            return 0, 0
        
        # Parse input tensors
        q_tensor = input_info[0] if len(input_info) > 0 else {}
        k_tensor = input_info[1] if len(input_info) > 1 else {}
        v_tensor = input_info[2] if len(input_info) > 2 else {}
        
        q_shape = q_tensor.get('shape', [])
        q_dtype = q_tensor.get('dtype', 'torch.float32')
        
        # If K/V not provided or same shape as Q, use Q shape
        k_shape = k_tensor.get('shape', q_shape) if k_tensor else q_shape
        v_shape = v_tensor.get('shape', q_shape) if v_tensor else q_shape
        
        if not q_shape or len(q_shape) < 2:
            return 0, 0
        
        dtype_bytes = self._get_dtype_bytes(q_dtype)
        
        # Get sequence length and hidden dimension
        if len(q_shape) == 2:  # [seq_len, hidden_size]
            seq_len_q, hidden_size_q = q_shape
            seq_len_k, hidden_size_k = k_shape if len(k_shape) == 2 else (seq_len_q, hidden_size_q)
            seq_len_v, hidden_size_v = v_shape if len(v_shape) == 2 else (seq_len_q, hidden_size_q)
            batch_size = 1
            num_heads = 1  # Single head attention
            head_dim = hidden_size_q
        elif len(q_shape) == 3:  # [batch_size, seq_len, hidden_size]
            batch_size, seq_len_q, hidden_size_q = q_shape
            _, seq_len_k, hidden_size_k = k_shape if len(k_shape) == 3 else (batch_size, seq_len_q, hidden_size_q)
            _, seq_len_v, hidden_size_v = v_shape if len(v_shape) == 3 else (batch_size, seq_len_q, hidden_size_q)
            
            # Infer number of heads and head dimension
            if hidden_size_q % 64 == 0:  # Assume head dim 64
                num_heads = hidden_size_q // 64
                head_dim = 64
            elif hidden_size_q % 128 == 0:  # Assume head dim 128
                num_heads = hidden_size_q // 128
                head_dim = 128
            else:
                # Default single head
                num_heads = 1
                head_dim = hidden_size_q
        elif len(q_shape) == 4:  # [batch_size, num_heads, seq_len, head_dim]
            batch_size, num_heads, seq_len_q, head_dim = q_shape
            _, _, seq_len_k, _ = k_shape if len(k_shape) == 4 else (batch_size, num_heads, seq_len_q, head_dim)
            _, _, seq_len_v, _ = v_shape if len(v_shape) == 4 else (batch_size, num_heads, seq_len_q, head_dim)
            hidden_size_q = num_heads * head_dim
        else:
            # Unsupported shape
            return 0, 0
        
        # FLOPs calculation
        # 1. Q*K^T: 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
        qk_flops = 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
        
        # 2. Softmax: 3 * batch_size * num_heads * seq_len_q * seq_len_k
        softmax_flops = 3 * batch_size * num_heads * seq_len_q * seq_len_k
        
        # 3. Attention*V: 2 * batch_size * num_heads * seq_len_q * seq_len_v * head_dim
        attn_v_flops = 2 * batch_size * num_heads * seq_len_q * seq_len_v * head_dim
        
        # 4. Output projection
        output_proj_flops = 2 * batch_size * seq_len_q * hidden_size_q * hidden_size_q
        
        total_flops = qk_flops + softmax_flops + attn_v_flops + output_proj_flops
        
        # Memory Access calculation
        # Input memory
        input_memory = 0
        for i, tensor_info in enumerate(input_info):
            if i >= 3:  # Only process first 3 inputs (Q, K, V)
                break
            shape = tensor_info.get('shape', [])
            if shape:
                elements = 1
                for dim in shape:
                    elements *= dim
                input_memory += elements * dtype_bytes
        
        # Output memory
        output_memory = 0
        if output_info and isinstance(output_info, dict):
            output_shape = output_info.get('shape', [])
            if output_shape:
                output_elements = 1
                for dim in output_shape:
                    output_elements *= dim
                output_memory = output_elements * dtype_bytes
        else:
            # Default output shape same as Q
            output_memory = batch_size * seq_len_q * hidden_size_q * dtype_bytes
        
        # Weight memory (QKV projection + output projection)
        hidden_size = hidden_size_q
        weight_memory = (3 * hidden_size * hidden_size +  # QKV projection
                        hidden_size * hidden_size) * dtype_bytes  # Output projection
        
        # Intermediate memory (Attention matrix)
        attention_matrix_memory = batch_size * num_heads * seq_len_q * seq_len_k * dtype_bytes
        
        total_memory_access = input_memory + output_memory + weight_memory + attention_matrix_memory
        
        return total_flops, total_memory_access

    def _calculate_silu_and_mul_ops(self, input_info, output_info) -> tuple:
        """Calculate FLOPs and Memory Access for SiLUAndMul operation."""
        if not input_info:
            return 0, 0
        
        input_tensor = input_info[0] if isinstance(input_info, list) else input_info
        input_shape = input_tensor.get('shape', [])
        input_dtype = input_tensor.get('dtype', 'torch.float32')
        
        if not input_shape or len(input_shape) != 2:
            return 0, 0
        
        batch_size, input_features = input_shape
        
        # Verify output shape
        if output_info and isinstance(output_info, dict):
            output_shape = output_info.get('shape', [])
            if output_shape and len(output_shape) == 2:
                expected_output_features = input_features // 2
                if output_shape[1] != expected_output_features:
                    print(f"[Warning] Unexpected output shape for SiluAndMul: {output_shape}, expected features: {expected_output_features}")
        
        # Actual processed elements: half of input
        half_features = input_features // 2
        num_elements = batch_size * half_features
        
        # FLOPs calculation:
        # 1. SiLU activation: 9 FLOPs per element (sigmoid ≈8 + mul 1)
        # 2. Element-wise multiplication: 1 FLOP per element
        # Total: 10 FLOPs per element
        flops_per_element = 10
        flops = num_elements * flops_per_element
        
        # Memory Access (bytes)
        dtype_bytes = self._get_dtype_bytes(input_dtype)
        
        # Input memory: full input tensor
        input_memory = batch_size * input_features * dtype_bytes
        
        # Output memory: half size
        output_memory = batch_size * half_features * dtype_bytes
        
        total_memory_access = input_memory + output_memory
        
        return flops, total_memory_access
    
    def _calculate_rotary_embedding_ops(self, input_info, output_info) -> tuple:
        """Calculate FLOPs and Memory Access for RotaryEmbedding operation."""
        if not input_info or not isinstance(input_info, list) or len(input_info) < 3:
            return 0, 0
        
        # Parse input tensors
        positions_tensor = input_info[0]  # Position indices, usually int64
        query_tensor = input_info[1]      # Query vectors
        key_tensor = input_info[2]        # Key vectors
        
        positions_shape = positions_tensor.get('shape', [])
        query_shape = query_tensor.get('shape', [])
        key_shape = key_tensor.get('shape', [])
        query_dtype = query_tensor.get('dtype', 'torch.float32')
        
        if not query_shape or len(query_shape) < 2:
            return 0, 0
        
        dtype_bytes = self._get_dtype_bytes(query_dtype)
        
        # Get sequence length and hidden dimension
        if len(query_shape) == 2:  # [seq_len, hidden_size]
            seq_len, hidden_size = query_shape
            batch_size = 1
        elif len(query_shape) == 3:  # [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = query_shape
        else:
            return 0, 0
        
        # Rotary positional encoding calculation
        # Split hidden dimension into real and imaginary parts
        # Apply rotation matrix for each position
        
        # Assume head dimension
        if hidden_size % 128 == 0:
            head_dim = 128
            num_heads = hidden_size // head_dim
        elif hidden_size % 64 == 0:
            head_dim = 64
            num_heads = hidden_size // head_dim
        else:
            head_dim = hidden_size
            num_heads = 1
        
        # FLOPs calculation
        flops_per_position_per_head = 4 * head_dim  # Complex multiplication: 4 real multiplications
        
        total_positions = batch_size * seq_len
        total_flops = total_positions * num_heads * flops_per_position_per_head
        
        # Memory Access calculation
        input_memory = (2 * batch_size * seq_len * hidden_size * dtype_bytes)
        
        output_memory = 0
        if output_info and isinstance(output_info, list):
            for output_tensor in output_info:
                output_shape = output_tensor.get('shape', [])
                if output_shape:
                    elements = 1
                    for dim in output_shape:
                        elements *= dim
                    output_memory += elements * dtype_bytes
        else:
            output_memory = 2 * batch_size * seq_len * hidden_size * dtype_bytes
        
        # Rotary table memory access
        max_seq_len = 32768  # Common max sequence length
        rotary_table_memory = 2 * max_seq_len * (head_dim // 2) * dtype_bytes
        
        total_memory_access = input_memory + output_memory + rotary_table_memory
        
        return total_flops, total_memory_access
    
    def _calculate_vocab_embedding_ops(self, input_info, output_info) -> tuple:
        """Calculate FLOPs and Memory Access for VocabParallelEmbedding operation."""
        if not input_info or not isinstance(input_info, list) or len(input_info) == 0:
            return 0, 0
        
        # Parse input tensor
        input_tensor = input_info[0]  # Input token IDs, usually int32 or int64
        input_shape = input_tensor.get('shape', [])
        input_dtype = input_tensor.get('dtype', 'torch.int32')
        
        # Parse output tensor
        output_shape = output_info.get('shape', []) if isinstance(output_info, dict) else []
        output_dtype = output_info.get('dtype', 'torch.float32') if isinstance(output_info, dict) else 'torch.float32'
        
        if not input_shape or not output_shape:
            return 0, 0
        
        # Input [256], output [256, 2048]
        if len(input_shape) == 1:
            batch_size = input_shape[0]
            hidden_size = output_shape[1] if len(output_shape) == 2 else 0
        elif len(input_shape) == 2:  # [batch_size, seq_len]
            batch_size = input_shape[0] * input_shape[1]  # Total tokens
            hidden_size = output_shape[-1] if output_shape else 0
        else:
            return 0, 0
        
        if hidden_size == 0:
            return 0, 0
        
        # Embedding operations: mainly table lookup
        flops_per_token = 1  # Simplified calculation
        total_flops = batch_size * flops_per_token
        
        # Memory Access calculation
        input_dtype_bytes = 4  # int32 is usually 4 bytes
        output_dtype_bytes = self._get_dtype_bytes(output_dtype)
        
        input_memory = batch_size * input_dtype_bytes
        output_memory = batch_size * hidden_size * output_dtype_bytes
        
        # Weight memory: embedding matrix
        vocab_size = 100000  # Common vocabulary size
        world_size = 8       # Common GPU count
        
        local_vocab_size = vocab_size // world_size
        accessed_weight_memory = batch_size * hidden_size * output_dtype_bytes
        
        total_memory_access = input_memory + output_memory + accessed_weight_memory
        
        return total_flops, total_memory_access