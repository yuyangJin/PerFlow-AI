'''
Test enhanced TorchProfiler trace parsing with Python parent id and Python id
'''

from perflowai.padoc.mst_mapper import MSTMapper
from perflowai.padoc import ModelStructureTree

def test_enhanced_torchprofiler_parsing():
    '''Test parsing TorchProfiler traces with Python id hierarchy'''
    print("\n=== Test: Enhanced TorchProfiler Parsing ===")
    
    # Create a mock TorchProfiler trace with Python id/parent id
    mock_trace = {
        'traceEvents': [
            {
                'name': 'model_forward',
                'ph': 'X',
                'cat': 'python_function',
                'ts': 1000,
                'dur': 100,
                'tid': 0,
                'pid': 0,
                'args': {
                    'Python id': 1,
                    'Python parent id': None,
                    'External id': 100,
                }
            },
            {
                'name': 'transformer_layer',
                'ph': 'X',
                'cat': 'python_function',
                'ts': 1010,
                'dur': 80,
                'tid': 0,
                'pid': 0,
                'args': {
                    'Python id': 2,
                    'Python parent id': 1,  # Child of model_forward
                    'External id': 101,
                }
            },
            {
                'name': 'attention_forward',
                'ph': 'X',
                'cat': 'python_function',
                'ts': 1020,
                'dur': 30,
                'tid': 0,
                'pid': 0,
                'args': {
                    'Python id': 3,
                    'Python parent id': 2,  # Child of transformer_layer
                    'External id': 102,
                }
            },
            {
                'name': 'matmul',
                'ph': 'X',
                'cat': 'kernel',
                'ts': 1025,
                'dur': 20,
                'tid': 0,
                'pid': 0,
                'args': {
                    'Python id': 4,
                    'Python parent id': 3,  # Child of attention_forward
                    'External id': 103,
                    'device': 0,
                    'stream': 7,
                }
            },
            {
                'name': 'ffn_forward',
                'ph': 'X',
                'cat': 'python_function',
                'ts': 1060,
                'dur': 25,
                'tid': 0,
                'pid': 0,
                'args': {
                    'Python id': 5,
                    'Python parent id': 2,  # Child of transformer_layer (sibling of attention)
                    'External id': 104,
                }
            },
        ]
    }
    
    print("\n1. Extracting call stacks with Python id hierarchy:")
    callstacks = MSTMapper.extract_callstack_from_torchprofiler(mock_trace)
    
    # Verify hierarchical call stacks
    for i, (event_idx, stack) in enumerate(callstacks.items()):
        event_name = mock_trace['traceEvents'][event_idx]['name']
        print(f"   Event {event_idx} ({event_name}): {' -> '.join(stack)}")
    
    # Verify the call stack hierarchy
    assert len(callstacks) == 5, "Should have 5 call stacks"
    
    # Check that matmul has the full hierarchy
    matmul_stack = callstacks[3]
    expected_hierarchy = ['model_forward', 'transformer_layer', 'attention_forward', 'matmul']
    assert matmul_stack == expected_hierarchy, f"Expected {expected_hierarchy}, got {matmul_stack}"
    print(f"\n2. ✓ Verified hierarchical call stack for matmul: {' -> '.join(matmul_stack)}")
    
    # Check that ffn_forward has correct parent
    ffn_stack = callstacks[4]
    expected_ffn = ['model_forward', 'transformer_layer', 'ffn_forward']
    assert ffn_stack == expected_ffn, f"Expected {expected_ffn}, got {ffn_stack}"
    print(f"3. ✓ Verified sibling relationship for ffn_forward: {' -> '.join(ffn_stack)}")
    
    print("\n✓ Enhanced TorchProfiler parsing test passed!")
    return callstacks


def test_fallback_to_traditional_parsing():
    '''Test fallback to traditional parsing when Python id not available'''
    print("\n=== Test: Fallback to Traditional Parsing ===")
    
    # Mock trace without Python id (old format)
    mock_trace = {
        'traceEvents': [
            {
                'name': 'forward_step',
                'ts': 1000,
                'dur': 100,
                'tid': 0,
                'args': {
                    'Python call stack': 'model\nlayer1\nforward'
                }
            },
            {
                'name': 'backward_step',
                'ts': 1100,
                'dur': 150,
                'tid': 0,
                'args': {}  # No Python id, no call stack
            },
        ]
    }
    
    callstacks = MSTMapper.extract_callstack_from_torchprofiler(mock_trace)
    
    print("\n1. Extracted call stacks:")
    for idx, stack in callstacks.items():
        print(f"   Event {idx}: {' -> '.join(stack)}")
    
    # Verify fallback behavior
    assert callstacks[0] == ['model', 'layer1', 'forward'], "Should parse Python call stack"
    assert callstacks[1] == ['backward_step'], "Should fallback to event name"
    
    print("\n✓ Fallback parsing test passed!")


if __name__ == '__main__':
    print("="*70)
    print("Testing Enhanced TorchProfiler Trace Parsing")
    print("="*70)
    
    test_enhanced_torchprofiler_parsing()
    test_fallback_to_traditional_parsing()
    
    print("\n" + "="*70)
    print("✓ All enhanced parsing tests passed!")
    print("="*70)
