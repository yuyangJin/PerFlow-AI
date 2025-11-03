'''
Test Model Structure Tree (MST)
'''

from perflowai.padoc import ModelStructureTree, MSTNode

def test_mst_basic():
    '''Test basic MST creation and operations'''
    mst = ModelStructureTree()
    
    # Add nodes
    module1 = mst.add_node('LinearLayer1', 'module', 0)
    module2 = mst.add_node('LinearLayer2', 'module', 0)
    
    # Add children
    op1 = mst.add_node('matmul', 'operator', module1.node_id)
    op2 = mst.add_node('bias_add', 'operator', module1.node_id)
    
    assert len(mst.nodes) == 5  # root + 4 nodes
    assert len(module1.children) == 2
    assert op1.parent == module1
    
    print("✓ Basic MST creation test passed")

def test_mst_trace_mapping():
    '''Test mapping trace events to MST nodes'''
    mst = ModelStructureTree()
    
    # Create hierarchy
    layer1 = mst.add_node('layer1', 'module', 0)
    op1 = mst.add_node('forward', 'operation', layer1.node_id)
    
    # Map trace events
    op1.add_trace_event(1)
    op1.add_trace_event(2)
    
    assert len(op1.trace_events) == 2
    assert 1 in op1.trace_events
    
    print("✓ Trace event mapping test passed")

def test_mst_callstack_matching():
    '''Test call stack matching for event mapping'''
    mst = ModelStructureTree()
    
    # Create nodes with call stacks
    layer1 = mst.add_node('layer1', 'module', 0)
    layer1.set_call_stack(['model', 'layer1'])
    
    op1 = mst.add_node('forward', 'operation', layer1.node_id)
    op1.set_call_stack(['model', 'layer1', 'forward'])
    
    # Test matching
    matched = mst.map_trace_events_by_callstack(100, ['model', 'layer1', 'forward'])
    assert matched == op1
    assert 100 in op1.trace_events
    
    # Test partial match
    matched2 = mst.map_trace_events_by_callstack(101, ['model', 'layer1', 'forward', 'matmul'])
    assert matched2 == op1  # Should match the longest prefix
    
    print("✓ Call stack matching test passed")

def test_mst_visualization():
    '''Test MST visualization'''
    mst = ModelStructureTree()
    
    layer1 = mst.add_node('layer1', 'module', 0)
    op1 = mst.add_node('forward', 'operation', layer1.node_id)
    op1.add_trace_event(1)
    op1.add_trace_event(2)
    
    viz = mst.visualize()
    
    assert 'root' in viz
    assert 'layer1' in viz
    assert 'forward' in viz
    assert '[events: 2]' in viz
    
    print("✓ MST visualization test passed")
    print("Visualization output:")
    print(viz)

def test_mst_serialization():
    '''Test MST JSON serialization and deserialization'''
    import tempfile
    import os
    
    mst = ModelStructureTree()
    layer1 = mst.add_node('layer1', 'module', 0)
    op1 = mst.add_node('forward', 'operation', layer1.node_id)
    op1.add_trace_event(1)
    op1.set_call_stack(['model', 'layer1', 'forward'])
    
    # Save to JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        mst.to_json(temp_path)
        
        # Load from JSON
        loaded_mst = ModelStructureTree.from_json(temp_path)
        
        assert len(loaded_mst.nodes) == len(mst.nodes)
        
        # Find the operation node
        op_nodes = loaded_mst.find_nodes_by_name('forward')
        assert len(op_nodes) == 1
        assert op_nodes[0].trace_events == [1]
        assert op_nodes[0].call_stack == ['model', 'layer1', 'forward']
        
        print("✓ MST serialization test passed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_mst_node_queries():
    '''Test querying nodes by name and type'''
    mst = ModelStructureTree()
    
    mst.add_node('layer1', 'module', 0)
    mst.add_node('layer2', 'module', 0)
    mst.add_node('forward', 'operation', 0)
    
    modules = mst.find_nodes_by_type('module')
    assert len(modules) == 2
    
    ops = mst.find_nodes_by_type('operation')
    assert len(ops) == 1
    
    layer1_nodes = mst.find_nodes_by_name('layer1')
    assert len(layer1_nodes) == 1
    assert layer1_nodes[0].node_type == 'module'
    
    print("✓ MST node queries test passed")

if __name__ == '__main__':
    test_mst_basic()
    test_mst_trace_mapping()
    test_mst_callstack_matching()
    test_mst_visualization()
    test_mst_serialization()
    test_mst_node_queries()
    print("\n✓ All MST tests passed!")
