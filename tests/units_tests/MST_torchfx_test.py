'''
Test MST with torch.fx symbolic tracing, Graphviz visualization, and TorchProfiler mapping
'''

import torch
import torch.nn as nn
import torch.fx as fx
from perflowai.padoc import ModelStructureTree
import os
import tempfile
import json

class SimpleModel(nn.Module):
    '''A simple model for testing torch.fx tracing'''
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def test_mst_from_torch_fx():
    '''Test building MST from torch.fx symbolic trace'''
    print("\n=== Test 1: Building MST from torch.fx graph ===")
    
    # Create and trace model
    model = SimpleModel()
    traced_model = fx.symbolic_trace(model)
    
    # Build MST from the traced graph
    mst = ModelStructureTree.from_torch_fx_graph(traced_model.graph, 'SimpleModel')
    
    # Verify the structure
    assert len(mst.nodes) > 1, "MST should have more than just root node"
    
    # Check that model node exists
    model_nodes = mst.find_nodes_by_type('model')
    assert len(model_nodes) == 1, "Should have exactly one model node"
    assert model_nodes[0].name == 'SimpleModel'
    
    # Check for input nodes
    input_nodes = mst.find_nodes_by_type('input')
    assert len(input_nodes) >= 1, "Should have at least one input node"
    
    # Check for module nodes (linear layers)
    module_nodes = mst.find_nodes_by_type('module')
    print(f"  Found {len(module_nodes)} module nodes")
    
    # Check for function/method nodes
    function_nodes = mst.find_nodes_by_type('function')
    method_nodes = mst.find_nodes_by_type('method')
    print(f"  Found {len(function_nodes)} function nodes and {len(method_nodes)} method nodes")
    
    # Check for output nodes
    output_nodes = mst.find_nodes_by_type('output')
    assert len(output_nodes) >= 1, "Should have at least one output node"
    
    # Verify call stacks are set
    for node in mst.nodes.values():
        if node.node_id != 0 and node.node_type != 'root':  # Skip root
            assert len(node.call_stack) > 0, f"Node {node.name} (type: {node.node_type}) should have a call stack"
    
    # Print visualization
    print("\n  MST Structure:")
    viz = mst.visualize()
    print(viz)
    
    print("✓ torch.fx MST building test passed")
    return mst


def test_mst_graphviz_visualization():
    '''Test MST visualization with Graphviz (PDF format)'''
    print("\n=== Test 2: Graphviz PDF Visualization ===")
    
    # Create a sample MST
    mst = ModelStructureTree()
    
    # Add some nodes with different types
    model_node = mst.add_node('TransformerLayer', 'model', 0)
    
    attention_node = mst.add_node('Attention', 'module', model_node.node_id)
    attention_node.set_call_stack(['model', 'attention'])
    
    qkv_node = mst.add_node('qkv_projection', 'operation', attention_node.node_id)
    qkv_node.add_trace_event(1)
    qkv_node.add_trace_event(2)
    
    ffn_node = mst.add_node('FeedForward', 'module', model_node.node_id)
    ffn_node.set_call_stack(['model', 'ffn'])
    
    linear1_node = mst.add_node('linear1', 'operation', ffn_node.node_id)
    linear1_node.add_trace_event(3)
    
    linear2_node = mst.add_node('linear2', 'operation', ffn_node.node_id)
    linear2_node.add_trace_event(4)
    
    # Generate visualization
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'mst_test')
        
        # Test PDF output
        pdf_file = mst.visualize_graphviz(output_path, format='pdf')
        assert os.path.exists(pdf_file), "PDF file should be created"
        print(f"  ✓ PDF visualization created: {pdf_file}")
        
        # Test PNG output
        png_path = os.path.join(tmpdir, 'mst_test_png')
        png_file = mst.visualize_graphviz(png_path, format='png')
        assert os.path.exists(png_file), "PNG file should be created"
        print(f"  ✓ PNG visualization created: {png_file}")
    
    print("✓ Graphviz visualization test passed")


def test_torchfx_with_graphviz():
    '''Test combining torch.fx MST building with Graphviz visualization'''
    print("\n=== Test 3: torch.fx + Graphviz Visualization ===")
    
    # Create and trace model
    model = SimpleModel()
    traced_model = fx.symbolic_trace(model)
    
    # Build MST from torch.fx
    mst = ModelStructureTree.from_torch_fx_graph(traced_model.graph, 'SimpleModel')
    
    # Visualize with Graphviz
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'torchfx_mst')
        
        pdf_file = mst.visualize_graphviz(output_path, format='pdf')
        assert os.path.exists(pdf_file), "PDF file should be created"
        print(f"  ✓ torch.fx MST visualized as PDF: {pdf_file}")
        
        # Also save as SVG for easy viewing
        svg_path = os.path.join(tmpdir, 'torchfx_mst_svg')
        svg_file = mst.visualize_graphviz(svg_path, format='svg')
        print(f"  ✓ torch.fx MST visualized as SVG: {svg_file}")
    
    print("✓ torch.fx + Graphviz test passed")


def test_torchprofiler_mapping_with_callstack():
    '''Test mapping TorchProfiler JSON traces to MST by call stacks'''
    print("\n=== Test 4: TorchProfiler JSON Mapping to MST ===")
    
    # Create a mock TorchProfiler trace JSON
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
                'args': {
                    'Python call stack': 'model\nlayer1\nbackward'
                }
            },
            {
                'name': 'matmul',
                'ts': 1250,
                'dur': 50,
                'tid': 0,
                'args': {
                    'Python call stack': 'model\nlayer1\nforward\nmatmul'
                }
            }
        ],
        'distributedInfo': {
            'world_size': 1
        }
    }
    
    # Create MST
    mst = ModelStructureTree()
    model_node = mst.add_node('model', 'model', 0)
    
    layer1_node = mst.add_node('layer1', 'layer', model_node.node_id)
    layer1_node.set_call_stack(['model', 'layer1'])
    
    forward_node = mst.add_node('forward', 'operation', layer1_node.node_id)
    forward_node.set_call_stack(['model', 'layer1', 'forward'])
    
    backward_node = mst.add_node('backward', 'operation', layer1_node.node_id)
    backward_node.set_call_stack(['model', 'layer1', 'backward'])
    
    # Map trace events to MST by call stack
    from perflowai.padoc.mst_mapper import MSTMapper
    
    event_id = 0
    for event in mock_trace['traceEvents']:
        if 'args' in event and 'Python call stack' in event['args']:
            stack_str = event['args']['Python call stack']
            call_stack = [line.strip() for line in stack_str.split('\n') if line.strip()]
            
            matched_node = mst.map_trace_events_by_callstack(event_id, call_stack)
            
            if matched_node:
                print(f"  Event '{event['name']}' (ID {event_id}) mapped to node '{matched_node.name}'")
            else:
                print(f"  Event '{event['name']}' (ID {event_id}) not mapped (no matching call stack)")
        
        event_id += 1
    
    # Verify mappings
    assert len(forward_node.trace_events) >= 1, "Forward node should have mapped events"
    assert len(backward_node.trace_events) >= 1, "Backward node should have mapped events"
    
    print(f"\n  Total events mapped: {sum(len(node.trace_events) for node in mst.nodes.values())}")
    
    # Visualize the mapped MST
    print("\n  MST with mapped events:")
    print(mst.visualize())
    
    print("✓ TorchProfiler mapping with call stack test passed")


def test_full_workflow():
    '''Test the complete workflow: torch.fx -> MST -> Graphviz -> TorchProfiler mapping'''
    print("\n=== Test 5: Full Workflow Integration ===")
    
    # Step 1: Build MST from torch.fx
    model = SimpleModel()
    traced_model = fx.symbolic_trace(model)
    mst = ModelStructureTree.from_torch_fx_graph(traced_model.graph, 'SimpleModel')
    print("  Step 1: ✓ Built MST from torch.fx")
    
    # Step 2: Visualize with Graphviz
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'full_workflow_mst')
        pdf_file = mst.visualize_graphviz(output_path, format='pdf')
        print(f"  Step 2: ✓ Generated visualization: {pdf_file}")
        
        # Step 3: Simulate mapping trace events
        # In a real scenario, these would come from TorchProfiler
        linear1_nodes = [n for n in mst.nodes.values() if 'linear1' in n.name.lower()]
        if linear1_nodes:
            linear1_nodes[0].add_trace_event(100)
            linear1_nodes[0].add_trace_event(101)
        
        relu_nodes = [n for n in mst.nodes.values() if 'relu' in n.name.lower()]
        if relu_nodes:
            relu_nodes[0].add_trace_event(102)
        
        print(f"  Step 3: ✓ Mapped trace events to MST nodes")
        
        # Step 4: Generate final visualization with trace info
        final_path = os.path.join(tmpdir, 'full_workflow_final')
        final_file = mst.visualize_graphviz(final_path, format='pdf')
        print(f"  Step 4: ✓ Generated final visualization: {final_file}")
    
    print("\n  Final MST structure:")
    print(mst.visualize())
    
    print("✓ Full workflow integration test passed")


if __name__ == '__main__':
    print("="*70)
    print("Testing MST with torch.fx, Graphviz, and TorchProfiler Integration")
    print("="*70)
    
    test_mst_from_torch_fx()
    test_mst_graphviz_visualization()
    test_torchfx_with_graphviz()
    test_torchprofiler_mapping_with_callstack()
    test_full_workflow()
    
    print("\n" + "="*70)
    print("✓ All MST torch.fx and Graphviz tests passed!")
    print("="*70)
