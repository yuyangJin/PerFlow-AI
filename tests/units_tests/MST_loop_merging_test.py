#!/usr/bin/env python3
'''
Unit tests for MST loop merging feature
'''

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import torch
    import torch.nn as nn
    import torch.fx as fx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from perflowai.padoc import ModelStructureTree


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestMSTLoopMerging(unittest.TestCase):
    '''Test cases for MST loop merging functionality'''
    
    def test_loop_merging_basic(self):
        '''Test that repeated layers are merged into loop nodes'''
        
        # Create a simple model with repeated layers
        class SimpleLayer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                return self.relu(self.linear(x))
        
        class ModelWithLayers(nn.Module):
            def __init__(self, num_layers=4):
                super().__init__()
                self.layers = nn.ModuleList([SimpleLayer(64) for _ in range(num_layers)])
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        model = ModelWithLayers(num_layers=4)
        traced = fx.symbolic_trace(model)
        
        # Build MST with loop merging enabled
        mst = ModelStructureTree.from_torch_fx_graph(
            traced.graph, 
            'ModelWithLayers',
            merge_similar_layers=True
        )
        
        # Check that loop nodes were created
        loop_nodes = [node for node in mst.nodes.values() if node.node_type == 'loop']
        self.assertGreater(len(loop_nodes), 0, "Loop nodes should be created for repeated layers")
        
        # Check loop node attributes
        for loop_node in loop_nodes:
            self.assertIn('loop_count', loop_node.attributes, "Loop node should have loop_count attribute")
            self.assertGreater(loop_node.attributes['loop_count'], 1, "Loop count should be > 1")
    
    def test_loop_merging_disabled(self):
        '''Test that loop merging can be disabled'''
        
        class SimpleLayer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)
                
            def forward(self, x):
                return self.linear(x)
        
        class ModelWithLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([SimpleLayer(64) for _ in range(3)])
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        model = ModelWithLayers()
        traced = fx.symbolic_trace(model)
        
        # Build MST with loop merging disabled
        mst = ModelStructureTree.from_torch_fx_graph(
            traced.graph, 
            'ModelWithLayers',
            merge_similar_layers=False
        )
        
        # Check that no loop nodes were created
        loop_nodes = [node for node in mst.nodes.values() if node.node_type == 'loop']
        self.assertEqual(len(loop_nodes), 0, "No loop nodes should be created when merging is disabled")
    
    def test_loop_detection_consecutive(self):
        '''Test that only consecutive numeric indices are merged'''
        
        # Test the _detect_loop_patterns function directly
        module_list = [
            'layers.0.linear',
            'layers.1.linear',
            'layers.2.linear',
            'layers.3.linear',
            'other.5.linear',  # Not consecutive
        ]
        
        loop_groups = ModelStructureTree._detect_loop_patterns(module_list)
        
        # Should detect 'layers' as a loop (0,1,2,3 are consecutive)
        self.assertGreater(len(loop_groups), 0, "Should detect at least one loop pattern")
        
        # Find the 'layers' loop
        layers_loop = None
        for prefix, info in loop_groups.items():
            if 'layers' in prefix:
                layers_loop = info
                break
        
        self.assertIsNotNone(layers_loop, "Should detect 'layers' as a loop pattern")
        self.assertEqual(layers_loop['count'], 4, "Should have 4 iterations")
    
    def test_loop_visualization(self):
        '''Test that loop nodes are properly visualized'''
        
        class SimpleLayer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)
                
            def forward(self, x):
                return self.linear(x)
        
        class ModelWithLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([SimpleLayer(64) for _ in range(3)])
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        model = ModelWithLayers()
        traced = fx.symbolic_trace(model)
        
        mst = ModelStructureTree.from_torch_fx_graph(
            traced.graph, 
            'ModelWithLayers',
            merge_similar_layers=True
        )
        
        # Get text visualization
        text_viz = mst.visualize()
        
        # Should contain loop information
        self.assertIn('loop', text_viz.lower(), "Visualization should show loop nodes")
        
        # Check that loop nodes have proper formatting
        loop_nodes = [node for node in mst.nodes.values() if node.node_type == 'loop']
        for loop_node in loop_nodes:
            self.assertIn('_loop', loop_node.name, "Loop nodes should have '_loop' suffix")


if __name__ == '__main__':
    unittest.main()
