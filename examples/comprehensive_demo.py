#!/usr/bin/env python3
'''
Comprehensive demo showing all three PADoC enhancements:
1. Building MST from torch.fx symbolic trace
2. Visualizing MST with Graphviz (PDF format)
3. Mapping TorchProfiler JSON traces to MST by call stacks
'''

import torch
import torch.nn as nn
import torch.fx as fx
from perflowai.padoc import ModelStructureTree, MSTMapper
import json
import os

def demo_torchfx_mst():
    '''Demo: Build MST from torch.fx symbolic trace'''
    print("\n" + "="*70)
    print("DEMO 1: Building MST from torch.fx symbolic trace")
    print("="*70)
    
    # Define a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Create and trace model
    model = SimpleNN()
    print("\n1. Created SimpleNN model")
    
    traced = fx.symbolic_trace(model)
    print("2. Traced model with torch.fx.symbolic_trace()")
    
    # Build MST from torch.fx graph
    mst = ModelStructureTree.from_torch_fx_graph(traced.graph, 'SimpleNN')
    print(f"3. Built MST with {len(mst.nodes)} nodes")
    
    # Display structure
    print("\n4. MST Structure:")
    print(mst.visualize())
    
    return mst


def demo_graphviz_visualization(mst):
    '''Demo: Visualize MST with Graphviz'''
    print("\n" + "="*70)
    print("DEMO 2: Visualizing MST with Graphviz (PDF, PNG, SVG)")
    print("="*70)
    
    output_dir = '/home/runner/work/PerFlow-AI/PerFlow-AI'
    
    # Generate PDF
    pdf_path = os.path.join(output_dir, 'demo_mst')
    pdf_file = mst.visualize_graphviz(pdf_path, format='pdf')
    print(f"\n1. Generated PDF: {pdf_file}")
    print("   - Small circles represent nodes")
    print("   - Different colors for different node types")
    print("   - Tree layout (top to bottom)")
    
    # Generate PNG
    png_path = os.path.join(output_dir, 'demo_mst_png')
    png_file = mst.visualize_graphviz(png_path, format='png')
    print(f"\n2. Generated PNG: {png_file}")
    
    # Generate SVG
    svg_path = os.path.join(output_dir, 'demo_mst_svg')
    svg_file = mst.visualize_graphviz(svg_path, format='svg')
    print(f"\n3. Generated SVG: {svg_file}")
    
    # Show color scheme
    print("\n4. Color Scheme:")
    print("   - Root: Gray")
    print("   - Model: Blue")
    print("   - Module: Green")
    print("   - Function: Purple")
    print("   - Input: Yellow")
    print("   - Output: Pink")
    
    return pdf_file


def demo_torchprofiler_mapping(mst):
    '''Demo: Map TorchProfiler JSON traces to MST by call stacks'''
    print("\n" + "="*70)
    print("DEMO 3: Mapping TorchProfiler JSON traces to MST by call stacks")
    print("="*70)
    
    # Create a mock TorchProfiler trace
    mock_trace = {
        'traceEvents': [
            {
                'name': 'forward',
                'ts': 1000,
                'dur': 100,
                'tid': 0,
                'args': {
                    'Python call stack': 'SimpleNN\nfc1'
                }
            },
            {
                'name': 'relu',
                'ts': 1100,
                'dur': 50,
                'tid': 0,
                'args': {
                    'Python call stack': 'SimpleNN\nrelu'
                }
            },
            {
                'name': 'dropout',
                'ts': 1150,
                'dur': 30,
                'tid': 0,
                'args': {
                    'Python call stack': 'SimpleNN\ndropout'
                }
            },
            {
                'name': 'forward',
                'ts': 1180,
                'dur': 80,
                'tid': 0,
                'args': {
                    'Python call stack': 'SimpleNN\nfc2'
                }
            }
        ]
    }
    
    print("\n1. Created mock TorchProfiler trace with 4 events")
    
    # Map events to MST by call stack
    print("\n2. Mapping events to MST nodes by call stack:")
    
    event_id = 0
    for event in mock_trace['traceEvents']:
        if 'args' in event and 'Python call stack' in event['args']:
            stack_str = event['args']['Python call stack']
            call_stack = [line.strip() for line in stack_str.split('\n') if line.strip()]
            
            # Try to map by call stack
            matched_node = mst.map_trace_events_by_callstack(event_id, call_stack)
            
            if matched_node:
                print(f"   Event '{event['name']}' (ts={event['ts']}, dur={event['dur']}) → {matched_node.name}")
            else:
                print(f"   Event '{event['name']}' → No match (creating mapping to parent)")
        
        event_id += 1
    
    # Show MST with mapped events
    print("\n3. MST with mapped trace events:")
    print(mst.visualize())
    
    # Count total mapped events
    total_events = sum(len(node.trace_events) for node in mst.nodes.values())
    print(f"\n4. Total events mapped: {total_events}")
    
    return mst


def demo_full_integration():
    '''Demo: Full integration of all three features'''
    print("\n" + "="*70)
    print("DEMO 4: Full Integration - torch.fx → MST → Graphviz → Mapping")
    print("="*70)
    
    # Step 1: Build MST from torch.fx
    mst = demo_torchfx_mst()
    
    # Step 2: Visualize with Graphviz
    pdf_file = demo_graphviz_visualization(mst)
    
    # Step 3: Map TorchProfiler traces
    mst = demo_torchprofiler_mapping(mst)
    
    # Step 4: Generate final visualization with trace info
    print("\n" + "="*70)
    print("DEMO 4: Final Visualization with Trace Event Info")
    print("="*70)
    
    output_dir = '/home/runner/work/PerFlow-AI/PerFlow-AI'
    final_path = os.path.join(output_dir, 'demo_final_mst')
    final_file = mst.visualize_graphviz(final_path, format='pdf')
    
    print(f"\n1. Generated final visualization: {final_file}")
    print("   - Includes all trace event mappings")
    print("   - Shows event counts for each node")
    
    print("\n2. Summary:")
    print(f"   - Total MST nodes: {len(mst.nodes)}")
    total_events = sum(len(node.trace_events) for node in mst.nodes.values())
    print(f"   - Total mapped events: {total_events}")
    nodes_with_events = sum(1 for node in mst.nodes.values() if node.trace_events)
    print(f"   - Nodes with events: {nodes_with_events}")
    
    return final_file


def main():
    '''Main demo runner'''
    print("\n" + "="*70)
    print("PADoC Enhancement Demo: torch.fx + Graphviz + TorchProfiler Mapping")
    print("="*70)
    print("\nThis demo shows three new features:")
    print("1. Building MST from torch.fx symbolic trace")
    print("2. Visualizing MST with Graphviz (PDF format with colored circles)")
    print("3. Mapping TorchProfiler JSON traces to MST by call stacks")
    
    # Run full integration demo
    final_file = demo_full_integration()
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - demo_mst.pdf (Initial MST visualization)")
    print("  - demo_mst_png.png (PNG version)")
    print("  - demo_mst_svg.svg (SVG version)")
    print(f"  - {final_file} (Final visualization with trace events)")
    print("\nTry running the demo yourself:")
    print("  python examples/comprehensive_demo.py")
    print("\n")


if __name__ == '__main__':
    main()
