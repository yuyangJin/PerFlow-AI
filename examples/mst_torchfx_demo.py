'''
Demo script for torch.fx MST building and Graphviz visualization
'''

import torch
import torch.nn as nn
import torch.fx as fx
from perflowai.padoc import ModelStructureTree
import os

class DemoTransformerLayer(nn.Module):
    '''A more complex model to demonstrate MST features'''
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed forward
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = self.norm2(x + ff_output)
        
        return x


def main():
    print("="*70)
    print("PADoC: MST Building from torch.fx and Graphviz Visualization Demo")
    print("="*70)
    
    # Create model
    print("\n1. Creating a Transformer layer model...")
    model = DemoTransformerLayer()
    print(f"   Model created: {model.__class__.__name__}")
    
    # Trace model with torch.fx
    print("\n2. Tracing model with torch.fx.symbolic_trace...")
    try:
        traced_model = fx.symbolic_trace(model)
        print("   ✓ Model traced successfully")
    except Exception as e:
        print(f"   Note: torch.fx cannot trace MultiheadAttention directly")
        print(f"   Using a simpler model instead...")
        
        # Use a simpler model that can be traced
        class SimpleFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(512, 2048)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
                self.linear2 = nn.Linear(2048, 512)
                self.norm = nn.LayerNorm(512)
                
            def forward(self, x):
                residual = x
                x = self.linear1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.linear2(x)
                x = self.norm(x + residual)
                return x
        
        model = SimpleFFN()
        traced_model = fx.symbolic_trace(model)
        print("   ✓ Simplified model traced successfully")
    
    # Build MST from torch.fx graph
    print("\n3. Building Model Structure Tree from torch.fx graph...")
    mst = ModelStructureTree.from_torch_fx_graph(traced_model.graph, 'TransformerFFN')
    print(f"   ✓ MST built with {len(mst.nodes)} nodes")
    
    # Print text visualization
    print("\n4. Text visualization of MST:")
    print("-" * 70)
    print(mst.visualize())
    print("-" * 70)
    
    # Generate Graphviz visualizations
    print("\n5. Generating Graphviz visualizations...")
    
    output_dir = '/home/runner/work/PerFlow-AI/PerFlow-AI'
    
    # PDF format
    pdf_path = os.path.join(output_dir, 'mst_torchfx_demo')
    pdf_file = mst.visualize_graphviz(pdf_path, format='pdf')
    print(f"   ✓ PDF visualization: {pdf_file}")
    
    # PNG format
    png_path = os.path.join(output_dir, 'mst_torchfx_demo_png')
    png_file = mst.visualize_graphviz(png_path, format='png')
    print(f"   ✓ PNG visualization: {png_file}")
    
    # SVG format
    svg_path = os.path.join(output_dir, 'mst_torchfx_demo_svg')
    svg_file = mst.visualize_graphviz(svg_path, format='svg')
    print(f"   ✓ SVG visualization: {svg_file}")
    
    # Demonstrate trace event mapping
    print("\n6. Simulating trace event mapping...")
    linear_nodes = [n for n in mst.nodes.values() if 'linear' in n.name.lower()]
    for i, node in enumerate(linear_nodes[:2]):
        node.add_trace_event(100 + i)
        node.add_trace_event(200 + i)
        print(f"   Mapped events to {node.name}: {node.trace_events}")
    
    # Generate final visualization with trace info
    print("\n7. Generating final visualization with trace event info...")
    final_path = os.path.join(output_dir, 'mst_torchfx_final')
    final_file = mst.visualize_graphviz(final_path, format='pdf')
    print(f"   ✓ Final visualization: {final_file}")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {pdf_file}")
    print(f"  - {png_file}")
    print(f"  - {svg_file}")
    print(f"  - {final_file}")


if __name__ == '__main__':
    main()
