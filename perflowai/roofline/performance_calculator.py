import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

class PerformanceCalculator:
    """
    A class for calculating MFU (Model FLOPs Utilization) and Roofline performance metrics.
    Supports automatic device detection and custom device specifications.
    """
    
    def __init__(self, device_type: Optional[str] = None, device_model: Optional[str] = None):
        """
        Initialize the PerformanceCalculator.
        
        Args:
            device_type: Device type ('cuda', 'cpu', etc.). If None, auto-detects.
            device_model: Specific device model. If None, auto-detects or uses default.
        """
        self.device_info = {}
    
    def get_peak_flops(self, precision: Optional[str] = None) -> float:
        """
        Get peak FLOPS for the current device and precision.

        Args:
            precision: Precision type (e.g., 'FP32', 'FP16', 'INT8').
                    If None, fallback to class precision or FP32.
        
        Returns:
            Peak FLOPS in operations per second
        """
        specs = self.device_info.get('specs', {})
        compute_specs = specs.get('compute', {})

        precision = (precision or getattr(self, "precision", "FP32")).upper()

        # 优先查 TFLOPS
        if f"{precision}_TFLOPS" in compute_specs:
            return compute_specs[f"{precision}_TFLOPS"] * 1e12
        # 其次查 TOPS
        elif f"{precision}_TOPS" in compute_specs:
            return compute_specs[f"{precision}_TOPS"] * 1e12
        # fallback 到 FP32
        elif "FP32_TFLOPS" in compute_specs:
            return compute_specs["FP32_TFLOPS"] * 1e12
        else:
            # 没有找到任何信息，返回 0
            return 0.0


    def get_memory_bandwidth(self) -> float:
        """
        Get memory bandwidth for the current device.

        Returns:
            Memory bandwidth in bytes per second
        """
        specs = self.device_info.get('specs', {})
        bw_TBps = specs.get('memory_bandwidth_TBps', None)
        if bw_TBps is not None:
            return bw_TBps * 1e12
        return 0.0
    
    def get_critical_arithmetic_intensity(self) -> float:
        """
        Calculate the critical arithmetic intensity for roofline analysis.
        
        Returns:
            Critical arithmetic intensity in FLOPs/byte
        """
        peak_flops = self.get_peak_flops()
        memory_bandwidth = self.get_memory_bandwidth()
        return peak_flops / memory_bandwidth if memory_bandwidth > 0 else float('inf')
    
    def calculate_node_performance(self, node: Any):
        """
        Calculate MFU and Roofline metrics for a node and store them in the node.
        
        Args:
            node: TreeNode object or dictionary with flops, bytes, and elapsed attributes
        """
        if not hasattr(node, 'flops') or not hasattr(node, 'bytes') or not hasattr(node, 'elapsed'):
            return
        
        if node.flops is None or node.bytes is None or node.elapsed is None or node.elapsed <= 0:
            return
        
        # Calculate achieved performance
        achieved_flops = node.flops / node.elapsed
        achieved_memory_bw = node.bytes / node.elapsed
        
        # Get device peaks
        peak_flops = self.get_peak_flops()
        peak_memory_bw = self.get_memory_bandwidth()
        
        # Calculate MFU (Model FLOPs Utilization)
        node.mfu = (achieved_flops / peak_flops) * 100 if peak_flops > 0 else 0
        
        # Calculate arithmetic intensity
        arithmetic_intensity = node.flops / node.bytes if node.bytes > 0 else float('inf')
        critical_ai = self.get_critical_arithmetic_intensity()
        
        # Determine roofline bound and performance percentage
        if arithmetic_intensity > critical_ai:
            node.roofline_bound = 'compute_bound'
            node.roofline_percentage = (achieved_flops / peak_flops) * 100 if peak_flops > 0 else 0
        else:
            node.roofline_bound = 'memory_bound'
            theoretical_flops = peak_memory_bw * arithmetic_intensity
            node.roofline_percentage = (achieved_flops / theoretical_flops) * 100 if theoretical_flops > 0 else 0
        
        # Store additional metrics
        node.arithmetic_intensity = arithmetic_intensity
        node.achieved_flops = achieved_flops
        node.achieved_memory_bw = achieved_memory_bw
    
    def calculate_performance_metrics(self, flops: float, memory_access: float, 
                                    elapsed_time: float) -> Dict[str, float]:
        """
        Calculate performance metrics for given operations.
        
        Args:
            flops: Total FLOPs performed
            memory_access: Total memory access in bytes
            elapsed_time: Time taken in seconds
            
        Returns:
            Dictionary containing all performance metrics
        """
        if elapsed_time <= 0:
            return {}
        
        achieved_flops = flops / elapsed_time
        achieved_memory_bw = memory_access / elapsed_time
        
        peak_flops = self.get_peak_flops()
        peak_memory_bw = self.get_memory_bandwidth()
        critical_ai = self.get_critical_arithmetic_intensity()
        arithmetic_intensity = flops / memory_access if memory_access > 0 else float('inf')
        
        mfu = (achieved_flops / peak_flops) * 100 if peak_flops > 0 else 0
        
        if arithmetic_intensity > critical_ai:
            roofline_bound = 'compute_bound'
            roofline_percentage = (achieved_flops / peak_flops) * 100 if peak_flops > 0 else 0
        else:
            roofline_bound = 'memory_bound'
            theoretical_flops = peak_memory_bw * arithmetic_intensity
            roofline_percentage = (achieved_flops / theoretical_flops) * 100 if theoretical_flops > 0 else 0
        
        return {
            'mfu_percentage': mfu,
            'roofline_bound': roofline_bound,
            'roofline_percentage': roofline_percentage,
            'arithmetic_intensity': arithmetic_intensity,
            'achieved_flops': achieved_flops,
            'achieved_memory_bw': achieved_memory_bw,
            'peak_flops': peak_flops,
            'peak_memory_bw': peak_memory_bw,
            'critical_ai': critical_ai
        }

    def plot_roofline(self, nodes: List[Any], device_info: Dict, 
                    save_path: Optional[str] = None, show_plot: bool = True):
        """
        Plot Roofline performance analysis.
        
        Args:
            nodes: List of node objects or dictionaries
            device: Device information dictionary. If None, uses current device settings
            save_path: Path to save the plot image. If None, doesn't save
            show_plot: Whether to display the plot
        
        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        self.device_info = device_info
        
        # Group nodes by operation type (same name, input_info, output_info)
        node_groups = self._group_nodes(nodes)
        print(f"Found {len(node_groups)} operation types")
        
        # Prepare plot data
        arithmetic_intensities = []
        achieved_flops = []
        labels = []
        colors = []
        
        # Assign colors to different operation types
        color_map = self._get_operation_color_map()
        
        for group_key, group_nodes in node_groups.items():
            # Calculate average performance metrics
            avg_flops = np.mean([n.flops for n in group_nodes if n.flops is not None])
            avg_memory_access = np.mean([n.bytes for n in group_nodes if n.bytes is not None])
            avg_elapsed = np.mean([n.elapsed for n in group_nodes if n.elapsed is not None])
            
            if avg_flops <= 0 or avg_memory_access <= 0 or avg_elapsed <= 0:
                continue
            
            # Calculate arithmetic intensity and achieved performance
            ai = avg_flops / avg_memory_access
            performance = avg_flops / avg_elapsed
            
            arithmetic_intensities.append(ai)
            achieved_flops.append(performance)
            
            # Get operation type and create label
            op_type = group_key[0]  # name
            labels.append(f"{op_type}\n({len(group_nodes)} runs)")
            
            # Assign color
            colors.append(color_map.get(op_type, 'gray'))
        
        if not arithmetic_intensities:
            print("No valid data to plot")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot theoretical roofline curve
        self._plot_roofline_curve()
        
        # Plot data points for each operation
        scatter = plt.scatter(arithmetic_intensities, achieved_flops, 
                            c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Add annotations
        for i, (ai, flops, label) in enumerate(zip(arithmetic_intensities, achieved_flops, labels)):
            plt.annotate(label, (ai, flops), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Configure axes
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
        plt.ylabel('Performance (FLOPs/s)')
        
        # Get device specs for title
        peak_flops = self.get_peak_flops()
        memory_bw = self.get_memory_bandwidth()
        
        plt.title(f'Roofline Model - {self.device_info['type'].upper()} {self.device_info["model"]}\n'
                f'Compute Peak: {peak_flops/1e12:.1f} TFLOPS, '
                f'Memory BW: {memory_bw/1e9:.1f} GB/s')
        
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        
        # Add color legend
        self._add_color_legend(color_map)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Roofline plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return plt.gcf()

    def _group_nodes(self, nodes: List[Any]) -> Dict[Tuple, List[Any]]:
        """
        Group nodes by operation type.
        
        Args:
            nodes: List of node objects or dictionaries
            
        Returns:
            Dictionary grouping nodes by (name, input_info, output_info) tuple
        """
        groups = defaultdict(list)
        
        for node in nodes:
            if isinstance(node, dict):
                name = node.get('name', 'Unknown')
                input_info = str(node.get('input_info', []))
                output_info = str(node.get('output_info', {}))
                flops = node.get('flops', 0)
                memory_access = node.get('bytes', 0)
                elapsed = node.get('elapsed', 0)
            else:
                name = getattr(node, 'name', 'Unknown')
                input_info = str(getattr(node, 'input_info', []))
                output_info = str(getattr(node, 'output_info', {}))
                flops = getattr(node, 'flops', 0)
                memory_access = getattr(node, 'bytes', 0)
                elapsed = getattr(node, 'elapsed', 0)
            
            # Skip invalid data
            if flops <= 0 or memory_access <= 0 or elapsed <= 0:
                continue
            
            # Create grouping key
            group_key = (name, input_info, output_info)
            groups[group_key].append(node)
        
        return groups

    def _plot_roofline_curve(self):
        """
        Plot the theoretical roofline curve.
        """
        # Calculate critical arithmetic intensity
        critical_ai = self.get_critical_arithmetic_intensity()
        peak_flops = self.get_peak_flops()
        memory_bw = self.get_memory_bandwidth()
        
        # Generate curve data points
        ai_values = np.logspace(-3, 3, 1000)  # From 10^-3 to 10^3
        performance_values = np.zeros_like(ai_values)
        
        # Calculate theoretical performance
        for i, ai in enumerate(ai_values):
            if ai <= critical_ai:
                # Memory-bound region
                performance_values[i] = memory_bw * ai
            else:
                # Compute-bound region
                performance_values[i] = peak_flops
        
        # Plot theoretical curve
        plt.plot(ai_values, performance_values, 'r-', linewidth=2, 
                label='Theoretical Roofline')
        
        # Mark critical point
        plt.axvline(x=critical_ai, color='green', linestyle='--', alpha=0.7,
                label=f'Critical AI: {critical_ai:.2f} FLOPs/Byte')
        
        line_legend = plt.legend(loc="upper left")
        plt.gca().add_artist(line_legend)

        # Add region labels
        plt.text(critical_ai/10, peak_flops/10, 'Memory Bound', 
                fontsize=12, ha='center', va='center', rotation=45,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.text(critical_ai*10, peak_flops*0.9, 'Compute Bound', 
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    def _get_operation_color_map(self) -> Dict[str, str]:
        """
        Get color mapping for different operation types.
        
        Returns:
            Dictionary mapping operation types to colors
        """
        return {
            'Linear': 'blue',
            'Conv': 'red',
            'Attention': 'green',
            'Norm': 'purple',
            'ReLU': 'orange',
            'Pool': 'brown',
            'SiluAndMul': 'pink',
            'RotaryEmbedding': 'cyan',
            'VocabParallelEmbedding': 'magenta',
            'FusedMoE': 'gold',
            'Unknown': 'gray'
        }

    def _add_color_legend(self, color_map: Dict[str, str]):
        """
        Add color legend to the plot.
        
        Args:
            color_map: Dictionary mapping operation types to colors
        """
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=op_type, alpha=0.7)
            for op_type, color in color_map.items()
        ]
        plt.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(1.0, 0))