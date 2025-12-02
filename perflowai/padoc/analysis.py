"""TODO
"""

from typing import Tuple
import pandas as pd
from perflowai.padoc.trace import BaseTrace
from perflowai.padoc.hta.breakdown_analysis import BreakdownAnalysis
from perflowai.padoc.hta.communication_analysis import CommunicationAnalysis

class TraceAnalysis:
    """Trace analysis class.
    """
    def __init__(self, trace: BaseTrace):
        self.trace = trace

    def get_comm_comp_overlap(self, visualize: bool = True):
        r"""
        Compute the communication-computation overlap percentage for each rank.

        Args:
            visualize (bool): Set to True to display the graph. Default = True.

        Returns:
            pd.DataFrame
                A dataframe containing the communication computation overlap 
                percentage for each rank.
        """
        return CommunicationAnalysis.get_comm_comp_overlap(self.trace, visualize)

    def get_gpu_kernel_breakdown(
        self,
        visualize: bool = True,
        duration_ratio: float = 0.8,
        num_kernels: int = 10,
        include_memory_kernels: bool = True,
        image_renderer: str = "",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        r"""
        Summarizes the time spent by each kernel and by kernel type. Outputs the following graphs:

        1. Pie chart indicating the percentage of time taken by each kernel type.
        2. Pie charts showing the most time consuming kernels for each rank for each kernel type.
        3. Bar graphs showing the average duration for the most time consuming kernels for each rank 
           and each kernel type.

        Args:
            visualize (boolean): Set to True to display the graphs. Default = True.
            duration_ratio (float): Floating point value between 0 and 1 specifying the ratio of 
                time taken by top COMM/COMP/MEMORY kernels. Default = 0.8.
            num_kernels (int): Maximum number of COMM/COMP/MEMORY kernels to show. Default = 10.
            include_memory_kernels (bool): Whether to include MEMORY kernels in the analysis. 
                Default = True.
            image_renderer (str): Set to ``notebook`` when using jupyter and ``jupyterlab`` 
                when using jupyter-lab.
                To see all available options execute: ``import plotly; plotly.io.renderers`` 
                in a python shell.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]
                Returns two dataframes. The first dataframe shows the percentage of time spent 
                by kernel type. The second dataframe shows the min, max, mean, standard deviation, 
                total time taken by each kernel on each rank. This dataframe will be summarized 
                based on values of ``duration_ratio`` and ``num_kernels``. If both 
                ``duration_ratio`` and ``num_kernels`` are specified, ``num_kernels`` 
                takes precedence.
        """

        return BreakdownAnalysis.get_gpu_kernel_breakdown(
            self.trace,
            visualize,
            duration_ratio,
            num_kernels,
            include_memory_kernels,
            image_renderer,
        )

    def get_temporal_breakdown(self, visualize: bool = True):
        r"""
        Compute the idle time, compute time and non-compute time for each rank. Time is measured in
        nanoseconds (ns). non-compute time is defined as the total time the GPU is not executing a
        compute operation such as data transfers, copying to/from memory and communication 
        collectives. (In the strictest sense communication collectives do some compute but we 
        classify it as a communication operation).

        Args:
            visualize (bool): Set to True to display the graphs. Default = True.

        Returns:
            pd.DataFrame
                A dataframe containing the raw value and percentage of idle time, compute time and 
                non-compute time for each rank.
        """

        return BreakdownAnalysis.get_temporal_breakdown(self.trace, visualize)
