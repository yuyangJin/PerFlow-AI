# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..trace import BaseTrace, Trace, CompressedTrace
from ..visitor import StreamMergedEventsIterator
from .types import is_compute_kernel


# This configures the threshold under which we consider gaps between
# kernels to be due to realistic delays in launching back-back kernels on the GPU


class BreakdownAnalysis:
    def __init__(self):
        pass

    @classmethod
    def get_gpu_kernel_breakdown(
        cls,
        t: "Trace",
        visualize: bool = True,
        duration_ratio: float = 0.8,
        num_kernels: int = 10,
        include_memory_kernels: bool = False,
        image_renderer="notebook",
    ):
        """
        GPU kernel breakdown implementation. See `get_gpu_kernel_breakdown` in `trace_analysis.py` for details.
        """
        pass

    @classmethod
    def _get_gpu_kernel_interval_dataframe(
        cls,
        trace_df: pd.DataFrame,
        t: "Trace",
    ):
        """Obtains all GPU kernels in the trace dataframe and assigns them
        an interval index that can be used for analyzing overlap.
            @args: trace_df (pd.DataFrame) : trace df for specific rank
                Please make sure this includes "end" column.
            @args: t (Trace) : trace object

        Returns: pd.DataFrame with GPU kernels subset with an interval index
                of [start, end) intervals.
        """
        pass

    @classmethod
    def _get_gpu_user_anno_interval_dataframe(
        cls,
        trace_df: pd.DataFrame,
        t: "Trace",
    ):
        """Obtains all GPU user annotations in the trace dataframe and assigns them
        an interval index that can be used for analyzing overlap.
            @args: trace_df (pd.DataFrame) : trace df for specific rank
                Please make sure this includes "end" column.
            @args: t (Trace) : trace object

        Returns: pd.DataFrame with GPU kernels subset with an interval index
                of [start, end) intervals.
                None if the trace does not have user annotations.
        """
        pass

    @classmethod
    def _associate_gpu_kernels_with_user_annotations(
        cls,
        gpu_kernels_df: pd.DataFrame,
        gpu_user_anno_df: pd.DataFrame,
    ) -> None:
        """Assigns each gpu_kernel user annotation. If the kernel overlaps with multiple
        user annotations, we will pick the lowest/leaf annotation in the stack to attribute to.
            @args: gpu_kernels_df (pd.DataFrame) : kernel df with interval index.
            @args: gpu_user_anno_df (pd.DataFrame) : gpu user annotation df with interval index.
        """
        # OPTIMIZATION: Pre-filter by pid/tid and use integer-based indexing
        # Original: O(n*m) with expensive .loc[] calls using triple boolean masks
        # Optimized: O(n*m) overlap detection but with direct integer indexing updates

        # Get the pid tid combinations to scan over
        pass

    @classmethod
    def get_gpu_kernels_with_user_annotations(
        cls,
        t: "Trace",
        rank: int,
        expand_names: bool = True,
        shortern_names: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Returns a dataframe of all GPU kernels and associates them to closest or leaf
        GPU user annotation. If the kernel overlaps with multiple user annotations,
        we will pick the lowest/leaf annotation in the stack to attribute to.
        Read more in get_gpu_kernels_with_user_annotations in hta/trace_analysis.py."""
        pass

    @classmethod
    def get_gpu_user_annotation_breakdown(
        cls,
        t: "Trace",
        use_gpu_annotation: bool = True,
        visualize: bool = True,
        duration_ratio: float = 0.8,
        num_kernels: int = 1000,
        allowlist_patterns: Optional[List[str]] = None,
        image_renderer: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Summarizes the time spent by each GPU user annotation. Outputs the following graphs:

        1. Pie charts showing the most time consuming user annotations for each rank.
        2. Bar graphs showing the average duration for the most time user annotations for each rank.

        Args:
            use_gpu_annotation (boolean): Use time on GPU for each user annotation, if false use the time on CPU instead. Default = True,
            visualize (boolean): Set to True to display the graphs. Default = True.
            duration_ratio (float): Floating point value between 0 and 1 specifying the ratio of time taken
                                    by top user annotations. Default = 0.8.
            num_kernels (int): Maximum number of user annotations to show. Default = 1000. Rest get grouped into "other".
            allowlist_patterns (list(str)): if user annotations match any of the patterns in this list, they will not be aggregated into "other" catgory. This argument is meant to keep some events as distinct in the aggregation. Supports strings as well as regular expressions.
            image_renderer (str): Set to ``notebook`` when using jupyter and ``jupyterlab`` when using jupyter-lab.
                To see all available options execute: ``import plotly; plotly.io.renderers`` in a python shell.

        Returns:
            Optional[pd.DataFrame]
                Returns a dataframe that shows the min, max, mean, standard deviation, total time taken by each
                user annotation on each rank. This dataframe will be summarized based on values of ``duration_ratio``
                and ``num_kernels``. If both ``duration_ratio`` and ``num_kernels`` are specified,
                ``num_kernels`` takes precedence.
                If user_annotations are not present on CPU or GPU (according to use_gpu_annotation flag), return None.
        """
        pass

    @classmethod
    def _get_gpu_kernel_type_time(
        cls, gpu_kernels: pd.DataFrame, kernel_type_to_analysis: List[str]
    ) -> pd.DataFrame:
        pass

    @classmethod
    def _aggr_gpu_kernel_time(
        cls,
        gpu_kernel_time: pd.DataFrame,
        num_kernels: int = 10,
        duration_ratio: float = 0.8,
        allowlist_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregates GPU kernel/events

            @args: gpu_kernel_time: flat dataframe of events to consider
            @args: num_kernels (int) : Max number of kernels to show in result. If the
                aggregate exceeds this the rest of the kernels are grouped into "other",
                by first sorting by duration in descending order.
            @args: duration_ratio (float) : a quantile threshold above which kernels are grouped
                into "other" category. For example, setting to 0.8 will result is all kernels
                past > p80 to be grouped together.
            @args: allowlist_names (list(str): if kernel names are in this list, they will not be aggregated into "other" catgory.
                This argument is meant to keep some kernel/events as distinct in the aggregation

        Returns:
            aggregated pd.DataFrame by "name" column with ["sum", "max", "min", "mean", "std"]
        """

        pass

    @classmethod
    def _get_idle_time_for_kernels(cls, kernels_df: pd.DataFrame) -> Tuple[int, int]:
        """
        Compute idle time for given set of GPU kernels :
          returns :
            idle time (us) = kernel time - merged execution time of all kernels
            kernel time (us) = defined as the time difference between end of the
                         last kernel and start of the first kernel.
            PS: we exclude the last profiler iteration while reading trace
            so total time is exclusive of that.
        """
        pass

    @classmethod
    def get_temporal_breakdown(cls, t: Union[Trace, CompressedTrace],
                               visualize: bool = True) -> pd.DataFrame:
        """
        Temporal breakdown implementation. See `get_temporal_breakdown` 
        in `trace_analysis.py` for details.
        """
        results: Dict[str, List[Union[str, int, float]]] = defaultdict(list)
        for rank in t.get_ranks():
            results["rank"].append(rank)
            visitor = StreamMergedEventsIterator(t, rank)
            total_time = 0.0
            idle_time = 0.0
            compute_time = 0.0

            start_time = -1
            end_time = 0

            last_kernel_end_time = -1
            last_compute_kernel_end_time = -1
            last_compute_kernel_start_time = -1

            for e, _ in visitor:
                ts = e["ts"]
                dur = e["dur"]
                cat = e["cat"]
                if cat == "gpu_user_annotation":
                    continue
                if start_time == -1:
                    start_time = ts
                end_time = max(end_time, ts + dur)
                if last_kernel_end_time == -1:
                    last_kernel_end_time = ts + dur
                else:
                    if ts > last_kernel_end_time:
                        idle_time += ts - last_kernel_end_time
                        last_kernel_end_time = ts + dur
                    else:
                        last_kernel_end_time = max(last_kernel_end_time, ts + dur)

                if is_compute_kernel(e["name"]):
                    if last_compute_kernel_end_time == -1:
                        last_compute_kernel_end_time = ts + dur
                        last_compute_kernel_start_time = ts
                    else:
                        if ts > last_compute_kernel_end_time:
                            compute_time += \
                                last_compute_kernel_end_time - last_compute_kernel_start_time
                            last_compute_kernel_start_time = ts
                            last_compute_kernel_end_time = ts + dur
                        else:
                            last_compute_kernel_end_time = max(
                                last_compute_kernel_end_time, ts + dur
                            )

            total_time = end_time - start_time
            results["kernel_time(us)"].append(float(total_time))
            results["idle_time(us)"].append(idle_time)
            results["compute_time(us)"].append(compute_time)
            results["non_compute_time(us)"].append(total_time - idle_time - compute_time)

        result_df = pd.DataFrame(results)
        result_df["idle_time"] = (
            result_df["idle_time(us)"] / result_df["kernel_time(us)"]
        )
        result_df["idle_time_pctg"] = round(100 * result_df["idle_time"], 2)
        result_df["compute_time"] = (
            result_df["compute_time(us)"] / result_df["kernel_time(us)"]
        )
        result_df["compute_time_pctg"] = round(100 * result_df["compute_time"], 2)
        result_df["non_compute_time"] = (
            result_df["non_compute_time(us)"] / result_df["kernel_time(us)"]
        )
        result_df["non_compute_time_pctg"] = round(
            100 * result_df["non_compute_time"], 2
        )

        if visualize:  # pragma: no cover
            fig = px.bar(
                result_df,
                x="rank",
                y=["idle_time", "compute_time", "non_compute_time"],
                title="Temporal breakdown across ranks",
                labels={
                    "rank": "Rank",
                },
            )
            fig.update_layout(
                yaxis_tickformat=".2%",
                yaxis_title="Percentage",
                legend_title="Time Breakdown",
            )
            fig.show()

        return result_df[
            [
                "rank",
                "idle_time(us)",
                "compute_time(us)",
                "non_compute_time(us)",
                "kernel_time(us)",
                "idle_time_pctg",
                "compute_time_pctg",
                "non_compute_time_pctg",
            ]
        ]

    @classmethod
    def _analyze_idle_time_for_stream(
        cls,
        stream: int,
        gpu_kernels: pd.DataFrame,
        consecutive_kernel_delay: int,
        show_idle_interval_stats=False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Analyze a specific CUDA stream for idle time breakdown on it.

        stream (int): CUDA stream to consider.
        gpu_kernels: a dataframe of GPU kernels in a rank.

        returns
        1) dataframe with idle time breakdown.
        1) optional dataframe showing idle interval statistics.
        """
        pass

    @classmethod
    def get_idle_time_breakdown(
        cls,
        t: "Trace",
        consecutive_kernel_delay: int,
        rank: int = 0,
        streams: Optional[List[int]] = None,
        visualize: bool = True,
        visualize_pctg: bool = True,
        show_idle_interval_stats=False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Breakdown Idle time by host wait, kernel wait and other categories. See full description in trace_analysis.py

        consecutive_kernel_delay (int): configures the threshold under which we consider gaps between
           kernels to be due to realistic delays in launching back-back kernels on the GPU. Time is in ns.
        rank (int): the rank to analyze
        streams (List[int]): list of streams to provide analysis for.
            Defaults to all streams.
        visualize (bool): show the visualization chart or not (default = True).
        visualize_pctg (bool): show relative percentage across streams (default = True).
        show_idle_interval_stats (bool): prints statistics of the idle intervals like the min, max
           and median of idle intervals between kernels on a CUDA stream, also broken down by
           the idleness category (default = False).
        """
        pass
