"""TODO
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
import itertools

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from perflowai.padoc.trace import BaseTrace
from perflowai.padoc.visitor import StreamMergedEventsIterator
from perflowai.padoc.hta.types import KernelType, is_compute_kernel, get_kernel_type
from perflowai.padoc.hta.analysis_helper import AnalysisHelper


# This configures the threshold under which we consider gaps between
# kernels to be due to realistic delays in launching back-back kernels on the GPU


class BreakdownAnalysis:
    """TODO
    """
    def __init__(self):
        pass

    @classmethod
    def get_gpu_kernel_breakdown(
        cls,
        t: BaseTrace,
        visualize: bool = True,
        duration_ratio: float = 0.8,
        num_kernels: int = 10,
        include_memory_kernels: bool = False,
        image_renderer="notebook",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        GPU kernel breakdown implementation. See `get_gpu_kernel_breakdown` 
        in `trace_analysis.py` for details.
        """

        all_kernel_df = pd.DataFrame(
            {
                "name": pd.Series(dtype="str"),
                "sum": pd.Series(dtype="int"),
                "max": pd.Series(dtype="int"),
                "min": pd.Series(dtype="int"),
                "std": pd.Series(dtype="float"),
                "mean": pd.Series(dtype="int"),
                "kernel_type": pd.Series(dtype="str"),
                "rank": pd.Series(dtype="int"),
            }
        )

        kernel_type_to_analysis: List[str] = [
            KernelType.COMPUTATION.name,
            KernelType.COMMUNICATION.name,
        ]
        if include_memory_kernels:
            kernel_type_to_analysis.append(KernelType.MEMORY.name)

        results_by_rank: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        kernel_per_rank: Dict[str, Dict] = defaultdict(dict)

        for rank in t.get_ranks():
            visitor = StreamMergedEventsIterator(t, rank)

            rank_kernel_raw_data: List[Dict[str, Union[str, float, int]]] = []

            type_windows: Dict[str, Tuple[float, float]] = {
                k: (-1.0, -1.0) for k in kernel_type_to_analysis
            }

            type_total_duration: Dict[str, float] = {
                k: 0.0 for k in kernel_type_to_analysis
            }

            overlap_trackers: Dict[Tuple[str, str], Tuple[float, float]] = {}

            type_pairs = list(itertools.combinations(kernel_type_to_analysis, 2))
            for pair in type_pairs:
                overlap_trackers[pair] = (0.0, -1.0) # (duration, last_end_time)

            rank_start = -1.0
            rank_end = -1.0

            for e, _ in visitor:
                if e["cat"] == "gpu_user_annotation":
                    continue

                ts = e["ts"]
                dur = e["dur"]

                if rank_start == -1.0:
                    rank_start = ts
                rank_end = max(rank_end, ts + dur)

                current_type = get_kernel_type(e["name"])
                if current_type not in kernel_type_to_analysis:
                    continue

                rank_kernel_raw_data.append({
                    "name": e["name"],
                    "dur": float(dur),
                    "kernel_type": current_type,
                    "rank": int(rank),
                })

                current_start, current_end = type_windows[current_type]
                new_start, new_end, added_duration = AnalysisHelper.update_window(
                    current_start, current_end, ts, dur
                )
                type_windows[current_type] = (new_start, new_end)
                type_total_duration[current_type] += added_duration

                for other_type in kernel_type_to_analysis:
                    if other_type == current_type:
                        continue

                    currennt_start, currennt_end = new_start, new_end
                    other_start, other_end = type_windows[other_type]

                    pair = tuple((current_type, other_type))

                    if pair not in overlap_trackers:
                        pair = tuple((other_type, current_type))
                        other_start, other_end = new_start, new_end
                        currennt_start, currennt_end = type_windows[other_type]

                    if other_start != -1.0 and currennt_start != -1.0:

                        overlap_dur, last_end = overlap_trackers[pair]

                        new_overlap_dur, new_last_end = AnalysisHelper.add_overlap_duration(
                            overlap_dur, last_end,
                            currennt_start, currennt_end,
                            other_start, other_end
                        )

                        overlap_trackers[pair] = (new_overlap_dur, new_last_end)

            rank_results = results_by_rank[rank]

            for (type_a, type_b), (overlap_duration, _) in overlap_trackers.items():
                key = f"{type_a} overlapping {type_b}"
                rank_results[key] = overlap_duration

            for type_name in kernel_type_to_analysis:
                total_duration = type_total_duration[type_name]

                for (type_a, type_b), (overlap_duration, _) in overlap_trackers.items():
                    if type_name in (type_a, type_b):
                        total_duration -= overlap_duration

                rank_results[type_name] = total_duration

            if not rank_kernel_raw_data:
                continue
            rank_raw_df = pd.DataFrame(rank_kernel_raw_data)

            for kernel_type in kernel_type_to_analysis:
                gpu_kernel_time = rank_raw_df[rank_raw_df["kernel_type"] == kernel_type].copy()

                if gpu_kernel_time.empty:
                    continue

                top_kernels_df = cls._aggr_gpu_kernel_time(
                    gpu_kernel_time,
                    duration_ratio=duration_ratio,
                    num_kernels=num_kernels,
                )

                kernel_per_rank[kernel_type][rank] = top_kernels_df.copy()

                top_kernels_df["kernel_type"] = kernel_type
                top_kernels_df["rank"] = int(rank)
                all_kernel_df = pd.concat(
                    [all_kernel_df, top_kernels_df], ignore_index=True
                )

        final_list = []
        for rank, data in results_by_rank.items():
            for k, v in data.items():
                final_list.append({"kernel_type": k, "sum": int(v)})

        kernel_type_df = pd.DataFrame(final_list)

        kernel_type_df = kernel_type_df.groupby(by=["kernel_type"])["sum"].agg(["sum"])
        kernel_type_df.reset_index(inplace=True)

        kernel_type_df.sort_values(
            by=["sum"],
            inplace=True,
            ascending=False
        )

        kernel_type_df.reset_index(drop=True, inplace=True)

        total_sum_of_all_events = kernel_type_df["sum"].sum()

        kernel_type_df["percentage"] = (
            kernel_type_df["sum"] / total_sum_of_all_events
        ) * 100
        kernel_type_df = kernel_type_df.round({"percentage": 1})
        kernel_type_df.sort_values(by=["sum"], inplace=True, ascending=False)

        kernel_type_df = kernel_type_df[kernel_type_df["sum"] > 0].copy()

        kernel_type_df.reset_index(drop=True, inplace=True)

        all_kernel_df.sort_values(
            by=["kernel_type", "name", "rank"], ignore_index=True, inplace=True
        )
        all_kernel_df.rename(
            columns={
                "sum": "sum (us)",
                "mean": "mean (us)",
                "max": "max (us)",
                "min": "min (us)",
                "std": "stddev",
            },
            inplace=True,
        )

        if visualize:  # pragma: no cover
            non_zero_kernel_df = kernel_type_df[(kernel_type_df["percentage"] > 0)]

            fig = px.pie(
                non_zero_kernel_df,
                values="percentage",
                names="kernel_type",
                height=500,
                title="Kernel Type Percentage Across All Ranks",
            )
            fig.update_layout(
                margin=dict(l=50, r=50, b=50, t=50),
                showlegend=True,
                legend=dict(yanchor="bottom", y=-0.4, xanchor="left", x=0),
            )
            fig.show(renderer=image_renderer)

            for kernel in kernel_per_rank:
                specs = []
                for count, _ in enumerate(kernel_per_rank[kernel]):
                    if count % 2 == 0:
                        specs.append([{"type": "domain"}, {"type": "domain"}])
                fig = make_subplots(
                    rows=int((len(kernel_per_rank[kernel]) + 1) / 2),
                    cols=2,
                    specs=specs,
                )
                for rank in kernel_per_rank[kernel]:
                    fig.add_trace(
                        go.Pie(
                            labels=kernel_per_rank[kernel][rank]["name"],
                            values=kernel_per_rank[kernel][rank]["sum"],
                            title=f"Rank {rank}",
                            automargin=False,
                        ),
                        int(int(rank) / 2) + 1,
                        int(int(rank) % 2) + 1,
                    )
                image_size_multiplier = 1 + (len(t.get_ranks())) / 2
                fig.update_layout(
                    title_text=f'Kernel type "{kernel}" - kernel distribution on each rank',
                    margin=dict(l=50, r=50, b=50, t=50),
                    showlegend=True,
                    height=400 * image_size_multiplier,
                    legend=dict(yanchor="bottom", y=-0.1, xanchor="left", x=0),
                )
                fig.show(renderer=image_renderer)

                kernel_df = all_kernel_df[all_kernel_df["kernel_type"].eq(kernel)]

                kernel_name = kernel_df["name"].unique()
                for name in kernel_name:
                    if name != "others":
                        kernel_name_df = kernel_df[kernel_df["name"].eq(name)]
                        fig = px.bar(
                            kernel_name_df,
                            x="rank",
                            y="mean (us)",
                            title=name,
                            labels={
                                "rank": "Rank",
                                "mean (us)": "Mean Duration (us)",
                            },
                            error_y=kernel_name_df["max (us)"]
                            - kernel_name_df["mean (us)"],
                            error_y_minus=kernel_name_df["mean (us)"]
                            - kernel_name_df["min (us)"],
                        )
                        fig.update_layout(
                            title_text=f'Kernel type "{kernel}" - {name}',
                            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
                        )
                        fig.show(renderer=image_renderer)

        return kernel_type_df, all_kernel_df

    @classmethod
    def get_temporal_breakdown(
        cls,
        t: BaseTrace,
        visualize: bool = True
    ) -> pd.DataFrame:
        """
        Temporal breakdown implementation. See `get_temporal_breakdown` 
        in `trace_analysis.py` for details.
        """

        results: Dict[str, List[Union[str, int, float]]] = defaultdict(list)

        for rank in t.get_ranks():
            results["rank"].append(rank)
            visitor = StreamMergedEventsIterator(t, rank)
            total_time = 0.0
            active_time = 0.0
            compute_time = 0.0

            start_time = -1
            end_time = 0

            gpu_active_window = (-1.0, -1.0)
            compute_active_window = (-1.0, -1.0)

            for e, _ in visitor:
                if e["cat"] == "gpu_user_annotation":
                    continue

                ts = e["ts"]
                dur = e["dur"]

                if start_time == -1:
                    start_time = ts
                end_time = max(end_time, ts + dur)

                new_start, new_end, added_duration = AnalysisHelper.update_window(
                    gpu_active_window[0],
                    gpu_active_window[1],
                    ts,
                    dur
                )
                gpu_active_window = (new_start, new_end)
                active_time += added_duration

                if is_compute_kernel(e["name"]):
                    new_start, new_end, added_duration = AnalysisHelper.update_window(
                        compute_active_window[0],
                        compute_active_window[1],
                        ts,
                        dur
                    )
                    compute_active_window = (new_start, new_end)
                    compute_time += added_duration

            total_time = end_time - start_time
            results["kernel_time(us)"].append(float(total_time))
            results["idle_time(us)"].append(total_time - active_time)
            results["compute_time(us)"].append(compute_time)
            results["non_compute_time(us)"].append(active_time - compute_time)

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
            @args: allowlist_names (list(str): if kernel names are in this list, 
                they will not be aggregated into "other" catgory.
                This argument is meant to keep some kernel/events as distinct in the aggregation

        Returns:
            aggregated pd.DataFrame by "name" column with ["sum", "max", "min", "mean", "std"]
        """

        gpu_kernel_time = gpu_kernel_time.groupby(by=["name"])["dur"].agg(
            ["sum", "max", "min", "mean", "std"]
        )
        gpu_kernel_time.reset_index(inplace=True)
        gpu_kernel_time = gpu_kernel_time.sort_values(
            by=["sum"], ascending=False, ignore_index=True
        )
        gpu_kernel_time.fillna({"std": 0}, inplace=True)

        # if there are more than num_kernels kernels, starting to aggregate kernels
        if gpu_kernel_time.shape[0] > num_kernels:
            if allowlist_names is not None:
                keep_idx = gpu_kernel_time.name.isin(allowlist_names)
            else:
                # always false
                keep_idx = gpu_kernel_time["sum"] < 0

            gpu_kernel_time["cumsum"] = gpu_kernel_time["sum"].cumsum()
            quantiles = gpu_kernel_time["cumsum"].quantile(duration_ratio)
            # fmt: off
            gpu_kernel_time.loc[~keep_idx & (gpu_kernel_time["cumsum"] > quantiles), "name"] = (
                "others"
            )
            # fmt: on
            gpu_kernel_time.loc[
                ~keep_idx & (gpu_kernel_time.index >= num_kernels), "name"
            ] = "others"
            gpu_kernel_time = gpu_kernel_time.groupby(by=["name"])["sum"].agg(
                ["sum", "max", "min", "mean", "std"]
            )
            gpu_kernel_time.reset_index(inplace=True)
            gpu_kernel_time.fillna({"std": 0}, inplace=True)

        return gpu_kernel_time
