# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Union

import pandas as pd
import plotly.express as px

from perflowai.padoc.trace import BaseTrace
from perflowai.padoc.visitor import StreamMergedEventsIterator
from perflowai.padoc.hta.types import is_compute_kernel, is_comm_kernel
from perflowai.padoc.hta.analysis_helper import AnalysisHelper


class CommunicationAnalysis:
    """TODO
    """
    def __init__(self):
        pass

    @classmethod
    def get_comm_comp_overlap(cls, t: BaseTrace, visualize: bool = True) -> pd.DataFrame:
        """
        Communication analysis implementation. 
        See `get_comm_comp_overlap` in `trace_analysis.py` for details.
        """

        result: Dict[str, List[Union[str, int, float]]] = defaultdict(list)

        for rank in t.get_ranks():
            result["rank"].append(rank)
            visitor = StreamMergedEventsIterator(t, rank)

            start_time = -1
            end_time = 0
            comm_time = 0.0
            overlap_tracker = (0.0, -1.0)
            comp_window = (-1.0, -1.0)
            comm_window = (-1.0, -1.0)

            for e, _ in visitor:
                if e["cat"] == "gpu_user_annotation":
                    continue

                ts = e["ts"]
                dur = e["dur"]

                if start_time == -1:
                    start_time = ts
                end_time = max(end_time, ts + dur)

                if is_compute_kernel(e["name"]):
                    new_start, new_end, _ = AnalysisHelper.update_window(
                        comp_window[0],
                        comp_window[1],
                        ts,
                        dur
                    )
                    comp_window = (new_start, new_end)

                    overlap_tracker = AnalysisHelper.add_overlap_duration(
                        overlap_tracker[0],
                        overlap_tracker[1],
                        comp_window[0],
                        comp_window[1],
                        comm_window[0],
                        comm_window[1],
                    )
                elif is_comm_kernel(e["name"]):
                    new_start, new_end, added_duration = AnalysisHelper.update_window(
                        comm_window[0],
                        comm_window[1],
                        ts,
                        dur
                    )
                    comm_window = (new_start, new_end)
                    comm_time += added_duration

                    overlap_tracker = AnalysisHelper.add_overlap_duration(
                        overlap_tracker[0],
                        overlap_tracker[1],
                        comp_window[0],
                        comp_window[1],
                        comm_window[0],
                        comm_window[1],
                    )
                else:
                    continue

            result["comp_comm_overlap_ratio"].append(
                overlap_tracker[0] / comm_time
            )

        result_df = pd.DataFrame(result)
        result_df["comp_comm_overlap_pctg"] = round(
            100 * result_df["comp_comm_overlap_ratio"], 2
        )

        if visualize:  # pragma: no cover
            fig = px.bar(
                result_df,
                x="rank",
                y="comp_comm_overlap_ratio",
                title="Computation-Communication Overlap",
                labels={
                    "rank": "Rank",
                    "comp_comm_overlap_ratio": (
                        "Computation-Communication Overlap Percentage"
                    ),
                },
            )

            fig.update_layout(yaxis_tickformat=".2%")
            fig.show()

        return result_df[["rank", "comp_comm_overlap_pctg"]]
