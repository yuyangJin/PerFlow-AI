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
            end_time = -1
            last_compute_start_time = -1
            last_compute_end_time = -1
            last_comm_start_time = -1
            last_comm_end_time = -1
            last_compute_comm_overlap_end_time = -1
            comp_comm_overlap_duration = 0.0
            comm_time = 0.0

            def _add_duration(s1, e1, s2, e2):
                nonlocal last_compute_comm_overlap_end_time, comp_comm_overlap_duration
                if s1 <= s2 <= e1:
                    if s2 >= last_compute_comm_overlap_end_time:
                        last_compute_comm_overlap_end_time = min(e1, e2)
                        comp_comm_overlap_duration += last_compute_comm_overlap_end_time - s2
                    else:
                        new_end_time = min(e1, e2)
                        if new_end_time > last_compute_comm_overlap_end_time:
                            comp_comm_overlap_duration += \
                                new_end_time - last_compute_comm_overlap_end_time
                            last_compute_comm_overlap_end_time = new_end_time
                if s2 <= s1 <= e2:
                    if s1 >= last_compute_comm_overlap_end_time:
                        last_compute_comm_overlap_end_time = min(e1, e2)
                        comp_comm_overlap_duration += last_compute_comm_overlap_end_time - s1
                    else:
                        new_end_time = min(e1, e2)
                        if new_end_time > last_compute_comm_overlap_end_time:
                            comp_comm_overlap_duration += \
                                new_end_time - last_compute_comm_overlap_end_time
                            last_compute_comm_overlap_end_time = new_end_time


            for e, _ in visitor:
                if e["cat"] == "gpu_user_annotation":
                    continue
                ts = e["ts"]
                dur = e["dur"]
                if start_time == -1:
                    start_time = ts
                end_time = max(end_time, ts + dur)

                if is_compute_kernel(e["name"]):
                    if ts > last_compute_end_time:
                        last_compute_start_time = ts
                        last_compute_end_time = ts + dur
                    else:
                        last_compute_end_time = max(last_compute_end_time, ts + dur)
                    _add_duration(last_compute_start_time, last_compute_end_time, \
                                  last_comm_start_time, last_comm_end_time)
                elif is_comm_kernel(e["name"]):
                    if ts > last_comm_end_time:
                        last_comm_start_time = ts
                        last_comm_end_time = ts + dur
                        comm_time += dur
                    else:
                        if ts + dur > last_comm_end_time:
                            comm_time += ts + dur - last_comm_end_time
                            last_comm_end_time = ts + dur
                    _add_duration(last_compute_start_time, last_compute_end_time, \
                                  last_comm_start_time, last_comm_end_time)
                else:
                    continue

            result["comp_comm_overlap_ratio"].append(
                comp_comm_overlap_duration / comm_time
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
