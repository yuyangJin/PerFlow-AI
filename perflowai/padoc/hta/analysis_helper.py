"""Analysis Helper Module
"""

from typing import Tuple, Union

class AnalysisHelper:
    """Analysis Helper Class
    """

    def __init__(self):
        pass

    @classmethod
    def update_window(
        cls,
        window_start: Union[int, float],
        window_end: Union[int, float],
        event_ts: Union[int, float],
        event_dur: Union[int, float],
    ) -> Tuple[Union[int, float], Union[int, float], Union[int, float]]:
        """Update the window based on the current event."""

        new_start = event_ts
        new_end = event_ts + event_dur
        added_duration = 0.0

        if new_start > window_end:
            added_duration = event_dur
            window_start = new_start
            window_end = new_end
        else:
            if new_end > window_end:
                added_duration = new_end - window_end
                window_end = new_end

        return window_start, window_end, added_duration


    @classmethod
    def add_overlap_duration(
        cls,
        overlap_accumulator: Union[int, float],
        last_overlap_end: Union[int, float],
        window1_start: Union[int, float],
        window1_end: Union[int, float],
        window2_start: Union[int, float],
        window2_end: Union[int, float],
    ) -> Tuple[Union[int, float], Union[int, float]]:
        """Calculate the overlap duration between two time windows."""

        # Find the overlap between the two windows
        overlap_start = max(window1_start, window2_start)
        overlap_end = min(window1_end, window2_end)

        if overlap_start < overlap_end:
            # There is overlap
            if overlap_start >= last_overlap_end:
                # The overlap starts after the last overlap
                added_overlap = overlap_end - overlap_start
                last_overlap_end = overlap_end
                overlap_accumulator += added_overlap
            else:
                # Overlap partially or fully intersects with the previous recorded segment.
                if overlap_end > last_overlap_end:
                    # Only add the part that is beyond the last overlap
                    added_overlap = overlap_end - last_overlap_end
                    last_overlap_end = overlap_end
                    overlap_accumulator += added_overlap

        return overlap_accumulator, last_overlap_end
