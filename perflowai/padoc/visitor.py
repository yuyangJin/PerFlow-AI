"""TODO
"""

import heapq
from typing import List, Generator, Tuple, Dict
from .trace import BaseTrace
from .event import BaseEvent, Event

class StreamMergedEventsIterator:
    """TODO
    """

    def __init__(self, trace: BaseTrace, rank: str):
        self.priority_queue: List[Tuple[int, str, BaseEvent]] = []
        self.streams: List[Generator[BaseEvent, None, None]] = []
        self.index_dict: Dict[str, int] = {}
        self.rank = rank

        for pid in trace.get_pids(rank):
            for tid in trace.get_tids(rank, pid):
                is_stream = tid.startswith("stream")
                if is_stream:
                    x_node = trace.get_node(rank, pid, tid, "X")
                    stream_id = tid.split(" ")[1]
                    unique_stream_key = f"{pid}_{stream_id}"
                    self.index_dict[unique_stream_key] = len(self.streams)
                    stream = x_node.events_visitor()
                    self.streams.append(stream)
                    try:
                        first_event = next(stream)
                        heapq.heappush(self.priority_queue,
                                       (first_event.get_ts(), unique_stream_key, first_event))
                    except StopIteration:
                        pass

    def __iter__(self) -> Generator[Event, None, None]:
        return self

    def __next__(self) -> Tuple[Event, str]:
        if not self.priority_queue:
            raise StopIteration

        _, stream_id, event_to_yield = heapq.heappop(self.priority_queue)

        stream = self.streams[self.index_dict[stream_id]]
        try:
            next_event = next(stream)

            heapq.heappush(
                self.priority_queue,
                (next_event.get_ts(), stream_id, next_event)
            )

        except StopIteration:
            pass

        return event_to_yield, stream_id
