import heapq
from typing import List, Generator, Tuple, Dict
from .trace import BaseTrace
from .event import BaseEvent, Event

class StreamMergedEventsIterator:
    """TODO
    """

    def __init__(self, trace: BaseTrace, rank: str):
        self.priority_queue: List[Tuple[int, int, BaseEvent]] = []
        self.streams: List[Generator[BaseEvent, None, None]] = []
        self.index_dict: Dict[int, int] = {}
        self.rank = rank

        for pid in trace.get_pids(rank):
            for tid in trace.get_tids(rank, pid):
                node = trace.get_node(rank, pid, tid, "M")
                events = node.get_all_events() if node else []
                is_stream = False
                for e in events:
                    args = e.to_dict().get("args", {})
                    name = args.get("name", "")
                    if "stream" in name:
                        is_stream = True
                        break
                if is_stream:
                    x_node = trace.get_node(rank, pid, tid, "X")
                    stream_id = int(tid)
                    self.index_dict[stream_id] = len(self.streams)
                    stream = x_node.events_visitor()
                    self.streams.append(stream)
                    try:
                        first_event = next(stream)
                        heapq.heappush(self.priority_queue,
                                       (first_event.get_ts(), stream_id, first_event))
                    except StopIteration:
                        pass


    def __iter__(self) -> Generator[Event, None, None]:
        return self

    def __next__(self) -> Event:
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

        return event_to_yield
