import time
from dataclasses import dataclass
from typing import Dict
from contextlib import contextmanager
from collections import defaultdict


class TimeTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.time_dict = defaultdict(lambda: 0.0)
        self.count_dict = defaultdict(lambda: 0)

    def add_time(self, name: str, dt: float):
        self.time_dict[name] += dt
        self.count_dict[name] += 1

    def print_summary(self):
        for k in self.time_dict.keys():
            print(
                f"[{k}] tot={self.time_dict[k]:0.5f}; avg={self.time_dict[k] / self.count_dict[k]:0.5f}"
            )


TIMING_ENABLED = True
TIME_TRACKER = TimeTracker()


@contextmanager
def time_block(name="default", instant: bool = False):
    if TIMING_ENABLED:
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            if instant:
                print(f"[{name}] instant={end-start:0.5f}")
            TIME_TRACKER.add_time(name, end - start)
    else:
        yield


@dataclass
class MicroTimer:

    time_stamps: Dict[str, float]
    time_durations: Dict[str, float]

    def __init__(self):
        self.reset()

    def start(self, name: str):
        self.time_stamps[name] = time.time()

    def stop(self, name: str):
        if name not in self.time_stamps:
            raise ValueError(f"{name} was not started!")
        self.time_durations[name] = time.time() - self.time_stamps[name]

    def reset(self):
        self.time_stamps = {}
        self.time_durations = {}

    def collect(self):
        all_durations = self.time_durations
        all_durations["time_total"] = sum([v for v in self.time_durations.values()])
        return all_durations
