"""Features use only requests observed at or before the decision time."""
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from .workload import Request

FEATURE_NAMES = ("recency_seconds", "count_last_10", "count_last_50", "total_count",
                 "size_bytes", "mean_gap_seconds", "std_gap_seconds", "cold_start")

@dataclass
class ObjectHistory:
    count: int = 0
    last_time: float | None = None
    indices: deque = field(default_factory=lambda: deque(maxlen=50))
    gaps: deque = field(default_factory=lambda: deque(maxlen=10))

class History:
    def __init__(self) -> None:
        self.objects: dict[int, ObjectHistory] = {}
        self.sizes: dict[int, int] = {}

    def observe(self, request: Request, index: int) -> None:
        h = self.objects.setdefault(request.object_id, ObjectHistory())
        if h.last_time is not None:
            h.gaps.append(request.timestamp - h.last_time)
        h.count += 1
        h.last_time = request.timestamp
        h.indices.append(index)
        self.sizes[request.object_id] = request.size_bytes

    def features(self, object_id: int, size: int, index: int, timestamp: float) -> list[float]:
        h = self.objects.get(object_id)
        if h is None:
            return [0, 0, 0, 0, size, 0, 0, 1]
        return [timestamp - h.last_time,
                sum(i > index - 10 for i in h.indices),
                sum(i > index - 50 for i in h.indices), h.count, size,
                float(np.mean(h.gaps)) if h.gaps else 0,
                float(np.std(h.gaps)) if h.gaps else 0, 0]


def supervised_samples(trace: list[Request], horizon: int = 30, seed: int = 42,
                       candidates: int = 5, stride: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Sample known catalog objects at decision times; purge incomplete label horizons.

    History starts empty for each chronological partition. Only labels inspect
    future requests. Candidate sampling is independent of future reuse.
    """
    if horizon < 1 or len(trace) <= horizon or candidates < 1 or stride < 1:
        raise ValueError("Partition must exceed positive horizon; sampling parameters must be positive")
    rng = np.random.default_rng(seed)
    history = History()
    rows, labels = [], []
    for i, request in enumerate(trace[:-horizon]):
        history.observe(request, i)
        if i % stride:
            continue
        known = list(history.sizes)
        selected = rng.choice(known, min(candidates, len(known)), replace=False)
        future = {r.object_id for r in trace[i + 1:i + horizon + 1]}
        for obj in selected:
            rows.append(history.features(int(obj), history.sizes[obj], i, request.timestamp))
            labels.append(int(obj in future))
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)
