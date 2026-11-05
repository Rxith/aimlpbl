"""Reproducible, chronological synthetic web request traces."""
from dataclasses import dataclass
import numpy as np

WORKLOADS = ("uniform", "zipf", "bursty", "shifting")

@dataclass(frozen=True)
class Request:
    timestamp: float
    object_id: int
    size_bytes: int
    backend_latency_ms: float


def generate_trace(kind: str = "zipf", n_requests: int = 3000,
                   n_objects: int = 100, seed: int = 42,
                   variable_sizes: bool = True, zipf_alpha: float = 1.2,
                   mean_interval: float = 1.0, phase_length: int = 250) -> list[Request]:
    if kind not in WORKLOADS:
        raise ValueError(f"Unknown workload: {kind}")
    if n_requests < 1 or n_objects < 2 or mean_interval <= 0 or phase_length < 1 or zipf_alpha <= 0:
        raise ValueError("Invalid workload parameters")
    rng = np.random.default_rng(seed)
    sizes = rng.integers(512, 8193, n_objects) if variable_sizes else np.full(n_objects, 4096)
    # Catalog metadata stays fixed; latency includes transfer cost and fixed server cost.
    latencies = 20 + sizes / 1024 * 2 + rng.uniform(0, 10, n_objects)
    popularity = np.arange(1, n_objects + 1, dtype=float) ** -zipf_alpha
    popularity /= popularity.sum()
    order = rng.permutation(n_objects)
    times = np.cumsum(rng.exponential(mean_interval, n_requests))
    result = []
    hot = order[:max(2, n_objects // 10)]
    for i, timestamp in enumerate(times):
        if kind == "uniform":
            obj = int(rng.integers(n_objects))
        elif kind == "zipf":
            obj = int(rng.choice(n_objects, p=popularity))
        elif kind == "bursty":
            if i % phase_length == 0:
                hot = rng.choice(n_objects, max(2, n_objects // 10), replace=False)
            obj = int(rng.choice(hot)) if rng.random() < .85 else int(rng.integers(n_objects))
        else:
            # Rotate popularity ranks without changing catalog sizes or latency.
            phase_order = np.roll(order, (i // phase_length) * max(1, n_objects // 4))
            obj = int(phase_order[rng.choice(n_objects, p=popularity)])
        result.append(Request(float(timestamp), obj, int(sizes[obj]), float(latencies[obj])))
    return result


def chronological_split(trace: list[Request]) -> tuple[list[Request], list[Request], list[Request]]:
    if len(trace) < 30:
        raise ValueError("Use at least 30 requests for chronological splits")
    a, b = int(len(trace) * .5), int(len(trace) * .7)
    return trace[:a], trace[a:b], trace[b:]
