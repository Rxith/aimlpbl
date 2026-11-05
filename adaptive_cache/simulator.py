"""Trace-driven byte-limited cache. No real HTTP traffic or wall-clock sleeps."""
from dataclasses import dataclass
from time import perf_counter
import math
import numpy as np
from .features import History
from .policies import Entry, LRU, LFU, ML
from .workload import Request

@dataclass
class SimulationResult:
    metrics: dict
    hit_flags: list[bool]
    occupancy_bytes: list[int]
    evictions: list[int]


def simulate(trace: list[Request], capacity_bytes: int, policy: LRU | LFU | ML,
             hit_latency_ms: float = 1.0) -> SimulationResult:
    if not trace or capacity_bytes <= 0 or hit_latency_ms < 0 or not math.isfinite(hit_latency_ms):
        raise ValueError("Require a nonempty trace, positive capacity, and finite nonnegative hit latency")
    entries: dict[int, Entry] = {}
    history = History()
    used = 0
    hits, occupancy, evictions, response = [], [], [], []
    hit_bytes = total_bytes = 0
    policy_seconds = 0.0
    previous_time = -float("inf")
    catalog = {}
    for i, request in enumerate(trace):
        if (not math.isfinite(request.timestamp) or request.timestamp < previous_time
                or request.size_bytes <= 0 or not math.isfinite(request.backend_latency_ms)
                or request.backend_latency_ms < 0):
            raise ValueError("Trace must be chronological with positive sizes and finite latency")
        previous_time = request.timestamp
        if request.object_id in catalog and catalog[request.object_id] != request.size_bytes:
            raise ValueError("Object sizes must remain stable within a trace")
        catalog[request.object_id] = request.size_bytes
        start = perf_counter()
        history.observe(request, i)
        hit = request.object_id in entries
        hits.append(hit)
        total_bytes += request.size_bytes
        if hit:
            entry = entries[request.object_id]
            entry.last_access = i
            entry.frequency += 1
            hit_bytes += request.size_bytes
        elif request.size_bytes <= capacity_bytes:
            if used + request.size_bytes > capacity_bytes:
                for victim in policy.eviction_order(entries, history, i, request.timestamp):
                    used -= entries.pop(victim).size_bytes
                    evictions.append(victim)
                    if used + request.size_bytes <= capacity_bytes:
                        break
            entries[request.object_id] = Entry(request.object_id, request.size_bytes, i, i)
            used += request.size_bytes
        policy_seconds += perf_counter() - start
        assert 0 <= used <= capacity_bytes
        assert used == sum(e.size_bytes for e in entries.values())
        # All lookups pay cache latency; misses additionally pay backend retrieval.
        response.append(hit_latency_ms + (0 if hit else request.backend_latency_ms))
        occupancy.append(used)
    hit_count = sum(hits)
    return SimulationResult({"policy": policy.name, "capacity_bytes": capacity_bytes,
        "requests": len(trace), "hits": hit_count, "misses": len(trace) - hit_count,
        "hit_rate_pct": 100 * hit_count / len(trace), "miss_rate_pct": 100 * (1 - hit_count / len(trace)),
        "backend_accesses": len(trace) - hit_count,
        "average_response_ms": float(np.mean(response)), "p95_response_ms": float(np.percentile(response, 95)),
        "byte_hit_rate_pct": 100 * hit_bytes / total_bytes,
        "cache_utilization_pct": 100 * float(np.mean(occupancy)) / capacity_bytes,
        "policy_overhead_ms": policy_seconds * 1000,
        "policy_overhead_us_per_request": policy_seconds * 1e6 / len(trace)}, hits, occupancy, evictions)
