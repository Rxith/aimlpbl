"""Paired held-out experiments and seed-level aggregate reporting."""
from time import perf_counter
import pandas as pd
from .workload import WORKLOADS, generate_trace, chronological_split
from .model import train_model, predictive_metrics
from .policies import LRU, LFU, ML
from .simulator import simulate

METRICS = ["hit_rate_pct", "miss_rate_pct", "backend_accesses", "average_response_ms",
           "p95_response_ms", "byte_hit_rate_pct", "cache_utilization_pct",
           "policy_overhead_us_per_request"]


def run_experiments(workloads=WORKLOADS, seeds=(11, 22, 33),
                    capacities=(16384, 32768, 65536), n_requests=3000,
                    n_objects=100, horizon=30, hit_latency_ms=1.0,
                    variable_sizes=True, algorithms=("LRU", "LFU", "ML")) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not workloads or not seeds or not capacities or not algorithms:
        raise ValueError("Select at least one workload, seed, capacity and algorithm")
    if any(a not in ("LRU", "LFU", "ML") for a in algorithms):
        raise ValueError("Unknown algorithm")
    rows, predictions = [], []
    for kind in workloads:
        for seed in seeds:
            trace = generate_trace(kind, n_requests, n_objects, seed, variable_sizes)
            train, validation, test = chronological_split(trace)
            model = None
            if "ML" in algorithms:
                start = perf_counter()
                model, val = train_model(train, validation, horizon, seed)
                training_seconds = perf_counter() - start
                for partition, metrics in (("validation", val),
                                            ("test", predictive_metrics(model, test, horizon, seed))):
                    predictions.append({"workload": kind, "seed": seed, "partition": partition,
                                        "training_seconds": training_seconds, **metrics})
            for capacity in capacities:
                for name in algorithms:
                    policy = {"LRU": LRU, "LFU": LFU}.get(name)
                    instance = ML(model) if name == "ML" else policy()
                    # Each call starts with empty cache AND empty observable history.
                    result = simulate(test, capacity, instance, hit_latency_ms)
                    rows.append({"workload": kind, "seed": seed, **result.metrics})
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["workload", "capacity_bytes", "policy"])[METRICS].agg(["mean", "std"])


def winners(results: pd.DataFrame) -> pd.DataFrame:
    """Report all tied winners on mean hit rate, without privileging ML."""
    means = results.groupby(["workload", "capacity_bytes", "policy"], as_index=False).hit_rate_pct.mean()
    best = means.groupby(["workload", "capacity_bytes"]).hit_rate_pct.transform("max")
    return means[abs(means.hit_rate_pct - best) < 1e-10].reset_index(drop=True)
