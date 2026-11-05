import pytest
from adaptive_cache.workload import Request, generate_trace, WORKLOADS
from adaptive_cache.policies import LRU, LFU, ML
from adaptive_cache.simulator import simulate
from adaptive_cache.model import train_model


def trace(ids, sizes=None):
    sizes = sizes or {}
    return [Request(float(i), k, sizes.get(k, 1), 10.0) for i, k in enumerate(ids)]


def test_lru_hits_misses_and_eviction():
    result = simulate(trace([1, 2, 1, 3, 2]), 2, LRU())
    assert result.hit_flags == [False, False, True, False, False]
    assert result.evictions == [2, 1]
    assert result.metrics["backend_accesses"] == 4
    assert result.metrics["average_response_ms"] == 9
    assert result.metrics["hit_rate_pct"] == 20
    assert result.metrics["p95_response_ms"] == 11


def test_lfu_counts_and_lru_ties():
    result = simulate(trace([1, 1, 2, 3, 1, 2]), 2, LFU())
    assert result.evictions == [2, 3]
    assert result.hit_flags == [False, True, False, False, True, False]
    tied = simulate(trace([2, 1, 3]), 2, LFU())
    assert tied.evictions == [2]


def test_lfu_resets_count_after_reinsertion():
    result = simulate(trace([1, 2, 3, 1, 4]), 2, LFU())
    assert result.evictions == [1, 2, 3]

@pytest.mark.parametrize("policy", [LRU, LFU])
def test_bytes_multiple_evictions_and_oversize_bypass(policy):
    result = simulate(trace([1, 2, 3, 4, 3], {1: 3, 2: 3, 3: 5, 4: 20}), 6, policy())
    assert result.evictions == [1, 2]
    assert result.occupancy_bytes == [3, 6, 5, 5, 5]
    assert result.hit_flags == [False, False, False, False, True]
    assert result.metrics["byte_hit_rate_pct"] == pytest.approx(100 * 5 / 36)
    assert result.metrics["cache_utilization_pct"] == pytest.approx(80)

@pytest.mark.parametrize("kind", WORKLOADS)
def test_reproducible_workloads_and_capacity(kind):
    requests = generate_trace(kind, 400, 20, 5)
    assert requests == generate_trace(kind, 400, 20, 5)
    assert requests != generate_trace(kind, 400, 20, 6)
    for policy in (LRU(), LFU()):
        result = simulate(requests, 10000, policy)
        assert max(result.occupancy_bytes) <= 10000
        assert result.metrics["hits"] + result.metrics["misses"] == 400
        assert simulate(requests, 10000, policy).hit_flags == result.hit_flags


def test_single_class_models_and_ml_capacity():
    requests = trace([1] * 60)
    model, metrics = train_model(requests, requests, horizon=3, trees=5)
    assert metrics["roc_auc"] is None
    result = simulate(trace([1, 2, 3, 4, 1]), 2, ML(model))
    assert max(result.occupancy_bytes) <= 2

@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        simulate(trace([1]), capacity, LRU())


def test_invalid_trace():
    with pytest.raises(ValueError, match="stable"):
        simulate([Request(0, 1, 1, 1), Request(1, 1, 2, 1)], 2, LRU())
    with pytest.raises(ValueError, match="chronological"):
        simulate([Request(2, 1, 1, 1), Request(1, 2, 1, 1)], 2, LRU())
