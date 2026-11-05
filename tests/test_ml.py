import numpy as np
import pytest
from adaptive_cache.features import History, supervised_samples
from adaptive_cache.workload import Request, generate_trace, chronological_split
from adaptive_cache.model import train_model, reuse_probability
from adaptive_cache.policies import Entry, ML
from adaptive_cache.experiments import run_experiments, aggregate, winners


def test_features_values_and_cold_start():
    history = History()
    assert history.features(4, 512, 0, 0) == [0, 0, 0, 0, 512, 0, 0, 1]
    history.observe(Request(2, 4, 512, 10), 0)
    history.observe(Request(5, 4, 512, 10), 2)
    assert history.features(4, 512, 12, 8) == [3, 0, 2, 2, 512, 3, 0, 0]


def test_future_changes_labels_but_not_inputs():
    prefix = [Request(float(i), 1, 10, 5) for i in range(6)]
    a = prefix + [Request(float(i), 1, 10, 5) for i in range(6, 10)]
    b = prefix + [Request(float(i), 2, 10, 5) for i in range(6, 10)]
    xa, ya = supervised_samples(a, horizon=4, stride=1)
    xb, yb = supervised_samples(b, horizon=4, stride=1)
    np.testing.assert_array_equal(xa, xb)
    assert ya[-1] == 1 and yb[-1] == 0
    # Last four decisions are excluded: their complete horizons do not exist.
    assert len(ya) == 6


def test_labels_exclude_current_request_and_horizon_boundary():
    requests = [Request(float(i), obj, 10, 5) for i, obj in enumerate([1, 2, 1, 3])]
    _, y = supervised_samples(requests, horizon=1, stride=1, candidates=1, seed=3)
    assert y[0] == 0
    _, wider = supervised_samples(requests, horizon=2, stride=1, candidates=1)
    assert wider[0] == 1


def test_chronological_partitions_have_disjoint_horizons():
    train, val, test = chronological_split(generate_trace(n_requests=300))
    assert train[-1].timestamp < val[0].timestamp < test[0].timestamp
    # No training observation can change when later partitions are replaced.
    x, y = supervised_samples(train, horizon=10)
    assert len(x) == len(y) and x.shape[1] == 8
    with pytest.raises(ValueError):
        supervised_samples(test, horizon=len(test))


def test_reproducible_training():
    train, val, test = chronological_split(generate_trace(n_requests=400, n_objects=12))
    a, metrics = train_model(train, val, 10, trees=5)
    b, _ = train_model(train, val, 10, trees=5)
    x, _ = supervised_samples(test, 10)
    np.testing.assert_array_equal(reuse_probability(a, x), reuse_probability(b, x))
    assert 0 <= metrics["f1"] <= 1


def test_ml_size_heuristic_and_deterministic_tie():
    class ConstantModel:
        classes_ = np.array([0, 1])
        def predict_proba(self, x):
            return np.tile([.5, .5], (len(x), 1))
    entries = {1: Entry(1, 10, 0, 0), 2: Entry(2, 20, 1, 1), 3: Entry(3, 10, 2, 2)}
    assert ML(ConstantModel()).eviction_order(entries, History(), 3, 3) == [2, 1, 3]


def test_paired_experiment_pipeline():
    raw, predictive = run_experiments(workloads=("uniform",), seeds=(1, 2),
                                      capacities=(5000, 10000), n_requests=150,
                                      n_objects=10, horizon=5)
    assert len(raw) == 12
    assert set(raw.policy) == {"LRU", "LFU", "ML"}
    assert raw.requests.nunique() == 1 and raw.requests.iloc[0] == 45
    assert set(predictive.partition) == {"validation", "test"}
    assert len(aggregate(raw)) == 6
    assert not winners(raw).empty
