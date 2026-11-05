"""Random forest reuse prediction, distinct from the cache eviction heuristic."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from .features import supervised_samples
from .workload import Request


def reuse_probability(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
    if 1 not in model.classes_:
        return np.zeros(len(features))
    return model.predict_proba(features)[:, list(model.classes_).index(1)]


def predictive_metrics(model: RandomForestClassifier, trace: list[Request], horizon: int,
                       seed: int) -> dict:
    x, y = supervised_samples(trace, horizon, seed)
    p = reuse_probability(model, x)
    pred = p >= .5
    return {"samples": len(y), "positive_fraction": float(y.mean()),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}


def train_model(train: list[Request], validation: list[Request], horizon: int = 30,
                seed: int = 42, trees: int = 60) -> tuple[RandomForestClassifier, dict]:
    x, y = supervised_samples(train, horizon, seed)
    model = RandomForestClassifier(n_estimators=trees, max_depth=10, min_samples_leaf=4,
                                   random_state=seed, n_jobs=1, class_weight="balanced_subsample")
    model.fit(x, y)
    # Fixed model settings; validation is diagnostic, never used as training data.
    return model, predictive_metrics(model, validation, horizon, seed)
