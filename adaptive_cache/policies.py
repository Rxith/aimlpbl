"""Deterministic baseline and learned eviction ranking."""
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .features import History
from .model import reuse_probability

@dataclass
class Entry:
    object_id: int
    size_bytes: int
    inserted: int
    last_access: int
    frequency: int = 1  # LFU frequency since insertion, reset on reinsertion.

class LRU:
    name = "LRU"
    def eviction_order(self, entries: dict[int, Entry], history: History,
                       index: int, timestamp: float) -> list[int]:
        return sorted(entries, key=lambda k: (entries[k].last_access, k))

class LFU:
    name = "LFU"
    def eviction_order(self, entries: dict[int, Entry], history: History,
                       index: int, timestamp: float) -> list[int]:
        return sorted(entries, key=lambda k: (entries[k].frequency, entries[k].last_access, k))

class ML:
    name = "ML"
    def __init__(self, model: RandomForestClassifier) -> None:
        self.model = model

    def eviction_order(self, entries: dict[int, Entry], history: History,
                       index: int, timestamp: float) -> list[int]:
        keys = list(entries)
        x = np.asarray([history.features(k, entries[k].size_bytes, index, timestamp) for k in keys])
        probability = reuse_probability(self.model, x)
        # Reuse probability per byte is a heuristic, not an optimal knapsack solution.
        values = {k: p / entries[k].size_bytes for k, p in zip(keys, probability)}
        return sorted(keys, key=lambda k: (values[k], entries[k].last_access, k))
