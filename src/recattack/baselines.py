"""Non-personalised baselines: global mean, damped item mean, popularity, random."""
from __future__ import annotations

import numpy as np

from .base import Recommender


def _global_mean(R: np.ndarray, M: np.ndarray) -> float:
    return float(R[M].mean())


class GlobalMean(Recommender):
    name = "GlobalMean"

    def _fit(self, R, M):
        return np.full(R.shape, _global_mean(R, M), dtype=np.float32)


class DampedItemMean(Recommender):
    """Item mean shrunk towards the global mean: (sum + damp * mu) / (n + damp)."""

    name = "ItemMean"

    def __init__(self, damp: float = 5.0) -> None:
        super().__init__()
        if damp < 0:
            raise ValueError("damp must be >= 0")
        self.damp = damp

    def _fit(self, R, M):
        mu = _global_mean(R, M)
        n = M.sum(axis=0)
        item_mean = (R.sum(axis=0) + self.damp * mu) / np.maximum(n + self.damp, 1e-12)
        item_mean = np.where(n + self.damp > 0, item_mean, mu)
        return np.broadcast_to(item_mean, R.shape).copy()


class Popularity(DampedItemMean):
    """Ranks by number of ratings (most-rated first). Rating predictions use the damped item mean."""

    name = "Popularity"

    def _fit(self, R, M):
        self._counts = M.sum(axis=0).astype(np.float32)
        return super()._fit(R, M)

    def scores(self) -> np.ndarray:
        self.predictions()  # raises if not fitted
        return np.broadcast_to(self._counts, self._pred.shape)


class RandomRanker(GlobalMean):
    """Floor for ranking metrics: uniformly random scores, global-mean rating predictions."""

    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    def scores(self) -> np.ndarray:
        shape = self.predictions().shape
        return np.random.default_rng(self.seed).random(shape, dtype=np.float32)
