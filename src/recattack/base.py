"""Common recommender interface.

A model is fitted on a dense rating matrix R (0 = missing) with observed mask M and exposes
a full user x item prediction matrix. Rating prediction (RMSE/MAE) and top-N ranking both read
from it, so attack and defence code can treat every model the same way.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .data import RATING_MAX, RATING_MIN


class Recommender(ABC):
    name = "recommender"

    def __init__(self) -> None:
        self._pred: np.ndarray | None = None

    def fit(self, R: np.ndarray, M: np.ndarray) -> "Recommender":
        if R.shape != M.shape or R.ndim != 2:
            raise ValueError(f"R and M must be 2-D with equal shape, got {R.shape} and {M.shape}")
        if not M.any():
            raise ValueError("cannot fit on an empty rating matrix")
        pred = self._fit(R, M)
        self._pred = np.clip(pred, RATING_MIN, RATING_MAX).astype(np.float32, copy=False)
        return self

    @abstractmethod
    def _fit(self, R: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Return an unclipped (n_users, n_items) rating prediction matrix."""

    def predictions(self) -> np.ndarray:
        if self._pred is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        return self._pred

    def scores(self) -> np.ndarray:
        """Ranking scores, higher is better. Defaults to predicted ratings."""
        return self.predictions()
