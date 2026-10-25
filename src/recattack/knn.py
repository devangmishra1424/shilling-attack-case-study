"""User-based and item-based neighbourhood collaborative filtering.

Similarity is cosine on mean-centred ratings (users centred on their own mean, so item-based
is the usual adjusted cosine), shrunk by n_common / (n_common + shrink) to damp similarities
supported by few co-ratings. The k neighbours are chosen among entities that rated the target,
positive similarity only. Ties at the k-th similarity are all kept, so a neighbourhood can
slightly exceed k.
"""
from __future__ import annotations

import numpy as np

from .base import Recommender


def _similarity(X: np.ndarray, Mx: np.ndarray, shrink: float, col_weight: np.ndarray | None = None) -> np.ndarray:
    """Shrunk cosine between rows of X. `col_weight` (one weight per column, e.g. user trust when
    rows are items) scales each column's contribution to the dot products, norms and common counts."""
    if col_weight is not None:
        X = X * np.sqrt(col_weight).astype(np.float32)[None, :]
    num = X @ X.T
    norm = np.sqrt((X * X).sum(axis=1))
    denom = np.outer(norm, norm)
    sim = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    Mf = Mx.astype(np.float32)
    if col_weight is None:
        common = Mf @ Mf.T
    else:
        common = (Mf * col_weight.astype(np.float32)[None, :]) @ Mf.T
    shrink_den = common + shrink
    sim *= np.divide(common, shrink_den, out=np.zeros_like(common), where=shrink_den > 0)
    np.fill_diagonal(sim, 0.0)
    return sim


def _neighbour_deviation(
    S: np.ndarray, X: np.ndarray, Mx: np.ndarray, k: int, row_trust: np.ndarray | None = None
) -> np.ndarray:
    """For each row entity r and column c: similarity-weighted mean of X[v, c] over r's top-k
    neighbours v that observed column c. Returns 0 where no neighbour qualifies. With `row_trust`
    the weight is sim*trust, used both to pick the top-k and to average: sum(sim*trust*x) /
    sum(sim*trust). Ranking by sim*trust matters: a zero-trust profile must not take a
    neighbour slot from a trusted one."""
    n_rows, n_cols = X.shape
    out = np.zeros((n_rows, n_cols), dtype=np.float32)
    kth = max(n_rows - k, 0)
    for r in range(n_rows):
        s = S[r] if row_trust is None else S[r] * row_trust
        W = np.where(Mx & (s[:, None] > 0), s[:, None], -np.inf)
        if n_rows > k:
            thr = np.partition(W, kth, axis=0)[kth]
            W = np.where(W >= thr, W, -np.inf)
        W = np.where(np.isfinite(W), W, 0.0)
        den = W.sum(axis=0)
        num = (W * X).sum(axis=0)
        np.divide(num, den, out=out[r], where=den > 0)
    return out


class KNNRecommender(Recommender):
    def __init__(self, mode: str = "user", k: int = 40, shrink: float = 25.0, user_trust: np.ndarray | None = None) -> None:
        """`user_trust` in [0, 1] per user (the class deck's trust weighting). User mode weights each
        neighbour's vote by its trust; item mode weights each user's contribution to item similarity."""
        super().__init__()
        if mode not in ("user", "item"):
            raise ValueError("mode must be 'user' or 'item'")
        if k < 1 or shrink < 0:
            raise ValueError("need k >= 1 and shrink >= 0")
        if user_trust is not None and (np.any(user_trust < 0) or np.any(user_trust > 1)):
            raise ValueError("user_trust must lie in [0, 1]")
        self.mode, self.k, self.shrink = mode, k, shrink
        self.user_trust = None if user_trust is None else np.asarray(user_trust, dtype=np.float32)
        self.name = f"{mode.capitalize()}KNN"

    def _fit(self, R, M):
        Rf = R.astype(np.float32)
        n_u = M.sum(axis=1)
        global_mean = float(Rf[M].mean())
        mu_u = np.where(n_u > 0, (Rf * M).sum(axis=1) / np.maximum(n_u, 1), global_mean)
        mu_u = mu_u.astype(np.float32)
        C = (Rf - mu_u[:, None]) * M
        trust = self.user_trust
        if trust is not None and trust.shape[0] != R.shape[0]:
            raise ValueError("user_trust must have one entry per user")
        if self.mode == "user":
            X, Mx = C, M
            S = _similarity(X, Mx, self.shrink)
            dev = _neighbour_deviation(S, X, Mx, self.k, row_trust=trust)
        else:
            X, Mx = np.ascontiguousarray(C.T), np.ascontiguousarray(M.T)
            S = _similarity(X, Mx, self.shrink, col_weight=trust)
            dev = _neighbour_deviation(S, X, Mx, self.k)
        if self.mode == "item":
            dev = dev.T
        return mu_u[:, None] + dev
