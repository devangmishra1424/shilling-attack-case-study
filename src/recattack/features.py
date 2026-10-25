"""Profile features for attack detection, and the MAD outlier score from the class notes.

All features are computed from the rating matrix alone (no labels), over every profile the
system holds, genuine or fake. Features follow the class deck's list (profile size, variance,
share of maximum/minimum ratings, popular items rated, deviation from item means, similarity
with other users) plus mean item popularity.
"""
from __future__ import annotations

import numpy as np

from .data import RATING_MAX, RATING_MIN
from .knn import _similarity

FEATURE_NAMES = (
    "log_n_ratings",
    "mean_rating",
    "var_rating",
    "frac_max",
    "frac_min",
    "frac_popular",
    "mean_log_item_pop",
    "rdma",
    "degsim",
)

MAD_THRESHOLD = 3.5  # class deck: |modified z| > 3.5 marks an outlier
_MAD_CONSTANT = 0.6745


def profile_features(R: np.ndarray, M: np.ndarray, n_popular: int = 100, k_sim: int = 10, shrink: float = 10.0) -> np.ndarray:
    """Return an (n_users, len(FEATURE_NAMES)) float64 matrix. Users with no ratings get zeros.

    rdma     mean of |rating - item mean| / item rating count over the profile (Chirita et al. 2005)
    degsim   mean of the k highest shrunk cosine similarities with other profiles; coordinated
             fakes resemble each other, so they score high
    """
    Rf = R.astype(np.float64)
    Mf = M.astype(np.float64)
    n = Mf.sum(axis=1)
    safe_n = np.maximum(n, 1.0)
    item_n = Mf.sum(axis=0)
    item_mean = np.where(item_n > 0, (Rf * Mf).sum(axis=0) / np.maximum(item_n, 1.0), 0.0)

    mean_r = (Rf * Mf).sum(axis=1) / safe_n
    var_r = (((Rf - mean_r[:, None]) ** 2) * Mf).sum(axis=1) / safe_n
    frac_max = ((R == RATING_MAX) & M).sum(axis=1) / safe_n
    frac_min = ((R == RATING_MIN) & M).sum(axis=1) / safe_n
    popular = np.argsort(-item_n, kind="stable")[:n_popular]
    frac_popular = Mf[:, popular].sum(axis=1) / safe_n
    mean_log_pop = (Mf * np.log1p(item_n)[None, :]).sum(axis=1) / safe_n
    dev = np.abs(Rf - item_mean[None, :]) / np.maximum(item_n, 1.0)[None, :]
    rdma = (dev * Mf).sum(axis=1) / safe_n

    C = ((R - mean_r[:, None]) * M).astype(np.float32)
    S = _similarity(C, M, shrink)
    k = min(k_sim, S.shape[0] - 1)
    degsim = np.sort(S, axis=1)[:, -k:].mean(axis=1) if k > 0 else np.zeros(S.shape[0])

    F = np.column_stack([np.log1p(n), mean_r, var_r, frac_max, frac_min, frac_popular, mean_log_pop, rdma, degsim])
    F[n == 0] = 0.0
    return F


def modified_z(F: np.ndarray) -> np.ndarray:
    """Modified z-score per feature: 0.6745 * (x - median) / MAD (class deck formula).

    If a feature's MAD is zero (more than half the profiles share one value) the scale falls
    back to 1.253314 * mean absolute deviation; if that is zero too the feature scores 0.
    """
    med = np.median(F, axis=0)
    dev = F - med
    mad = np.median(np.abs(dev), axis=0)
    mean_ad = np.mean(np.abs(dev), axis=0)
    z = np.zeros_like(F, dtype=np.float64)
    ok = mad > 0
    z[:, ok] = _MAD_CONSTANT * dev[:, ok] / mad[ok]
    fallback = (~ok) & (mean_ad > 0)
    z[:, fallback] = dev[:, fallback] / (1.253314 * mean_ad[fallback])
    return z


def mad_score(F: np.ndarray) -> np.ndarray:
    """Per-profile outlier score: the largest |modified z| over all features."""
    return np.abs(modified_z(F)).max(axis=1)


def trust_from_mad(score: np.ndarray, threshold: float = MAD_THRESHOLD) -> np.ndarray:
    """Trust in [0, 1]: 1 up to the threshold, then shrinking like threshold / score."""
    return np.minimum(1.0, threshold / np.maximum(score, 1e-12))
