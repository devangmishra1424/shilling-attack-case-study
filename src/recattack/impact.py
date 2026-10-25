"""Measure what an injected attack does to a fitted recommender."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .attacks import AttackSpec, Knowledge, inject, make_profiles, selected_items
from .base import Recommender
from .metrics import hit_ratio, item_ranks, prediction_shift


def segment_users(R: np.ndarray, M: np.ndarray, selected: np.ndarray, thr: int = 4) -> np.ndarray:
    """Genuine users who rated every selected item at least thr: the audience a segment attack targets."""
    if selected.size == 0:
        return np.empty(0, dtype=np.int64)
    ok = M[:, selected].all(axis=1) & (R[:, selected] >= thr).all(axis=1)
    return np.flatnonzero(ok)


def compare_models(
    before: Recommender,
    after: Recommender,
    R: np.ndarray,
    M: np.ndarray,
    spec: AttackSpec,
    kn: Knowledge,
    k: int = 10,
    users: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare two fitted models on the genuine users (rows 0..n_genuine-1 of both).

    Only genuine users who have not rated the target are evaluated, optionally restricted to
    `users`. Returns prediction shift, hit ratio@k before and after, and mean rank gain
    (positive = the target moved up the list).
    """
    n_genuine = R.shape[0]
    eligible = ~M[:, spec.target]
    if users is not None:
        mask = np.zeros(n_genuine, dtype=bool)
        mask[np.asarray(users)] = True
        eligible &= mask
    users = np.flatnonzero(eligible)
    if users.size == 0:
        raise ValueError("every genuine user already rated the target")
    s_before, s_after = before.scores()[:n_genuine], after.scores()[:n_genuine]
    p_before, p_after = before.predictions()[:n_genuine], after.predictions()[:n_genuine]

    def block(us: np.ndarray, tag: str) -> dict[str, float]:
        if us.size == 0:
            return {f"{tag}n_users": 0}
        r0 = item_ranks(s_before, M, spec.target, us)
        r1 = item_ranks(s_after, M, spec.target, us)
        return {
            f"{tag}n_users": int(us.size),
            f"{tag}pred_shift": prediction_shift(p_before, p_after, spec.target, us),
            f"{tag}hit_before": hit_ratio(s_before, M, spec.target, us, k),
            f"{tag}hit_after": hit_ratio(s_after, M, spec.target, us, k),
            f"{tag}rank_gain": float((r0 - r1).mean()),
        }

    out = block(users, "")
    if spec.kind == "segment":
        seg = segment_users(R, M, selected_items(spec, kn))
        out.update(block(np.intersect1d(seg, users), "seg_"))
    return out


def measure_attack(
    make_model: Callable[[], Recommender],
    before: Recommender,
    R: np.ndarray,
    M: np.ndarray,
    spec: AttackSpec,
    kn: Knowledge,
    k: int = 10,
) -> dict[str, float]:
    """Fit a fresh model on genuine + fake profiles and compare it with `before` (genuine only)."""
    R_aug, M_aug = inject(R, M, make_profiles(spec, kn, R.shape[1]))
    after = make_model().fit(R_aug, M_aug)
    return compare_models(before, after, R, M, spec, kn, k)
