"""Accuracy, ranking and attack-impact metrics.

Conventions: scores are (n_users, n_items), higher is better. `train_mask` marks items a user
already rated; those are never candidates. Ties are broken by lower item id, which matches a
stable descending sort, so every function here ranks identically.
"""
from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError("pred and true must be non-empty and the same shape")
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError("pred and true must be non-empty and the same shape")
    return float(np.mean(np.abs(pred - true)))


def _top_k(scores: np.ndarray, train_mask: np.ndarray, k: int) -> np.ndarray:
    masked = np.where(train_mask, -np.inf, scores)
    return np.argsort(-masked, axis=1, kind="stable")[:, :k]


def ranking_metrics(
    scores: np.ndarray, train_mask: np.ndarray, relevant: np.ndarray, k: int = 10
) -> dict[str, float]:
    """Precision@k, Recall@k and NDCG@k over users with at least one relevant held-out item.

    Full-catalogue ranking (no negative sampling). `relevant` is a boolean user x item matrix
    of held-out relevant items; it must not overlap `train_mask`.
    """
    if not (scores.shape == train_mask.shape == relevant.shape):
        raise ValueError("scores, train_mask and relevant must share one shape")
    if k < 1 or k > scores.shape[1]:
        raise ValueError("k must be in [1, n_items]")
    if (relevant & train_mask).any():
        raise ValueError("relevant items overlap the training mask")
    n_rel = relevant.sum(axis=1)
    users = np.flatnonzero(n_rel > 0)
    if users.size == 0:
        raise ValueError("no user has a relevant held-out item")
    top = _top_k(scores[users], train_mask[users], k)
    hits = np.take_along_axis(relevant[users], top, axis=1).astype(np.float64)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discount).sum(axis=1)
    ideal_len = np.minimum(n_rel[users], k)
    idcg = np.array([discount[:n].sum() for n in ideal_len])
    return {
        f"precision@{k}": float(hits.sum(axis=1).mean() / k),
        f"recall@{k}": float((hits.sum(axis=1) / n_rel[users]).mean()),
        f"ndcg@{k}": float((dcg / idcg).mean()),
        "n_eval_users": int(users.size),
    }


def item_ranks(scores: np.ndarray, train_mask: np.ndarray, item: int, users: np.ndarray) -> np.ndarray:
    """1-based rank of `item` in each listed user's candidate list. Users who already rated
    the item are not valid and raise, so callers cannot silently mix them in."""
    users = np.asarray(users)
    if train_mask[users, item].any():
        raise ValueError("some users already rated the target item; filter them out first")
    s = scores[users]
    s_t = s[:, item][:, None]
    idx = np.arange(scores.shape[1])[None, :]
    ahead = (s > s_t) | ((s == s_t) & (idx < item))
    ahead &= ~train_mask[users]
    return 1 + ahead.sum(axis=1)


def hit_ratio(scores: np.ndarray, train_mask: np.ndarray, item: int, users: np.ndarray, n: int = 10) -> float:
    """Fraction of users whose top-n list (unrated items only) contains `item`."""
    return float((item_ranks(scores, train_mask, item, users) <= n).mean())


def prediction_shift(before: np.ndarray, after: np.ndarray, item: int, users: np.ndarray) -> float:
    """Mean change in predicted rating of `item` over `users` caused by an attack."""
    users = np.asarray(users)
    return float((after[users, item] - before[users, item]).mean())
