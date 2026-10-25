"""Evaluation harness: fit on a training table, score on a held-out table, tune on a validation tail."""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
import pandas as pd

from .base import Recommender
from .data import Split, per_user_temporal_split, to_dense
from .metrics import mae, ranking_metrics, rmse

log = logging.getLogger(__name__)


def evaluate_model(
    model: Recommender,
    train_mask: np.ndarray,
    test: pd.DataFrame,
    k: int = 10,
    rel_threshold: int = 4,
    n_users: int | None = None,
) -> dict[str, float]:
    """Rating accuracy (RMSE, MAE) on every test pair, plus full-catalogue top-k ranking
    metrics where test ratings >= rel_threshold count as relevant. `n_users` keeps only the
    first n rows of the model, for models that also hold appended fake profiles."""
    n = n_users if n_users is not None else model.predictions().shape[0]
    pred = model.predictions()[:n]
    u, i = test["user"].to_numpy(), test["item"].to_numpy()
    truth = test["rating"].to_numpy(dtype=np.float64)
    out = {"rmse": rmse(pred[u, i], truth), "mae": mae(pred[u, i], truth)}
    relevant = np.zeros(pred.shape, dtype=bool)
    keep = truth >= rel_threshold
    relevant[u[keep], i[keep]] = True
    out.update(ranking_metrics(model.scores()[:n], train_mask[:n], relevant, k=k))
    return out


def fit_evaluate(
    factory: Callable[[], Recommender],
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_users: int,
    n_items: int,
    k: int = 10,
) -> dict[str, float]:
    R, M = to_dense(train, n_users, n_items)
    t0 = time.perf_counter()
    model = factory().fit(R, M)
    fit_s = time.perf_counter() - t0
    res = evaluate_model(model, M, test, k=k)
    res["fit_seconds"] = fit_s
    return res


def select_by_validation(
    make: Callable[..., Recommender],
    grid: list[dict],
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    val_frac: float = 0.15,
    splitter: Callable[..., Split] = per_user_temporal_split,
) -> tuple[dict, list[dict]]:
    """Pick the grid point with the lowest validation RMSE.

    Validation is the most recent `val_frac` of the training data (by the same splitter as the
    train/test split), so tuning never sees the test period. Returns (best_params, per-point results).
    """
    if not grid:
        raise ValueError("empty hyper-parameter grid")
    inner = splitter(train, test_frac=val_frac)
    R, M = to_dense(inner.train, n_users, n_items)
    rows = []
    for params in grid:
        model = make(**params).fit(R, M)
        pred = model.predictions()
        u, i = inner.test["user"].to_numpy(), inner.test["item"].to_numpy()
        score = rmse(pred[u, i], inner.test["rating"].to_numpy())
        rows.append({"params": params, "val_rmse": score})
        log.info("tune %s %s -> val_rmse=%.4f", make.__name__, params, score)
    best = min(rows, key=lambda r: r["val_rmse"])
    return best["params"], rows
