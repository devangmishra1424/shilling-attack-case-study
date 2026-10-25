"""Bias baseline and ALS matrix factorisation (ALS-WR) in plain numpy."""
from __future__ import annotations

import numpy as np

from .base import Recommender


def fit_biases(
    R: np.ndarray,
    M: np.ndarray,
    reg_item: float = 10.0,
    reg_user: float = 15.0,
    n_iter: int = 10,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Regularised baseline predictor mu + b_u + b_i (Koren 2009), by alternating means.

    `weights` (n_users, n_items, non-negative, only read where M is True) down-weights ratings,
    e.g. by user trust or a robust loss; None means every observed rating counts 1.
    """
    if reg_item < 0 or reg_user < 0:
        raise ValueError("regularisation must be >= 0")
    Rf = R.astype(np.float64)
    Mf = M.astype(np.float64) if weights is None else np.where(M, weights, 0.0).astype(np.float64)
    if Mf.sum() <= 0:
        raise ValueError("total rating weight is zero")
    mu = float((Rf * Mf).sum() / Mf.sum())
    n_i, n_u = Mf.sum(axis=0), Mf.sum(axis=1)
    b_u = np.zeros(R.shape[0])
    b_i = np.zeros(R.shape[1])
    for _ in range(n_iter):
        b_i = ((Rf - mu - b_u[:, None]) * Mf).sum(axis=0) / (reg_item + n_i)
        b_u = ((Rf - mu - b_i[None, :]) * Mf).sum(axis=1) / (reg_user + n_u)
    return mu, b_u, b_i


class BiasOnly(Recommender):
    name = "BiasOnly"

    def __init__(self, reg_item: float = 10.0, reg_user: float = 15.0) -> None:
        super().__init__()
        self.reg_item, self.reg_user = reg_item, reg_user

    def _fit(self, R, M):
        mu, b_u, b_i = fit_biases(R, M, self.reg_item, self.reg_user)
        return mu + b_u[:, None] + b_i[None, :]


class ALSMatrixFactorization(Recommender):
    """Biased MF fitted by alternating least squares with count-weighted L2 (ALS-WR).

    Biases are estimated first and held fixed; the factors model the residual. Factor
    initialisation is the only source of randomness, controlled by `seed`.
    """

    name = "ALS-MF"

    def __init__(
        self,
        n_factors: int = 20,
        reg: float = 0.1,
        n_iter: int = 15,
        seed: int = 0,
        reg_item: float = 10.0,
        reg_user: float = 15.0,
    ) -> None:
        super().__init__()
        if n_factors < 1 or n_iter < 1 or reg <= 0:
            raise ValueError("need n_factors >= 1, n_iter >= 1 and reg > 0")
        self.n_factors, self.reg, self.n_iter, self.seed = n_factors, reg, n_iter, seed
        self.reg_item, self.reg_user = reg_item, reg_user
        self.train_rmse_history_: list[float] = []
        self.objective_history_: list[float] = []

    def _solve_side(self, fixed: np.ndarray, E: np.ndarray, M: np.ndarray, out: np.ndarray) -> None:
        """Update `out` row by row: each row solves a ridge regression on the rated entries."""
        k = self.n_factors
        eye = np.eye(k)
        for r in range(out.shape[0]):
            idx = np.flatnonzero(M[r])
            if idx.size == 0:
                out[r] = 0.0
                continue
            F = fixed[idx]
            A = F.T @ F + self.reg * idx.size * eye
            out[r] = np.linalg.solve(A, F.T @ E[r, idx])

    def _fit(self, R, M):
        mu, b_u, b_i = fit_biases(R, M, self.reg_item, self.reg_user)
        base = mu + b_u[:, None] + b_i[None, :]
        E = np.where(M, R.astype(np.float64) - base, 0.0)
        rng = np.random.default_rng(self.seed)
        P = 0.1 * rng.standard_normal((R.shape[0], self.n_factors))
        Q = 0.1 * rng.standard_normal((R.shape[1], self.n_factors))
        self.train_rmse_history_ = []
        self.objective_history_ = []
        n_obs = M.sum()
        n_u, n_i = M.sum(axis=1), M.sum(axis=0)
        for _ in range(self.n_iter):
            self._solve_side(Q, E, M, P)
            self._solve_side(P, E.T, M.T, Q)
            sq = float((np.where(M, E - P @ Q.T, 0.0) ** 2).sum())
            self.train_rmse_history_.append(float(np.sqrt(sq / n_obs)))
            # The quantity ALS-WR minimises; it is non-increasing, raw train RMSE need not be.
            penalty = self.reg * (float((n_u * (P**2).sum(axis=1)).sum()) + float((n_i * (Q**2).sum(axis=1)).sum()))
            self.objective_history_.append(sq + penalty)
        return base + P @ Q.T
