"""Robust and trust-weighted ALS matrix factorisation.

Two defences from the class notes, usable alone or together:
  trust weighting  each user's ratings are weighted by trust in [0, 1] (0 = ignore the profile)
  robust loss      Huber loss via iteratively re-weighted least squares: a rating whose residual
                   exceeds delta gets weight delta / |residual|, so extreme injected ratings pull less
With neither option the model equals ALSMatrixFactorization for the same seed.
"""
from __future__ import annotations

import numpy as np

from .base import Recommender
from .mf import fit_biases


class RobustALS(Recommender):
    name = "RobustALS"

    def __init__(
        self,
        n_factors: int = 20,
        reg: float = 0.1,
        n_iter: int = 15,
        seed: int = 0,
        huber_delta: float | None = None,
        user_trust: np.ndarray | None = None,
        irls_rounds: int = 3,
        reg_item: float = 10.0,
        reg_user: float = 15.0,
    ) -> None:
        super().__init__()
        if n_factors < 1 or n_iter < 1 or reg <= 0 or irls_rounds < 1:
            raise ValueError("need n_factors >= 1, n_iter >= 1, irls_rounds >= 1 and reg > 0")
        if huber_delta is not None and huber_delta <= 0:
            raise ValueError("huber_delta must be > 0")
        if user_trust is not None and (np.any(user_trust < 0) or np.any(user_trust > 1)):
            raise ValueError("user_trust must lie in [0, 1]")
        self.n_factors, self.reg, self.n_iter, self.seed = n_factors, reg, n_iter, seed
        self.huber_delta, self.irls_rounds = huber_delta, irls_rounds
        self.reg_item, self.reg_user = reg_item, reg_user
        self.user_trust = None if user_trust is None else np.asarray(user_trust, dtype=np.float64)

    def _solve_side(self, fixed: np.ndarray, E: np.ndarray, M: np.ndarray, Wt: np.ndarray, out: np.ndarray) -> None:
        eye = np.eye(self.n_factors)
        for r in range(out.shape[0]):
            idx = np.flatnonzero(M[r])
            if idx.size == 0:
                out[r] = 0.0
                continue
            w = Wt[r, idx]
            sw = w.sum()
            if sw < 1e-9:
                out[r] = 0.0
                continue
            F = fixed[idx]
            Fw = F * w[:, None]
            out[r] = np.linalg.solve(Fw.T @ F + self.reg * sw * eye, Fw.T @ E[r, idx])

    def _fit(self, R, M):
        n_users, n_items = R.shape
        trust = np.ones(n_users) if self.user_trust is None else self.user_trust
        if trust.shape != (n_users,):
            raise ValueError("user_trust must have one entry per user")
        W0 = np.where(M, trust[:, None], 0.0)
        W = W0.copy()
        rng = np.random.default_rng(self.seed)
        P = 0.1 * rng.standard_normal((n_users, self.n_factors))
        Q = 0.1 * rng.standard_normal((n_items, self.n_factors))
        rounds = self.irls_rounds if self.huber_delta is not None else 1
        Rf = R.astype(np.float64)
        for rnd in range(rounds):
            mu, b_u, b_i = fit_biases(R, M, self.reg_item, self.reg_user, weights=W)
            base = mu + b_u[:, None] + b_i[None, :]
            E = np.where(M, Rf - base, 0.0)
            for _ in range(self.n_iter if rnd == 0 else max(3, self.n_iter // 3)):
                self._solve_side(Q, E, M, W, P)
                self._solve_side(P, E.T, M.T, W.T, Q)
            if self.huber_delta is not None:
                resid = np.abs(np.where(M, E - P @ Q.T, 0.0))
                huber = np.where(resid <= self.huber_delta, 1.0, self.huber_delta / np.maximum(resid, 1e-12))
                W = W0 * huber
        return base + P @ Q.T
