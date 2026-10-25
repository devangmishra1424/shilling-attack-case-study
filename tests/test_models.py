import numpy as np
import pytest

from recattack.baselines import DampedItemMean, GlobalMean, Popularity, RandomRanker
from recattack.knn import KNNRecommender
from recattack.mf import ALSMatrixFactorization, BiasOnly, fit_biases


def random_matrix(n_u=18, n_i=14, density=0.5, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.random((n_u, n_i)) < density
    M[:, 0] = True  # every item column and user row has at least one rating
    M[0, :] = True
    R = np.where(M, rng.integers(1, 6, size=(n_u, n_i)), 0).astype(np.float32)
    return R, M


def low_rank_matrix(seed=1, n_u=60, n_i=40, rank=3, density=0.4):
    rng = np.random.default_rng(seed)
    full = rng.standard_normal((n_u, rank)) @ rng.standard_normal((rank, n_i))
    full = np.clip(np.round(3 + full), 1, 5)
    M = rng.random((n_u, n_i)) < density
    M[:, 0] = True
    M[0, :] = True
    return np.where(M, full, 0).astype(np.float32), M, full


# ---------- naive loop reference for the neighbourhood models ----------

def naive_knn_predict(R, M, mode, k, shrink):
    n_u, n_i = R.shape
    mu_g = R[M].mean()
    mu_u = np.array([R[u, M[u]].mean() if M[u].any() else mu_g for u in range(n_u)])
    C = np.where(M, R - mu_u[:, None], 0.0)

    def sim(a, b, X, Mx):
        co = (Mx[a] & Mx[b]).sum()
        na, nb = np.linalg.norm(X[a]), np.linalg.norm(X[b])
        if na == 0 or nb == 0:
            return 0.0
        return float(X[a] @ X[b]) / (na * nb) * co / (co + shrink) if co + shrink > 0 else 0.0

    pred = np.zeros_like(R, dtype=np.float64)
    for u in range(n_u):
        for i in range(n_i):
            if mode == "user":
                cands = [(sim(u, v, C, M), C[v, i]) for v in range(n_u) if v != u and M[v, i]]
            else:
                Xi, Mi = C.T, M.T
                cands = [(sim(i, j, Xi, Mi), C[u, j]) for j in range(n_i) if j != i and M[u, j]]
            cands = [(s, d) for s, d in cands if s > 0]
            cands.sort(key=lambda t: -t[0])
            if len(cands) > k:
                kth = cands[k - 1][0]
                cands = [(s, d) for s, d in cands if s >= kth]
            den = sum(s for s, _ in cands)
            dev = sum(s * d for s, d in cands) / den if den > 0 else 0.0
            pred[u, i] = mu_u[u] + dev
    return np.clip(pred, 1, 5)


@pytest.mark.parametrize("mode", ["user", "item"])
def test_knn_matches_naive_reference(mode):
    R, M = random_matrix()
    model = KNNRecommender(mode, k=4, shrink=3.0).fit(R, M)
    expected = naive_knn_predict(R, M, mode, k=4, shrink=3.0)
    np.testing.assert_allclose(model.predictions(), expected, atol=1e-4)


def test_knn_user_with_no_ratings_falls_back_to_global_mean():
    R, M = random_matrix()
    R[5], M[5] = 0, False
    pred = KNNRecommender("user", k=3).fit(R, M).predictions()
    assert np.allclose(pred[5], R[M].mean(), atol=1e-5)


def test_knn_rejects_bad_arguments():
    with pytest.raises(ValueError):
        KNNRecommender("both")
    with pytest.raises(ValueError):
        KNNRecommender("user", k=0)


# ---------- baselines ----------

def test_global_and_item_mean():
    R = np.array([[5, 0], [3, 1]], dtype=np.float32)
    M = R > 0
    assert np.allclose(GlobalMean().fit(R, M).predictions(), 3.0)
    p = DampedItemMean(damp=0.0).fit(R, M).predictions()
    assert np.allclose(p[:, 0], 4.0) and np.allclose(p[:, 1], 1.0)
    shrunk = DampedItemMean(damp=1.0).fit(R, M).predictions()
    assert shrunk[0, 1] == pytest.approx((1 + 3.0) / 2)  # pulled towards the global mean 3


def test_popularity_ranks_by_count_and_random_is_seeded():
    R, M = random_matrix()
    pop = Popularity().fit(R, M)
    assert np.array_equal(pop.scores()[0], M.sum(axis=0))
    a = RandomRanker(seed=1).fit(R, M).scores()
    b = RandomRanker(seed=1).fit(R, M).scores()
    c = RandomRanker(seed=2).fit(R, M).scores()
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_predictions_before_fit_raises():
    with pytest.raises(RuntimeError, match="not been fitted"):
        GlobalMean().predictions()


def test_fit_rejects_bad_inputs():
    with pytest.raises(ValueError, match="equal shape"):
        GlobalMean().fit(np.zeros((2, 2)), np.zeros((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="empty"):
        GlobalMean().fit(np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))


# ---------- biases and matrix factorisation ----------

def test_biases_recover_additive_structure():
    b_u = np.array([0.5, -0.5, 0.0, 1.0])
    b_i = np.array([-1.0, 0.0, 1.0])
    R = (3 + b_u[:, None] + b_i[None, :]).astype(np.float32)
    mu, bu, bi = fit_biases(R, np.ones_like(R, dtype=bool), reg_item=0.0, reg_user=0.0, n_iter=50)
    np.testing.assert_allclose(mu + bu[:, None] + bi[None, :], R, atol=1e-4)


def test_als_beats_bias_only_on_low_rank_data_and_loss_decreases():
    R, M, _ = low_rank_matrix()
    bias = BiasOnly().fit(R, M).predictions()
    als = ALSMatrixFactorization(n_factors=3, reg=0.02, n_iter=25, seed=0)
    pred = als.fit(R, M).predictions()
    rmse = lambda p: float(np.sqrt(np.mean((p[M] - R[M]) ** 2)))
    assert rmse(pred) < rmse(bias) - 0.05
    # ALS-WR guarantees the regularised objective never increases; raw train RMSE may wobble.
    obj = als.objective_history_
    assert obj[-1] < obj[0] and all(b <= a * (1 + 1e-9) for a, b in zip(obj, obj[1:]))
    assert als.train_rmse_history_[-1] < als.train_rmse_history_[0]


def test_als_is_deterministic_for_a_seed_and_varies_across_seeds():
    R, M, _ = low_rank_matrix()
    a = ALSMatrixFactorization(n_factors=3, n_iter=5, seed=3).fit(R, M).predictions()
    b = ALSMatrixFactorization(n_factors=3, n_iter=5, seed=3).fit(R, M).predictions()
    c = ALSMatrixFactorization(n_factors=3, n_iter=5, seed=4).fit(R, M).predictions()
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_als_cold_user_and_item_fall_back_to_biases_only():
    R, M, _ = low_rank_matrix()
    R[7], M[7] = 0, False
    als = ALSMatrixFactorization(n_factors=3, n_iter=5).fit(R, M)
    base = BiasOnly().fit(R, M).predictions()
    np.testing.assert_allclose(als.predictions()[7], base[7], atol=1e-5)


def test_als_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ALSMatrixFactorization(n_factors=0)
    with pytest.raises(ValueError):
        ALSMatrixFactorization(reg=0.0)
