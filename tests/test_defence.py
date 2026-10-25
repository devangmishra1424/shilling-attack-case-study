import numpy as np
import pytest

from recattack.attacks import AttackSpec, knowledge_from
from recattack.baselines import DampedItemMean
from recattack.detect import ProfileDetector, build_world, detection_metrics, split_users
from recattack.features import FEATURE_NAMES, mad_score, modified_z, profile_features, trust_from_mad
from recattack.impact import compare_models
from recattack.knn import KNNRecommender
from recattack.mf import ALSMatrixFactorization
from recattack.robust import RobustALS


def random_matrix(n_u=24, n_i=16, density=0.5, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.random((n_u, n_i)) < density
    M[:, 0] = True
    M[0, :] = True
    R = np.where(M, rng.integers(1, 6, size=(n_u, n_i)), 0).astype(np.float32)
    return R, M


# ---------- features and MAD ----------

def test_profile_features_known_values():
    R = np.array([[5, 5, 5, 0], [1, 3, 0, 0], [0, 0, 0, 0], [4, 0, 2, 5]], dtype=np.float32)
    M = R > 0
    F = profile_features(R, M, n_popular=2, k_sim=1)
    assert F.shape == (4, len(FEATURE_NAMES))
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert F[0, col["mean_rating"]] == pytest.approx(5.0) and F[0, col["var_rating"]] == pytest.approx(0.0)
    assert F[0, col["frac_max"]] == pytest.approx(1.0) and F[1, col["frac_min"]] == pytest.approx(0.5)
    assert F[0, col["log_n_ratings"]] == pytest.approx(np.log1p(3))
    assert (F[2] == 0).all()  # a user with no ratings gets zeros, not NaN
    assert np.isfinite(F).all()


def test_modified_z_reproduces_the_class_worked_example():
    x = np.array([2, 3, 3, 4, 4, 5, 20], dtype=float)[:, None]
    z = modified_z(x)[:, 0]
    assert z[-1] == pytest.approx(0.6745 * 16 / 1)  # 10.79 on the slide
    s = mad_score(x)
    assert s[-1] > 3.5 and (s[:-1] <= 3.5).all()


def test_modified_z_zero_mad_fallback_and_constant_feature():
    mostly_same = np.array([3, 3, 3, 3, 9], dtype=float)[:, None]  # MAD = 0
    z = modified_z(mostly_same)[:, 0]
    assert z[-1] > 3.5 and np.isfinite(z).all()
    assert (modified_z(np.full((5, 1), 2.0)) == 0).all()


def test_trust_from_mad():
    t = trust_from_mad(np.array([0.0, 3.5, 7.0, 35.0]))
    assert t.tolist() == pytest.approx([1.0, 1.0, 0.5, 0.1])


# ---------- robust / trust-weighted ALS ----------

def test_robust_als_without_options_equals_plain_als():
    R, M = random_matrix()
    a = ALSMatrixFactorization(n_factors=4, n_iter=6, seed=2).fit(R, M).predictions()
    b = RobustALS(n_factors=4, n_iter=6, seed=2).fit(R, M).predictions()
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_zero_trust_equals_removing_the_profile():
    R, M = random_matrix()
    trust = np.ones(R.shape[0])
    trust[5] = 0.0
    weighted = RobustALS(n_factors=4, n_iter=6, seed=2, user_trust=trust).fit(R, M).predictions()
    R2, M2 = R.copy(), M.copy()
    R2[5], M2[5] = 0, False
    removed = RobustALS(n_factors=4, n_iter=6, seed=2).fit(R2, M2).predictions()
    keep = np.arange(R.shape[0]) != 5
    np.testing.assert_allclose(weighted[keep], removed[keep], atol=1e-4)


def test_huber_loss_limits_a_single_wild_rating():
    rng = np.random.default_rng(0)
    R = np.full((30, 12), 3.0, dtype=np.float32)
    M = np.ones_like(R, dtype=bool)
    R += rng.integers(-1, 2, size=R.shape).astype(np.float32) * 0.0  # flat world, the outlier is the only signal
    R[4, 7] = 5.0
    plain = ALSMatrixFactorization(n_factors=3, reg=0.05, seed=0).fit(R, M).predictions()[4, 7]
    robust = RobustALS(n_factors=3, reg=0.05, seed=0, huber_delta=0.5).fit(R, M).predictions()[4, 7]
    assert abs(robust - 3.0) < abs(plain - 3.0)


def test_robust_als_rejects_bad_arguments():
    R, M = random_matrix()
    with pytest.raises(ValueError):
        RobustALS(huber_delta=0.0)
    with pytest.raises(ValueError):
        RobustALS(user_trust=np.array([1.5, 0.2]))
    with pytest.raises(ValueError, match="one entry per user"):
        RobustALS(user_trust=np.ones(3)).fit(R, M)


# ---------- trust-weighted kNN ----------

@pytest.mark.parametrize("mode", ["user", "item"])
def test_knn_with_full_trust_matches_no_trust(mode):
    R, M = random_matrix()
    a = KNNRecommender(mode, k=5, shrink=3.0).fit(R, M).predictions()
    b = KNNRecommender(mode, k=5, shrink=3.0, user_trust=np.ones(R.shape[0])).fit(R, M).predictions()
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("mode", ["user", "item"])
def test_zero_trust_user_cannot_influence_anyone_else(mode):
    R, M = random_matrix()
    trust = np.ones(R.shape[0])
    trust[3] = 0.0
    R2 = R.copy()
    R2[3, M[3]] = 5.0  # rewrite the zero-trust user's ratings entirely
    a = KNNRecommender(mode, k=5, shrink=3.0, user_trust=trust).fit(R, M).predictions()
    b = KNNRecommender(mode, k=5, shrink=3.0, user_trust=trust).fit(R2, M).predictions()
    keep = np.arange(R.shape[0]) != 3
    np.testing.assert_allclose(a[keep], b[keep], atol=1e-5)


def test_knn_trust_validation():
    with pytest.raises(ValueError):
        KNNRecommender("user", user_trust=np.array([-0.1, 1.0]))
    R, M = random_matrix()
    with pytest.raises(ValueError, match="one entry per user"):
        KNNRecommender("user", user_trust=np.ones(3)).fit(R, M)


# ---------- detection ----------

def test_detection_metrics_hand_computed():
    y = np.array([1, 1, 0, 0, 0], dtype=bool)
    score = np.array([0.9, 0.4, 0.8, 0.2, 0.1])
    m = detection_metrics(score, y, threshold=0.5)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5) and m["n_flagged"] == 2 and m["false_positives"] == 1
    assert 0.5 < m["auc"] < 1.0


def test_detection_metrics_nothing_flagged_and_single_class():
    y = np.array([1, 0, 0], dtype=bool)
    m = detection_metrics(np.array([0.1, 0.1, 0.1]), y, threshold=0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    with pytest.raises(ValueError, match="both"):
        detection_metrics(np.array([0.1, 0.2]), np.array([False, False]), 0.5)


@pytest.mark.parametrize("kind", ["rf", "lr"])
def test_profile_detector_separates_an_easy_problem(kind):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (80, 3)), rng.normal(4, 1, (20, 3))])
    y = np.r_[np.zeros(80), np.ones(20)]
    s = ProfileDetector(kind).fit(X, y).score(X)
    assert detection_metrics(s, y.astype(bool), 0.5)["auc"] > 0.99


def test_profile_detector_rejects_bad_input():
    with pytest.raises(ValueError):
        ProfileDetector("svm")
    with pytest.raises(ValueError, match="both"):
        ProfileDetector("lr").fit(np.zeros((4, 2)), np.zeros(4))


def test_build_world_and_user_halves():
    R, M = random_matrix()
    kn = knowledge_from(R, M)
    w = build_world(R, M, kn, AttackSpec("random", 3, 4, 5, seed=1))
    assert w.R.shape[0] == R.shape[0] + 4 and w.is_fake.sum() == 4 and not w.is_fake[: R.shape[0]].any()
    a, b = split_users(25, seed=3)
    assert len(a) + len(b) == 25 and not set(a) & set(b)


def test_compare_models_can_be_restricted_to_a_user_subset():
    R = np.zeros((6, 4), dtype=np.float32)
    R[:3, 0] = [4, 5, 3]
    R[:, 1:] = 3
    M = R > 0
    kn = knowledge_from(R, M)
    spec = AttackSpec("random", 0, 2, 2)
    before = DampedItemMean(0.0).fit(R, M)
    w = build_world(R, M, kn, spec)
    after = DampedItemMean(0.0).fit(w.R, w.M)
    full = compare_models(before, after, R, M, spec, kn, k=2)
    sub = compare_models(before, after, R, M, spec, kn, k=2, users=np.array([3, 4]))
    assert full["n_users"] == 3 and sub["n_users"] == 2
    with pytest.raises(ValueError, match="already rated"):
        compare_models(before, after, R, M, spec, kn, k=2, users=np.array([0, 1]))
