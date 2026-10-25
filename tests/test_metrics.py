import numpy as np
import pytest

from recattack.metrics import hit_ratio, item_ranks, mae, prediction_shift, ranking_metrics, rmse


def test_rmse_mae_known_values():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)


def test_rmse_rejects_shape_mismatch_and_empty():
    with pytest.raises(ValueError):
        rmse([1, 2], [1])
    with pytest.raises(ValueError):
        mae([], [])


def test_ranking_metrics_hand_computed():
    # candidates after masking item 0: ranked [1, 2, 3, 4]; top-2 = [1, 2]; relevant = {1, 3}
    scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])
    train = np.array([[True, False, False, False, False]])
    rel = np.array([[False, True, False, True, False]])
    m = ranking_metrics(scores, train, rel, k=2)
    idcg = 1 + 1 / np.log2(3)
    assert m["precision@2"] == pytest.approx(0.5)
    assert m["recall@2"] == pytest.approx(0.5)
    assert m["ndcg@2"] == pytest.approx(1.0 / idcg)
    assert m["n_eval_users"] == 1


def test_ranking_metrics_skips_users_without_relevant_items():
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    train = np.zeros((2, 3), dtype=bool)
    rel = np.array([[True, False, False], [False, False, False]])
    m = ranking_metrics(scores, train, rel, k=1)
    assert m["n_eval_users"] == 1 and m["precision@1"] == 1.0


def test_ranking_metrics_failure_paths():
    s = np.zeros((1, 3))
    t = np.array([[True, False, False]])
    with pytest.raises(ValueError, match="overlap"):
        ranking_metrics(s, t, np.array([[True, False, False]]), k=1)
    with pytest.raises(ValueError, match="no user"):
        ranking_metrics(s, t, np.zeros((1, 3), dtype=bool), k=1)
    with pytest.raises(ValueError, match="k must"):
        ranking_metrics(s, t, np.array([[False, True, False]]), k=9)
    with pytest.raises(ValueError, match="one shape"):
        ranking_metrics(s, t[:, :2], np.array([[False, True, False]]), k=1)


def test_item_ranks_agree_with_stable_sort_including_ties():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 4, size=(30, 12)).astype(np.float32)  # many ties on purpose
    train = rng.random((30, 12)) < 0.3
    item = 5
    users = np.flatnonzero(~train[:, item])
    ranks = item_ranks(scores, train, item, users)
    masked = np.where(train, -np.inf, scores)
    order = np.argsort(-masked, axis=1, kind="stable")
    expected = np.array([np.flatnonzero(order[u] == item)[0] + 1 for u in users])
    assert (ranks == expected).all()


def test_item_ranks_rejects_users_who_already_rated_target():
    scores = np.zeros((2, 3))
    train = np.array([[False, True, False], [False, False, False]])
    with pytest.raises(ValueError, match="already rated"):
        item_ranks(scores, train, 1, np.array([0, 1]))


def test_hit_ratio_and_prediction_shift():
    scores = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.3], [0.1, 0.2, 0.9]])
    train = np.zeros((3, 3), dtype=bool)
    users = np.array([0, 1, 2])
    assert hit_ratio(scores, train, item=1, users=users, n=1) == pytest.approx(1 / 3)
    assert hit_ratio(scores, train, item=1, users=users, n=2) == pytest.approx(2 / 3)
    before = np.full((3, 3), 3.0)
    after = before.copy()
    after[:, 1] = [4.0, 5.0, 3.0]
    assert prediction_shift(before, after, 1, users) == pytest.approx(1.0)
