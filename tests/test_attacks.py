from pathlib import Path

import numpy as np
import pytest

from recattack.attacks import (
    AttackSpec,
    attack_size_percent,
    choose_targets,
    inject,
    knowledge_from,
    make_profiles,
    n_fakes_for,
    pick_genre,
    selected_items,
)
from recattack.baselines import DampedItemMean
from recattack.data import load_genres
from recattack.impact import measure_attack, segment_users

ML_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-100k"


def world(n_u=40, n_i=30, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.random((n_u, n_i)) < 0.4
    M[:, 0] = True
    R = np.where(M, rng.integers(1, 6, size=(n_u, n_i)), 0).astype(np.float32)
    genres = np.zeros((n_i, 3), dtype=bool)
    genres[:, 0] = True  # 'unknown'
    genres[5:20, 1] = True
    genres[:, 2] = np.arange(n_i) % 2 == 0
    return R, M, genres


def test_knowledge_matches_direct_statistics():
    R, M, g = world()
    kn = knowledge_from(R, M, g)
    i = 3
    vals = R[M[:, i], i]
    assert kn.item_mean[i] == pytest.approx(vals.mean())
    assert kn.item_std[i] == pytest.approx(vals.std(ddof=1))
    assert kn.popularity[i] == M[:, i].sum()


@pytest.mark.parametrize("kind", ["random", "average", "bandwagon", "segment"])
def test_profile_structure(kind):
    R, M, g = world()
    kn = knowledge_from(R, M, g)
    spec = AttackSpec(kind, target=7, n_fakes=6, filler_size=10, n_selected=3, genre=1, seed=4)
    P = make_profiles(spec, kn, R.shape[1])
    sel = selected_items(spec, kn)
    assert P.shape == (6, 30)
    assert (P[:, 7] == 5).all()
    assert ((P[:, sel] == 5).all()) if sel.size else True
    assert P[P > 0].min() >= 1 and P.max() <= 5
    assert ((P > 0).sum(axis=1) == 10 + 1 + sel.size).all()  # filler + target + selected
    if kind == "segment":
        filler = (P > 0) & (P != 5)
        assert (P[filler] == 1).all() and filler.any()


def test_nuke_rates_target_one_and_seed_controls_randomness():
    R, M, g = world()
    kn = knowledge_from(R, M, g)
    a = make_profiles(AttackSpec("random", 7, 4, 8, direction="nuke", seed=1), kn, 30)
    b = make_profiles(AttackSpec("random", 7, 4, 8, direction="nuke", seed=1), kn, 30)
    c = make_profiles(AttackSpec("random", 7, 4, 8, direction="nuke", seed=2), kn, 30)
    assert (a[:, 7] == 1).all() and np.array_equal(a, b) and not np.array_equal(a, c)


def test_bandwagon_selects_most_popular_items_excluding_target():
    R, M, g = world()
    kn = knowledge_from(R, M, g)
    top = np.argsort(-kn.popularity, kind="stable")
    target = int(top[0])  # the most popular item cannot be selected for its own push
    sel = selected_items(AttackSpec("bandwagon", target, 2, 5, n_selected=3), kn)
    assert target not in sel and list(sel) == list(top[1:4])


def test_segment_selects_only_from_the_chosen_genre():
    R, M, g = world()
    kn = knowledge_from(R, M, g)
    sel = selected_items(AttackSpec("segment", 0, 2, 5, n_selected=3, genre=1), kn)
    assert g[sel, 1].all()


def test_spec_validation_and_error_paths():
    with pytest.raises(ValueError):
        AttackSpec("bogus", 0, 1)
    with pytest.raises(ValueError):
        AttackSpec("random", 0, 0)
    with pytest.raises(ValueError, match="genre"):
        AttackSpec("segment", 0, 1)
    R, M, g = world()
    kn = knowledge_from(R, M, g)
    with pytest.raises(ValueError, match="filler_size"):
        make_profiles(AttackSpec("random", 0, 1, filler_size=500), kn, 30)
    with pytest.raises(ValueError, match="genre data"):
        selected_items(AttackSpec("segment", 0, 1, 5, genre=1), knowledge_from(R, M))


def test_pick_genre_skips_unknown_and_tiny_genres():
    R, M, g = world()
    kn = knowledge_from(R, M, g)
    assert pick_genre(kn, 10, 3) in (1, 2)
    with pytest.raises(ValueError, match="no usable genre"):
        pick_genre(kn, 0, 100)


def test_inject_appends_fake_rows_below_genuine_users():
    R, M, _ = world()
    P = np.ones((3, 30), dtype=np.float32)
    R2, M2 = inject(R, M, P)
    assert R2.shape == (43, 30) and M2[40:].all() and np.array_equal(R2[:40], R)
    with pytest.raises(ValueError, match="width"):
        inject(R, M, np.ones((3, 9)))


def test_sizes():
    assert attack_size_percent(200, 10_000) == pytest.approx(2.0)  # worked example from the class deck
    assert n_fakes_for(0.03, 943) == 29 and n_fakes_for(1e-9, 10) == 1


def test_choose_targets_in_range_and_deterministic():
    R, M, _ = world(n_u=200, n_i=60)
    t1, t2 = choose_targets(M, 5, lo=60, hi=100, seed=3), choose_targets(M, 5, lo=60, hi=100, seed=3)
    assert np.array_equal(t1, t2) and ((M.sum(0)[t1] >= 60) & (M.sum(0)[t1] <= 100)).all()
    with pytest.raises(ValueError, match="only"):
        choose_targets(M, 5, lo=10_000, hi=20_000)


def test_slide_four_arithmetic_through_the_real_pipeline():
    """Five genuine ratings 4,5,3,4,3 (mean 3.8) plus three fake 5s -> 4.25, the number on slide 4."""
    n_u, n_i = 8, 6
    R = np.zeros((n_u, n_i), dtype=np.float32)
    R[:5, 0] = [4, 5, 3, 4, 3]  # the target item
    R[:, 1:] = 3
    M = R > 0
    kn = knowledge_from(R, M)
    spec = AttackSpec("random", target=0, n_fakes=3, filler_size=2, seed=0)
    make = lambda: DampedItemMean(damp=0.0)
    before = make().fit(R, M)
    res = measure_attack(make, before, R, M, spec, kn, k=3)
    assert before.predictions()[7, 0] == pytest.approx(3.8)
    assert res["pred_shift"] == pytest.approx(0.45)
    assert res["n_users"] == 3  # only the genuine users who never rated the target


def test_measure_attack_raises_if_nobody_can_be_evaluated():
    R = np.full((4, 5), 3, dtype=np.float32)
    M = np.ones_like(R, dtype=bool)
    kn = knowledge_from(R, M)
    make = lambda: DampedItemMean(damp=0.0)
    with pytest.raises(ValueError, match="already rated"):
        measure_attack(make, make().fit(R, M), R, M, AttackSpec("random", 0, 1, 2), kn)


def test_segment_users():
    R = np.array([[5, 4, 0], [5, 3, 0], [0, 5, 5]], dtype=np.float32)
    M = R > 0
    assert list(segment_users(R, M, np.array([0, 1]), thr=4)) == [0]
    assert segment_users(R, M, np.empty(0, dtype=int)).size == 0


@pytest.mark.skipif(not (ML_DIR / "u.item").is_file(), reason="ml-100k not downloaded")
def test_real_genres_load():
    flags, names = load_genres(ML_DIR)
    assert flags.shape == (1682, 19) and names[0] == "unknown" and "Action" in names
