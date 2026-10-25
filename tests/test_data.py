from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recattack.data import (
    DataError,
    load_ml100k,
    per_user_temporal_split,
    temporal_split,
    to_dense,
    unseen_fractions,
    validate_ratings,
)

ML_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-100k"


def table(rows):
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])


def test_validate_accepts_clean_table():
    validate_ratings(table([(0, 0, 5, 1), (0, 1, 1, 2), (1, 0, 3, 3)]), 2, 2)


@pytest.mark.parametrize(
    "rows,msg",
    [
        ([(0, 0, 6, 1)], "ratings must be"),
        ([(0, 0, 0, 1)], "ratings must be"),
        ([(0, 0, 3, 1), (0, 0, 4, 2)], "duplicate"),
        ([(-1, 0, 3, 1)], "user ids"),
        ([(0, 5, 3, 1)], "item ids"),
    ],
)
def test_validate_rejects_bad_tables(rows, msg):
    with pytest.raises(DataError, match=msg):
        validate_ratings(table(rows), 2, 2)


def test_validate_rejects_empty_and_missing_columns():
    with pytest.raises(DataError, match="empty"):
        validate_ratings(table([]), 2, 2)
    with pytest.raises(DataError, match="missing columns"):
        validate_ratings(pd.DataFrame({"user": [0]}), 2, 2)


def test_load_missing_directory_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ml100k(tmp_path)


def test_load_truncated_file_is_rejected(tmp_path):
    (tmp_path / "u.data").write_text("1\t1\t5\t100\n2\t1\t3\t101\n")
    with pytest.raises(DataError, match="expected 100000"):
        load_ml100k(tmp_path)


def test_temporal_split_has_no_time_leakage():
    ts = np.arange(100)
    df = table([(i % 5, i % 7, 1 + i % 5, t) for i, t in enumerate(ts)]).drop_duplicates(["user", "item"])
    s = temporal_split(df, 0.2)
    assert s.train["ts"].max() < s.cutoff_ts <= s.test["ts"].min()
    assert len(s.train) + len(s.test) == len(df)


def test_temporal_split_rejects_bad_fraction_and_degenerate_time():
    df = table([(0, 0, 3, 5), (1, 1, 3, 5)])
    with pytest.raises(ValueError):
        temporal_split(df, 1.5)
    with pytest.raises(DataError, match="empty side"):
        temporal_split(df, 0.5)  # every timestamp equal, so nothing precedes the cutoff


def test_to_dense_places_ratings_and_mask():
    R, M = to_dense(table([(0, 1, 4, 1), (1, 0, 2, 2)]), 2, 2)
    assert R[0, 1] == 4 and R[1, 0] == 2 and M.sum() == 2 and not M[0, 0]


def test_unseen_fractions():
    train = table([(0, 0, 3, 1)])
    test = table([(0, 0, 3, 2), (1, 0, 3, 3), (0, 1, 3, 4), (1, 1, 3, 5)])
    f = unseen_fractions(train, test)
    assert f == {"test_unseen_user": 0.5, "test_unseen_item": 0.5}


@pytest.mark.skipif(not (ML_DIR / "u.data").is_file(), reason="ml-100k not downloaded")
def test_real_dataset_matches_published_counts():
    df = load_ml100k(ML_DIR)
    assert len(df) == 100_000 and df["user"].nunique() == 943 and df["item"].nunique() == 1682


def test_per_user_split_holds_out_each_users_latest_ratings():
    rows = [(u, i, 3, 100 * u + i) for u in range(3) for i in range(10)]  # user u rates items 0..9 in order
    df = table(rows).sample(frac=1.0, random_state=0)  # shuffled input must not matter
    s = per_user_temporal_split(df, 0.2)
    assert len(s.test) == 6 and len(s.train) == 24
    for u in range(3):
        tr, te = s.train[s.train.user == u], s.test[s.test.user == u]
        assert len(te) == 2 and tr["ts"].max() < te["ts"].min()
    assert s.cutoff_ts is None
    assert not s.train.merge(s.test, on=["user", "item"]).shape[0]


def test_per_user_split_edge_cases():
    df = table([(0, 0, 3, 1), (1, 0, 3, 1), (1, 1, 3, 2)])
    s = per_user_temporal_split(df, 0.5)
    assert set(s.train.user) == {0, 1} and set(s.test.user) == {1}  # single-rating user stays in train
    with pytest.raises(ValueError):
        per_user_temporal_split(df, 0.0)
    with pytest.raises(DataError, match="empty side"):
        per_user_temporal_split(table([(0, 0, 3, 1)]), 0.2)


def test_per_user_split_gives_every_real_user_training_history():
    if not (ML_DIR / "u.data").is_file():
        pytest.skip("ml-100k not downloaded")
    s = per_user_temporal_split(load_ml100k(ML_DIR), 0.2)
    assert s.train["user"].nunique() == 943
    assert unseen_fractions(s.train, s.test)["test_unseen_user"] == 0.0
