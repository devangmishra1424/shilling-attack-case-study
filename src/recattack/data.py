"""MovieLens 100K loading, validation and temporal splitting.

Everything downstream assumes ids are 0-based and contiguous, ratings are integers
in 1..5, and there is at most one rating per (user, item).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_USERS, N_ITEMS, N_RATINGS = 943, 1682, 100_000
RATING_MIN, RATING_MAX = 1, 5
COLUMNS = ["user", "item", "rating", "ts"]


class DataError(ValueError):
    """The ratings table violates an assumption the experiments rely on."""


def validate_ratings(df: pd.DataFrame, n_users: int, n_items: int) -> None:
    """Raise DataError unless df is a clean 0-based ratings table. Interior code assumes this."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"missing columns: {missing}")
    if df.empty:
        raise DataError("ratings table is empty")
    if df[COLUMNS].isna().any().any():
        raise DataError("ratings table contains missing values")
    if (df["user"] < 0).any() or (df["user"] >= n_users).any():
        raise DataError(f"user ids must be in [0, {n_users})")
    if (df["item"] < 0).any() or (df["item"] >= n_items).any():
        raise DataError(f"item ids must be in [0, {n_items})")
    if (df["rating"] < RATING_MIN).any() or (df["rating"] > RATING_MAX).any():
        raise DataError(f"ratings must be in [{RATING_MIN}, {RATING_MAX}]")
    if df.duplicated(subset=["user", "item"]).any():
        raise DataError("duplicate (user, item) pairs found")


def load_ml100k(data_dir: str | Path) -> pd.DataFrame:
    """Load u.data from an extracted ml-100k folder and return a validated 0-based table."""
    path = Path(data_dir) / "u.data"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; extract ml-100k.zip first")
    df = pd.read_csv(path, sep="\t", names=COLUMNS, dtype="int64")
    df["user"] -= 1
    df["item"] -= 1
    validate_ratings(df, N_USERS, N_ITEMS)
    if len(df) != N_RATINGS:
        raise DataError(f"expected {N_RATINGS} ratings, found {len(df)}; file may be corrupt")
    return df


def load_genres(data_dir: str | Path) -> tuple[np.ndarray, list[str]]:
    """Item x genre boolean matrix (0-based item ids) and genre names, from u.item and u.genre."""
    base = Path(data_dir)
    item_path, genre_path = base / "u.item", base / "u.genre"
    for p in (item_path, genre_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found")
    names = [ln.split("|")[0] for ln in genre_path.read_text(encoding="latin-1").splitlines() if ln.strip()]
    items = pd.read_csv(item_path, sep="|", header=None, encoding="latin-1")
    flags = items.iloc[:, 5:].to_numpy()
    if len(items) != N_ITEMS or flags.shape[1] != len(names):
        raise DataError("u.item does not match the expected item count or genre list")
    if not np.isin(flags, (0, 1)).all():
        raise DataError("genre flags must be 0 or 1")
    return flags.astype(bool), names


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff_ts: int | None  # None for the per-user split, which has no single cutoff


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2) -> Split:
    """Global time split: ratings before the cutoff train, ratings at or after it test.

    One global cutoff (not per-user) so no test-period behaviour reaches training. Timestamp
    ties go to the test side, so the realised test fraction can exceed test_frac slightly.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    cutoff = int(np.quantile(df["ts"].to_numpy(), 1.0 - test_frac, method="lower"))
    train = df[df["ts"] < cutoff].reset_index(drop=True)
    test = df[df["ts"] >= cutoff].reset_index(drop=True)
    if train.empty or test.empty:
        raise DataError("temporal split produced an empty side; timestamps may be degenerate")
    return Split(train=train, test=test, cutoff_ts=cutoff)


def per_user_temporal_split(df: pd.DataFrame, test_frac: float = 0.2) -> Split:
    """Each user's most recent test_frac of ratings (at least one) go to test, the rest to train.

    This is the main protocol: in ML-100K most users joined within a short window, so a single
    global cutoff leaves ~85% of test ratings with users who have no training history (see
    README). Trade-off: a user's test period can overlap other users' training periods, so
    some collaborative signal from 'the future' can reach training. Users with one rating
    stay in train only.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    ordered = df.sort_values(["user", "ts", "item"], kind="stable").reset_index(drop=True)
    grp = ordered.groupby("user")["item"]
    n = grp.transform("size").to_numpy()
    pos = grp.cumcount().to_numpy()
    n_test = np.minimum(np.maximum(1, np.rint(test_frac * n).astype(np.int64)), n - 1)
    is_test = pos >= n - n_test
    train = ordered[~is_test].reset_index(drop=True)
    test = ordered[is_test].reset_index(drop=True)
    if train.empty or test.empty:
        raise DataError("per-user split produced an empty side")
    return Split(train=train, test=test, cutoff_ts=None)


def to_dense(df: pd.DataFrame, n_users: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, M): float32 rating matrix with 0 for missing, and a boolean observed mask."""
    validate_ratings(df, n_users, n_items)
    R = np.zeros((n_users, n_items), dtype=np.float32)
    R[df["user"].to_numpy(), df["item"].to_numpy()] = df["rating"].to_numpy(dtype=np.float32)
    return R, R > 0


def unseen_fractions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    """Share of test ratings whose user or item never appears in train (cold-start exposure)."""
    seen_u = set(train["user"].unique())
    seen_i = set(train["item"].unique())
    return {
        "test_unseen_user": float((~test["user"].isin(seen_u)).mean()),
        "test_unseen_item": float((~test["item"].isin(seen_i)).mean()),
    }
