"""Shilling (profile injection) attack generators.

Profile = target item + selected items + filler items (the anatomy from the class notes).
The attacker knows only what is public: item popularity, item rating mean and spread, and
(for the segment attack) item genres. It never sees individual genuine ratings.

Rating rules by attack:
  random    filler ~ uniform integer 1..5                           (class deck)
  average   filler ~ round(N(item mean, item std)), clipped to 1..5
  bandwagon selected = most popular items rated 5; filler as random
  segment   selected = most popular items of one genre rated 5; filler rated 1
The target is rated 5 for a push and 1 for a nuke.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import RATING_MAX, RATING_MIN

KINDS = ("random", "average", "bandwagon", "segment")


@dataclass(frozen=True)
class Knowledge:
    """Public information the attacker is assumed to have."""

    item_mean: np.ndarray
    item_std: np.ndarray
    popularity: np.ndarray
    global_mean: float
    global_std: float
    genre_items: np.ndarray | None = None  # bool (n_items, n_genres)


def knowledge_from(R: np.ndarray, M: np.ndarray, genre_items: np.ndarray | None = None) -> Knowledge:
    vals = R[M].astype(np.float64)
    g_mean, g_std = float(vals.mean()), float(vals.std())
    n = M.sum(axis=0)
    Rf = R.astype(np.float64)
    mean = np.where(n > 0, (Rf * M).sum(axis=0) / np.maximum(n, 1), g_mean)
    var = np.where(n > 1, (((Rf - mean) ** 2) * M).sum(axis=0) / np.maximum(n - 1, 1), g_std**2)
    return Knowledge(mean, np.sqrt(var), n.astype(np.int64), g_mean, g_std, genre_items)


@dataclass(frozen=True)
class AttackSpec:
    kind: str
    target: int
    n_fakes: int
    filler_size: int = 84  # about 5% of the 1,682 items
    n_selected: int = 3
    direction: str = "push"
    genre: int | None = None  # required for the segment attack
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.direction not in ("push", "nuke"):
            raise ValueError("direction must be 'push' or 'nuke'")
        if self.n_fakes < 1 or self.filler_size < 1 or self.n_selected < 0:
            raise ValueError("need n_fakes >= 1, filler_size >= 1, n_selected >= 0")
        if self.kind == "segment" and self.genre is None:
            raise ValueError("segment attack needs a genre")


def attack_size_percent(n_fakes: int, n_genuine: int) -> float:
    """Class-deck metric: fake profiles / genuine profiles x 100."""
    return 100.0 * n_fakes / n_genuine


def n_fakes_for(fraction: float, n_genuine: int) -> int:
    return max(1, int(np.ceil(fraction * n_genuine)))


def selected_items(spec: AttackSpec, kn: Knowledge) -> np.ndarray:
    """Items the attacker rates 5 to look like real fans: popular items (bandwagon) or the
    most popular items of one genre (segment). Empty for random and average."""
    if spec.kind not in ("bandwagon", "segment") or spec.n_selected == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-kn.popularity, kind="stable")
    if spec.kind == "segment":
        if kn.genre_items is None:
            raise ValueError("segment attack needs genre data in Knowledge")
        order = order[kn.genre_items[order, spec.genre]]
    order = order[order != spec.target]
    if order.size < spec.n_selected:
        raise ValueError("not enough candidate items to select from")
    return order[: spec.n_selected]


def pick_genre(kn: Knowledge, target: int, n_selected: int) -> int:
    """Genre for a segment attack: the target's own genre with the most items, skipping
    'unknown' (column 0) and genres too small to supply n_selected other items."""
    if kn.genre_items is None:
        raise ValueError("no genre data")
    cands = [g for g in np.flatnonzero(kn.genre_items[target]) if g != 0]
    cands = [g for g in cands if kn.genre_items[:, g].sum() - 1 >= n_selected]
    if not cands:
        raise ValueError(f"item {target} has no usable genre for a segment attack")
    return int(max(cands, key=lambda g: kn.genre_items[:, g].sum()))


def make_profiles(spec: AttackSpec, kn: Knowledge, n_items: int) -> np.ndarray:
    """Return an (n_fakes, n_items) float32 matrix of fake ratings, 0 where unrated."""
    rng = np.random.default_rng(spec.seed)
    sel = selected_items(spec, kn)
    pool = np.setdiff1d(np.arange(n_items), np.append(sel, spec.target))
    if spec.filler_size > pool.size:
        raise ValueError("filler_size exceeds the number of available items")
    out = np.zeros((spec.n_fakes, n_items), dtype=np.float32)
    target_rating = RATING_MAX if spec.direction == "push" else RATING_MIN
    for f in range(spec.n_fakes):
        filler = rng.choice(pool, size=spec.filler_size, replace=False)
        if spec.kind == "average":
            draw = rng.normal(kn.item_mean[filler], kn.item_std[filler])
            out[f, filler] = np.clip(np.rint(draw), RATING_MIN, RATING_MAX)
        elif spec.kind == "segment":
            out[f, filler] = RATING_MIN
        else:  # random and bandwagon share the uniform filler rule
            out[f, filler] = rng.integers(RATING_MIN, RATING_MAX + 1, size=filler.size)
        out[f, sel] = RATING_MAX
        out[f, spec.target] = target_rating
    return out


def inject(R: np.ndarray, M: np.ndarray, profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append fake rows below the genuine users. Fake user ids are n_genuine .. n_genuine + n_fakes - 1."""
    if profiles.shape[1] != R.shape[1]:
        raise ValueError("profile width must equal the number of items")
    R_aug = np.vstack([R, profiles.astype(R.dtype)])
    return R_aug, R_aug > 0


def choose_targets(M: np.ndarray, n: int, lo: int = 10, hi: int = 50, seed: int = 0) -> np.ndarray:
    """Sample n target items that are lightly but not barely rated (lo..hi training ratings),
    so a push is a meaningful change and neighbourhood models have some signal to work with."""
    counts = M.sum(axis=0)
    pool = np.flatnonzero((counts >= lo) & (counts <= hi))
    if pool.size < n:
        raise ValueError(f"only {pool.size} items have between {lo} and {hi} ratings")
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))
