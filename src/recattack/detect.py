"""Attack detection: worlds with known fakes, a supervised profile classifier, detection metrics.

The detector is trained and tested on disjoint genuine users and on attacks that differ in target,
seed and (for leave-one-kind-out) attack kind, so it cannot memorise either the honest users or
the attack it is later judged on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .attacks import AttackSpec, Knowledge, inject, make_profiles
from .features import profile_features


@dataclass(frozen=True)
class World:
    """A rating matrix with fake profiles appended; labels mark the fakes."""

    R: np.ndarray
    M: np.ndarray
    is_fake: np.ndarray
    n_genuine: int
    spec: AttackSpec


def build_world(R: np.ndarray, M: np.ndarray, kn: Knowledge, spec: AttackSpec) -> World:
    R_aug, M_aug = inject(R, M, make_profiles(spec, kn, R.shape[1]))
    labels = np.arange(R_aug.shape[0]) >= R.shape[0]
    return World(R_aug, M_aug, labels, R.shape[0], spec)


def split_users(n_users: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random halves A and B of the genuine users (sorted ids)."""
    perm = np.random.default_rng(seed).permutation(n_users)
    half = n_users // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


class ProfileDetector:
    """Supervised fake-profile classifier on profile features. Scores are P(fake)."""

    def __init__(self, kind: str = "rf", seed: int = 0) -> None:
        if kind == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=2
            )
        elif kind == "lr":
            self.model = make_pipeline(
                StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
            )
        else:
            raise ValueError("kind must be 'rf' or 'lr'")
        self.kind = kind

    def fit(self, F: np.ndarray, y: np.ndarray) -> "ProfileDetector":
        if len(np.unique(y)) != 2:
            raise ValueError("training data needs both genuine and fake profiles")
        self.model.fit(F, y.astype(int))
        return self

    def score(self, F: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(F)[:, 1]


def detection_metrics(score: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    """Threshold-free AUC and average precision, plus precision/recall/F1 at `threshold`."""
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        raise ValueError("need both fake and genuine profiles to evaluate a detector")
    flagged = score > threshold
    tp = int((flagged & y).sum())
    precision = tp / flagged.sum() if flagged.any() else 0.0
    recall = tp / y.sum()
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "auc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_flagged": int(flagged.sum()),
        "false_positives": int((flagged & ~y).sum()),
    }


def world_rows(world: World, F: np.ndarray, genuine_users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels for the chosen genuine users plus every fake in the world."""
    rows = np.concatenate([genuine_users, np.flatnonzero(world.is_fake)])
    return F[rows], world.is_fake[rows]


def training_pool(
    R: np.ndarray,
    M: np.ndarray,
    kn: Knowledge,
    kinds: tuple[str, ...],
    fractions: tuple[float, ...],
    targets: np.ndarray,
    fillers: tuple[int, ...],
    genuine_users: np.ndarray,
    seed: int,
    genres_for=None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per attack kind, stacked (features, labels) from many simulated attacks on `genuine_users`.

    `genres_for(target)` supplies the genre for segment attacks. Filler sizes cycle through
    `fillers` so the detector does not tie itself to one profile length.
    """
    from .attacks import n_fakes_for  # local import keeps module import order simple

    pool: dict[str, list] = {k: ([], []) for k in kinds}
    j = 0
    for kind in kinds:
        for frac in fractions:
            for t in targets:
                genre = genres_for(int(t)) if kind == "segment" else None
                spec = AttackSpec(
                    kind, int(t), n_fakes_for(frac, R.shape[0]), filler_size=fillers[j % len(fillers)],
                    genre=genre, seed=seed + j,
                )
                j += 1
                w = build_world(R, M, kn, spec)
                X, y = world_rows(w, profile_features(w.R, w.M), genuine_users)
                pool[kind][0].append(X)
                pool[kind][1].append(y)
    return {k: (np.vstack(v[0]), np.concatenate(v[1])) for k, v in pool.items()}
