"""Detection experiment: can fake profiles be told apart from genuine ones?

Detectors compared: MAD outlier score (no training), logistic regression, random forest.
Settings: 'in-distribution' (trained on all four attack kinds) and leave-one-kind-out, where the
detector is trained on three kinds and tested on the fourth ('unseen attack'). Train and test use
disjoint genuine users (random halves A and B) and disjoint target items.

Usage (from case_study/):  python scripts/run_detection.py
Writes results/detection.csv and results/detection_importance.csv.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recattack.attacks import KINDS, AttackSpec, choose_targets, knowledge_from, n_fakes_for, pick_genre  # noqa: E402
from recattack.data import N_ITEMS, N_USERS, load_genres, load_ml100k, per_user_temporal_split, to_dense  # noqa: E402
from recattack.detect import ProfileDetector, build_world, detection_metrics, split_users, training_pool, world_rows  # noqa: E402
from recattack.features import FEATURE_NAMES, MAD_THRESHOLD, mad_score, profile_features  # noqa: E402

log = logging.getLogger("detection")
FRACTIONS = (0.01, 0.03, 0.05, 0.10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "ml-100k"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_ml100k(args.data_dir)
    R, M = to_dense(per_user_temporal_split(df, 0.2).train, N_USERS, N_ITEMS)
    genres, _ = load_genres(args.data_dir)
    kn = knowledge_from(R, M, genres)
    users_a, users_b = split_users(N_USERS, args.seed)
    targets = np.random.default_rng(args.seed).permutation(choose_targets(M, 8, seed=args.seed))
    train_targets, test_targets = targets[:3], targets[3:]
    genre_of = lambda t: pick_genre(kn, t, 3)

    t0 = time.perf_counter()
    pool = training_pool(R, M, kn, KINDS, FRACTIONS, train_targets, (50, 84, 120), users_a, seed=1000, genres_for=genre_of)
    log.info("training pool built in %.0fs: %s", time.perf_counter() - t0, {k: v[0].shape for k, v in pool.items()})

    tests = []  # (kind, frac, target, F rows, labels, MAD score rows)
    for kind in KINDS:
        for frac in FRACTIONS:
            for t in test_targets:
                genre = genre_of(int(t)) if kind == "segment" else None
                spec = AttackSpec(kind, int(t), n_fakes_for(frac, N_USERS), genre=genre, seed=5000 + int(t))
                w = build_world(R, M, kn, spec)
                F = profile_features(w.R, w.M)
                X, y = world_rows(w, F, users_b)
                rows = np.concatenate([users_b, np.flatnonzero(w.is_fake)])
                tests.append((kind, frac, int(t), X, y, mad_score(F)[rows]))
    log.info("%d test worlds built (%.0fs elapsed)", len(tests), time.perf_counter() - t0)

    records = []
    for kind, frac, t, X, y, mad in tests:
        records.append({"setting": "none", "detector": "MAD", "kind": kind, "frac": frac, "target": t,
                        **detection_metrics(mad, y, MAD_THRESHOLD)})

    importance = {}
    for held_out in (None, *KINDS):
        train_kinds = [k for k in KINDS if k != held_out]
        Xtr = np.vstack([pool[k][0] for k in train_kinds])
        ytr = np.concatenate([pool[k][1] for k in train_kinds])
        setting = "in-distribution" if held_out is None else f"held-out:{held_out}"
        for dkind in ("lr", "rf"):
            det = ProfileDetector(dkind, seed=args.seed).fit(Xtr, ytr)
            if dkind == "rf" and held_out is None:
                importance = dict(zip(FEATURE_NAMES, det.model.feature_importances_.round(4)))
            for kind, frac, t, X, y, _ in tests:
                if held_out is not None and kind != held_out:
                    continue
                records.append({"setting": setting, "detector": dkind.upper(), "kind": kind, "frac": frac,
                                "target": t, **detection_metrics(det.score(X), y, 0.5)})
        log.info("%s done (%.0fs elapsed)", setting, time.perf_counter() - t0)

    res = pd.DataFrame(records)
    res.to_csv(out / "detection.csv", index=False)
    pd.Series(importance, name="rf_importance").to_csv(out / "detection_importance.csv")

    view = res.copy()
    view["scenario"] = np.where(view["setting"].str.startswith("held-out"), "unseen attack", np.where(view["setting"] == "none", "no training", "seen attacks"))
    summ = view.groupby(["scenario", "detector", "kind"])[["auc", "ap", "precision", "recall", "f1"]].mean().round(3)
    print(summ.to_string())
    print("\nby attack size (mean over kinds):")
    print(view.groupby(["scenario", "detector", "frac"])[["auc", "recall", "precision"]].mean().round(3).to_string())
    print("\nrandom-forest feature importance (in-distribution):", importance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
