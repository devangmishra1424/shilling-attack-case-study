"""Fit and evaluate all baseline recommenders on a temporal split of MovieLens 100K.

Usage (from case_study/):  python scripts/run_baselines.py
Writes results/baselines_<split>.csv and .log. Hyper-parameters are chosen on a
validation tail of the training period; the test period is touched once per model.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recattack.baselines import DampedItemMean, GlobalMean, Popularity, RandomRanker  # noqa: E402
from recattack.data import (  # noqa: E402
    N_ITEMS, N_USERS, load_ml100k, per_user_temporal_split, temporal_split, unseen_fractions,
)
from recattack.evaluate import fit_evaluate, select_by_validation  # noqa: E402
from recattack.knn import KNNRecommender  # noqa: E402
from recattack.mf import ALSMatrixFactorization, BiasOnly  # noqa: E402

log = logging.getLogger("run_baselines")


def grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "ml-100k"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--seeds", type=int, default=5, help="random seeds for stochastic models")
    ap.add_argument("--split", choices=["user", "global"], default="user",
                    help="user: each user's latest 20%% is test (main); global: one time cutoff (stress test)")
    ap.add_argument("--skip-knn", action="store_true", help="skip the slow neighbourhood models")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"baselines_{args.split}.log", mode="w")],
    )
    log.info("python %s numpy %s pandas %s", platform.python_version(), np.__version__, pd.__version__)

    df = load_ml100k(args.data_dir)
    splitter = per_user_temporal_split if args.split == "user" else temporal_split
    split = splitter(df, test_frac=0.2)
    log.info(
        "split=%s train=%d test=%d cutoff_ts=%s %s",
        args.split, len(split.train), len(split.test), split.cutoff_ts, unseen_fractions(split.train, split.test),
    )

    rows: list[dict] = []

    def record(name: str, params: dict, seed, res: dict) -> None:
        rows.append({"model": name, "params": str(params), "seed": seed, **res})
        log.info("%-12s seed=%s rmse=%.4f ndcg@10=%.4f fit=%.1fs", name, seed, res["rmse"], res["ndcg@10"], res["fit_seconds"])

    def run(name, make, g, seeds=(None,)):
        best, _ = select_by_validation(make, g, split.train, N_USERS, N_ITEMS, splitter=splitter)
        log.info("%s best params: %s", name, best)
        for s in seeds:
            kw = dict(best) if s is None else {**best, "seed": s}
            res = fit_evaluate(lambda kw=kw: make(**kw), split.train, split.test, N_USERS, N_ITEMS)
            record(name, best, s, res)

    seeds = list(range(args.seeds))
    run("GlobalMean", GlobalMean, [{}])
    run("ItemMean", DampedItemMean, grid(damp=[0.0, 5.0, 25.0]))
    run("Popularity", Popularity, grid(damp=[5.0]))
    run("Random", RandomRanker, [{}], seeds=seeds)
    run("BiasOnly", BiasOnly, grid(reg_item=[5.0, 10.0, 25.0], reg_user=[5.0, 15.0, 25.0]))
    run("ALS-MF", ALSMatrixFactorization, grid(n_factors=[10, 20], reg=[0.05, 0.1, 0.2]), seeds=seeds)
    if not args.skip_knn:
        run("UserKNN", lambda **kw: KNNRecommender("user", **kw), grid(k=[20, 50], shrink=[10.0, 50.0]))
        run("ItemKNN", lambda **kw: KNNRecommender("item", **kw), grid(k=[20, 50], shrink=[10.0, 50.0]))

    res = pd.DataFrame(rows)
    res.to_csv(out / f"baselines_{args.split}.csv", index=False)
    cols = ["rmse", "mae", "precision@10", "recall@10", "ndcg@10"]
    summary = res.groupby("model", sort=False)[cols].agg(["mean", "std"]).round(4)
    log.info("\n%s", summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
