"""Pilot: push attacks (4 kinds x 4 sizes x several targets) against one model on the temporal-split
training matrix. Quick sanity run before the full experiment grid.

Usage (from case_study/):  python scripts/pilot_attack.py --model als --targets 5
"""
from __future__ import annotations

import argparse
import ast
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recattack.attacks import (  # noqa: E402
    KINDS, AttackSpec, attack_size_percent, choose_targets, knowledge_from, n_fakes_for, pick_genre,
)
from recattack.data import N_ITEMS, N_USERS, load_genres, load_ml100k, per_user_temporal_split, to_dense  # noqa: E402
from recattack.impact import measure_attack  # noqa: E402
from recattack.knn import KNNRecommender  # noqa: E402
from recattack.mf import ALSMatrixFactorization  # noqa: E402

log = logging.getLogger("pilot")

def best(model: str) -> dict:
    """Best hyper-parameters chosen on the validation tail, as saved by run_baselines.py."""
    rows = pd.read_csv(ROOT / "results" / "baselines_user.csv")
    return ast.literal_eval(rows.loc[rows["model"] == model, "params"].iloc[0])


MODELS = {
    "als": lambda: ALSMatrixFactorization(seed=0, **best("ALS-MF")),
    "userknn": lambda: KNNRecommender("user", **best("UserKNN")),
    "itemknn": lambda: KNNRecommender("item", **best("ItemKNN")),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="als")
    ap.add_argument("--targets", type=int, default=5)
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "ml-100k"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = load_ml100k(args.data_dir)
    R, M = to_dense(per_user_temporal_split(df, 0.2).train, N_USERS, N_ITEMS)
    genres, genre_names = load_genres(args.data_dir)
    kn = knowledge_from(R, M, genres)
    targets = choose_targets(M, args.targets, seed=args.seed)
    make = MODELS[args.model]
    before = make().fit(R, M)
    log.info("model=%s targets=%s", args.model, targets.tolist())

    rows = []
    t0 = time.perf_counter()
    for kind in KINDS:
        for frac in args.sizes:
            for t in targets:
                n_f = n_fakes_for(frac, N_USERS)
                genre = pick_genre(kn, int(t), 3) if kind == "segment" else None
                spec = AttackSpec(kind, int(t), n_f, genre=genre, seed=args.seed)
                res = measure_attack(make, before, R, M, spec, kn)
                rows.append({"model": args.model, "kind": kind, "frac": frac, "n_fakes": n_f,
                             "attack_size_pct": attack_size_percent(n_f, N_USERS), "target": int(t), **res})
        log.info("%s done (%.0fs elapsed)", kind, time.perf_counter() - t0)

    out = pd.DataFrame(rows)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.out_dir) / f"pilot_attack_{args.model}.csv", index=False)
    cols = ["pred_shift", "hit_before", "hit_after", "rank_gain"]
    summ = out.groupby(["kind", "frac"], sort=False)[cols].mean().round(3)
    print(summ.to_string())
    seg = out[out["kind"] == "segment"]
    if "seg_pred_shift" in seg and seg["seg_n_users"].fillna(0).gt(0).any():
        print("\nsegment attack, restricted to its target audience:")
        print(seg.groupby("frac")[["seg_n_users", "seg_pred_shift", "seg_hit_before", "seg_hit_after"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
