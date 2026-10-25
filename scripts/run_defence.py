"""Defence experiment: attack a recommender, defend it, and measure damage and cost.

For every (attack kind, size, target) the script builds a world with fake profiles, then fits the
recommender under several defences and compares it with the clean undefended model:
  none          fit on everything
  filter_mad    drop profiles whose MAD outlier score exceeds 3.5
  filter_clf    drop profiles a random-forest detector gives P(fake) > 0.5
  trust_clf     weight each profile by trust = 1 - P(fake)
  trust_mad     weight each profile by min(1, 3.5 / MAD score)
  huber         Huber-loss robust MF (ALS only)
  trust_clf+huber  both (ALS only)
Detectors are cross-fitted: genuine users are split into halves A and B; a detector trained with
half A as its genuine examples scores half B and vice versa, so no genuine user is scored by a
detector that saw them. By default the detector has never seen the attack kind it faces
(leave-one-kind-out). Targets used to train detectors differ from the targets attacked here.
A 'clean' row per model shows what each defence costs honest users when there is no attack.

Usage (from case_study/):  python scripts/run_defence.py --models ALS-MF --targets 5
Updates results/defence_<model>.csv after every world, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import ast
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
from recattack.detect import ProfileDetector, build_world, detection_metrics, split_users, training_pool  # noqa: E402
from recattack.evaluate import evaluate_model  # noqa: E402
from recattack.features import MAD_THRESHOLD, mad_score, profile_features, trust_from_mad  # noqa: E402
from recattack.impact import compare_models  # noqa: E402
from recattack.knn import KNNRecommender  # noqa: E402
from recattack.robust import RobustALS  # noqa: E402

log = logging.getLogger("defence")
FRACTIONS = (0.01, 0.03, 0.05, 0.10)


def best_params(baselines_csv: Path, model: str) -> dict:
    df = pd.read_csv(baselines_csv)
    rows = df[df["model"] == model]
    if rows.empty:
        raise ValueError(f"{model} not found in {baselines_csv}")
    return ast.literal_eval(rows["params"].iloc[0])


def make_model(model: str, params: dict, trust=None, huber=None):
    if model == "ALS-MF":
        return RobustALS(n_factors=params["n_factors"], reg=params["reg"], seed=0, user_trust=trust, huber_delta=huber)
    if model in ("UserKNN", "ItemKNN"):
        return KNNRecommender(model[:-3].lower(), k=params["k"], shrink=params["shrink"], user_trust=trust)
    raise ValueError(f"unknown model {model}")


def defences(model: str, flag_mad, flag_clf, trust_clf, trust_mad) -> dict[str, dict]:
    d = {
        "none": {},
        "filter_mad": {"flag": flag_mad},
        "filter_clf": {"flag": flag_clf},
        "trust_clf": {"trust": trust_clf},
        "trust_mad": {"trust": trust_mad},
    }
    if model == "ALS-MF":
        d["huber"] = {"huber": 1.0}
        d["trust_clf+huber"] = {"trust": trust_clf, "huber": 1.0}
    return d


def fit_defended(model, params, R, M, spec: dict):
    flag = spec.get("flag")
    if flag is not None:
        R, M = R.copy(), M.copy()
        R[flag], M[flag] = 0, False
    return make_model(model, params, trust=spec.get("trust"), huber=spec.get("huber")).fit(R, M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ALS-MF"], choices=["ALS-MF", "UserKNN", "ItemKNN"])
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    ap.add_argument("--targets", type=int, default=5)
    ap.add_argument("--detector-mode", choices=["loko", "seen"], default="loko")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "ml-100k"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    split = per_user_temporal_split(load_ml100k(args.data_dir), 0.2)
    R, M = to_dense(split.train, N_USERS, N_ITEMS)
    genres, _ = load_genres(args.data_dir)
    kn = knowledge_from(R, M, genres)
    users_a, users_b = split_users(N_USERS, args.seed)
    rng = np.random.default_rng(args.seed)
    targets = rng.permutation(choose_targets(M, 3 + args.targets, seed=args.seed))
    train_targets, attack_targets = targets[:3], targets[3:]
    genre_of = lambda t: pick_genre(kn, t, 3)

    t0 = time.perf_counter()
    pools = {h: training_pool(R, M, kn, KINDS, FRACTIONS, train_targets, (50, 84, 120), u, seed=1000 + 7 * i, genres_for=genre_of)
             for i, (h, u) in enumerate((("A", users_a), ("B", users_b)))}
    log.info("detector training pools built (%.0fs)", time.perf_counter() - t0)

    def train_pair(held_out):
        kinds = [k for k in KINDS if k != held_out]
        dets = {}
        for half in ("A", "B"):
            X = np.vstack([pools[half][k][0] for k in kinds])
            y = np.concatenate([pools[half][k][1] for k in kinds])
            dets[half] = ProfileDetector("rf", seed=args.seed).fit(X, y)
        return dets

    detectors = {h: train_pair(h) for h in (None, *KINDS)}

    def cross_fit_scores(F, n_genuine, held_out):
        d = detectors[held_out if args.detector_mode == "loko" else None]
        p = np.zeros(F.shape[0])
        p[users_b] = d["A"].score(F[users_b])  # detector trained on A scores B
        p[users_a] = d["B"].score(F[users_a])
        if F.shape[0] > n_genuine:
            p[n_genuine:] = 0.5 * (d["A"].score(F[n_genuine:]) + d["B"].score(F[n_genuine:]))
        return p

    for model in args.models:
        params = best_params(out / "baselines_user.csv", model)
        csv = out / f"defence_{model}.csv"
        done = set()
        if csv.exists():
            prev = pd.read_csv(csv)
            done = set(zip(prev["kind"], prev["frac"].round(4), prev["target"]))
        before = make_model(model, params).fit(R, M)
        base_acc = evaluate_model(before, M, split.test)
        log.info("%s clean accuracy %s params %s", model, {k: round(v, 4) for k, v in base_acc.items() if k in ("rmse", "ndcg@10")}, params)

        def emit(rows):
            new = pd.DataFrame(rows)
            if csv.exists():  # rewrite with the union of columns so the file keeps one schema
                new = pd.concat([pd.read_csv(csv), new], ignore_index=True)
            new.to_csv(csv, index=False)

        # cost of each defence on honest users when nothing is wrong
        if ("clean", 0.0, -1) not in done:
            F = profile_features(R, M)
            p = cross_fit_scores(F, N_USERS, None)
            mad = mad_score(F)
            flag_mad, flag_clf = mad > MAD_THRESHOLD, p > 0.5
            rows = []
            for name, spec in defences(model, flag_mad, flag_clf, 1.0 - p, trust_from_mad(mad)).items():
                after = fit_defended(model, params, R, M, spec)
                acc = evaluate_model(after, M, split.test)
                rows.append({"model": model, "kind": "clean", "frac": 0.0, "target": -1, "defence": name,
                             "rmse": acc["rmse"], "ndcg@10": acc["ndcg@10"], "rmse_clean_ref": base_acc["rmse"],
                             "ndcg_clean_ref": base_acc["ndcg@10"], "mad_fp": int(flag_mad.sum()), "clf_fp": int(flag_clf.sum())})
            emit(rows)
            log.info("%s clean-data cost rows written", model)

        for kind in args.kinds:
            for frac in args.fracs:
                for t in attack_targets:
                    key = (kind, round(frac, 4), int(t))
                    if key in done:
                        continue
                    tw = time.perf_counter()
                    genre = genre_of(int(t)) if kind == "segment" else None
                    spec_a = AttackSpec(kind, int(t), n_fakes_for(frac, N_USERS), genre=genre, seed=9000 + int(t))
                    w = build_world(R, M, kn, spec_a)
                    F = profile_features(w.R, w.M)
                    p = cross_fit_scores(F, N_USERS, kind)
                    mad = mad_score(F)
                    flag_mad, flag_clf = mad > MAD_THRESHOLD, p > 0.5
                    det = {
                        "mad_recall": float(flag_mad[w.is_fake].mean()), "mad_fp": int(flag_mad[~w.is_fake].sum()),
                        "clf_recall": float(flag_clf[w.is_fake].mean()), "clf_fp": int(flag_clf[~w.is_fake].sum()),
                        "clf_auc": detection_metrics(p, w.is_fake, 0.5)["auc"],
                    }
                    rows = []
                    for name, dspec in defences(model, flag_mad, flag_clf, 1.0 - p, trust_from_mad(mad)).items():
                        after = fit_defended(model, params, w.R, w.M, dspec)
                        imp = compare_models(before, after, R, M, spec_a, kn)
                        acc = evaluate_model(after, M, split.test, n_users=N_USERS)
                        rows.append({"model": model, "kind": kind, "frac": frac, "n_fakes": spec_a.n_fakes, "target": int(t),
                                     "defence": name, **imp, "rmse": acc["rmse"], "ndcg@10": acc["ndcg@10"],
                                     "rmse_clean_ref": base_acc["rmse"], **det})
                    emit(rows)
                    log.info("%s %s frac=%.2f target=%d done in %.0fs (elapsed %.0fs)", model, kind, frac, t, time.perf_counter() - tw, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
