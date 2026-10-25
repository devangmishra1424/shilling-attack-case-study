"""Follow one real target movie through a push attack and the MAD trust-weight defence.

Uses the same targets, attack seeds and ALS settings as run_defence.py, so the figures agree with the
defence table. Usage (from case_study/): python scripts/worked_example.py [target_index 0|1] [kind]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recattack.attacks import AttackSpec, choose_targets, knowledge_from, n_fakes_for, pick_genre, selected_items  # noqa: E402
from recattack.data import N_ITEMS, N_USERS, load_genres, load_ml100k, per_user_temporal_split, to_dense  # noqa: E402
from recattack.detect import build_world  # noqa: E402
from recattack.features import MAD_THRESHOLD, mad_score, profile_features, trust_from_mad  # noqa: E402
from recattack.metrics import item_ranks  # noqa: E402
from recattack.robust import RobustALS  # noqa: E402

data = ROOT / "data" / "ml-100k"
titles = pd.read_csv(data / "u.item", sep="|", header=None, encoding="latin-1")[1].tolist()
R, M = to_dense(per_user_temporal_split(load_ml100k(data), 0.2).train, N_USERS, N_ITEMS)
genres, gnames = load_genres(data)
kn = knowledge_from(R, M, genres)
perm = np.random.default_rng(0).permutation(choose_targets(M, 5, seed=0))
targets = perm[3:]
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
kinds = sys.argv[2:] or ["random", "segment"]
t = int(targets[idx])
import ast
params = ast.literal_eval(pd.read_csv(ROOT / "results" / "baselines_user.csv").query("model == 'ALS-MF'")["params"].iloc[0])
make = lambda trust=None: RobustALS(seed=0, user_trust=trust, **params)
before = make().fit(R, M)
users = np.flatnonzero(~M[:, t])
print(f"TARGET {t}: {titles[t]} | genuine ratings {int(M[:, t].sum())} mean {R[M[:, t], t].mean():.2f} | genres "
      f"{[gnames[g] for g in np.flatnonzero(genres[t])]} | eligible users {users.size}")
r0 = item_ranks(before.scores()[:N_USERS], M, t, users)
print(f"BEFORE: mean rank {r0.mean():.0f} of ~{N_ITEMS - int(M[users].sum(1).mean())} candidates, hit@10 {(r0 <= 10).mean():.3f}, mean predicted rating {before.predictions()[users, t].mean():.2f}")
for kind in kinds:
    genre = pick_genre(kn, t, 3) if kind == "segment" else None
    spec = AttackSpec(kind, t, n_fakes_for(0.05, N_USERS), genre=genre, seed=9000 + t)
    w = build_world(R, M, kn, spec)
    F = profile_features(w.R, w.M)
    mad = mad_score(F)
    flag = mad > MAD_THRESHOLD
    sel = selected_items(spec, kn)
    fake0 = N_USERS
    rated = np.flatnonzero(w.M[fake0])
    print(f"\nATTACK {kind}: {spec.n_fakes} fakes, genre {gnames[genre] if genre is not None else '-'}, selected {[titles[i] for i in sel]}")
    print("  example fake rates", len(rated), "movies, e.g.", [(titles[i], int(w.R[fake0, i])) for i in rated[:4]], "target", int(w.R[fake0, t]))
    for name, trust in (("attacked", None), ("defended (MAD trust)", trust_from_mad(mad))):
        m = make(trust).fit(w.R, w.M)
        r = item_ranks(m.scores()[:N_USERS], M, t, users)
        print(f"  {name}: mean rank {r.mean():.0f}, hit@10 {(r <= 10).mean():.3f}, mean predicted rating {m.predictions()[:N_USERS][users, t].mean():.2f}")
    print(f"  MAD flags {int(flag[w.is_fake].sum())} of {spec.n_fakes} fakes and {int(flag[~w.is_fake].sum())} of {N_USERS} honest users; "
          f"mean trust fakes {trust_from_mad(mad)[w.is_fake].mean():.2f}, honest {trust_from_mad(mad)[~w.is_fake].mean():.2f}")
