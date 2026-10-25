"""Build the report figures from results/*.csv. Usage (from case_study/): python scripts/make_figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DEEP, OCEAN, TEAL, AQUA, CORAL, SUN, GREY = "#06323F", "#0E6E80", "#14A3A8", "#2EC4B6", "#E8553F", "#E0A800", "#7E9AA0"
plt.rcParams.update({"font.family": ["Calibri", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9FB4B8", "axes.labelcolor": DEEP,
                     "xtick.color": DEEP, "ytick.color": DEEP, "axes.titleweight": "bold", "axes.titlecolor": DEEP})

# 1. attack impact -------------------------------------------------------------------------
p = pd.read_csv(RES / "pilot_attack_als.csv")
g = p.groupby(["kind", "frac"])["hit_after"].mean().unstack(0) * 100
fig, ax = plt.subplots(figsize=(6.4, 3.5))
style = {"random": (GREY, "o"), "average": (TEAL, "s"), "bandwagon": (OCEAN, "^"), "segment": (CORAL, "D")}
for kind, (c, m) in style.items():
    ax.plot(g.index * 100, g[kind], color=c, marker=m, lw=2, ms=6, label=kind.capitalize())
    ax.annotate(f"{g[kind].iloc[-1]:.0f}%", (10, g[kind].iloc[-1]), xytext=(6, 0), textcoords="offset points", color=c, va="center", fontsize=9)
ax.set_xlabel("Fake profiles as % of genuine users"); ax.set_ylabel("Users with the target in their top-10 (%)")
ax.set_xticks([1, 3, 5, 10]); ax.set_xlim(0.5, 11.5); ax.set_ylim(0, 105); ax.grid(axis="y", color="#E3F1F0")
ax.legend(frameon=False, loc="upper left"); ax.set_title("A few fake profiles push one movie into many top-10 lists", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "fig1_attack_impact.png", dpi=220); plt.close(fig)

# 2. detection ----------------------------------------------------------------------------
d = pd.read_csv(RES / "detection.csv")
d["scenario"] = np.where(d["detector"] == "MAD", "MAD score (no training)",
                         np.where((d["detector"] == "RF") & d["setting"].str.startswith("held-out"), "Random forest, unseen attack",
                                  np.where((d["detector"] == "RF") & (d["setting"] == "in-distribution"), "Random forest, seen attacks", "other")))
d = d[d["scenario"] != "other"]
order = ["MAD score (no training)", "Random forest, seen attacks", "Random forest, unseen attack"]
cols = [GREY, OCEAN, CORAL]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharex=True)
for ax, metric, title in zip(axes, ["precision", "recall"], ["Precision (flagged that are fake)", "Recall (fakes caught)"]):
    t = d.groupby(["scenario", "frac"])[metric].mean().unstack(0)[order] * 100
    w = 0.26
    for i, (s, c) in enumerate(zip(order, cols)):
        ax.bar(np.arange(4) + (i - 1) * w, t[s].values, w, color=c, label=s)
    ax.set_xticks(range(4)); ax.set_xticklabels(["1%", "3%", "5%", "10%"]); ax.set_ylim(0, 105)
    ax.set_xlabel("Fake profiles, % of genuine users", fontsize=8.5); ax.set_title(title, fontsize=9.5, loc="left"); ax.grid(axis="y", color="#E3F1F0"); ax.set_axisbelow(True)
axes[0].set_ylabel("%")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "fig2_detection.png", dpi=220); plt.close(fig)

# 3 and 4. defences -----------------------------------------------------------------------
f = pd.read_csv(RES / "defence_ALS-MF.csv")
clean = f[f["kind"] == "clean"].set_index("defence")
att = f[f["kind"] != "clean"]
ref = clean["rmse_clean_ref"].iloc[0]
hit = att.groupby("defence")["hit_after"].mean() * 100
cost = clean["rmse"] - ref
names = {"none": "No defence", "filter_mad": "Filter (MAD)", "filter_clf": "Filter (classifier)", "trust_clf": "Trust (classifier)",
         "trust_mad": "Trust (MAD)", "huber": "Huber robust MF", "trust_clf+huber": "Trust + Huber"}
fig, ax = plt.subplots(figsize=(6.4, 3.6))
offs = {"none": (8, -4), "filter_mad": (8, -3), "filter_clf": (8, 4), "trust_clf": (8, -12), "trust_mad": (8, 6),
        "huber": (-8, 8), "trust_clf+huber": (8, 8)}
for k in names:
    c = CORAL if k in ("none", "huber") else (OCEAN if "trust" in k else TEAL)
    ax.scatter(hit[k], cost[k], s=70, color=c, zorder=3)
    ha = "right" if offs[k][0] < 0 else "left"
    ax.annotate(names[k], (hit[k], cost[k]), xytext=offs[k], textcoords="offset points", fontsize=8.5, color=DEEP, ha=ha)
ax.set_xlabel("Users with the target in their top-10 under attack (%)  (lower is safer)")
ax.set_ylabel("Extra RMSE for honest users, no attack")
ax.set_xlim(-6, 112); ax.set_ylim(-0.012, 0.062); ax.grid(color="#E3F1F0"); ax.set_axisbelow(True)
ax.set_title("Best defences sit bottom-left: safe and cheap", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "fig3_defence_tradeoff.png", dpi=220); plt.close(fig)

h2 = att.pivot_table(index="defence", columns="kind", values="hit_after", aggfunc="mean").loc[list(names)] * 100
h2 = h2[["random", "average", "bandwagon", "segment"]]
fig, ax = plt.subplots(figsize=(6.0, 3.3))
im = ax.imshow(h2.values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([c.capitalize() for c in h2.columns]); ax.set_yticks(range(len(h2))); ax.set_yticklabels([names[i] for i in h2.index])
for i in range(h2.shape[0]):
    for j in range(h2.shape[1]):
        v = h2.values[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 60 else DEEP)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Top-10 hit ratio (%) by defence and attack kind", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "fig4_defence_by_attack.png", dpi=220); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
