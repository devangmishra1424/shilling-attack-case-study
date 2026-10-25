# Can you game the algorithm? A red-team / blue-team case study

Shilling (profile injection) attacks and defences on a neighbourhood and matrix-factorisation recommender, MovieLens 100K. Course: CSE4143 Recommender System. Author: Devang Pradeep Mishra (23FE10CSE00659).

Status: baselines, evaluation harness, attack generators and impact measurement done. Defences, the full experiment grid and the report are next. `results/pilot_attack_als_GLOBALSPLIT_superseded.csv` was produced on the flawed global split and must not be cited.

## Get the data

The dataset is not in this repository (GroupLens does not allow redistribution). Download and unzip it into `data/`:

```bash
mkdir data && cd data
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip          # creates data/ml-100k/
```

Source and terms: https://grouplens.org/datasets/movielens/100k/ and the [README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt). If you use it, cite F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), https://doi.org/10.1145/2827872.

## Install

```bash
pip install -r requirements.txt
```

## Layout

| Path | What it is |
| --- | --- |
| `data/ml-100k/` | MovieLens 100K (GroupLens). Research use only; cite Harper and Konstan (2015); do not redistribute |
| `src/recattack/data.py` | Loader with validation, 0-based ids, global temporal split, dense matrix builder |
| `src/recattack/base.py` | Common recommender interface (fit on rating matrix, full prediction matrix) |
| `src/recattack/baselines.py` | GlobalMean, damped ItemMean, Popularity, Random |
| `src/recattack/mf.py` | Regularised bias baseline and ALS-WR matrix factorisation |
| `src/recattack/knn.py` | User-based and item-based kNN (shrunk cosine on mean-centred ratings) |
| `src/recattack/metrics.py` | RMSE, MAE, Precision/Recall/NDCG@k, hit ratio, rank, prediction shift |
| `src/recattack/evaluate.py` | Fit-and-evaluate harness and validation-tail tuning |
| `src/recattack/attacks.py` | Random, average, bandwagon and segment attack generators; injection; target choice |
| `src/recattack/impact.py` | Prediction shift, hit ratio and rank gain of an attack on a fitted model |
| `src/recattack/features.py` | Profile features and the MAD outlier score |
| `src/recattack/detect.py` | Fake-profile detectors (logistic regression, random forest) and detection metrics |
| `src/recattack/robust.py` | Trust-weighted and Huber-loss (robust) ALS matrix factorisation |
| `scripts/run_baselines.py` | Runs every baseline, writes `results/baselines.csv` and `.log` |
| `tests/` | pytest suite |
| `results/` | Outputs |

## Run

```bash
python -m pytest -q
python scripts/run_baselines.py --seeds 5          # add --skip-knn for a fast run
```

## Evaluation protocol

- **Main protocol, per-user chronological split:** each user's most recent 20% of ratings (at least one) are test, the rest train. Every user has training history. Trade-off: one user's test period can overlap another user's training period, so some collaborative signal from "the future" can reach training. Say this in the report.
- **Stress test, one global time cutoff** (`--split global`): earliest 80% of ratings train, latest 20% test. On ML-100K this is a poor protocol: 192 of 943 users have no training ratings and they hold about 85% of the test ratings, so personalised models fall back to the global mean. Kept as a documented negative result (`results/baselines_global.csv`).
- Hyper-parameters are chosen on the most recent 15% of each user's training data (same splitter); the test set is used once per final model.
- Rating accuracy: RMSE and MAE on all test pairs. Ranking: Precision, Recall and NDCG at 10 over the full catalogue (no negative sampling), test ratings of 4 or 5 count as relevant, items a user rated in training are excluded.
- Stochastic models (ALS-MF, Random) are run over several seeds and reported as mean and standard deviation.
- Ties in ranking are broken by lower item id everywhere.

## Memory note

The machine has about 7 GB RAM with little free; everything is plain numpy on a 943 x 1682 dense matrix (about 6 MB in float32). Keep parallel workers to 3 or 4 for the experiment grid.
