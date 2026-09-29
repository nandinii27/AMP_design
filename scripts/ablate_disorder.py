"""Ablation of the conditional-disorder descriptor block.

Tests whether modelling the disorder to order transition adds predictive signal
over the existing feature set, which already contains two crude helicity proxies
(mean Chou-Fasman propensity, and the spectral purity of the hydrophobicity
periodogram) that both failed the arrangement ablation.

Three comparisons:

    existing              the full current feature set, including topology
    +disorder             plus the conditional-disorder block
    +disorder, no helix   the block replacing the existing helicity proxies,
                          to check the block is not merely restating them

The kill criterion is the same as every other ablation here: the gain must
exceed the run to run standard deviation across five clustered splits.

Also reports the univariate Spearman of each new feature against the endpoint,
and the tree-based importance of switchability. The point of the block is to
express a turning point, so a near-zero univariate correlation alongside a high
importance is the signature worth looking for; the reverse means the block is
adding a monotone trend that cheaper features already capture.

    uv run python scripts/ablate_disorder.py
    uv run python scripts/ablate_disorder.py --endpoint gram_negative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design.constraints import library_feasible
from amp_design.disorder import (
    CONDITIONAL_DISORDER_FEATURES,
    conditional_disorder_features,
)
from amp_design.features import featurize_many
from amp_design.models import CensoredRegressor

CENSOR_NONE, CENSOR_RIGHT = 0, 1
EXISTING_HELIX_PROXIES = ["mean_helix_propensity", "periodicity_purity"]


def cluster_assign(sequences, identity: float = 0.40) -> dict[str, int]:
    ordered = sorted(set(sequences), key=lambda s: (-len(s), s))
    centroids: list[str] = []
    assignment: dict[str, int] = {}
    cutoff = identity * 100.0
    for s in ordered:
        for k, c in enumerate(centroids):
            if fuzz.ratio(s, c) >= cutoff:
                assignment[s] = k
                break
        else:
            assignment[s] = len(centroids)
            centroids.append(s)
    return assignment


def cluster_split(sequences, clusters, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    ids = sorted({clusters[s] for s in sequences if s in clusters})
    rng.shuffle(ids)
    test_ids = set(ids[: max(1, int(round(len(ids) * test_fraction)))])
    train = [s for s in sequences if clusters.get(s) not in test_ids]
    test = [s for s in sequences if clusters.get(s) in test_ids]
    return train, test


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activity", type=Path, default=Path("data/activity_wide.csv"))
    p.add_argument("--endpoint", default="hc50",
                   choices=["hc50", "gram_positive", "gram_negative", "resistant"])
    p.add_argument("--identity", type=float, default=0.40)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out_path = args.out or Path(f"checkpoint/disorder_{args.endpoint}.json")

    wide = pd.read_csv(args.activity, index_col=0)
    wide.index = wide.index.astype(str)
    wide = wide[[library_feasible(s) for s in wide.index]]
    if args.endpoint not in wide.columns:
        raise SystemExit(f"No {args.endpoint} column in {args.activity}.")

    rows = wide[wide[args.endpoint].notna()]
    sequences = sorted(rows.index)
    censor_col = f"{args.endpoint}_censor"

    print(f"endpoint {args.endpoint}   n = {len(sequences)}")
    print("featurising", flush=True)
    base = featurize_many(sequences, include_topology=True)
    disorder = pd.DataFrame(
        [conditional_disorder_features(s) for s in sequences], index=sequences
    )
    features = base.join(disorder)

    print(f"clustering at {args.identity:.0%} identity", flush=True)
    clusters = cluster_assign(sequences, args.identity)
    print(f"  {len(set(clusters.values()))} clusters\n", flush=True)

    y_all = rows.loc[sequences, args.endpoint].to_numpy(dtype=float)
    print("univariate Spearman of each new feature against the endpoint")
    print("(near zero here plus high importance below is the turning-point signature)")
    univariate = {}
    for name in CONDITIONAL_DISORDER_FEATURES:
        r = float(spearmanr(disorder[name].to_numpy(), y_all).statistic)
        univariate[name] = r
        print(f"  {name:24s} {r:+.3f}")
    print()

    base_cols = list(base.columns)
    disorder_cols = list(CONDITIONAL_DISORDER_FEATURES)
    no_proxy_cols = [c for c in base_cols if c not in EXISTING_HELIX_PROXIES]

    feature_sets = {
        "existing": base_cols,
        "+disorder": base_cols + disorder_cols,
        "+disorder, no helix proxies": no_proxy_cols + disorder_cols,
    }

    results: dict[str, dict[str, float]] = {}
    importances: dict[str, float] = {}

    for name, cols in feature_sets.items():
        runs = []
        for r in range(args.repeats):
            train, test = cluster_split(sequences, clusters, seed=args.seed + r)
            if not train or not test:
                continue

            def block(keys):
                y = rows.loc[keys, args.endpoint].to_numpy(dtype=float)
                c = (
                    rows.loc[keys, censor_col].to_numpy(dtype=int)
                    if censor_col in rows.columns
                    else np.zeros(len(keys), dtype=int)
                )
                return features.loc[keys, cols], y, np.where(c == CENSOR_RIGHT, CENSOR_RIGHT, CENSOR_NONE)

            Xtr, ytr, ctr = block(train)
            Xte, yte, cte = block(test)
            model = CensoredRegressor(seed=args.seed + r).fit(Xtr, ytr, ctr)
            runs.append(model.evaluate(Xte, yte, cte))

            if name == "+disorder" and r == 0:
                booster = model.models_[0]
                gains = booster.feature_importances_
                total = float(gains.sum()) or 1.0
                for col, g in zip(cols, gains):
                    if col in disorder_cols:
                        importances[col] = float(g) / total

        if runs:
            frame = pd.DataFrame(runs)
            results[name] = {
                **{f"{c}_mean": float(frame[c].mean()) for c in frame.columns},
                **{f"{c}_std": float(frame[c].std(ddof=0)) for c in frame.columns},
                "n_features": float(len(cols)),
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"performance": results, "univariate": univariate, "importance": importances},
            indent=2,
        )
    )

    print(f"{'feature set':30s} {'n':>4s}  {'spearman':>16s}  {'within 1 dilution':>18s}")
    print("-" * 76)
    for name, m in results.items():
        print(
            f"{name:30s} {m['n_features']:4.0f}  "
            f"{m.get('spearman_mean', float('nan')):7.3f} +/- {m.get('spearman_std', float('nan')):5.3f}  "
            f"{m.get('within_one_dilution_mean', float('nan')):10.3f} +/- "
            f"{m.get('within_one_dilution_std', float('nan')):5.3f}"
        )

    if importances:
        print("\nshare of total tree gain taken by each new feature")
        for name, share in sorted(importances.items(), key=lambda kv: -kv[1]):
            print(f"  {name:24s} {share:6.2%}")

    if "existing" in results and "+disorder" in results:
        gain = results["+disorder"]["spearman_mean"] - results["existing"]["spearman_mean"]
        noise = max(results["+disorder"]["spearman_std"], results["existing"]["spearman_std"])
        print(f"\ndisorder gain over existing: {gain:+.3f}   run to run sd: {noise:.3f}")
        if gain > noise:
            print("SUPPORTED: conditional-disorder descriptors add signal beyond the "
                  "existing feature set.")
        else:
            print("NOT SUPPORTED: the gain is within run to run noise. Report as a "
                  "tested and unsupported descriptor block.")

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
