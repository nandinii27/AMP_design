"""Ablation of the arrangement descriptor tiers.

Answers one question: does H0 persistence of the hydrophobic point cloud predict
anything that cheaper descriptors do not?

Four nested feature sets, each adding one tier:

    composition   charge, hydrophobicity, length, amino acid fractions
    +moment       hydrophobic moment at helical and sheet periodicity, windowed
                  moment, spectral purity
    +wheel        circular statistics on the helical wheel: arc width, largest
                  interrupting gap, resultant length
    +topology     H0 persistence summaries of the three dimensional hydrophobic
                  point cloud

The claim being tested is narrow and specific. The helical wheel is a projection
down the helix axis and discards the axial coordinate, so two hydrophobic
residues at the same wheel angle but ten positions apart project onto one point
while sitting roughly fifteen angstroms apart in space. The wheel therefore
reports a contiguous hydrophobic face where the molecule has two separate
patches. H0 persistence recovers that distinction.

If the topology tier does not clear the wheel tier by more than the run to run
standard deviation, the descriptor is reported as tested and unsupported, and the
write up says nothing about topology.

Reads the parsed activity table rather than the raw dump, so the unit
normalisation and the assay ceiling censoring are already applied.

    uv run python scripts/ablate.py
    uv run python scripts/ablate.py --endpoint gram_negative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design.constraints import library_feasible
from amp_design.features import TOPOLOGY_FEATURES, WHEEL_FEATURES, featurize_many
from amp_design.models import CensoredRegressor

ACTIVITY = Path("data/activity_wide.csv")
CENSOR_NONE, CENSOR_RIGHT = 0, 1

MOMENT_FEATURES = [
    "moment_helix", "moment_sheet", "moment_windowed", "periodicity_purity",
]


def cluster_assign(sequences, identity: float = 0.40) -> dict[str, int]:
    """Greedy single linkage clustering at a sequence identity threshold.

    Peptide databases are saturated with analogue series differing at one
    position. A random split puts siblings on both sides and inflates every
    metric, so held out performance has to be measured across clusters.
    """
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


def tiers(columns: list[str]) -> dict[str, list[str]]:
    topology = [c for c in TOPOLOGY_FEATURES if c in columns]
    wheel = [c for c in WHEEL_FEATURES if c in columns]
    moment = [c for c in MOMENT_FEATURES if c in columns]
    base = [c for c in columns if c not in set(topology + wheel + moment)]
    return {
        "composition": base,
        "+moment": base + moment,
        "+wheel": base + moment + wheel,
        "+topology": base + moment + wheel + topology,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activity", type=Path, default=ACTIVITY)
    p.add_argument("--endpoint", default="hc50",
                   choices=["hc50", "gram_positive", "gram_negative", "resistant"])
    p.add_argument("--identity", type=float, default=0.40)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("checkpoint/ablation.json"))
    args = p.parse_args()

    if not args.activity.exists():
        raise SystemExit(f"{args.activity} not found. Run scripts/parse_dbaasp.py first.")

    wide = pd.read_csv(args.activity, index_col=0)
    wide.index = wide.index.astype(str)
    wide = wide[[library_feasible(s) for s in wide.index]]

    if args.endpoint not in wide.columns:
        raise SystemExit(f"No {args.endpoint} column in {args.activity}.")

    rows = wide[wide[args.endpoint].notna()]
    sequences = sorted(rows.index)
    censor_col = f"{args.endpoint}_censor"
    n_censored = int((rows.get(censor_col, pd.Series(dtype=float)) == CENSOR_RIGHT).sum())

    print(f"endpoint {args.endpoint}   n = {len(sequences)}   "
          f"right-censored {n_censored} ({n_censored / max(len(sequences), 1):.0%})")
    print("featurising", flush=True)
    features = featurize_many(sequences, include_topology=True)

    print(f"clustering at {args.identity:.0%} identity", flush=True)
    clusters = cluster_assign(sequences, args.identity)
    print(f"  {len(set(clusters.values()))} clusters\n", flush=True)

    feature_sets = tiers(list(features.columns))
    results: dict[str, dict[str, float]] = {}

    for name, cols in feature_sets.items():
        runs = []
        for r in range(args.repeats):
            train, test = cluster_split(sequences, clusters, seed=args.seed + r)
            if not test or not train:
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

        if runs:
            frame = pd.DataFrame(runs)
            results[name] = {
                **{f"{c}_mean": float(frame[c].mean()) for c in frame.columns},
                **{f"{c}_std": float(frame[c].std(ddof=0)) for c in frame.columns},
                "n_features": float(len(cols)),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(f"{'feature set':16s} {'n':>4s}  {'spearman':>16s}  {'within 1 dilution':>18s}")
    print("-" * 62)
    for name, m in results.items():
        print(
            f"{name:16s} {m['n_features']:4.0f}  "
            f"{m.get('spearman_mean', float('nan')):7.3f} +/- {m.get('spearman_std', float('nan')):5.3f}  "
            f"{m.get('within_one_dilution_mean', float('nan')):10.3f} +/- "
            f"{m.get('within_one_dilution_std', float('nan')):5.3f}"
        )

    if "+wheel" in results and "+topology" in results:
        gain = results["+topology"]["spearman_mean"] - results["+wheel"]["spearman_mean"]
        noise = max(results["+topology"]["spearman_std"], results["+wheel"]["spearman_std"])
        print(f"\ntopology gain over wheel: {gain:+.3f}   run to run sd: {noise:.3f}")
        if gain > noise:
            print("SUPPORTED: the topological descriptor adds signal beyond the wheel "
                  "projection. Report the ablation and the mechanism.")
        else:
            print("NOT SUPPORTED: the gain is within run to run noise. Report as a "
                  "tested and unsupported descriptor; make no topological claim.")

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
