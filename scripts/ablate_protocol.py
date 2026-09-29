"""Evaluation protocol ablation.

Tests whether reported peptide activity predictability is inflated by two
independent artifacts:

  1. Homology leakage. Peptide databases are saturated with analogue series
     differing at a single position, so a random split places siblings on both
     sides of the partition. Vishnepolsky et al. (Brief Bioinform 23(5), bbac343)
     document a related bias for binary classification and conclude that "all the
     benchmark analyses that have been performed for AMP prediction models are
     significantly biased". This extends that test to quantitative endpoints.

  2. Assay ceilings recorded as measurements. Reported haemolysis values pile up
     at round levels (100 appears 1244 times bare against 254 times as ">100"),
     because most authors who tested to the top of their dilution series and saw
     no lysis omitted the inequality. Published work commonly averages such
     values, or averages ranges, which converts the experiment's budget into an
     apparent property of the molecule.

A two by two over the same peptides, the same features, the same model class:

                        ceilings as observed   ceilings as censored
    random split              cell A                 cell B
    clustered split           cell C                 cell D

Cell A is the protocol most published results use. Cell D is the protocol used
for the submission's predictors. B and C isolate each artifact.

The comparison tests the censoring assumption as much as it tests the field's
protocol: if cell A and cell D differ little, the artifacts are minor and this is
a footnote rather than a finding.

    uv run python scripts/ablate_protocol.py
    uv run python scripts/ablate_protocol.py --endpoint gram_negative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design.constraints import library_feasible
from amp_design.features import featurize_many
from amp_design.models import CensoredRegressor

CENSOR_NONE, CENSOR_RIGHT = 0, 1


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


def clustered_split(sequences, clusters, test_fraction, seed):
    rng = np.random.default_rng(seed)
    ids = sorted({clusters[s] for s in sequences if s in clusters})
    rng.shuffle(ids)
    test_ids = set(ids[: max(1, int(round(len(ids) * test_fraction)))])
    train = [s for s in sequences if clusters.get(s) not in test_ids]
    test = [s for s in sequences if clusters.get(s) in test_ids]
    return train, test


def random_split(sequences, test_fraction, seed):
    """The protocol most published work uses. Siblings land on both sides."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sequences))
    n_test = max(1, int(round(len(sequences) * test_fraction)))
    test = [sequences[i] for i in idx[:n_test]]
    train = [sequences[i] for i in idx[n_test:]]
    return train, test


def split_leakage(train, test, clusters) -> float:
    """Fraction of test sequences whose cluster also appears in training.

    This is the quantity the whole experiment turns on, so it is measured rather
    than assumed.
    """
    train_clusters = {clusters[s] for s in train if s in clusters}
    if not test:
        return float("nan")
    leaked = sum(1 for s in test if clusters.get(s) in train_clusters)
    return leaked / len(test)


def evaluate(model, X, y, censor, treat_censored_as_observed: bool) -> dict[str, float]:
    """Metrics on the test fold.

    Under the observed treatment, censored rows are scored as though their bound
    were a measurement, which is what a pipeline that ignores censoring does.
    Under the censored treatment they are excluded from point metrics and scored
    separately by whether the prediction respects the bound.
    """
    pred = model.predict(X)
    if treat_censored_as_observed:
        mask = np.ones(len(y), dtype=bool)
    else:
        mask = censor == CENSOR_NONE

    out: dict[str, float] = {"n_test": float(len(y)), "n_scored": float(mask.sum())}
    if mask.sum() > 2:
        out["spearman"] = float(spearmanr(y[mask], pred[mask]).statistic)
        out["pearson"] = float(pearsonr(y[mask], pred[mask]).statistic)
        out["mae_log2"] = float(np.mean(np.abs(y[mask] - pred[mask])))
        out["within_one_dilution"] = float(np.mean(np.abs(y[mask] - pred[mask]) <= 1.0))
    return out


def run_cell(sequences, features, rows, endpoint, censor_col, clusters,
             split_kind: str, censoring: str, repeats: int, seed: int,
             test_fraction: float) -> dict[str, float]:
    runs, leakages = [], []

    for r in range(repeats):
        if split_kind == "random":
            train, test = random_split(sequences, test_fraction, seed + r)
        else:
            train, test = clustered_split(sequences, clusters, test_fraction, seed + r)
        if not train or not test:
            continue
        leakages.append(split_leakage(train, test, clusters))

        def block(keys):
            y = rows.loc[keys, endpoint].to_numpy(dtype=float)
            c = (
                rows.loc[keys, censor_col].to_numpy(dtype=int)
                if censor_col in rows.columns
                else np.zeros(len(keys), dtype=int)
            )
            c = np.where(c == CENSOR_RIGHT, CENSOR_RIGHT, CENSOR_NONE)
            if censoring == "observed":
                # The bound is fed to the model as if it were a measurement.
                c = np.zeros_like(c)
            return features.loc[keys], y, c

        Xtr, ytr, ctr = block(train)
        Xte, yte, cte_model = block(test)
        cte_true = (
            rows.loc[test, censor_col].to_numpy(dtype=int)
            if censor_col in rows.columns
            else np.zeros(len(test), dtype=int)
        )
        cte_true = np.where(cte_true == CENSOR_RIGHT, CENSOR_RIGHT, CENSOR_NONE)

        model = CensoredRegressor(seed=seed + r).fit(Xtr, ytr, ctr)
        runs.append(
            evaluate(model, Xte, yte, cte_true,
                     treat_censored_as_observed=(censoring == "observed"))
        )

    if not runs:
        return {}
    frame = pd.DataFrame(runs)
    out = {f"{c}_mean": float(frame[c].mean()) for c in frame.columns}
    out.update({f"{c}_std": float(frame[c].std(ddof=0)) for c in frame.columns})
    out["train_test_leakage"] = float(np.mean(leakages)) if leakages else float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activity", type=Path, default=Path("data/activity_wide.csv"))
    p.add_argument("--endpoint", default="hc50",
                   choices=["hc50", "gram_positive", "gram_negative", "resistant"])
    p.add_argument("--identity", type=float, default=0.40)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out_path = args.out or Path(f"checkpoint/protocol_{args.endpoint}.json")

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

    cells = {
        "A random split, ceilings observed": ("random", "observed"),
        "B random split, ceilings censored": ("random", "censored"),
        "C clustered split, ceilings observed": ("clustered", "observed"),
        "D clustered split, ceilings censored": ("clustered", "censored"),
    }

    results: dict[str, dict[str, float]] = {}
    for label, (split_kind, censoring) in cells.items():
        print(f"  running {label}", flush=True)
        results[label] = run_cell(
            sequences, features, rows, args.endpoint, censor_col, clusters,
            split_kind, censoring, args.repeats, args.seed, args.test_fraction,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'cell':40s} {'spearman':>16s} {'pearson':>16s} {'leakage':>9s}")
    print("-" * 86)
    for label, m in results.items():
        if not m:
            continue
        print(
            f"{label:40s} "
            f"{m.get('spearman_mean', float('nan')):7.3f} +/- {m.get('spearman_std', float('nan')):5.3f} "
            f"{m.get('pearson_mean', float('nan')):7.3f} +/- {m.get('pearson_std', float('nan')):5.3f} "
            f"{m.get('train_test_leakage', float('nan')):8.0%}"
        )

    a = results.get("A random split, ceilings observed", {}).get("spearman_mean")
    b = results.get("B random split, ceilings censored", {}).get("spearman_mean")
    c = results.get("C clustered split, ceilings observed", {}).get("spearman_mean")
    d = results.get("D clustered split, ceilings censored", {}).get("spearman_mean")

    if None not in (a, b, c, d):
        print(f"\ntotal inflation, A minus D          {a - d:+.3f}")
        print(f"  attributable to homology leakage   {((a - c) + (b - d)) / 2:+.3f}")
        print(f"  attributable to ceiling labelling  {((a - b) + (c - d)) / 2:+.3f}")
        if a - d < 0.10:
            print("\nBelow the 0.10 threshold: the artifacts are minor. Report as a "
                  "footnote, not a finding.")
        else:
            print("\nAbove the 0.10 threshold: reported predictability on this endpoint "
                  "is materially inflated by evaluation protocol.")

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
