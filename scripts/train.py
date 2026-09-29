"""Train the generator and the four predictive heads.

    uv run python scripts/train.py lm
    uv run python scripts/train.py predictors
    uv run python scripts/train.py windows

The generator trains on data/antibacterial.fasta rather than on DBAASP
sequences. Measured overlap between the two is 92 percent, and the FASTA is
roughly four times larger and already filtered to the challenge alphabet and
length rules, so DBAASP contributes labels here and nothing else.

Splits are by sequence identity cluster, never at random. Peptide databases are
saturated with analogue series differing at one position; a random split puts
siblings on both sides and inflates every held-out metric. Clustered numbers are
lower and real, and they match the regime the competition tests, since submitted
candidates must be at least twenty percent divergent from everything known.
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

from amp_design.constants import POTENCY_THRESHOLD_UM, TOPK_MAX_LENGTH, TOPK_MIN_LENGTH
from amp_design.constraints import library_feasible
from amp_design.features import featurize_many, hydrophobic_moment, net_charge
from amp_design.models import (
    CensoredRegressor,
    ConformalInterval,
    InDistributionGate,
)

CHECKPOINT = Path("checkpoint")
ACTIVITY = Path("data/activity_wide.csv")
REFERENCE = Path("data/antibacterial.fasta")

HEADS = ("gram_positive", "gram_negative", "hc50", "resistant")
CENSOR_NONE, CENSOR_RIGHT = 0, 1


def read_fasta(path: Path) -> list[str]:
    seqs, cur = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return seqs


def cluster_assign(sequences, identity: float = 0.40) -> dict[str, int]:
    """Greedy single-linkage clustering at a sequence identity threshold.

    Longest first, so cluster representatives are the full-length members of an
    analogue series rather than an arbitrary truncation.
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


def load_activity() -> pd.DataFrame:
    if not ACTIVITY.exists():
        raise SystemExit(f"{ACTIVITY} not found. Run scripts/parse_dbaasp.py first.")
    df = pd.read_csv(ACTIVITY, index_col=0)
    df.index = df.index.astype(str)
    return df[[library_feasible(s) for s in df.index]]


def train_lm(args) -> None:
    import torch

    from amp_design import lm

    corpus = [s for s in read_fasta(REFERENCE) if library_feasible(s)]
    corpus = [s for s in corpus if "C" not in s] if args.exclude_cysteine else corpus
    corpus = sorted(set(corpus))
    if not corpus:
        raise SystemExit("Empty training corpus.")

    encoded = [lm.encode(s, net_charge(s)) for s in corpus]

    model = lm.PeptideLM(d_model=args.d_model, n_layers=args.layers)
    cfg = lm.TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)

    print(f"corpus {len(corpus)} sequences, device {cfg.device}")
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    lm.train(model, encoded, cfg, checkpoint_path=CHECKPOINT / "peptide_lm.pt")

    (CHECKPOINT / "lm_summary.json").write_text(
        json.dumps(
            {
                "corpus_size": len(corpus),
                "source": str(REFERENCE),
                "cysteine_excluded": bool(args.exclude_cysteine),
                "epochs": args.epochs,
                "seed": args.seed,
                "parameters": int(sum(p.numel() for p in model.parameters())),
            },
            indent=2,
        )
    )
    print(f"\nSaved {CHECKPOINT / 'peptide_lm.pt'}")


def train_predictors(args) -> None:
    wide = load_activity()
    sequences = sorted(wide.index)
    print(f"{len(sequences)} sequences with activity data")

    print("clustering at 40 percent identity (this is the slow step)")
    clusters = cluster_assign(sequences, identity=args.identity)
    print(f"  {len(set(clusters.values()))} clusters")

    print("featurising")
    features = featurize_many(sequences, include_topology=True)

    heads, metrics, conformal = {}, {}, {}
    for name in HEADS:
        if name not in wide.columns:
            print(f"  {name:16s} absent from activity table, skipped")
            continue
        rows = wide[wide[name].notna()]
        subset = sorted(rows.index)
        if len(subset) < 200:
            print(f"  {name:16s} only {len(subset)} rows, skipped")
            continue

        train, test = cluster_split(subset, clusters, seed=args.seed)
        censor_col = f"{name}_censor"

        def block(keys):
            X = features.loc[keys]
            y = rows.loc[keys, name].to_numpy(dtype=float)
            c = (
                rows.loc[keys, censor_col].to_numpy(dtype=int)
                if censor_col in rows.columns
                else np.zeros(len(keys), dtype=int)
            )
            c = np.where(c == CENSOR_RIGHT, CENSOR_RIGHT, CENSOR_NONE)
            return X, y, c

        Xtr, ytr, ctr = block(train)
        model = CensoredRegressor(seed=args.seed).fit(Xtr, ytr, ctr)

        Xte, yte, cte = block(test)
        head_metrics = model.evaluate(Xte, yte, cte)
        head_metrics["n_train"] = float(len(train))
        head_metrics["n_test"] = float(len(test))

        observed = cte == CENSOR_NONE
        if observed.sum() > 5:
            residuals = yte[observed] - model.predict(Xte)[observed]
            conformal[name] = ConformalInterval(alpha=0.2).calibrate(residuals)
            head_metrics["conformal_width_log2"] = float(conformal[name].width_)

        heads[name] = model
        metrics[name] = head_metrics
        print(
            f"  {name:16s} n={len(subset):6d}  "
            f"spearman {head_metrics.get('spearman', float('nan')):.3f}  "
            f"within-1-dilution {head_metrics.get('within_one_dilution', float('nan')):.3f}"
        )

    threshold = np.log2(POTENCY_THRESHOLD_UM)
    potent_mask = pd.Series(False, index=wide.index)
    for col in ("gram_positive", "gram_negative"):
        if col in wide.columns:
            potent_mask |= wide[col] <= threshold
    potent = [s for s in wide.index[potent_mask] if "C" not in s]
    print(f"\nin-distribution gate fitted on {len(potent)} potent cysteine-free peptides")
    gate = InDistributionGate(seed=args.seed).fit(features.loc[potent])

    import pickle

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    with (CHECKPOINT / "predictors.pkl").open("wb") as fh:
        pickle.dump(
            {
                "feature_names": list(features.columns),
                "heads": heads,
                "gate": gate,
                "conformal": conformal,
                "metrics": metrics,
            },
            fh,
        )
    (CHECKPOINT / "predictor_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float)
    )
    print(f"\nSaved {CHECKPOINT / 'predictors.pkl'}")


def calibrate_windows(args) -> None:
    """Refit the design windows on the subpopulation this design actually targets.

    Linear, cysteine free, potent, and where haemolysis data exists, selective.
    Quantiles rather than min and max, so a handful of unusual peptides cannot
    widen the window to uselessness.
    """
    wide = load_activity()
    threshold = np.log2(POTENCY_THRESHOLD_UM)

    potent = pd.Series(False, index=wide.index)
    for col in ("gram_positive", "gram_negative"):
        if col in wide.columns:
            potent |= wide[col] <= threshold

    base = [s for s in wide.index[potent] if "C" not in s]
    populations = {"potent, cysteine free": base}

    if "hc50" in wide.columns:
        mic_cols = [c for c in ("gram_positive", "gram_negative") if c in wide.columns]
        best = wide.loc[base, mic_cols].min(axis=1)
        window = wide.loc[base, "hc50"] - best
        populations["plus safety window >= 4x"] = [
            s for s in base if window.get(s, np.nan) >= 2.0
        ]
        populations["plus in top-k length band"] = [
            s
            for s in base
            if window.get(s, np.nan) >= 2.0 and TOPK_MIN_LENGTH <= len(s) <= TOPK_MAX_LENGTH
        ]

    out = {}
    for label, pop in populations.items():
        if len(pop) < 50:
            print(f"\n{label}: only {len(pop)} sequences, skipped")
            continue
        charges = np.array([net_charge(s) for s in pop])
        moments = np.array([hydrophobic_moment(s) for s in pop])
        lengths = np.array([len(s) for s in pop], dtype=float)
        entry = {
            "n": len(pop),
            "charge": [float(np.quantile(charges, 0.10)), float(np.quantile(charges, 0.90))],
            "charge_median": float(np.median(charges)),
            "moment_floor": float(np.quantile(moments, 0.10)),
            "moment_median": float(np.median(moments)),
            "length": [float(np.quantile(lengths, 0.10)), float(np.quantile(lengths, 0.90))],
        }
        out[label] = entry
        print(f"\n{label}  (n = {len(pop)})")
        print(f"  charge  10th {entry['charge'][0]:5.2f}  median {entry['charge_median']:5.2f}  90th {entry['charge'][1]:5.2f}")
        print(f"  moment  10th {entry['moment_floor']:5.3f}  median {entry['moment_median']:5.3f}")
        print(f"  length  10th {entry['length'][0]:5.0f}  90th {entry['length'][1]:5.0f}")

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT / "windows.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {CHECKPOINT / 'windows.json'}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("lm")
    a.add_argument("--epochs", type=int, default=30)
    a.add_argument("--batch-size", type=int, default=128)
    a.add_argument("--d-model", type=int, default=128)
    a.add_argument("--layers", type=int, default=4)
    a.add_argument("--exclude-cysteine", action="store_true",
                   help="drop the ~26 percent cysteine-containing corpus this design never emits")
    a.add_argument("--seed", type=int, default=42)
    a.set_defaults(func=train_lm)

    b = sub.add_parser("predictors")
    b.add_argument("--identity", type=float, default=0.40)
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=train_predictors)

    c = sub.add_parser("windows")
    c.set_defaults(func=calibrate_windows)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
