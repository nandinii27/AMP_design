"""Rebuild the top list as a factorial experimental design.

Reads the existing generate/library.fasta rather than regenerating, so the
library is untouched and the submission stays valid: every sequence in the new
top list is already a library member.

    uv run python scripts/design_top.py                 # report only, writes nothing
    uv run python scripts/design_top.py --apply          # rewrite generate/top.fasta

Run the report first. If the relative cost of choosing by design rather than by
rank exceeds a few percent, the information is not worth the score and the
greedy top list should stand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design.constraints import NoveltyFilter, candidate_feasible
from amp_design.design import FactorialDesign, factorial_select
from amp_design.features import featurize
from amp_design.funnel import Predictors, explain
from amp_design.scalarize import ChebyshevObjective
from amp_design.select import diversity_report

LIBRARY = Path("generate/library.fasta")
TOP = Path("generate/top.fasta")
REFERENCE = Path("data/antibacterial.fasta")
PREDICTORS = Path("checkpoint/predictors.pkl")
ACTIVITY = Path("data/activity_wide.csv")


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


def write_fasta(sequences, path: Path) -> None:
    with open(path, "w", newline="\n") as fh:
        for i, seq in enumerate(sequences, start=1):
            fh.write(f">seq{i}\n{seq}\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="rewrite generate/top.fasta; otherwise report only")
    p.add_argument("--tolerance", type=float, default=0.12)
    p.add_argument("--shortlist", type=int, default=4000,
                   help="how many library sequences to score for the design")
    p.add_argument("--max-cost", type=float, default=0.05,
                   help="refuse to apply if the design costs more than this fraction")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not LIBRARY.exists():
        raise SystemExit(f"{LIBRARY} not found. Run the generator first.")

    library = read_fasta(LIBRARY)
    current_top = read_fasta(TOP) if TOP.exists() else []
    print(f"library {len(library)}   current top {len(current_top)}")

    predictors = Predictors.load(PREDICTORS)
    objective = ChebyshevObjective.from_activity_table(ACTIVITY)
    novelty = NoveltyFilter(read_fasta(REFERENCE) if REFERENCE.exists() else [])

    # The design can only choose among library members, so that every selected
    # sequence is guaranteed to satisfy the "top list must appear in the library"
    # check the organisers' verifier performs.
    pool = [s for s in library if candidate_feasible(s)]
    print(f"candidate-eligible library members: {len(pool)}")

    # Score the current top first so they are always in contention, then a
    # shortlist of the remaining library.
    ordered = current_top + [s for s in pool if s not in set(current_top)]
    ordered = ordered[: max(args.shortlist, len(current_top))]

    print(f"scoring {len(ordered)} candidates", flush=True)
    feats = [featurize(s, include_topology=True) for s in ordered]
    preds = predictors.predict(feats)
    scores = [objective.risk_adjusted(pr, f) for pr, f in zip(preds, feats)]

    frame = pd.DataFrame(feats, index=ordered)
    frame["score"] = scores

    print("applying novelty filter", flush=True)
    novel = [s for s in ordered if novelty.is_novel(s)]
    frame = frame.loc[novel]
    print(f"  {len(frame)} novel candidates\n")

    design = FactorialDesign(tolerance=args.tolerance, seed=args.seed)
    selected, report = factorial_select(frame, "score", design)

    print("factorial design, cell = (moment_helix, h0_n_patches)")
    print(f"  band size {report['band_size']} at tolerance {report['band_tolerance']}")
    print(f"  thresholds {json.dumps({k: round(v, 3) for k, v in report['thresholds'].items()})}\n")
    print(f"{'cell':16s} {'avail':>6s} {'sel':>4s} {'mean score':>11s} "
          f"{'mean moment':>12s} {'mean charge':>12s}")
    print("-" * 66)
    for cell, m in sorted(report["cells"].items()):
        print(
            f"{cell:16s} {m['available']:6d} {m['selected']:4d} "
            f"{m['mean_score']:11.4f} {m.get('mean_moment_helix', float('nan')):12.3f} "
            f"{m.get('mean_net_charge', float('nan')):12.2f}"
        )

    print(f"\ndesign mean score {report['design_mean_score']:.4f}")
    print(f"greedy mean score {report['greedy_mean_score']:.4f}")
    print(f"relative cost     {report['relative_cost']:+.2%}")

    charge_spread = report["balance"].get("net_charge", {}).get("spread", float("nan"))
    print(f"net charge spread across cells: {charge_spread:.2f}")
    if charge_spread > 1.5:
        print("  WARNING: cells differ in charge by more than one unit. The contrast "
              "is confounded; widen the tolerance band or match on charge explicitly.")

    print("\n" + json.dumps(diversity_report(selected), indent=2))

    out = Path("generate/design_report.json")
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {out}")

    if not args.apply:
        print("\nReport only. Re-run with --apply to rewrite the top list.")
        return 0

    if report["relative_cost"] > args.max_cost:
        print(f"\nRefusing to apply: cost {report['relative_cost']:.2%} exceeds "
              f"--max-cost {args.max_cost:.2%}. The information is not worth the score.")
        return 1

    # Tail: next best novel candidates not already in the pool.
    chosen = set(selected)
    tail_source = frame.drop(index=list(chosen), errors="ignore").nlargest(
        100 - len(selected), "score"
    )
    new_top = selected + list(tail_source.index)

    library_set = set(library)
    missing = [s for s in new_top if s not in library_set]
    if missing:
        raise SystemExit(f"{len(missing)} selected sequences are not in the library.")
    if len(new_top) != 100:
        raise SystemExit(f"Top list has {len(new_top)} sequences, expected 100.")
    violations = novelty.verify_authoritative(new_top)
    if violations:
        raise SystemExit(f"{len(violations)} sequences exceed the identity ceiling.")

    write_fasta(new_top, TOP)
    print(f"\nRewrote {TOP} with the factorial design in positions 1 to {len(selected)}")

    table = explain(new_top, predictors, objective)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv("generate/top_ranking.csv", index=False)
    print("Rewrote generate/top_ranking.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
