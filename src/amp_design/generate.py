"""Entry point for `uv run generate`.

The organisers' harness runs this with no arguments at all, twice, from a fresh
clone, and compares the two outputs byte for byte. Every default therefore has
to be the competition's specification, every source of randomness has to derive
from the single seed, and everything the run reads must be committed to the
repository.

Output goes to a directory named after the entry point, matching the reference
implementation, so the harness finds generate/library.fasta and generate/top.fasta.

The final gate mirrors every check in scripts/verify_submission.py and raises
before writing anything. A file that will be rejected is worse than no file,
because it looks like success.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from .constants import ALPHABET, MAX_IDENTITY_TO_REFERENCE, MAX_LENGTH, MIN_LENGTH
from .constraints import NoveltyFilter, deduplicate, library_feasible
from .funnel import FunnelConfig, Predictors, explain, run
from .scalarize import ChebyshevObjective
from .select import diversity_report

CHECKPOINT_LM = Path("checkpoint/peptide_lm.pt")
CHECKPOINT_PREDICTORS = Path("checkpoint/predictors.pkl")
REFERENCE_FASTA = Path("data/antibacterial.fasta")
ACTIVITY_TABLE = Path("data/activity_wide.csv")


def set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


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


def write_fasta(sequences: list[str], path: Path) -> None:
    with open(path, "w", newline="\n") as fh:
        for i, seq in enumerate(sequences, start=1):
            fh.write(f">seq{i}\n{seq}\n")


def sample_pool(checkpoint: Path, target: int, seed: int, max_len: int) -> list[str]:
    """Draw a pool from the language model, spread across conditioning buckets.

    Oversampled relative to the library because the funnel discards heavily and
    the library must end at exactly the requested size. Spreading over charge
    buckets and temperatures widens coverage, which matters because the first
    screening phase scores the whole library for diversity and physicochemical
    distribution rather than for predicted activity.
    """
    from . import lm

    model = lm.load(checkpoint)
    charges = (2.0, 4.0, 6.0, 8.0, 10.0)
    temperatures = (0.9, 1.0, 1.15)
    per_combo = int(np.ceil(target / (len(charges) * len(temperatures))))

    out: list[str] = []
    k = 0
    for charge in charges:
        for temperature in temperatures:
            out.extend(
                lm.sample(model, n=per_combo, seed=seed + k, temperature=temperature,
                          charge_target=charge, max_len=max_len)
            )
            k += 1

    out = deduplicate([s for s in out if library_feasible(s)])
    stall = 0
    while len(out) < target and stall < 12:
        before = len(out)
        extra = lm.sample(model, n=target - len(out), seed=seed + 5000 + k,
                          temperature=1.2, max_len=max_len)
        out = deduplicate(out + [s for s in extra if library_feasible(s)])
        k += 1
        stall = stall + 1 if len(out) == before else 0
    return out


def validate(library, top, references, n_expected, top_k, novelty) -> None:
    """Mirror every check the organisers' verifier performs."""
    errors: list[str] = []
    alphabet = set(ALPHABET)

    if len(library) != n_expected:
        errors.append(f"library has {len(library)}, expected {n_expected}")
    if len(set(library)) != len(library):
        errors.append("library contains duplicates")
    for seq in library:
        if set(seq) - alphabet:
            errors.append(f"invalid characters in {seq[:24]}")
            break
        if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
            errors.append(f"length {len(seq)} out of range for {seq[:24]}")
            break

    if len(top) != top_k:
        errors.append(f"top list has {len(top)}, expected {top_k}")
    if len(set(top)) != len(top):
        errors.append("top list contains duplicates")

    library_set = set(library)
    missing = [s for s in top if s not in library_set]
    if missing:
        errors.append(f"{len(missing)} top sequences absent from the library")

    if references:
        overlap = library_set & set(references)
        if overlap:
            errors.append(f"{len(overlap)} library sequences exactly match a reference")
        violations = novelty.verify_authoritative(top)
        if violations:
            worst = max(violations, key=lambda v: v["identity"])
            errors.append(
                f"{len(violations)} top sequences exceed {MAX_IDENTITY_TO_REFERENCE} "
                f"identity (worst {worst['identity']})"
            )

    if errors:
        raise ValueError(
            "Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="generate")
    p.add_argument("--n-sequences", type=int, default=50_000)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--oversample", type=float, default=1.4)
    p.add_argument("--refine-seeds", type=int, default=150)
    p.add_argument("--anneal-steps", type=int, default=400)
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    entry_point = Path(sys.argv[0]).stem
    args = build_parser().parse_args()
    set_all_seeds(args.seed)

    out_dir = Path(entry_point)
    out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    references = read_fasta(REFERENCE_FASTA) if REFERENCE_FASTA.exists() else []
    if not references:
        print(f"WARNING: {REFERENCE_FASTA} not found, novelty checks skipped")
    novelty = NoveltyFilter(references)

    predictors = Predictors.load(CHECKPOINT_PREDICTORS)
    objective = ChebyshevObjective.from_activity_table(ACTIVITY_TABLE)
    config = FunnelConfig(
        library_size=args.n_sequences,
        top_k=args.top_k,
        refine_seeds=args.refine_seeds,
        anneal_steps=args.anneal_steps,
        seed=args.seed,
    )

    pool_target = int(args.n_sequences * args.oversample)
    if verbose:
        print(f"sampling {pool_target} candidates from the language model", flush=True)
    pool = sample_pool(CHECKPOINT_LM, pool_target, args.seed, args.length)
    if verbose:
        print(f"  pool {len(pool)}", flush=True)

    library, top, report = run(pool, predictors, novelty, objective, config, verbose)

    validate(library, top, references, args.n_sequences, args.top_k, novelty)

    write_fasta(library, out_dir / "library.fasta")
    print(f"Generated {len(library)} sequences -> {out_dir / 'library.fasta'}")
    write_fasta(top, out_dir / "top.fasta")
    print(f"Top {len(top)} sequences -> {out_dir / 'top.fasta'}")

    table = explain(top, predictors, objective)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv(out_dir / "top_ranking.csv", index=False)

    report["seed"] = args.seed
    report["safety_window_target"] = objective.ideals["safety_window"]
    report["draw_pool"] = config.draw_pool
    report["top_diversity"] = diversity_report(top)
    report["pool_diversity"] = diversity_report(top[:50])
    (out_dir / "run_summary.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
