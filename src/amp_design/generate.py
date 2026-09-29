"""Entry point for `uv run generate`.

STUB VERSION. The generator is a uniform random sampler and produces nothing of
scientific value. Its purpose is to prove the packaging contract end to end
against the organisers' validator before any modelling work exists, so that a
later failure can only be a modelling problem.

The seeding and the final validation gate below are the real implementation and
carry over unchanged.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MIN_LENGTH = 8
MAX_LENGTH = 50
MAX_IDENTITY = 0.80

REFERENCE_FASTA = Path("data/antibacterial.fasta")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_fasta(path: Path) -> list[str]:
    seqs, current = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                seqs.append("".join(current))
                current = []
        else:
            current.append(line.upper())
    if current:
        seqs.append("".join(current))
    return seqs


def write_fasta(sequences: list[str], path: Path) -> None:
    with open(path, "w", newline="\n") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n{seq}\n")


def generate(n_sequences: int, *, length: int = MAX_LENGTH, seed: int = 42) -> list[str]:
    """Uniform random sampling over the canonical alphabet.

    Placeholder. Draws more than requested and deduplicates, because the library
    must contain exactly n_sequences unique entries and collisions are possible
    at short lengths.
    """
    rng = np.random.default_rng(seed)
    alphabet = np.array(list(ALPHABET))

    seen: set[str] = set()
    ordered: list[str] = []
    while len(ordered) < n_sequences:
        batch = n_sequences - len(ordered) + 1000
        lengths = rng.integers(MIN_LENGTH, min(length, MAX_LENGTH) + 1, size=batch)
        draws = rng.integers(0, len(alphabet), size=(batch, MAX_LENGTH))
        for row, size in zip(draws, lengths):
            seq = "".join(alphabet[row[:size]])
            if seq not in seen:
                seen.add(seq)
                ordered.append(seq)
                if len(ordered) == n_sequences:
                    break
    return ordered


def score(sequences: list[str]) -> list[float]:
    """Placeholder ranking. Replaced by the Chebyshev objective."""
    return [float(i) for i in range(len(sequences), 0, -1)]


def validate(library: list[str], top: list[str], references: list[str],
             n_expected: int, top_k: int) -> None:
    """Fail here rather than in the organisers' harness.

    Mirrors every check in scripts/verify_submission.py. Raising before writing
    is deliberate: a file that will be rejected is worse than no file, because it
    looks like success.
    """
    errors: list[str] = []

    if len(library) != n_expected:
        errors.append(f"library has {len(library)} sequences, expected {n_expected}")
    if len(set(library)) != len(library):
        errors.append("library contains duplicates")
    for seq in library:
        if set(seq) - set(ALPHABET):
            errors.append(f"invalid characters in {seq[:20]}")
            break
        if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
            errors.append(f"length {len(seq)} out of range for {seq[:20]}")
            break

    if len(top) != top_k:
        errors.append(f"top list has {len(top)} sequences, expected {top_k}")
    if len(set(top)) != len(top):
        errors.append("top list contains duplicates")

    library_set = set(library)
    missing = [s for s in top if s not in library_set]
    if missing:
        errors.append(f"{len(missing)} top sequences are absent from the library")

    if references:
        reference_set = set(references)
        overlap = library_set & reference_set
        if overlap:
            errors.append(f"{len(overlap)} library sequences exactly match a reference")

        import Levenshtein

        violations = 0
        for seq in top:
            for ref in references:
                if Levenshtein.ratio(seq, ref) > MAX_IDENTITY:
                    violations += 1
                    break
        if violations:
            errors.append(f"{violations} top sequences exceed {MAX_IDENTITY} identity")

    if errors:
        raise ValueError(
            "Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def main() -> None:
    entry_point = Path(sys.argv[0]).stem

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sequences", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    out_dir = Path(entry_point)
    out_dir.mkdir(parents=True, exist_ok=True)

    library = generate(args.n_sequences, length=args.length, seed=args.seed)

    scores = score(library)
    ranked = sorted(zip(scores, library), key=lambda x: (-x[0], x[1]))
    top = [seq for _, seq in ranked[: args.top_k]]

    references = read_fasta(REFERENCE_FASTA) if REFERENCE_FASTA.exists() else []
    if not references:
        print(f"WARNING: {REFERENCE_FASTA} not found, novelty checks skipped")

    validate(library, top, references, args.n_sequences, args.top_k)

    write_fasta(library, out_dir / "library.fasta")
    print(f"Generated {len(library)} sequences -> {out_dir / 'library.fasta'}")

    write_fasta(top, out_dir / "top.fasta")
    print(f"Top {len(top)} sequences -> {out_dir / 'top.fasta'}")