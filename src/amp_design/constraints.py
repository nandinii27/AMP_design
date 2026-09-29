"""Hard constraints and the novelty filter.

Hard constraints are enforced by rejection, never by a penalty term. A penalty
lets the optimiser trade a violation against predicted quality, which is how a
search ends up recommending a sequence the model has never seen anything like.
Soft physicochemical preferences live in scalarize.py and are multiplicative,
so they cannot be bought off either.

Two identity implementations are used deliberately. rapidfuzz screens the pool,
because it batches across thousands of references with multiple workers. The
final list is re-verified with the Levenshtein package, which is what the
challenge pins as a dependency and therefore what the organisers' harness runs.
The two formulas agree, but agreement between a fast path and the authoritative
path should be asserted rather than assumed when the consequence is rejection of
the whole submission.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz, process

from .constants import (
    ALPHABET,
    FORBIDDEN_RESIDUES,
    MAX_IDENTITY_TO_REFERENCE,
    MAX_LENGTH,
    MIN_LENGTH,
    TOPK_MAX_LENGTH,
    TOPK_MIN_LENGTH,
)

_ALPHABET_SET = frozenset(ALPHABET)


def valid_alphabet(seq: str) -> bool:
    return bool(seq) and set(seq) <= _ALPHABET_SET


def library_feasible(seq: str) -> bool:
    """Rules every sequence in the library must satisfy."""
    return valid_alphabet(seq) and MIN_LENGTH <= len(seq) <= MAX_LENGTH


def candidate_feasible(seq: str) -> bool:
    """Additional rules for the ranked top list.

    The length window is tightened well inside the legal range. Solid phase
    synthesis adds one residue at a time and yields compound, so failure rate
    climbs steeply with length, and a peptide that fails is not retested.
    Sequences below twelve residues span fewer than three and a half helical
    turns and cannot present a usable amphipathic face without the terminal
    modifications the challenge forbids.
    """
    if not library_feasible(seq):
        return False
    if not TOPK_MIN_LENGTH <= len(seq) <= TOPK_MAX_LENGTH:
        return False
    if set(seq) & FORBIDDEN_RESIDUES:
        return False
    return True


@dataclass
class NoveltyFilter:
    """Identity screen against known antibacterial peptides.

    Two different bars. No sequence anywhere in the fifty thousand library may be
    identical to a reference. Top-100 candidates additionally may not exceed
    eighty percent Levenshtein identity to any reference, which on a twenty
    residue peptide means at least four edits from everything known. Single and
    double point mutants of natural peptides are ruled out by construction, so
    any local search must start from de novo scaffolds rather than templates.
    """

    references: list[str]
    max_identity: float = MAX_IDENTITY_TO_REFERENCE

    def __post_init__(self) -> None:
        self.references = [r.upper() for r in self.references if valid_alphabet(r.upper())]
        self._exact = frozenset(self.references)
        self._cutoff = self.max_identity * 100.0

    def is_exact_match(self, seq: str) -> bool:
        return seq in self._exact

    def max_identity_to_any(self, seq: str) -> float:
        hit = process.extractOne(seq, self.references, scorer=fuzz.ratio, score_cutoff=0.0)
        return (hit[1] / 100.0) if hit else 0.0

    def closest_reference(self, seq: str) -> tuple[str | None, float]:
        hit = process.extractOne(seq, self.references, scorer=fuzz.ratio, score_cutoff=0.0)
        return (hit[0], hit[1] / 100.0) if hit else (None, 0.0)

    def is_novel(self, seq: str) -> bool:
        hit = process.extractOne(
            seq, self.references, scorer=fuzz.ratio, score_cutoff=self._cutoff
        )
        return hit is None

    def filter(self, seqs):
        return [s for s in seqs if self.is_novel(s)]

    def verify_authoritative(self, seqs) -> list[dict]:
        """Re-check with the Levenshtein package the challenge pins.

        Returns one row per violating sequence. An empty list means the submitted
        set clears the ceiling under the same implementation the organisers run.

        Raises ImportError rather than falling back silently: a novelty check
        that quietly degrades to a different metric is worse than one that fails
        loudly, because the cost of being wrong is rejection of the whole list.
        """
        import Levenshtein

        violations = []
        for seq in seqs:
            worst_ref, worst = None, 0.0
            for ref in self.references:
                r = Levenshtein.ratio(seq, ref)
                if r > worst:
                    worst_ref, worst = ref, r
            if worst > self.max_identity:
                violations.append(
                    {"sequence": seq, "closest_reference": worst_ref,
                     "identity": round(worst, 4)}
                )
        return violations


def deduplicate(seqs):
    """Preserve order, drop repeats. The challenge requires unique sequences."""
    seen = set()
    out = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
