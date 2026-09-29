"""Diverse subset selection.

The evaluation draws twenty five peptides at random from the top fifty and
scores the arithmetic mean, so this is subset selection under random sampling,
not a top-k slice. Two consequences.

Position fifty matters exactly as much as position one, because expectation is
linear over the draw.

Correlated liabilities dominate the risk. If the fifty are variations on one
motif, a single shared failure mode removes most of the batch at once.
Diversity here is variance reduction, not presentation.

The objective is monotone submodular under a cardinality constraint, so greedy
selection carries the standard one minus one over e guarantee.

Everything iterates over sorted lists rather than sets. Set iteration order is
stable in practice in CPython but "stable in practice" is not what you want
standing between you and a byte-for-byte reproducibility check.
"""

from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz, process

from .constraints import candidate_feasible


def similarity_matrix(seqs) -> np.ndarray:
    return np.asarray(
        process.cdist(seqs, seqs, scorer=fuzz.ratio, workers=-1), dtype=float
    ) / 100.0


def greedy_diverse(seqs, scores, k: int, redundancy_weight: float = 1.0):
    """Facility-location style greedy selection.

    Marginal gain is a candidate's quality discounted by its maximum similarity
    to anything already chosen, so a near duplicate of a selected peptide
    contributes almost nothing however high its raw score.
    """
    seqs = list(seqs)
    scores = np.asarray(scores, dtype=float)
    if not seqs:
        return []
    k = min(k, len(seqs))

    sim = similarity_matrix(seqs)
    chosen: list[int] = []
    max_sim = np.zeros(len(seqs), dtype=float)
    available = np.ones(len(seqs), dtype=bool)

    for _ in range(k):
        gain = scores * (1.0 - redundancy_weight * max_sim)
        gain[~available] = -np.inf
        pick = int(np.argmax(gain))
        if not np.isfinite(gain[pick]):
            break
        chosen.append(pick)
        available[pick] = False
        max_sim = np.maximum(max_sim, sim[pick])

    return [seqs[i] for i in chosen]


def build_top_list(candidates, scores, novelty, k: int = 100, draw_pool: int = 50,
                   redundancy_weight: float = 1.0, shortlist: int = 600):
    """Assemble the ranked candidate list.

    Positions one to fifty are the draw pool and are selected for diversity as
    well as quality, because that is the set the experimental batch is sampled
    from. Positions fifty one to one hundred are insurance against identity
    filter rejections and are filled with the next best candidates that are also
    maximally divergent from the pool above them.

    Every returned sequence passes the candidate length and residue rules and the
    eighty percent identity ceiling against the reference database.
    """
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ranked = [(candidates[i], float(scores[i])) for i in order]

    eligible = []
    for seq, sc in ranked:
        if not candidate_feasible(seq):
            continue
        if not novelty.is_novel(seq):
            continue
        eligible.append((seq, sc))
        if len(eligible) >= shortlist:
            break

    if not eligible:
        return []

    head = eligible[: max(draw_pool * 4, draw_pool)]
    pool = greedy_diverse(
        [s for s, _ in head], [sc for _, sc in head],
        k=min(draw_pool, len(head)), redundancy_weight=redundancy_weight,
    )

    chosen = set(pool)
    tail_source = [(s, sc) for s, sc in eligible if s not in chosen]
    tail = greedy_diverse(
        [s for s, _ in tail_source], [sc for _, sc in tail_source],
        k=min(k - len(pool), len(tail_source)), redundancy_weight=redundancy_weight,
    )
    return pool + tail


def diversity_report(seqs) -> dict[str, float]:
    if len(seqs) < 2:
        return {"n_sequences": float(len(seqs))}
    sim = similarity_matrix(list(seqs))
    iu = np.triu_indices(len(seqs), k=1)
    values = sim[iu]
    lengths = np.array([len(s) for s in seqs], dtype=float)
    return {
        "n_sequences": float(len(seqs)),
        "mean_pairwise_identity": float(values.mean()),
        "max_pairwise_identity": float(values.max()),
        "mean_length": float(lengths.mean()),
        "min_length": float(lengths.min()),
        "max_length": float(lengths.max()),
    }
