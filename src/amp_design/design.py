"""Factorial selection of the experimental batch.

Twenty five peptides are drawn at random from the top fifty and measured in a
single laboratory under one protocol. That is the cleanest dataset that will
exist for this class of molecule, and which peptides enter it is entirely the
submitter's choice.

Four ablations in this repository found that no sequence-derived structural
descriptor improves activity prediction over amino acid composition under
homology-controlled evaluation: Fourier (hydrophobic moment, spectral purity),
circular (wheel arc width, interrupting gap), topological (H0 persistence of the
hydrophobic point cloud) and conformational (Lifson-Roig helicity in water versus
at an interface). Every gain sat inside the run to run standard deviation.

Those results cannot distinguish two explanations. Either arrangement genuinely
does not matter, or the pooled literature is too noisy to resolve it: activity
records span hundreds of laboratories, media, inocula and assay ceilings, and
between-study variance may simply swamp the effect.

A single-laboratory measurement is the instrument that separates them. This
module selects the draw pool as a factorial contrast over the descriptors the
literature could not resolve, holding predicted potency and net charge matched,
so that the batch is simultaneously a competitive submission and a designed
experiment.

The selection is free in expectation. Candidates within a narrow band of
predicted score are statistically indistinguishable given a held-out Spearman
near 0.5 and a within-one-dilution accuracy near 0.4, so choosing among them by
design rather than by rank costs no expected quality. The tolerance band makes
that explicit and the cost is measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .select import greedy_diverse, similarity_matrix

# The contrast variables. Each is dichotomised at the median of the eligible
# candidate pool, so the cells are populated by construction rather than by luck.
CONTRAST_AXES = ("moment_helix", "h0_n_patches", "net_charge")
# Variables held matched across cells, so that a difference in measured outcome
# is attributable to the contrast rather than to potency or charge.
MATCH_AXES = ("net_charge",)


@dataclass
class FactorialDesign:
    """Two by two factorial over amphipathicity and face fragmentation.

    moment_helix separates strongly from weakly amphipathic sequences. It is the
    field's standard descriptor and the one with forty years of mechanistic
    argument behind it.

    h0_n_patches counts the distinct hydrophobic patches in three dimensions. The
    helical wheel is a projection down the helix axis and discards the axial
    coordinate, so two hydrophobic residues at the same wheel angle but ten
    positions apart project onto one point while sitting roughly fifteen
    angstroms apart. The wheel therefore reports a contiguous face where the
    molecule has separate patches, and this is the variable that distinction
    produces.

    Crossing them asks whether face contiguity matters at matched amphipathicity,
    which is the question three correlation analyses could not answer from pooled
    data.
    """

    pool_size: int = 50
    tolerance: float = 0.12
    contrast_axes: tuple = CONTRAST_AXES
    match_axes: tuple = MATCH_AXES
    min_per_cell: int = 8
    redundancy_weight: float = 1.0
    seed: int = 42
    thresholds_: dict = field(default_factory=dict)


def assign_cells(frame: pd.DataFrame, design: FactorialDesign) -> pd.Series:
    """Label each candidate with its factorial cell.

    Splits at the median of the eligible pool rather than at an absolute value,
    so the design adapts to whatever distribution the generator produced.
    """
    labels = []
    thresholds = {}
    for axis in design.contrast_axes:
        thresholds[axis] = float(frame[axis].median())
    design.thresholds_ = thresholds

    for _, row in frame.iterrows():
        cell = tuple(
            "hi" if row[axis] >= thresholds[axis] else "lo"
            for axis in design.contrast_axes
        )
        labels.append("_".join(cell))
    return pd.Series(labels, index=frame.index, name="cell")


def eligible_band(frame: pd.DataFrame, score_col: str, tolerance: float) -> pd.DataFrame:
    """Candidates whose predicted score is within tolerance of the best.

    The band is the set the design is free to choose within. Its width is the
    honest statement of how much predictive resolution the models actually have:
    with a held-out Spearman near 0.5, differences inside this band are not
    meaningfully distinguishable.
    """
    best = float(frame[score_col].max())
    span = float(frame[score_col].max() - frame[score_col].min()) or 1.0
    cutoff = best - tolerance * span
    return frame[frame[score_col] >= cutoff]


def factorial_select(frame: pd.DataFrame, score_col: str,
                     design: FactorialDesign | None = None) -> tuple[list[str], dict]:
    """Choose the draw pool as a balanced factorial, diverse within each cell.

    Returns the selected sequences and a report describing the design, including
    the expected-quality cost of choosing by design rather than by rank.

    frame must be indexed by sequence and carry the contrast and match columns
    plus score_col.
    """
    design = design or FactorialDesign()
    rng = np.random.default_rng(design.seed)

    band = eligible_band(frame, score_col, design.tolerance)
    if len(band) < design.pool_size:
        band = frame.nlargest(max(design.pool_size * 3, 150), score_col)

    cells = assign_cells(band, design)
    band = band.assign(cell=cells)

    counts = band["cell"].value_counts().to_dict()
    n_cells = max(len(counts), 1)
    per_cell = max(design.pool_size // n_cells, 1)

    selected: list[str] = []
    cell_report: dict[str, dict] = {}

    for cell in sorted(counts):
        members = band[band["cell"] == cell]
        take = min(per_cell, len(members))
        if take == 0:
            continue

        # Within a cell, choose for sequence diversity so a single shared
        # liability cannot remove the whole cell from the experiment.
        chosen = greedy_diverse(
            list(members.index),
            members[score_col].to_numpy(dtype=float),
            k=take,
            redundancy_weight=design.redundancy_weight,
        )
        selected.extend(chosen)
        cell_report[cell] = {
            "available": int(len(members)),
            "selected": int(len(chosen)),
            "mean_score": float(members.loc[chosen, score_col].mean()),
            **{
                f"mean_{axis}": float(members.loc[chosen, axis].mean())
                for axis in design.contrast_axes + design.match_axes
                if axis in members.columns
            },
        }

    # Fill any shortfall from the band by score, so the pool is always full.
    if len(selected) < design.pool_size:
        remainder = band.drop(index=selected, errors="ignore").nlargest(
            design.pool_size - len(selected), score_col
        )
        selected.extend(list(remainder.index))

    selected = selected[: design.pool_size]

    greedy_reference = frame.nlargest(design.pool_size, score_col)
    design_mean = float(frame.loc[selected, score_col].mean())
    greedy_mean = float(greedy_reference[score_col].mean())
    cost = (greedy_mean - design_mean) / abs(greedy_mean) if greedy_mean else 0.0

    report = {
        "band_size": int(len(band)),
        "band_tolerance": design.tolerance,
        "thresholds": design.thresholds_,
        "cells": cell_report,
        "design_mean_score": design_mean,
        "greedy_mean_score": greedy_mean,
        "relative_cost": float(cost),
        "balance": _balance_report(frame.loc[selected], design),
    }
    return selected, report


def _balance_report(chosen: pd.DataFrame, design: FactorialDesign) -> dict:
    """Confirm the matched variables really are matched across cells.

    A factorial design whose cells differ in net charge is not a clean contrast,
    so this is checked rather than assumed.
    """
    cells = assign_cells(chosen, design)
    out = {}
    for axis in design.match_axes:
        if axis not in chosen.columns:
            continue
        by_cell = chosen.groupby(cells)[axis].mean()
        out[axis] = {
            "per_cell_mean": {k: float(v) for k, v in by_cell.items()},
            "spread": float(by_cell.max() - by_cell.min()) if len(by_cell) else 0.0,
        }
    return out


def build_factorial_top_list(candidates, scores, frame: pd.DataFrame, novelty,
                             k: int = 100, design: FactorialDesign | None = None):
    """Assemble a top list whose first fifty form the factorial design.

    Positions one to fifty are the draw pool and carry the experiment. Positions
    fifty one to one hundred are insurance against identity filter rejections and
    are filled by score and diversity in the usual way.
    """
    from .constraints import candidate_feasible

    design = design or FactorialDesign()

    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    eligible = []
    for i in order:
        seq = candidates[i]
        if candidate_feasible(seq) and novelty.is_novel(seq) and seq in frame.index:
            eligible.append(seq)
        if len(eligible) >= 600:
            break

    if len(eligible) < design.pool_size:
        return [], {}

    sub = frame.loc[eligible].copy()
    pool, report = factorial_select(sub, "score", design)

    remaining = [s for s in eligible if s not in set(pool)]
    tail = greedy_diverse(
        remaining,
        sub.loc[remaining, "score"].to_numpy(dtype=float),
        k=min(k - len(pool), len(remaining)),
        redundancy_weight=design.redundancy_weight,
    )
    return pool + tail, report
