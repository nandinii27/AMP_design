"""Multi-objective scalarisation.

Two facts about the competition's scoring rule drive everything here.

First, a team's category score is the arithmetic mean over twenty five peptides
drawn at random from the top fifty. Because it is a mean and not a maximum, a
specialist portfolio is strictly dominated: ten Gram-negative specialists at 0.9
mixed with fifteen peptides at 0.1 score 0.42 in the Gram-negative category,
while twenty five balanced peptides at 0.5 score 0.50. The generalist portfolio
wins in the specialist's own category. Targeting all five categories is
therefore a per-peptide scalarisation problem, not a portfolio allocation one.

Second, a weighted sum permits compensation, so the search abandons the hardest
objective to buy gains in the easiest. Chebyshev scalarisation maximises the
weakest normalised objective instead, which sends optimisation pressure wherever
a candidate is currently worst. The small augmentation term breaks ties among
solutions with an equally weak worst axis and avoids stalling on weakly Pareto
optimal points.

Targets are set at quantiles of the labelled actives rather than at the
competition's floor. Fifty three percent of known peptides already clear the
16 uM bar, so optimising to it is unambitious; the objective should be to beat
the incumbent distribution, not to match its median.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import DEFAULT_WINDOWS, MOMENT_FLOOR, POTENCY_THRESHOLD_UM

LOG2_POTENCY_THRESHOLD = float(np.log2(POTENCY_THRESHOLD_UM))

AXES = ("gram_positive", "gram_negative", "safety_window", "qc_survival")


def window_factor(x: float, lo: float, hi: float, softness: float = 0.5) -> float:
    """Multiplicative soft constraint: one inside the window, decaying outside.

    Multiplicative rather than additive on purpose. Under an additive penalty a
    candidate two charge units outside the window with predicted quality 12 beats
    a compliant candidate at quality 9, so the optimiser trades the constraint
    away and lands where the model is extrapolating. Here the factor decays
    faster than any quality term can grow, so the violation cannot be bought off.
    """
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return float(np.exp(-((d / max(softness, 1e-6)) ** 2)))


def potency_probability(log2_mic: float, spread: float = 0.0,
                        threshold: float = LOG2_POTENCY_THRESHOLD) -> float:
    """Probability the true value clears the potency threshold.

    Smoothed by the predictive spread so the objective has a gradient near the
    threshold rather than a step the search cannot navigate.
    """
    scale = max(spread, 0.75)
    return float(1.0 / (1.0 + np.exp((log2_mic - threshold) / scale)))


def qc_survival(feats: dict) -> float:
    """Probability of surviving synthesis, solubility and purity control.

    A failed candidate is not retested and its slot in the experimental batch is
    lost, so this multiplies expected value directly rather than adjusting rank.
    Length dominates: solid phase synthesis adds one residue at a time and yields
    compound, so failure rate climbs steeply with length.
    """
    p = 1.0
    p *= 0.995 ** max(feats.get("length", 20.0) - 12.0, 0.0)
    p *= window_factor(feats.get("frac_hydrophobic", 0.4), 0.0, 0.50, softness=0.08)
    p *= window_factor(feats.get("longest_hydrophobic_run", 0.0), 0.0, 4.0, softness=1.5)
    p *= window_factor(feats.get("longest_beta_branched_run", 0.0), 0.0, 2.0, softness=1.2)
    p *= 0.97 ** feats.get("n_oxidation_prone", 0.0)
    p *= 0.93 ** feats.get("n_liability_motifs", 0.0)
    p *= window_factor(feats.get("mean_sheet_propensity", 1.0), 0.0, 1.20, softness=0.15)
    return float(np.clip(p, 0.0, 1.0))


@dataclass
class ChebyshevObjective:
    """Worst-axis scalarisation over the four trainable objectives.

    The multidrug-resistant head exists but is underpowered: two thousand
    sequences against eleven thousand for each Gram class, and a held-out
    Spearman of 0.39 against 0.46 and 0.54. It enters as a weighted contribution
    to the potency axes rather than as an axis of its own, and the data statement
    should say so rather than claim a model the data cannot support.
    """

    ideals: dict[str, float] = field(
        default_factory=lambda: {
            "gram_positive": 1.0,
            "gram_negative": 1.0,
            "safety_window": 5.0,
            "qc_survival": 1.0,
        }
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "gram_positive": 1.0,
            "gram_negative": 1.0,
            "safety_window": 1.0,
            "qc_survival": 1.0,
        }
    )
    rho: float = 0.05
    resistant_weight: float = 0.35
    windows: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_WINDOWS)
    )
    moment_floor: float = MOMENT_FLOOR
    risk_lambda: float = 0.25

    @classmethod
    def from_activity_table(cls, path: str | Path, quantile: float = 0.75,
                            **kwargs) -> "ChebyshevObjective":
        """Set the safety-window target at a quantile of the labelled actives.

        Anchoring to the incumbent distribution rather than to the competition's
        floor. If the table is unavailable the defaults stand, so a fresh clone
        without it still runs.
        """
        obj = cls(**kwargs)
        path = Path(path)
        if not path.exists():
            return obj
        try:
            import pandas as pd

            wide = pd.read_csv(path, index_col=0)
            mic_cols = [c for c in ("gram_positive", "gram_negative") if c in wide.columns]
            if "hc50" in wide.columns and mic_cols:
                best = wide[mic_cols].min(axis=1)
                window = (wide["hc50"] - best).dropna()
                window = window[window > 0]
                if len(window) > 100:
                    obj.ideals["safety_window"] = float(np.quantile(window, quantile))
        except Exception:
            pass
        return obj

    def axes(self, pred: dict, feats: dict) -> dict[str, float]:
        mic_pos = pred["log2_mic_gram_positive"]
        mic_neg = pred["log2_mic_gram_negative"]
        hc50 = pred["log2_hc50"]

        # The resistant head shifts the potency axes rather than forming its own.
        res = pred.get("log2_mic_resistant")
        if res is not None and np.isfinite(res):
            w = self.resistant_weight
            mic_pos = (1 - w) * mic_pos + w * res
            mic_neg = (1 - w) * mic_neg + w * res

        best_mic = min(mic_pos, mic_neg)
        return {
            "gram_positive": potency_probability(mic_pos, pred.get("spread_gram_positive", 0.0)),
            "gram_negative": potency_probability(mic_neg, pred.get("spread_gram_negative", 0.0)),
            "safety_window": max(hc50 - best_mic, 0.0),
            "qc_survival": qc_survival(feats),
        }

    def soft_factor(self, feats: dict, softness_scale: float = 1.0) -> float:
        f = 1.0
        for name, (lo, hi) in self.windows.items():
            if name in feats:
                f *= window_factor(feats[name], lo, hi, softness=0.5 * softness_scale)
        f *= window_factor(
            feats.get("moment_helix", 0.0), self.moment_floor, 1.0,
            softness=0.12 * softness_scale,
        )
        return f

    def __call__(self, pred: dict, feats: dict, softness_scale: float = 1.0) -> float:
        raw = self.axes(pred, feats)
        norm = {k: min(raw[k] / max(self.ideals[k], 1e-9), 1.0) for k in raw}
        worst = min(self.weights[k] * norm[k] for k in norm)
        augmented = worst + self.rho * float(np.mean(list(norm.values())))
        return float(augmented * self.soft_factor(feats, softness_scale))

    def risk_adjusted(self, pred: dict, feats: dict) -> float:
        """Chebyshev value less a modest penalty on epistemic spread.

        Model error is correlated across similar sequences, so the realised batch
        mean has real variance and the downside is asymmetric. Most of the
        variance reduction should come from portfolio diversity rather than from
        a large coefficient here.
        """
        base = self(pred, feats)
        spread = float(
            np.mean([
                pred.get("spread_gram_positive", 0.0),
                pred.get("spread_gram_negative", 0.0),
                pred.get("spread_hc50", 0.0),
            ])
        )
        return float(base - self.risk_lambda * spread * base)


def category_report(pred: dict, feats: dict, objective: ChebyshevObjective) -> dict[str, float]:
    """Per-category breakdown, for the ranking documentation the submission
    has to include."""
    raw = objective.axes(pred, feats)
    raw["broad_spectrum"] = float(
        (15 * raw["gram_negative"] + 5 * raw["gram_positive"]) / 20.0
    )
    raw["chebyshev"] = objective(pred, feats)
    return raw
