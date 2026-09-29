"""The screening cascade, and the library-as-output inversion.

The organisers' verifier requires that every sequence in the top list also
appears in the fifty thousand member library. The local search produces new
strings that were never in the sampled pool, so a pipeline that builds the
library first and refines afterwards fails that check on every refined
candidate. The library is therefore an output of this module, not its input:
refined candidates are guaranteed members and the remainder is filled from the
sampled pool.

Cheap filters run at the wide end, expensive ones only at the narrow end. The
verifier runs generation twice, on its own hardware, with no GPU guarantee, so
the persistent homology descriptor and the ensemble predictors are kept out of
the search's inner loop and applied only to the few hundred candidates that
reach the final rescoring.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constraints import deduplicate, library_feasible
from .features import featurize
from .scalarize import ChebyshevObjective, category_report
from .search import AnnealingConfig, MutationOperator, refine_population
from .select import build_top_list

HEAD_KEYS = ("gram_positive", "gram_negative", "hc50", "resistant")


@dataclass
class Predictors:
    """Trained heads plus the in-distribution gate, loaded as one object.

    Keeping them together guarantees the gate was fitted on the same feature
    ordering the heads consume, which is the kind of mismatch that produces a
    plausible looking ranking built on shuffled columns.
    """

    feature_names: list[str]
    heads: dict
    gate: object
    conformal: dict
    metrics: dict

    @staticmethod
    def load(path: str | Path) -> "Predictors":
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        return Predictors(
            feature_names=blob["feature_names"],
            heads=blob["heads"],
            gate=blob["gate"],
            conformal=blob.get("conformal", {}),
            metrics=blob.get("metrics", {}),
        )

    def frame(self, feats: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(feats).reindex(columns=self.feature_names).fillna(0.0)

    def predict(self, feats: list[dict]) -> list[dict]:
        X = self.frame(feats)
        out: list[dict] = [{} for _ in range(len(X))]
        for key in HEAD_KEYS:
            model = self.heads.get(key)
            if model is None:
                continue
            mu, sd = model.predict_with_uncertainty(X)
            name = "log2_hc50" if key == "hc50" else f"log2_mic_{key}"
            for i in range(len(X)):
                out[i][name] = float(mu[i])
                out[i][f"spread_{key}"] = float(sd[i])
        for row in out:
            row.setdefault("log2_mic_gram_positive", 6.0)
            row.setdefault("log2_mic_gram_negative", 6.0)
            row.setdefault("log2_hc50", 6.0)
        return out

    def in_distribution(self, feats: list[dict]) -> np.ndarray:
        return self.gate.passes(self.frame(feats))


@dataclass
class FunnelConfig:
    library_size: int = 50_000
    top_k: int = 100
    draw_pool: int = 50
    stage1_keep: int = 20_000
    stage2_keep: int = 2_000
    refine_seeds: int = 150
    anneal_steps: int = 400
    seed: int = 42


def make_score_fn(predictors: Predictors, objective: ChebyshevObjective):
    """Sequence and softness scale to a scalar, for the search's inner loop.

    Topology features are computed here even though they cost time in the inner
    loop. The gate is fitted on the full feature vector, so omitting them sends
    six columns in as zeros, puts every candidate far outside the fitted
    distribution, and rejects the entire population. Consistency with the fitted
    feature set matters more than the milliseconds.

    Hard infeasibility returns negative infinity so the search rejects rather
    than trades, and the in-distribution gate is treated as hard for the same
    reason: outside it the heads are extrapolating and their output is not a
    quantity worth optimising.
    """
    cache: dict[str, float] = {}

    def score(seq: str, softness_scale: float = 1.0) -> float:
        key = f"{seq}|{softness_scale:.3f}"
        hit = cache.get(key)
        if hit is not None:
            return hit
        if not library_feasible(seq):
            cache[key] = -np.inf
            return -np.inf
        feats = featurize(seq, include_topology=True)
        if not bool(predictors.in_distribution([feats])[0]):
            cache[key] = -np.inf
            return -np.inf
        pred = predictors.predict([feats])[0]
        value = objective(pred, feats, softness_scale)
        cache[key] = value
        return value

    return score


def _batch_score(seqs, predictors, objective, include_topology: bool,
                 risk_adjusted: bool = False, chunk: int = 2000):
    """Vectorised scoring. One model call per chunk rather than per sequence."""
    scores = np.full(len(seqs), -np.inf, dtype=float)
    for start in range(0, len(seqs), chunk):
        block = seqs[start : start + chunk]
        feats = [featurize(s, include_topology=include_topology) for s in block]
        preds = predictors.predict(feats)
        for i, (p, f) in enumerate(zip(preds, feats)):
            scores[start + i] = (
                objective.risk_adjusted(p, f) if risk_adjusted else objective(p, f)
            )
    return scores


def run(pool: list[str], predictors: Predictors, novelty, objective: ChebyshevObjective,
        config: FunnelConfig, verbose: bool = True):
    """Take a sampled pool to a library and a ranked top list.

    Returns (library, top, report). The library is exactly config.library_size
    sequences and contains every member of top.
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    stage0 = deduplicate([s for s in pool if library_feasible(s)])
    stage0 = [s for s in stage0 if not novelty.is_exact_match(s)]
    log(f"  stage 0  legal, unique, not a known peptide      {len(stage0)}")

    feats0 = [featurize(s, include_topology=True) for s in stage0]
    keep = predictors.in_distribution(feats0)
    passed = [s for s, ok in zip(stage0, keep) if ok]
    rate = len(passed) / max(len(stage0), 1)
    stage1 = passed[: config.stage1_keep]
    log(f"  stage 1  inside the model's support               {len(stage1)}  ({rate:.1%})")
    if rate < 0.02:
        # No silent fallback. A gate that quietly disables itself is worse than
        # one that fails, because the failure then surfaces somewhere unrelated.
        raise RuntimeError(
            f"In-distribution gate rejected {1 - rate:.1%} of the pool. Either the "
            "gate was fitted on a different feature set than the funnel computes, "
            "or its covariance is degenerate. Refusing to proceed with it disabled."
        )

    scores1 = _batch_score(stage1, predictors, objective, include_topology=True)
    order = np.argsort(-scores1, kind="stable")[: config.stage2_keep]
    stage2 = [stage1[i] for i in order]
    log(f"  stage 2  ranked by Chebyshev objective            {len(stage2)}")

    seeds = stage2[: config.refine_seeds]
    score_fn = make_score_fn(predictors, objective)
    results = refine_population(
        seeds, score_fn,
        AnnealingConfig(n_steps=config.anneal_steps, seed=config.seed),
        MutationOperator(),
    )
    refined = deduplicate(
        [r.best_sequence for r in results if np.isfinite(r.best_score)]
        + [s for r in results for s in r.visited]
    )
    refined = [s for s in refined if library_feasible(s) and not novelty.is_exact_match(s)]
    log(f"  stage 3  refined by local search                  {len(refined)}")

    candidates = deduplicate(refined + stage2)
    scores = _batch_score(
        candidates, predictors, objective, include_topology=True, risk_adjusted=True
    )
    top = build_top_list(
        candidates, scores, novelty, k=config.top_k,
        draw_pool=min(config.draw_pool, config.top_k),
    )
    log(f"  stage 4  novel, diverse top list                  {len(top)}")

    # The library must contain the top list, so refined candidates go in first.
    library = deduplicate(top + candidates + stage1 + stage0)
    library = [s for s in library if library_feasible(s)]
    if len(library) > config.library_size:
        library = library[: config.library_size]
    if len(library) < config.library_size:
        raise RuntimeError(
            f"Library has {len(library)} sequences, need exactly "
            f"{config.library_size}. Raise --oversample."
        )
    log(f"  library  assembled                                {len(library)}")

    report = {
        "n_pool": len(pool),
        "n_stage0": len(stage0),
        "n_stage1": len(stage1),
        "n_refined": len(refined),
        "n_candidates": len(candidates),
        "n_top": len(top),
        "n_library": len(library),
        "mean_accept_rate": float(
            np.mean([r.accepted / max(config.anneal_steps, 1) for r in results])
        ) if results else 0.0,
    }
    return library, top, report


def explain(seqs, predictors: Predictors, objective: ChebyshevObjective) -> pd.DataFrame:
    """Per-candidate breakdown, for the required selection documentation."""
    feats = [featurize(s, include_topology=True) for s in seqs]
    preds = predictors.predict(feats)
    rows = []
    for s, p, f in zip(seqs, preds, feats):
        row = {"sequence": s, "length": len(s)}
        row.update(category_report(p, f, objective))
        for k in ("net_charge", "moment_helix", "wheel_arc_width",
                  "interfacial_preference", "h0_n_patches", "frac_hydrophobic"):
            row[k] = f.get(k, float("nan"))
        row["pred_log2_mic_gram_negative"] = p.get("log2_mic_gram_negative")
        row["pred_log2_mic_gram_positive"] = p.get("log2_mic_gram_positive")
        row["pred_log2_hc50"] = p.get("log2_hc50")
        rows.append(row)
    return pd.DataFrame(rows)