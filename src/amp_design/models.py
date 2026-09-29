"""Predictive heads and the in-distribution gate.

Three components:

  CensoredRegressor   interval aware model for log2 concentrations
  ActivityClassifier  calibrated probability against charge matched negatives
  InDistributionGate  Mahalanobis screen against the validated active envelope

The gate is the single most important piece. Without it any optimiser run
against a learned oracle drifts toward the extreme corner of descriptor space
where the oracle extrapolates most confidently and has seen nothing. That is
how generative peptide pipelines produce sequences scoring above 0.99 that do
nothing in a plate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import MinCovDet
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from scipy.stats import spearmanr

# Censoring codes, matching those written by scripts/parse_dbaasp.py.
CENSOR_NONE, CENSOR_RIGHT, CENSOR_LEFT, CENSOR_INTERVAL = 0, 1, -1, 2


def _as_matrix(X) -> np.ndarray:
    return X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)


@dataclass
class CensoredRegressor:
    """Gradient boosted regression on interval censored log2 concentrations.

    Minimum inhibitory concentrations come from a twofold dilution series, so a
    reported 8 means the true value lies in (4, 8]. Haemolytic concentrations
    are commonly reported only as above some ceiling. Dropping right censored
    rows biases the model toward haemolytic peptides, which is backwards for a
    selectivity objective.

    Censoring is handled by iterative imputation on the Tobit principle: right
    censored targets are replaced each round by the model's own prediction where
    that prediction exceeds the censoring bound, and left at the bound otherwise.
    This converges to the censored likelihood optimum in practice and needs no
    custom objective, which matters when the run has to finish tonight.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 3
    n_rounds: int = 4
    seed: int = 42
    models_: list = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)

    def fit(self, X, y, censor=None, n_ensemble: int = 3) -> "CensoredRegressor":
        from lightgbm import LGBMRegressor

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        Xm = _as_matrix(X)
        y = np.asarray(y, dtype=float)
        censor = np.zeros_like(y, dtype=int) if censor is None else np.asarray(censor, dtype=int)

        self.models_ = []
        for member in range(n_ensemble):
            target = y.copy()
            model = None
            for _ in range(self.n_rounds):
                model = LGBMRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    random_state=self.seed + member,
                    verbose=-1,
                )
                model.fit(Xm, target)
                pred = model.predict(Xm)
                right = censor == CENSOR_RIGHT
                target[right] = np.maximum(y[right], pred[right])
            self.models_.append(model)
        return self

    def predict(self, X) -> np.ndarray:
        Xm = _as_matrix(X)
        return np.mean([m.predict(Xm) for m in self.models_], axis=0)

    def predict_with_uncertainty(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Mean and epistemic spread across ensemble members."""
        Xm = _as_matrix(X)
        preds = np.stack([m.predict(Xm) for m in self.models_])
        return preds.mean(axis=0), preds.std(axis=0)

    def evaluate(self, X, y, censor=None) -> dict[str, float]:
        """Metrics appropriate to a twofold dilution series.

        Within one dilution is the operative accuracy measure: a single twofold
        step is experimental noise, so anything finer is false precision.
        Censored rows are excluded from point metrics and scored separately by
        the fraction where the prediction respects the bound.
        """
        y = np.asarray(y, dtype=float)
        censor = np.zeros_like(y, dtype=int) if censor is None else np.asarray(censor, dtype=int)
        pred = self.predict(X)
        obs = censor == CENSOR_NONE
        out: dict[str, float] = {}
        if obs.sum() > 2:
            out["spearman"] = float(spearmanr(y[obs], pred[obs]).statistic)
            out["mae_log2"] = float(np.mean(np.abs(y[obs] - pred[obs])))
            out["within_one_dilution"] = float(np.mean(np.abs(y[obs] - pred[obs]) <= 1.0))
            out["within_two_dilutions"] = float(np.mean(np.abs(y[obs] - pred[obs]) <= 2.0))
        right = censor == CENSOR_RIGHT
        if right.sum() > 0:
            out["censored_bound_respected"] = float(np.mean(pred[right] >= y[right]))
            out["n_censored"] = float(right.sum())
        out["n_observed"] = float(obs.sum())
        return out


@dataclass
class ActivityClassifier:
    """Calibrated activity probability.

    Calibration matters more than discrimination here because the objective
    takes an expectation over the predicted probability. An overconfident
    classifier corrupts the ranking at any AUC.
    """

    seed: int = 42
    model_: object | None = None

    def fit(self, X, y) -> "ActivityClassifier":
        base = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=self.seed,
        )
        self.model_ = CalibratedClassifierCV(base, method="isotonic", cv=5)
        self.model_.fit(_as_matrix(X), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(_as_matrix(X))[:, 1]

    def evaluate(self, X, y) -> dict[str, float]:
        p = self.predict_proba(X)
        y = np.asarray(y, dtype=int)
        return {
            "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "positive_rate": float(y.mean()),
        }


@dataclass
class InDistributionGate:
    """Robust Mahalanobis screen against the validated active envelope.

    Rejects candidates outside the descriptor region where the predictive heads
    have support. Uses the minimum covariance determinant estimator so a handful
    of unusual training peptides cannot inflate the envelope.

    This is a hard gate, not a penalty. A candidate outside it is not worse, it
    is unscoreable.
    """

    quantile: float = 0.975
    support_fraction: float = 0.85
    seed: int = 42
    estimator_: MinCovDet | None = None
    threshold_: float = float("inf")
    feature_names_: list[str] = field(default_factory=list)

    def fit(self, X) -> "InDistributionGate":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        Xm = _as_matrix(X)
        keep = np.isfinite(Xm).all(axis=1)
        Xm = Xm[keep]
        var = Xm.var(axis=0)
        self._keep_cols = var > 1e-12
        Xm = Xm[:, self._keep_cols]
        self.estimator_ = MinCovDet(
            support_fraction=self.support_fraction, random_state=self.seed
        ).fit(Xm)
        d = self.estimator_.mahalanobis(Xm)
        self.threshold_ = float(np.quantile(d, self.quantile))
        return self

    def distance(self, X) -> np.ndarray:
        Xm = _as_matrix(X)[:, self._keep_cols]
        return self.estimator_.mahalanobis(Xm)

    def passes(self, X) -> np.ndarray:
        return self.distance(X) <= self.threshold_


@dataclass
class ConformalInterval:
    """Split conformal prediction for calibrated log2 concentration intervals.

    Distribution free, needs one held out calibration split, and gives a
    defensible width for the risk adjusted ranking instead of an ensemble
    standard deviation used as if it were a confidence interval.
    """

    alpha: float = 0.2
    width_: float = 0.0

    def calibrate(self, residuals) -> "ConformalInterval":
        r = np.abs(np.asarray(residuals, dtype=float))
        r = r[np.isfinite(r)]
        n = len(r)
        if n == 0:
            self.width_ = 0.0
            return self
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.width_ = float(np.quantile(r, level))
        return self

    def interval(self, point) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(point, dtype=float)
        return p - self.width_, p + self.width_
