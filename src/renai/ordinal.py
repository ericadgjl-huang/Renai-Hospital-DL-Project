"""Ordinal-aware 4-class methods + metrics (no CNN retraining required).

Motivation
----------
Ficat stages 1<2<3<4 form an *ordered* scale, yet the current pipeline scores
everything with nominal macro-F1 and combines cut-probabilities with a
RandomForest that ignores order. On the internal 58-image test that RF combiner
looks best (macro-F1 0.72), but on the external AVNFH cohort it drops to 0.52 —
*below* the hard hierarchy (0.54). Classic small-sample overfitting of a
high-capacity, order-blind meta-learner.

This module adds order-aware, low-capacity alternatives that reuse the already
trained cut checkpoints (via their saved OOF / test / external P(class=1)),
so nothing here needs a GPU retrain:

1. Ordinal metrics: quadratic weighted kappa (QWK) — the standard metric for
   medical grading (diabetic-retinopathy Kaggle, etc.) and directly comparable
   to the human inter-observer kappa (0.39-0.46) the thesis already cites — plus
   off-by-one (±1 stage) accuracy and mean absolute stage error.

2. Isotonic probability calibration of each cut, fit on leak-free OOF.

3. Rank-consistent cumulative decode (CORAL/CORN-style, applied *post-hoc* to
   the three cumulative cuts [P>=2, P>=3, P>=4]) — ZERO trainable parameters, so
   it cannot overfit the tiny test set. Two decoders: rank-count and expected
   value.

4. A small learned ordinal meta-learner: K-1 cumulative logistic regressions
   over the cut-prob features, decoded rank-consistently. Far fewer parameters
   than RF/XGBoost.

5. Monotone-constrained gradient boosting — an *order-aware* version of the
   "boosting" the user already tried (HistGradientBoosting with monotonic_cst).

References
----------
- Cao, Mirjalili, Raschka (2020) "Rank consistent ordinal regression for neural
  networks" (CORAL). Pattern Recognition Letters.
- Shi, Cao, Raschka (2021/2023) "Deep NNs for rank-consistent ordinal regression
  based on conditional probabilities" (CORN). Pattern Anal. Applic.
- de la Torre et al. (2018) "Weighted kappa loss for ordinal data in deep
  learning"; QWK is the Kaggle DR-grading metric.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

STAGES = np.array([1, 2, 3, 4])
# The three cumulative cuts, in ascending threshold order:
#   1_vs_234 = P(stage >= 2), 12_vs_34 = P(stage >= 3), 123_vs_4 = P(stage >= 4)
ORDINAL_CUTS = ["1_vs_234", "12_vs_34", "123_vs_4"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Full ordinal report for 1..4 stage labels.

    qwk        : quadratic weighted kappa (order-aware agreement; the headline
                 metric for ordinal medical grading and comparable to human
                 inter-observer kappa).
    off_by_one : fraction predicted within +/-1 stage of truth.
    mae        : mean absolute error in stage units.
    accuracy / macro_f1 : kept for continuity with the existing reports.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return {k: float("nan") for k in
                ("qwk", "accuracy", "macro_f1", "off_by_one", "mae", "n")}
    # cohen_kappa with quadratic weights == QWK. labels pinned so a class absent
    # from a bootstrap resample doesn't shift the weighting.
    qwk = cohen_kappa_score(y_true, y_pred, labels=[1, 2, 3, 4], weights="quadratic")
    return {
        "n": int(len(y_true)),
        "qwk": float(qwk),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[1, 2, 3, 4],
                                   average="macro", zero_division=0)),
        "off_by_one": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(np.concatenate([y_true, y_pred]))) < 2:
        return 0.0
    return float(cohen_kappa_score(y_true, y_pred, labels=[1, 2, 3, 4],
                                   weights="quadratic"))


def bootstrap_ci(y_true, y_pred, metric="qwk", n_boot=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for one ordinal metric (default QWK)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    fn = {"qwk": _qwk,
          "accuracy": lambda a, b: float(accuracy_score(a, b)),
          "macro_f1": lambda a, b: float(f1_score(a, b, labels=[1, 2, 3, 4],
                                                  average="macro", zero_division=0)),
          "off_by_one": lambda a, b: float(np.mean(np.abs(a - b) <= 1))}[metric]
    point = fn(y_true, y_pred)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(y_true, y_pred_a, y_pred_b, metric="qwk",
                          n_boot=2000, seed=42):
    """Paired bootstrap of metric(B) - metric(A) on the SAME samples.

    Returns (delta, ci_low, ci_high, p_two_sided). This is the honest way to
    ask "is method B actually better than A" on a 58-image test set, instead of
    eyeballing two point estimates whose CIs overlap."""
    y_true = np.asarray(y_true, dtype=int)
    a = np.asarray(y_pred_a, dtype=int)
    b = np.asarray(y_pred_b, dtype=int)
    n = len(y_true)
    fn = {"qwk": _qwk,
          "accuracy": lambda t, p: float(accuracy_score(t, p)),
          "macro_f1": lambda t, p: float(f1_score(t, p, labels=[1, 2, 3, 4],
                                                  average="macro", zero_division=0))}[metric]
    point = fn(y_true, b) - fn(y_true, a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = fn(y_true[idx], b[idx]) - fn(y_true[idx], a[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p: fraction of resamples on the wrong side of 0, doubled.
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(point), float(lo), float(hi), float(min(p, 1.0))


# ---------------------------------------------------------------------------
# Per-cut isotonic calibration (fit on leak-free OOF)
# ---------------------------------------------------------------------------

class CutCalibrators:
    """One isotonic regressor per cut, fit on OOF P(class=1) vs the cut's binary
    target. Applying it to test/external probabilities is leak-free because the
    fit used only OOF predictions."""

    def __init__(self):
        self.iso: dict[int, IsotonicRegression] = {}

    def fit(self, X_oof: np.ndarray, cut_targets: dict[int, np.ndarray]):
        """X_oof: (N, n_cuts) OOF P(class=1). cut_targets[j]: binary 0/1 target
        for column j, restricted to the samples where that cut is defined
        (use np.nan in the target to mark 'not in this cut' -> skipped)."""
        for j in range(X_oof.shape[1]):
            t = cut_targets.get(j)
            if t is None:
                continue
            mask = ~np.isnan(t)
            if mask.sum() < 8 or len(np.unique(t[mask])) < 2:
                continue
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(X_oof[mask, j], t[mask].astype(float))
            self.iso[j] = ir
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = X.copy().astype(float)
        for j, ir in self.iso.items():
            Xc[:, j] = ir.predict(X[:, j])
        return Xc


# ---------------------------------------------------------------------------
# Rank-consistent cumulative decode (post-hoc CORAL/CORN, no parameters)
# ---------------------------------------------------------------------------

def enforce_monotone_cumulative(P: np.ndarray) -> np.ndarray:
    """Make each row non-increasing: P(>=2) >= P(>=3) >= P(>=4).

    CNN cuts are trained independently so their raw probabilities can violate
    this. We take the running min across thresholds — the standard fix that
    yields a coherent ordinal distribution."""
    Q = P.copy().astype(float)
    for k in range(1, Q.shape[1]):
        Q[:, k] = np.minimum(Q[:, k], Q[:, k - 1])
    return Q


def decode_rank_count(P_cumulative: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """CORAL/CORN decode: stage = 1 + #{thresholds with P >= 0.5}.

    P_cumulative columns = [P(>=2), P(>=3), P(>=4)]."""
    Q = enforce_monotone_cumulative(P_cumulative)
    return 1 + (Q >= thresh).sum(axis=1).astype(int)


def decode_rank_count_thresh(P_cumulative: np.ndarray, thresholds) -> np.ndarray:
    """Rank-count decode with a PER-THRESHOLD cutoff instead of a flat 0.5.

    thresholds = [t2, t3, t4] for crossing into [>=2, >=3, >=4]. Lowering t3
    makes it easier to reach stage 3 (rescues stage-3 vs stage-2 errors);
    raising t4 makes it harder to jump to stage 4 (rescues stage-3 vs stage-4
    errors). Tuned on OOF, this targets the weak stage-3 recall for free."""
    Q = enforce_monotone_cumulative(P_cumulative)
    thr = np.asarray(thresholds, dtype=float).reshape(1, -1)
    return 1 + (Q >= thr).sum(axis=1).astype(int)


def tune_decode_thresholds(P_oof, y_oof, objective="macro_f1", grid=None):
    """Grid-search per-threshold decode cutoffs on OOF. Returns the [t2,t3,t4]
    that maximises the objective (macro-F1 by default, which rewards lifting the
    weak stage-3 class). Only t3 and t4 are searched (t2 fixed at 0.5) since
    stage-3 errors live at the 3rd/4th boundaries."""
    if grid is None:
        grid = np.round(np.arange(0.30, 0.71, 0.05), 2)
    score = (lambda yt, yp: f1_score(yt, yp, labels=[1, 2, 3, 4],
                                     average="macro", zero_division=0)) \
        if objective == "macro_f1" else \
        (lambda yt, yp: cohen_kappa_score(yt, yp, labels=[1, 2, 3, 4], weights="quadratic"))
    best, best_thr = -1.0, [0.5, 0.5, 0.5]
    for t3 in grid:
        for t4 in grid:
            pred = decode_rank_count_thresh(P_oof, [0.5, t3, t4])
            s = score(y_oof, pred)
            if s > best:
                best, best_thr = s, [0.5, float(t3), float(t4)]
    return best_thr, best


def decode_expected(P_cumulative: np.ndarray) -> np.ndarray:
    """Expected-value decode: E[stage] = 1 + sum_k P(>=k+1), rounded to 1..4.

    Smoother than rank-count; often better when probabilities are calibrated."""
    Q = enforce_monotone_cumulative(P_cumulative)
    ev = 1.0 + Q.sum(axis=1)
    return np.clip(np.rint(ev), 1, 4).astype(int)


# ---------------------------------------------------------------------------
# Learned ordinal meta-learner (K-1 cumulative logits over cut-prob features)
# ---------------------------------------------------------------------------

class OrdinalLogisticMeta(BaseEstimator, ClassifierMixin):
    """All-thresholds ordinal meta over cut-probability features.

    Fits K-1 binary logistic regressions for the cumulative targets
    [y>=2], [y>=3], [y>=4], then decodes rank-consistently (running-min +
    expected value). This is the "learned CORN" over cut features: it respects
    stage order and has only K-1 small linear models, so it generalizes far
    better than an order-blind RandomForest on ~230 samples.
    """

    def __init__(self, C: float = 1.0, decode: str = "expected"):
        self.C = C
        self.decode = decode

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.models_ = []
        for thr in (2, 3, 4):                     # P(y >= thr)
            target = (y >= thr).astype(int)
            lr = LogisticRegression(class_weight="balanced", C=self.C,
                                    max_iter=2000)
            lr.fit(X, target)
            # column index of the positive class
            pos = list(lr.classes_).index(1)
            self.models_.append((lr, pos))
        self.classes_ = STAGES
        return self

    def cumulative_probs(self, X) -> np.ndarray:
        cols = [m.predict_proba(X)[:, pos] for m, pos in self.models_]
        return enforce_monotone_cumulative(np.column_stack(cols))

    def predict(self, X):
        Q = self.cumulative_probs(X)
        if self.decode == "rank_count":
            return decode_rank_count(Q)
        return decode_expected(Q)

    def predict_proba(self, X):
        """Turn the monotone cumulative probs into a proper 4-class simplex:
        P(y=1)=1-P(>=2), P(y=k)=P(>=k)-P(>=k+1), P(y=4)=P(>=4)."""
        Q = self.cumulative_probs(X)          # [P>=2, P>=3, P>=4]
        p1 = 1.0 - Q[:, 0]
        p2 = Q[:, 0] - Q[:, 1]
        p3 = Q[:, 1] - Q[:, 2]
        p4 = Q[:, 2]
        P = np.clip(np.column_stack([p1, p2, p3, p4]), 0, None)
        P /= P.sum(axis=1, keepdims=True)
        return P
