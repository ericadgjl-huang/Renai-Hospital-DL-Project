"""Stage 14 — Ordinal-aware combiners vs the current baselines (NO retraining).

Reuses the already-trained cut checkpoints. It extracts three leak-free
P(class=1) feature matrices once (OOF train_val, internal test, external
AVNFH), caches them, then compares order-aware low-capacity methods against the
current hierarchy hard-routing and the RandomForest combiner.

Everything is *selected/tuned on OOF only*; the internal test and the external
cohort are report-only. The headline metric is quadratic weighted kappa (QWK),
the standard for ordinal medical grading and directly comparable to the human
inter-observer kappa (~0.39-0.46) the thesis already cites.

The key question this answers: does an order-aware, low-capacity combiner close
the internal->external overfitting gap that the RandomForest combiner suffers
(internal macro-F1 0.72 but external 0.52, below the hierarchy's 0.54)?

Usage (repo root, conda env unet_labeling):
    python scripts/14_ordinal_methods.py --device 0
    python scripts/14_ordinal_methods.py --device 0 --force     # recompute feats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from torchvision.datasets import ImageFolder

from renai.combiner import build_features
from renai.cuts_registry import CUTS
from renai.data import get_4class_labels, make_4class_eval_loader, make_outer_split
from renai.hierarchy import TOPOLOGIES, CutPredictor, route_samples
from renai.ordinal import (
    CutCalibrators,
    OrdinalLogisticMeta,
    ORDINAL_CUTS,
    bootstrap_ci,
    decode_expected,
    decode_rank_count,
    enforce_monotone_cumulative,
    ordinal_metrics,
    paired_bootstrap_diff,
)
from renai.seed import SEED, set_seed

STAGE_NAMES = ["stage_1", "stage_2", "stage_3", "stage_4"]
CANON_CUTS = sorted(CUTS.keys())          # fixed column order for every matrix
ORD_IDX = [CANON_CUTS.index(c) for c in ORDINAL_CUTS]


# ---------------------------------------------------------------------------
# Feature extraction (GPU inference; cached to .npy so methods iterate fast)
# ---------------------------------------------------------------------------

def _external_features(out_root, data_root, device, batch_size):
    """Per-cut P(class=1) matrix for an external ImageFolder cohort (+ labels)."""
    base = ImageFolder(data_root)
    if base.classes != STAGE_NAMES:
        raise SystemExit(f"Expected classes {STAGE_NAMES}, got {base.classes}")
    n = len(base.samples)
    predictors = {}
    for cn in CANON_CUTS:
        wp = out_root / "cuts" / cn / "ensemble" / "winner.json"
        if wp.exists():
            predictors[cn] = CutPredictor(out_root / "cuts" / cn, device)
    cols, y_true = [], None
    for cn in CANON_CUTS:
        loader = make_4class_eval_loader(data_root, list(range(n)), batch_size=batch_size)
        probs, ys = [], []
        for imgs, lbls in loader:
            probs.append(predictors[cn].prob_class1(imgs))
            ys.extend(int(l) + 1 for l in lbls.numpy().tolist())
        cols.append(np.concatenate(probs))
        if y_true is None:
            y_true = np.asarray(ys, dtype=np.int64)
        print(f"  [ext] {cn} done", flush=True)
    return np.column_stack(cols), y_true


def load_or_build_features(args):
    cache = args.out_root / "ordinal" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    need = ["oof_X", "oof_y", "test_X", "test_y", "ext_X", "ext_y"]
    if not args.force and all((cache / f"{k}.npy").exists() for k in need):
        print("[feat] loading cached feature matrices", flush=True)
        d = {k: np.load(cache / f"{k}.npy") for k in need}
        return d

    print("[feat] extracting OOF + internal-test features (GPU) ...", flush=True)
    X_oof, y_oof, X_test, y_test, used = build_features(
        CANON_CUTS, args.out_root, args.data_root,
        args.splits_dir, args.device, args.batch_size,
    )
    assert used == CANON_CUTS, f"cut order mismatch: {used}"

    print("[feat] extracting external AVNFH features (GPU) ...", flush=True)
    X_ext, y_ext = _external_features(
        args.out_root, args.ext_root, args.device, args.batch_size)

    d = {"oof_X": X_oof, "oof_y": y_oof, "test_X": X_test, "test_y": y_test,
         "ext_X": X_ext, "ext_y": y_ext}
    for k, v in d.items():
        np.save(cache / f"{k}.npy", v)
    (cache / "cut_order.json").write_text(json.dumps(CANON_CUTS, indent=2))
    print(f"[feat] cached to {cache}", flush=True)
    return d


# ---------------------------------------------------------------------------
# Calibration targets (leak-free, from OOF only)
# ---------------------------------------------------------------------------

def build_cut_targets(y_oof: np.ndarray) -> dict[int, np.ndarray]:
    """Per-cut binary target for isotonic calibration; np.nan where the cut is
    undefined for that stage (so it is skipped)."""
    targets = {}
    for j, cn in enumerate(CANON_CUTS):
        cut = CUTS[cn]
        t = np.full(len(y_oof), np.nan)
        for i, stg in enumerate(y_oof):
            if stg in cut.positives_zero:
                t[i] = 0.0
            elif stg in cut.positives_one:
                t[i] = 1.0
        targets[j] = t
    return targets


# ---------------------------------------------------------------------------
# Methods. Each returns OOF / test / external predictions (1..4).
# Selection uses OOF only.
# ---------------------------------------------------------------------------

def method_hierarchy(feats):
    """Current production baseline: hard routing through the selected topology."""
    best = json.loads((ARGS.out_root / "hierarchy" / "best_topology.json").read_text())
    topo = TOPOLOGIES[best["name"]]

    def route(X):
        per_cut = {CANON_CUTS[j]: X[:, j] for j in range(X.shape[1])}
        return route_samples(topo, per_cut, X.shape[0])
    return (route(feats["oof_X"]), route(feats["test_X"]), route(feats["ext_X"]),
            f"hierarchy[{topo.name}] hard-routing")


def method_rf_all(feats):
    """Current best experimental baseline: RandomForest over 10 raw cut probs."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=400,
                                class_weight="balanced_subsample", random_state=SEED)
    oof = cross_val_predict(rf, feats["oof_X"], feats["oof_y"], cv=cv)
    rf.fit(feats["oof_X"], feats["oof_y"])
    return (oof, rf.predict(feats["test_X"]), rf.predict(feats["ext_X"]),
            "combiner[all] RandomForest (order-blind)")


def method_ordinal_decode(feats, cal, decode, tag):
    """Post-hoc CORAL/CORN: calibrated ordinal-3 -> monotone -> decode. 0 params."""
    def run(X):
        Xc = cal.transform(X)
        P = Xc[:, ORD_IDX]                       # [P>=2, P>=3, P>=4]
        return decode(P)
    return (run(feats["oof_X"]), run(feats["test_X"]), run(feats["ext_X"]), tag)


def method_ordinal_logit(feats, cal, cols, C, tag):
    """Learned ordinal meta (K-1 cumulative logits) over calibrated cut probs."""
    Xo = cal.transform(feats["oof_X"])[:, cols]
    Xt = cal.transform(feats["test_X"])[:, cols]
    Xe = cal.transform(feats["ext_X"])[:, cols]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    meta = OrdinalLogisticMeta(C=C, decode="expected")
    oof = cross_val_predict(meta, Xo, feats["oof_y"], cv=cv)
    meta.fit(Xo, feats["oof_y"])
    return oof, meta.predict(Xt), meta.predict(Xe), tag


def method_mono_boost(feats, cal, tag):
    """Order-aware boosting: HistGradientBoosting with monotone_cst forcing each
    cumulative prob to push the prediction toward higher stages. Contrast with
    the order-blind boosting the user already tried."""
    cols = ORD_IDX
    Xo = cal.transform(feats["oof_X"])[:, cols]
    Xt = cal.transform(feats["test_X"])[:, cols]
    Xe = cal.transform(feats["ext_X"])[:, cols]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    # multiclass HGB doesn't accept monotone_cst; use it as a *regressor* on the
    # ordinal target with +1 monotonicity on every cumulative prob, then round.
    from sklearn.ensemble import HistGradientBoostingRegressor
    reg = HistGradientBoostingRegressor(
        monotonic_cst=[1] * len(cols), max_iter=300, learning_rate=0.05,
        max_depth=3, random_state=SEED)

    def to_stage(v):
        return np.clip(np.rint(v), 1, 4).astype(int)
    oof_reg = cross_val_predict(reg, Xo, feats["oof_y"].astype(float), cv=cv)
    reg.fit(Xo, feats["oof_y"].astype(float))
    return (to_stage(oof_reg), to_stage(reg.predict(Xt)),
            to_stage(reg.predict(Xe)), tag)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _tune_logit_C(feats, cal, cols):
    """Pick C by OOF QWK (report-only sets never touched)."""
    from renai.ordinal import _qwk
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    Xo = cal.transform(feats["oof_X"])[:, cols]
    best_C, best_q = 1.0, -1.0
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        oof = cross_val_predict(OrdinalLogisticMeta(C=C), Xo, feats["oof_y"], cv=cv)
        q = _qwk(feats["oof_y"], oof)
        if q > best_q:
            best_q, best_C = q, C
    return best_C


def main():
    global ARGS
    ap = argparse.ArgumentParser(description="Ordinal-aware combiners (no retraining).")
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--data-root", type=Path, default=Path("stage_cls_dataset"))
    ap.add_argument("--ext-root", type=Path, default=Path("stage_cls_avnfh"))
    ap.add_argument("--splits-dir", type=Path, default=Path("outputs/splits"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true", help="recompute cached features")
    ARGS = ap.parse_args()
    if ARGS.device.isdigit():
        ARGS.device = f"cuda:{ARGS.device}"
    set_seed(SEED)

    feats = load_or_build_features(ARGS)
    print(f"[feat] OOF={feats['oof_X'].shape} test={feats['test_X'].shape} "
          f"ext={feats['ext_X'].shape}", flush=True)

    # Calibrate every cut on OOF (leak-free), reuse across methods.
    cal = CutCalibrators().fit(feats["oof_X"], build_cut_targets(feats["oof_y"]))
    print(f"[calib] isotonic-calibrated {len(cal.iso)} cuts", flush=True)

    C_all = _tune_logit_C(feats, cal, list(range(len(CANON_CUTS))))
    C_ord = _tune_logit_C(feats, cal, ORD_IDX)
    print(f"[tune] ordinal-logit C: all={C_all} ord3={C_ord}", flush=True)

    methods = [
        method_hierarchy(feats),
        method_rf_all(feats),
        method_ordinal_decode(feats, cal, decode_rank_count,
                              "ordinal cumulative + rank-count (0 params)"),
        method_ordinal_decode(feats, cal, decode_expected,
                              "ordinal cumulative + expected-value (0 params)"),
        method_ordinal_logit(feats, cal, ORD_IDX, C_ord,
                             f"ordinal-logit meta / ord3 (C={C_ord})"),
        method_ordinal_logit(feats, cal, list(range(len(CANON_CUTS))), C_all,
                             f"ordinal-logit meta / all-10 (C={C_all})"),
        method_mono_boost(feats, cal, "monotone boosting / ord3 (order-aware)"),
    ]

    # Baseline for paired tests = the production hierarchy (first method).
    base_test = methods[0][1]
    base_ext = methods[0][2]

    rows = []
    for oof_p, test_p, ext_p, tag in methods:
        m_oof = ordinal_metrics(feats["oof_y"], oof_p)
        m_test = ordinal_metrics(feats["test_y"], test_p)
        m_ext = ordinal_metrics(feats["ext_y"], ext_p)
        q_t, ql_t, qh_t = bootstrap_ci(feats["test_y"], test_p, "qwk", ARGS.n_boot)
        q_e, ql_e, qh_e = bootstrap_ci(feats["ext_y"], ext_p, "qwk", ARGS.n_boot)
        d_t, dl_t, dh_t, p_t = paired_bootstrap_diff(
            feats["test_y"], base_test, test_p, "qwk", ARGS.n_boot)
        d_e, dl_e, dh_e, p_e = paired_bootstrap_diff(
            feats["ext_y"], base_ext, ext_p, "qwk", ARGS.n_boot)
        rows.append({
            "method": tag,
            "oof_qwk": round(m_oof["qwk"], 3),
            "oof_acc": round(m_oof["accuracy"], 3),
            "test_qwk": round(m_test["qwk"], 3),
            "test_qwk_ci": f"{ql_t:.2f}-{qh_t:.2f}",
            "test_acc": round(m_test["accuracy"], 3),
            "test_off1": round(m_test["off_by_one"], 3),
            "test_macroF1": round(m_test["macro_f1"], 3),
            "test_dQWK_vs_hier": f"{d_t:+.3f} (p={p_t:.2f})",
            "ext_qwk": round(m_ext["qwk"], 3),
            "ext_qwk_ci": f"{ql_e:.2f}-{qh_e:.2f}",
            "ext_acc": round(m_ext["accuracy"], 3),
            "ext_off1": round(m_ext["off_by_one"], 3),
            "ext_macroF1": round(m_ext["macro_f1"], 3),
            "ext_dQWK_vs_hier": f"{d_e:+.3f} (p={p_e:.2f})",
        })

    df = pd.DataFrame(rows)
    out_dir = ARGS.out_root / "ordinal"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison.csv", index=False, encoding="utf-8-sig")

    # Human-readable report.
    key = ["method", "oof_qwk", "test_qwk", "test_qwk_ci", "test_acc",
           "test_macroF1", "ext_qwk", "ext_qwk_ci", "ext_acc", "ext_macroF1"]
    print("\n================ ORDINAL METHODS COMPARISON ================")
    print("(selection on OOF; internal test n=58 and external AVNFH n=164 are report-only)")
    print("QWK = quadratic weighted kappa (human inter-observer kappa ~ 0.39-0.46)\n")
    print(df[key].to_string(index=False))

    def to_md(frame, cols):
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = ["| " + " | ".join(str(r[c]) for c in cols) + " |"
                for _, r in frame.iterrows()]
        return "\n".join([head, sep, *body])

    md = ["# Ordinal-aware methods — internal test + external AVNFH (no retraining)",
          "",
          "Selection on OOF only. QWK = quadratic weighted kappa "
          "(order-aware; human inter-observer kappa ~ 0.39-0.46).",
          "`dQWK_vs_hier` = paired-bootstrap QWK gain over the production hierarchy.",
          "",
          "## Headline (QWK + accuracy)", "",
          to_md(df, key),
          "",
          "## Full table (with paired-bootstrap tests)", "",
          to_md(df, list(df.columns))]
    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[ordinal] wrote {out_dir/'comparison.csv'} and REPORT.md", flush=True)


if __name__ == "__main__":
    main()
