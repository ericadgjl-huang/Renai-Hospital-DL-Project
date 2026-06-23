"""Bootstrap confidence-interval report for the final pipeline.

Why this module exists
----------------------
The outer test set is tiny (n≈58, 10–23 samples per class). A single macro-F1
number is therefore very noisy. This module resamples the test set with
replacement to attach a 95% CI to every headline metric, so the thesis can
report e.g. "macro-F1 0.67 (95% CI 0.55–0.79)" instead of a bare 0.67.

It reads predictions already on disk:
  * the SELECTED topology  -> outputs/hierarchy/<T>/test_y_true.npy / test_y_pred.npy
  * every binary cut       -> outputs/cuts/<cut>/ensemble/test_probs.npy + test_y_true.npy

Patient-level grouping
----------------------
GroupKFold (grouping all images of one patient into the same fold) is NOT done,
because the filenames `S<stage>_<side><n>.jpg` carry no patient id — `S1_L1` and
`S1_R1` need not be the same person. See `renai.data.patient_groups` for the
hook to switch grouping on once a filename→patient_id mapping is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .eval import bootstrap_classification_ci, dump_json

STAGE_NAMES = ["stage_1", "stage_2", "stage_3", "stage_4"]

# Reported human reliability of Ficat staging on radiographs (for context only).
HUMAN_FICAT_KAPPA = "inter-observer kappa approx 0.39-0.46 (Smith 1996; Ricci 2007)"


def _fmt(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} (95% CI {lo:.3f}-{hi:.3f})"


def build_ci_report(
    out_root: Path,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    out_root = Path(out_root)
    report_dir = out_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # --- 1. selected 4-class topology -------------------------------------
    best_path = out_root / "hierarchy" / "best_topology.json"
    topo_block: dict = {}
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        topo_dir = out_root / "hierarchy" / best["name"]
        yt_p = topo_dir / "test_y_true.npy"
        yp_p = topo_dir / "test_y_pred.npy"
        if yt_p.exists() and yp_p.exists():
            y_true = np.load(yt_p)
            y_pred = np.load(yp_p)
            ci = bootstrap_classification_ci(y_true, y_pred, n_boot=n_boot, seed=seed)
            per_class = f1_score(
                y_true, y_pred, labels=[1, 2, 3, 4], average=None, zero_division=0
            )
            topo_block = {
                "name": best["name"],
                "description": best.get("description", ""),
                "selected_by": best.get("selected_by", "oof_macro_f1"),
                **ci,
                "per_class_f1": {STAGE_NAMES[i]: float(per_class[i]) for i in range(4)},
            }
            rows.append({
                "scope": "4class_topology",
                "name": best["name"],
                "n": ci["n"],
                "accuracy": ci["accuracy"],
                "accuracy_ci_low": ci["accuracy_ci_low"],
                "accuracy_ci_high": ci["accuracy_ci_high"],
                "macro_f1": ci["macro_f1"],
                "macro_f1_ci_low": ci["macro_f1_ci_low"],
                "macro_f1_ci_high": ci["macro_f1_ci_high"],
            })
        else:
            print(f"[ci] missing {yt_p.name}/{yp_p.name} — re-run 06_search_hierarchy.py", flush=True)

    # --- 2. each binary cut -----------------------------------------------
    cuts_root = out_root / "cuts"
    for probs_path in sorted(cuts_root.glob("*/ensemble/test_probs.npy")):
        cut_name = probs_path.parent.parent.name
        yt_path = probs_path.parent / "test_y_true.npy"
        if not yt_path.exists():
            continue
        probs = np.load(probs_path)
        y_true = np.load(yt_path)
        y_pred = probs.argmax(axis=1) if probs.ndim == 2 else (probs >= 0.5).astype(int)
        ci = bootstrap_classification_ci(y_true, y_pred, n_boot=n_boot, seed=seed)
        rows.append({
            "scope": "binary_cut",
            "name": cut_name,
            "n": ci["n"],
            "accuracy": ci["accuracy"],
            "accuracy_ci_low": ci["accuracy_ci_low"],
            "accuracy_ci_high": ci["accuracy_ci_high"],
            "macro_f1": ci["macro_f1"],
            "macro_f1_ci_low": ci["macro_f1_ci_low"],
            "macro_f1_ci_high": ci["macro_f1_ci_high"],
        })

    df = pd.DataFrame(rows)
    csv_path = report_dir / "ci_report.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # --- 3. human-readable markdown ---------------------------------------
    lines = ["# Test-set metrics with 95% bootstrap CI", ""]
    lines.append(f"- bootstrap resamples: {n_boot}, seed {seed}")
    lines.append(f"- human Ficat reliability (context): {HUMAN_FICAT_KAPPA}")
    lines.append("")
    if topo_block:
        lines.append(f"## Final 4-class — topology {topo_block['name']} {topo_block['description']}")
        lines.append(f"- selected by: {topo_block['selected_by']} (test set never used for selection)")
        lines.append(f"- n = {topo_block['n']}")
        lines.append(f"- accuracy = {_fmt(topo_block['accuracy'], topo_block['accuracy_ci_low'], topo_block['accuracy_ci_high'])}")
        lines.append(f"- macro-F1 = {_fmt(topo_block['macro_f1'], topo_block['macro_f1_ci_low'], topo_block['macro_f1_ci_high'])}")
        lines.append("- per-class F1: " + ", ".join(
            f"{k}={v:.3f}" for k, v in topo_block["per_class_f1"].items()
        ))
        lines.append("")
    if not df.empty:
        cuts_df = df[df["scope"] == "binary_cut"]
        if not cuts_df.empty:
            lines.append("## Binary cuts (test)")
            lines.append("")
            lines.append("| cut | n | accuracy (95% CI) | macro-F1 (95% CI) |")
            lines.append("| --- | --- | --- | --- |")
            for _, r in cuts_df.iterrows():
                lines.append(
                    f"| {r['name']} | {int(r['n'])} | "
                    f"{_fmt(r['accuracy'], r['accuracy_ci_low'], r['accuracy_ci_high'])} | "
                    f"{_fmt(r['macro_f1'], r['macro_f1_ci_low'], r['macro_f1_ci_high'])} |"
                )
    md_path = report_dir / "ci_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dump_json(
        {"topology": topo_block, "n_boot": n_boot, "seed": seed},
        report_dir / "ci_report.json",
    )

    print(f"[ci] wrote {csv_path}", flush=True)
    print(f"[ci] wrote {md_path}", flush=True)
    return {"csv": str(csv_path), "md": str(md_path), "topology": topo_block}
