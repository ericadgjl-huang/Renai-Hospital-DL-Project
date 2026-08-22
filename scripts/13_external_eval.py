"""External-cohort validation on a NEW dataset (e.g. AVNFH), with a full
misclassification log.

This does NOT train or change any model. It loads the frozen cut checkpoints
plus the already-selected hierarchy topology and the fitted combiner, runs them
over an external, never-seen cohort, and reports honest generalization numbers.

Because the cohort is fully independent (no image was in train/val/test), this
is the strongest evidence that the model generalizes — much stronger than the
internal 58-image test split.

For the professor's request, every wrong prediction is recorded three ways:
  1. the crop is copied into  misclassified/<model>/true{T}_pred{P}/  folders,
  2. each copied file is renamed  true{T}_as_pred{P}__<original>.jpg,
  3. a CSV lists every error with true stage, predicted stage, the raw source
     X-ray path, and the per-cut P(class=1) that drove the decision.

Two models are evaluated side by side:
  * hierarchy[<topo>]   — the model the web app serves (hard routing).
  * combiner[all]       — the best experimental model (learned 4-class head).

Usage (from repo root, conda env unet_labeling):
    python scripts/13_external_eval.py --data-root stage_cls_avnfh --device 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from torchvision.datasets import ImageFolder

from renai.data import make_4class_eval_loader
from renai.eval import (
    bootstrap_classification_ci,
    save_classification_report,
    save_confusion_matrix,
)
from renai.hierarchy import TOPOLOGIES, CutPredictor, route_samples
from renai.seed import SEED, set_seed

STAGE_NAMES = ["stage_1", "stage_2", "stage_3", "stage_4"]


def _load_roi_source_map(roi_csv: Path) -> dict[str, str]:
    """crop filename -> original raw X-ray path (so a doctor can find the source)."""
    if not roi_csv.exists():
        return {}
    df = pd.read_csv(roi_csv)
    if "filename" not in df.columns or "source" not in df.columns:
        return {}
    return dict(zip(df["filename"].astype(str), df["source"].astype(str)))


def _per_cut_probs(predictors, data_root, n, batch_size, device):
    """Return (per_cut[cut] -> P(class=1) over all n images, y_true_1based)."""
    per_cut: dict[str, np.ndarray] = {}
    y_true: list[int] | None = None
    for cn in sorted(predictors):
        loader = make_4class_eval_loader(data_root, list(range(n)), batch_size=batch_size)
        probs, ys = [], []
        for imgs, lbls in loader:
            probs.append(predictors[cn].prob_class1(imgs))
            ys.extend(int(l) + 1 for l in lbls.numpy().tolist())
        per_cut[cn] = np.concatenate(probs)
        if y_true is None:
            y_true = ys
        print(f"  [probs] cut={cn} done ({len(per_cut[cn])} imgs)", flush=True)
    return per_cut, np.asarray(y_true, dtype=np.int64)


def _record(model_tag, y_true, y_pred, paths, per_cut, cut_order, out_dir, roi_map):
    """Copy each misclassified crop into true{T}_pred{P}/ + write a CSV log."""
    mis_root = out_dir / "misclassified" / model_tag
    if mis_root.exists():
        shutil.rmtree(mis_root)
    rows = []
    for i in range(len(y_true)):
        t, p = int(y_true[i]), int(y_pred[i])
        if t == p:
            continue
        folder = mis_root / f"true{t}_pred{p}"
        folder.mkdir(parents=True, exist_ok=True)
        src = Path(paths[i])
        annotated = f"true{t}_as_pred{p}__{src.name}"
        shutil.copy2(src, folder / annotated)
        row = {
            "crop_filename": src.name,
            "true_stage": t,
            "pred_stage": p,
            "note": f"stage {t} -> misclassified as stage {p}",
            "annotated_name": annotated,
            "raw_source_xray": roi_map.get(src.name, ""),
        }
        for cn in cut_order:
            row[f"P1_{cn}"] = round(float(per_cut[cn][i]), 4)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["true_stage", "pred_stage", "crop_filename"])
    df.to_csv(out_dir / f"misclassified_{model_tag}.csv", index=False, encoding="utf-8-sig")
    return len(rows)


def _metrics_block(model_tag, y_true, y_pred, out_dir, n_boot, seed):
    ci = bootstrap_classification_ci(y_true, y_pred, n_boot=n_boot, seed=seed)
    from sklearn.metrics import f1_score

    per_class_f1 = f1_score(
        y_true, y_pred, labels=[1, 2, 3, 4], average=None, zero_division=0
    )
    save_confusion_matrix(
        y_true - 1, y_pred - 1, STAGE_NAMES,
        out_dir / f"confusion_matrix_{model_tag}.png",
        title=f"{model_tag} (external AVNFH, n={len(y_true)})",
    )
    save_classification_report(
        y_true - 1, y_pred - 1, STAGE_NAMES,
        out_dir / f"classification_report_{model_tag}.txt",
    )
    return {
        "model": model_tag,
        "n": int(len(y_true)),
        "accuracy": ci["accuracy"],
        "accuracy_ci": [ci["accuracy_ci_low"], ci["accuracy_ci_high"]],
        "macro_f1": ci["macro_f1"],
        "macro_f1_ci": [ci["macro_f1_ci_low"], ci["macro_f1_ci_high"]],
        "per_class_f1": {STAGE_NAMES[k]: round(float(per_class_f1[k]), 4) for k in range(4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="External-cohort validation + misclassification log.")
    ap.add_argument("--data-root", type=Path, default=Path("stage_cls_avnfh"),
                    help="ImageFolder with stage_1..stage_4 subfolders (the NEW cohort).")
    ap.add_argument("--out-root", type=Path, default=Path("outputs"),
                    help="Where the trained cut checkpoints / topology / combiner live.")
    ap.add_argument("--roi-csv", type=Path, default=Path("roi_avnfh.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/external_avnfh"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    set_seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- files/labels aligned with the eval loader order ---
    base = ImageFolder(args.data_root)
    if base.classes != STAGE_NAMES:
        raise SystemExit(f"Expected classes {STAGE_NAMES}, got {base.classes}")
    paths = [p for p, _ in base.samples]
    n = len(paths)
    roi_map = _load_roi_source_map(args.roi_csv)
    print(f"[external] {n} images from {args.data_root}", flush=True)

    # --- load all available cut predictors (read-only) ---
    all_cuts = [d.name for d in (args.out_root / "cuts").iterdir() if d.is_dir()]
    predictors: dict[str, CutPredictor] = {}
    for cn in sorted(all_cuts):
        if (args.out_root / "cuts" / cn / "ensemble" / "winner.json").exists():
            predictors[cn] = CutPredictor(args.out_root / "cuts" / cn, args.device)
    print(f"[external] loaded {len(predictors)} cut predictors", flush=True)

    per_cut, y_true = _per_cut_probs(predictors, args.data_root, n, args.batch_size, args.device)
    cut_order = sorted(per_cut)

    results = []

    # --- model 1: production hierarchy topology (hard routing) ---
    best_topo_path = args.out_root / "hierarchy" / "best_topology.json"
    if best_topo_path.exists():
        best = json.loads(best_topo_path.read_text(encoding="utf-8"))
        topo = TOPOLOGIES[best["name"]]
        tag = f"hierarchy_{topo.name}"
        y_pred = route_samples(topo, per_cut, n)
        results.append(_metrics_block(tag, y_true, y_pred, args.out_dir, args.n_boot, SEED))
        n_mis = _record(tag, y_true, y_pred, paths, per_cut, cut_order, args.out_dir, roi_map)
        print(f"[external] {tag}: {n_mis} misclassified", flush=True)

    # --- model 2: best combiner (learned 4-class head) ---
    comb_path = args.out_root / "combiner" / "all" / "combiner_model.joblib"
    if comb_path.exists():
        bundle = joblib.load(comb_path)
        model, comb_cuts = bundle["model"], list(bundle["cuts"])
        missing = [c for c in comb_cuts if c not in per_cut]
        if missing:
            print(f"[external] combiner skipped, missing cuts: {missing}", flush=True)
        else:
            X = np.column_stack([per_cut[c] for c in comb_cuts])
            y_pred = model.predict(X).astype(np.int64)
            tag = "combiner_all"
            results.append(_metrics_block(tag, y_true, y_pred, args.out_dir, args.n_boot, SEED))
            n_mis = _record(tag, y_true, y_pred, paths, per_cut, cut_order, args.out_dir, roi_map)
            print(f"[external] {tag}: {n_mis} misclassified", flush=True)

    # --- write summary ---
    (args.out_dir / "external_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    class_counts = {STAGE_NAMES[k]: int((y_true == k + 1).sum()) for k in range(4)}
    lines = [
        "# External-cohort validation (AVNFH) — no retraining, models used as-is",
        "",
        f"- cohort: `{args.data_root}`  (n = {n})",
        f"- class counts: {class_counts}",
        "- This cohort was never seen in train/val/test, so these numbers reflect",
        "  true generalization to a new batch of patients.",
        "",
        "| model | accuracy (95% CI) | macro-F1 (95% CI) | per-class F1 (1/2/3/4) |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        pc = r["per_class_f1"]
        lines.append(
            f"| {r['model']} | {r['accuracy']:.3f} "
            f"({r['accuracy_ci'][0]:.3f}-{r['accuracy_ci'][1]:.3f}) | "
            f"{r['macro_f1']:.3f} ({r['macro_f1_ci'][0]:.3f}-{r['macro_f1_ci'][1]:.3f}) | "
            f"{pc['stage_1']:.2f}/{pc['stage_2']:.2f}/{pc['stage_3']:.2f}/{pc['stage_4']:.2f} |"
        )
    lines += [
        "",
        "## Misclassification log (professor's request)",
        "- `misclassified/<model>/true{T}_pred{P}/` — every wrong crop, foldered by true→pred.",
        "- each file renamed `true{T}_as_pred{P}__<original>.jpg`.",
        "- `misclassified_<model>.csv` — true stage, predicted stage, raw source X-ray, per-cut P(class=1).",
    ]
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\n[external] wrote results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
