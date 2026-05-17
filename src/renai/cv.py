"""5-fold cross-validation for one cut — no final retrain.

Pipeline per cut:
    1. Run 5-fold stratified CV (on the 80% train+val pool) for every backbone.
    2. For each (backbone, fold) save the validation-best checkpoint under
       outputs/cuts/<cut>/cv/fold_{fi}/<backbone>/best_<backbone>.pth.
    3. Aggregate mean/std macro-F1 across folds for diagnostics only.

The 25 per-fold best checkpoints (5 backbones x 5 folds) are the artefacts that
the ensemble step (renai.ensemble) and the hierarchy step (renai.hierarchy)
soft-vote together at inference time."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from .data import (
    Cut,
    filter_indices_for_cut,
    make_cv_folds,
    make_loaders_for_cut,
    make_outer_split,
)
from .eval import (
    binary_metrics,
    dump_json,
    predict_loader,
    save_classification_report,
    save_confusion_matrix,
)
from .models import DEFAULT_BACKBONES, create_model
from .seed import SEED, set_seed
from .train import train_one


@dataclass
class CutCVResult:
    cut: str
    per_backbone: dict      # backbone -> dict of cv mean/std stats
    summary_csv: str
    cv_root: str


def _round(v: float, n: int = 4) -> float:
    return float("nan") if (v is None or math.isnan(v)) else round(float(v), n)


def run_cv_for_cut(
    cut: Cut,
    data_root: Path,
    out_root: Path,
    splits_dir: Path,
    backbones: Sequence[str] = DEFAULT_BACKBONES,
    n_splits: int = 5,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 16,
    device: str = "cuda",
    test_ratio: float = 0.2,
    smoke: bool = False,
) -> CutCVResult:
    """Run 5-fold CV for a single cut. Reproducible — fixed seed."""
    set_seed(SEED)

    cut_root = out_root / "cuts" / cut.name
    cv_root = cut_root / "cv"
    cut_root.mkdir(parents=True, exist_ok=True)

    outer = make_outer_split(
        data_root=data_root,
        out_path=splits_dir / "outer_split.json",
        test_ratio=test_ratio,
        seed=SEED,
    )
    train_val_idx = filter_indices_for_cut(data_root, outer["train_val_idx"], cut)
    folds = make_cv_folds(data_root, train_val_idx, n_splits=n_splits, seed=SEED)

    if smoke:
        folds = folds[:1]
        epochs = max(1, min(epochs, 2))

    cv_rows = []
    per_backbone: dict[str, dict] = {}

    for backbone in backbones:
        print(f"\n=== CUT={cut.name}  BACKBONE={backbone}  ===", flush=True)
        fold_results = []
        for fi, (tr_idx, va_idx) in enumerate(folds):
            print(f"  fold {fi}: |train|={len(tr_idx)}  |val|={len(va_idx)}", flush=True)
            set_seed(SEED + fi)
            fold_dir = cv_root / f"fold_{fi}" / backbone
            tr_loader, va_loader = make_loaders_for_cut(
                data_root, cut, tr_idx, va_idx, batch_size=batch_size,
            )
            tr = train_one(
                backbone=backbone,
                train_loader=tr_loader,
                val_loader=va_loader,
                out_dir=fold_dir,
                device=device,
                epochs=epochs,
                lr=lr,
            )

            model = create_model(backbone, num_classes=2).to(device)
            model.load_state_dict(torch.load(tr.ckpt_path, map_location=device))
            y_true, y_pred, probs = predict_loader(model, va_loader, device)
            m = binary_metrics(y_true, y_pred, probs)
            dump_json({"fold": fi, **m}, fold_dir / "val_metrics.json")
            save_confusion_matrix(
                y_true, y_pred, list(cut.class_names),
                fold_dir / "confusion_matrix_val.png",
                title=f"{cut.name} | {backbone} | fold {fi} val",
            )
            save_classification_report(
                y_true, y_pred, list(cut.class_names),
                fold_dir / "classification_report_val.txt",
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            fold_results.append({
                "fold": fi,
                "best_epoch": tr.best_epoch,
                **m,
                "train_time_sec": tr.train_time_sec,
            })
            cv_rows.append({"cut": cut.name, "backbone": backbone, **fold_results[-1]})

        if fold_results:
            mean_macro_f1 = statistics.fmean(r["macro_f1"] for r in fold_results)
            std_macro_f1 = (
                statistics.pstdev(r["macro_f1"] for r in fold_results)
                if len(fold_results) > 1 else 0.0
            )
            mean_acc = statistics.fmean(r["accuracy"] for r in fold_results)
            auc_vals = [r.get("auc", float("nan")) for r in fold_results]
            auc_vals = [v for v in auc_vals if not math.isnan(v)]
            mean_auc = statistics.fmean(auc_vals) if auc_vals else float("nan")
            mean_best_epoch = statistics.fmean(r["best_epoch"] for r in fold_results)
        else:
            mean_macro_f1 = std_macro_f1 = mean_acc = mean_auc = float("nan")
            mean_best_epoch = float("nan")

        print(
            f"  >> CV mean macro_f1={mean_macro_f1:.4f} +/- {std_macro_f1:.4f} | "
            f"acc={mean_acc:.4f} | auc={mean_auc:.4f}",
            flush=True,
        )

        per_backbone[backbone] = {
            "cv_mean_macro_f1": _round(mean_macro_f1),
            "cv_std_macro_f1": _round(std_macro_f1),
            "cv_mean_acc": _round(mean_acc),
            "cv_mean_auc": _round(mean_auc),
            "cv_mean_best_epoch": _round(mean_best_epoch, 2),
            "n_folds": len(fold_results),
        }

    summary_path = cut_root / "summary.csv"
    rows = [{"cut": cut.name, "backbone": bb, **stats} for bb, stats in per_backbone.items()]
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    fold_summary_path = cut_root / "cv_per_fold.csv"
    pd.DataFrame(cv_rows).to_csv(fold_summary_path, index=False, encoding="utf-8-sig")

    return CutCVResult(
        cut=cut.name,
        per_backbone=per_backbone,
        summary_csv=str(summary_path),
        cv_root=str(cv_root),
    )
