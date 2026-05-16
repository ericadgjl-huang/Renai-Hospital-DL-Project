"""5-fold cross-validation orchestrator for one cut.

Pipeline per cut:
    1. Run 5-fold stratified CV (on the 80% train+val pool) for every backbone.
    2. Aggregate mean ± std macro-F1 across folds.
    3. Take mean(best_epoch) per backbone, retrain on the full 80% for that
       many epochs (no val held out), save as the `final/` checkpoint.
    4. Evaluate `final/` checkpoints on the 20% test set."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from .data import (
    Cut,
    filter_indices_for_cut,
    make_cv_folds,
    make_eval_loader_for_cut,
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
    per_backbone: dict      # backbone -> dict (cv mean/std + final test metrics)
    summary_csv: str
    final_dir: str


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
    """Run the full CV+final pipeline for a single cut. Reproducible — fixed seed."""
    set_seed(SEED)

    cut_root = out_root / "cuts" / cut.name
    cv_root = cut_root / "cv"
    final_root = cut_root / "final"
    cut_root.mkdir(parents=True, exist_ok=True)

    outer = make_outer_split(
        data_root=data_root,
        out_path=splits_dir / "outer_split.json",
        test_ratio=test_ratio,
        seed=SEED,
    )
    # Restrict to the stages that this cut actually distinguishes.  Non-cut
    # stages would only contaminate training; the outer split already locks
    # which raw samples are off-limits for training.
    train_val_idx = filter_indices_for_cut(data_root, outer["train_val_idx"], cut)
    test_idx      = filter_indices_for_cut(data_root, outer["test_idx"],      cut)
    folds = make_cv_folds(data_root, train_val_idx, n_splits=n_splits, seed=SEED)

    if smoke:
        folds = folds[:1]
        epochs = max(1, min(epochs, 2))

    test_loader = make_eval_loader_for_cut(
        data_root, cut, test_idx, batch_size=batch_size,
    )

    cv_rows = []
    per_backbone: dict[str, dict] = {}

    for backbone in backbones:
        print(f"\n=== CUT={cut.name}  BACKBONE={backbone}  ===", flush=True)
        fold_results = []
        for fi, (tr_idx, va_idx) in enumerate(folds):
            print(f"  fold {fi}: |train|={len(tr_idx)}  |val|={len(va_idx)}", flush=True)
            set_seed(SEED + fi)  # deterministic but fold-dependent
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

            # Eval on val with full metrics
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

        # Aggregate over folds
        if fold_results:
            mean_macro_f1 = statistics.fmean(r["macro_f1"] for r in fold_results)
            std_macro_f1 = (
                statistics.pstdev(r["macro_f1"] for r in fold_results)
                if len(fold_results) > 1 else 0.0
            )
            mean_acc = statistics.fmean(r["accuracy"] for r in fold_results)
            mean_auc = statistics.fmean(
                r.get("auc", float("nan")) for r in fold_results
                if not math.isnan(r.get("auc", float("nan")))
            ) if any(not math.isnan(r.get("auc", float("nan"))) for r in fold_results) else float("nan")
            mean_best_epoch = max(1, round(statistics.fmean(r["best_epoch"] for r in fold_results)))
        else:
            mean_macro_f1 = std_macro_f1 = mean_acc = mean_auc = float("nan")
            mean_best_epoch = epochs

        print(
            f"  >> CV mean macro_f1={mean_macro_f1:.4f} ± {std_macro_f1:.4f} | "
            f"acc={mean_acc:.4f} | auc={mean_auc:.4f} | mean_best_epoch={mean_best_epoch}",
            flush=True,
        )

        # === Final retrain on full train+val with mean_best_epoch (no val) ===
        print(f"  -- retraining on full 80% for {mean_best_epoch} epochs --", flush=True)
        set_seed(SEED + 9999)
        full_loader, _ = make_loaders_for_cut(
            data_root, cut, train_val_idx, [], batch_size=batch_size,
        )
        final_dir = final_root / backbone
        tr_final = train_one(
            backbone=backbone,
            train_loader=full_loader,
            val_loader=None,
            out_dir=final_dir,
            device=device,
            epochs=mean_best_epoch,
            lr=lr,
            epochs_override=mean_best_epoch,
        )

        # Eval final on outer test set
        model = create_model(backbone, num_classes=2).to(device)
        model.load_state_dict(torch.load(tr_final.ckpt_path, map_location=device))
        y_true, y_pred, probs = predict_loader(model, test_loader, device)
        test_metrics = binary_metrics(y_true, y_pred, probs)
        dump_json(test_metrics, final_dir / "test_metrics.json")
        save_confusion_matrix(
            y_true, y_pred, list(cut.class_names),
            final_dir / "confusion_matrix_test.png",
            title=f"{cut.name} | {backbone} | TEST",
        )
        save_classification_report(
            y_true, y_pred, list(cut.class_names),
            final_dir / "classification_report_test.txt",
        )
        # Save test probs for later ensembling
        import numpy as np
        np.save(final_dir / "test_probs.npy", probs)
        np.save(final_dir / "test_y_true.npy", y_true)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        per_backbone[backbone] = {
            "cv_mean_macro_f1": _round(mean_macro_f1),
            "cv_std_macro_f1": _round(std_macro_f1),
            "cv_mean_acc": _round(mean_acc),
            "cv_mean_auc": _round(mean_auc),
            "cv_mean_best_epoch": int(mean_best_epoch),
            "test_acc": _round(test_metrics["accuracy"]),
            "test_macro_f1": _round(test_metrics["macro_f1"]),
            "test_auc": _round(test_metrics.get("auc", float("nan"))),
            "final_ckpt": tr_final.ckpt_path,
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
        final_dir=str(final_root),
    )
