"""Training loop for one (cut, backbone, fold).

The best checkpoint by validation macro-F1 is kept; that ckpt is the artefact
the downstream fold-voting ensemble consumes."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from .models import create_model


@dataclass
class TrainResult:
    backbone: str
    best_epoch: int
    best_val_macro_f1: float
    best_val_acc: float
    ckpt_path: str
    log_path: str
    train_time_sec: float
    epochs_run: int


def _eval_loop(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())
    if not y_true:
        return 0.0, 0.0
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", zero_division=0)


def train_one(
    backbone: str,
    train_loader,
    val_loader,
    out_dir: Path,
    device: str,
    num_classes: int = 2,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    class_weights=None,
) -> TrainResult:
    """Train one (backbone, fold). Save the best-by-val-macro-F1 ckpt.

    Small-data regularization knobs (defaults are the project's new baseline):
      * AdamW weight_decay (decoupled L2);
      * class-weighted cross-entropy to counter class imbalance;
      * early stopping after `patience` epochs without val-macro-F1 improvement.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_{backbone}.pth"
    log_path = out_dir / "train_log.csv"

    model = create_model(backbone, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    best_macro_f1 = -1.0
    best_acc = 0.0
    best_epoch = -1
    epochs_since_improve = 0
    logs = []
    t0 = time.time()

    for ep in range(1, int(epochs) + 1):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels, weight=weight_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_acc, val_macro_f1 = _eval_loop(model, val_loader, device)
        else:
            val_acc, val_macro_f1 = float("nan"), float("nan")

        logs.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        })
        improved = val_macro_f1 > best_macro_f1
        marker = "  <- new best (val_macro_f1)" if improved else ""
        print(
            f"  [{backbone}] epoch {ep:02d}/{int(epochs)}  loss={train_loss:.4f} "
            f"acc={train_acc:.3f}  val_acc={val_acc:.3f} val_macro_f1={val_macro_f1:.4f}{marker}",
            flush=True,
        )

        if improved:
            best_macro_f1 = val_macro_f1
            best_acc = val_acc
            best_epoch = ep
            epochs_since_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_since_improve += 1
            if patience and epochs_since_improve >= patience:
                print(
                    f"  [{backbone}] early stop at epoch {ep} "
                    f"(no val_macro_f1 gain for {patience} epochs; best={best_macro_f1:.4f}@{best_epoch})",
                    flush=True,
                )
                break

    # Guard: if val was empty/NaN throughout, no ckpt was saved — save final.
    if best_epoch < 0:
        torch.save(model.state_dict(), ckpt_path)
        best_epoch = len(logs)

    pd.DataFrame(logs).to_csv(log_path, index=False, encoding="utf-8-sig")

    return TrainResult(
        backbone=backbone,
        best_epoch=int(best_epoch),
        best_val_macro_f1=float(best_macro_f1),
        best_val_acc=float(best_acc),
        ckpt_path=str(ckpt_path),
        log_path=str(log_path),
        train_time_sec=float(time.time() - t0),
        epochs_run=int(epochs),
    )


def train_result_to_row(tr: TrainResult) -> dict:
    return asdict(tr)
