"""Training loop for one (cut, backbone, fold)."""

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
    epochs_override: int | None = None,
    early_stop: bool = True,
) -> TrainResult:
    """Train one (backbone, fold).

    Best checkpoint is selected by val macro-F1.  When `epochs_override` is
    provided we ignore best-tracking and simply train for that many epochs,
    saving only the last checkpoint — used by the post-CV "retrain on full
    train+val" stage where there is no held-out validation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_{backbone}.pth"
    log_path = out_dir / "train_log.csv"

    model = create_model(backbone, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_epochs = int(epochs_override) if epochs_override else int(epochs)

    best_macro_f1 = -1.0
    best_acc = 0.0
    best_epoch = -1
    logs = []
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        if epochs_override is None and val_loader is not None and len(val_loader.dataset) > 0:
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
        retrain_mode = epochs_override is not None
        improved = (not retrain_mode) and val_macro_f1 > best_macro_f1
        if retrain_mode:
            val_tag = "  [retrain on full 80%, no val]"
        else:
            marker = "  <- new best (val_macro_f1)" if improved else ""
            val_tag = f"  val_acc={val_acc:.3f} val_macro_f1={val_macro_f1:.4f}{marker}"
        print(
            f"  [{backbone}] epoch {ep:02d}/{n_epochs}  loss={train_loss:.4f} "
            f"acc={train_acc:.3f}{val_tag}",
            flush=True,
        )

        if improved:
            best_macro_f1 = val_macro_f1
            best_acc = val_acc
            best_epoch = ep
            torch.save(model.state_dict(), ckpt_path)

    if epochs_override is not None:
        torch.save(model.state_dict(), ckpt_path)
        best_epoch = n_epochs
        best_macro_f1 = float("nan")
        best_acc = float("nan")

    pd.DataFrame(logs).to_csv(log_path, index=False, encoding="utf-8-sig")

    return TrainResult(
        backbone=backbone,
        best_epoch=int(best_epoch),
        best_val_macro_f1=float(best_macro_f1),
        best_val_acc=float(best_acc),
        ckpt_path=str(ckpt_path),
        log_path=str(log_path),
        train_time_sec=float(time.time() - t0),
        epochs_run=n_epochs,
    )


def train_result_to_row(tr: TrainResult) -> dict:
    return asdict(tr)
