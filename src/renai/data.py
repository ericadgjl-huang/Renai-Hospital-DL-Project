"""Datasets, transforms, splits.

A `cut` re-labels the 4 raw stages (0..3) into a 2-class problem.  A cut is
described by a list of stages that map to label 0; the remaining stages map to
label 1.  E.g. cut "1_vs_234" => positives_zero=[1], so stage 1 -> 0 and
stages 2/3/4 -> 1.

The 80/20 outer split is computed once with stratification on the 4-class
labels, frozen to disk (outputs/splits/outer_split.json), and reused by every
cut.  Inside the 80% we run 5-fold stratified CV on the same 4-class labels so
that every cut sees the same fold assignments — that is what makes results
comparable across cuts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torchvision.transforms as T
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from .seed import SEED, torch_generator

IMG_SIZE = 384
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
NUM_STAGES = 4


def make_train_transform(img_size: int = IMG_SIZE) -> T.Compose:
    """Augmentation for tiny radiograph data.

    Deliberately NO horizontal flip: the dataset pipeline already flips every
    right hip to the left orientation, so a flip would reverse the medial/
    lateral sides of the femoral head and undo that normalization. We use mild
    affine jitter (rotation + small translate/scale) plus brightness/contrast
    jitter to mimic positioning and exposure variation."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_STD),
    ])


def make_eval_transform(img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_STD),
    ])


@dataclass(frozen=True)
class Cut:
    """A binary cut: stages in `positives_zero` map to label 0, the rest to 1."""

    name: str
    positives_zero: tuple[int, ...]    # 1-indexed stage numbers (1..4)
    positives_one: tuple[int, ...]
    description: str = ""

    def relabel(self, stage_idx0: int) -> int:
        """Map a 0-indexed stage class (0..3) to the binary cut label (0/1)."""
        stage_1based = stage_idx0 + 1
        if stage_1based in self.positives_zero:
            return 0
        if stage_1based in self.positives_one:
            return 1
        raise ValueError(f"Stage {stage_1based} is not assigned in cut {self.name}")

    @property
    def class_names(self) -> tuple[str, str]:
        a = "_".join(str(s) for s in self.positives_zero)
        b = "_".join(str(s) for s in self.positives_one)
        return (f"stages_{a}", f"stages_{b}")


class BinaryCutDataset(Dataset):
    """Wraps an ImageFolder and re-labels its 4-class targets to a binary cut."""

    def __init__(self, base: ImageFolder, cut: Cut):
        self.base = base
        self.cut = cut

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y4 = self.base[idx]
        return x, self.cut.relabel(int(y4))


def load_imagefolder(data_root: Path, transform=None) -> ImageFolder:
    ds = ImageFolder(data_root, transform=transform)
    if len(ds.classes) != NUM_STAGES:
        raise RuntimeError(
            f"Expected {NUM_STAGES} class subfolders in {data_root}, got {ds.classes}"
        )
    return ds


def get_4class_labels(data_root: Path) -> torch.Tensor:
    base = ImageFolder(data_root)
    return torch.tensor([y for _, y in base.samples], dtype=torch.long)


def patient_groups(data_root: Path, roi_csv: Path | None = None):
    """Return a group id per ImageFolder sample (for GroupKFold), or None.

    Looks for a ``patient`` / ``patient_id`` column in ``roi_csv`` and maps it
    onto the ImageFolder sample order by filename. Returns None (and prints a
    warning) when no such column exists — which is the CURRENT state of this
    dataset: filenames `S<stage>_<side><n>.jpg` encode stage+side only, so two
    images CANNOT be linked to the same patient. Provide a CSV with a patient
    column to enable true patient-level splits and remove leakage risk."""
    if roi_csv is None or not Path(roi_csv).exists():
        print("[groups] no roi_csv -> patient grouping disabled "
              "(image-level split; possible patient leakage).", flush=True)
        return None
    try:
        import pandas as pd
        df = pd.read_csv(roi_csv)
    except Exception as e:  # noqa: BLE001
        print(f"[groups] could not read {roi_csv}: {e}", flush=True)
        return None

    pid_col = next((c for c in ("patient", "patient_id", "pid") if c in df.columns), None)
    if pid_col is None or "filename" not in df.columns:
        print(f"[groups] {Path(roi_csv).name} has no patient column "
              f"(cols={list(df.columns)}) -> grouping disabled.", flush=True)
        return None

    fname_to_pid = dict(zip(df["filename"].astype(str), df[pid_col].astype(str)))
    base = ImageFolder(data_root)
    groups = [fname_to_pid.get(Path(p).name, Path(p).name) for p, _ in base.samples]
    n_groups = len(set(groups))
    print(f"[groups] patient grouping ENABLED: {n_groups} groups over {len(groups)} images.",
          flush=True)
    return groups


def filter_indices_for_cut(
    data_root: Path,
    indices: Sequence[int],
    cut: "Cut",
) -> list[int]:
    """Restrict global indices to samples whose 4-class stage is part of the cut.

    Cuts that only distinguish between stages 2/3 must not see stage 1 or 4
    samples during training — otherwise the binary classifier learns from
    irrelevant data.  Inference-time leakage (a parent cut routes a stage-1
    sample into a 2-vs-3 child) is a hierarchy error, not a training one."""
    labels = get_4class_labels(data_root).numpy()
    allowed = set(cut.positives_zero) | set(cut.positives_one)  # 1-indexed
    return [int(i) for i in indices if int(labels[i]) + 1 in allowed]


def make_outer_split(
    data_root: Path,
    out_path: Path,
    test_ratio: float = 0.2,
    seed: int = SEED,
    groups=None,
) -> dict:
    """Compute (and cache) an 80/20 outer split on the 4-class labels.

    When `groups` is given (one id per sample), the split is made with
    GroupShuffleSplit so no patient appears in both train_val and test. Without
    groups it falls back to a stratified image-level split (current behaviour)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))

    labels = get_4class_labels(data_root).numpy()
    indices = list(range(len(labels)))
    grouped = groups is not None
    if grouped:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(gss.split(indices, labels, groups=groups))
    else:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=labels,
        )
    payload = {
        "seed": seed,
        "test_ratio": test_ratio,
        "grouped": bool(grouped),
        "train_val_idx": sorted(map(int, train_val_idx)),
        "test_idx": sorted(map(int, test_idx)),
        "n_total": int(len(labels)),
        "labels_per_class_total": [int((labels == c).sum()) for c in range(NUM_STAGES)],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def make_cv_folds(
    data_root: Path,
    train_val_idx: Sequence[int],
    n_splits: int = 5,
    seed: int = SEED,
    groups=None,
) -> list[tuple[list[int], list[int]]]:
    """5-fold CV on the 4-class labels restricted to train_val_idx.

    Returns a list of (train_idx, val_idx) pairs in *global* dataset indices.
    When `groups` is given (one id per global sample), uses StratifiedGroupKFold
    so a patient never spans train and val; otherwise plain StratifiedKFold."""
    labels = get_4class_labels(data_root).numpy()
    sub_idx = list(train_val_idx)
    sub_labels = labels[sub_idx]

    folds: list[tuple[list[int], list[int]]] = []
    if groups is not None:
        sub_groups = [groups[i] for i in sub_idx]
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = sgkf.split(sub_idx, sub_labels, groups=sub_groups)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = skf.split(sub_idx, sub_labels)
    for tr, va in splitter:
        tr_global = [int(sub_idx[i]) for i in tr]
        va_global = [int(sub_idx[i]) for i in va]
        folds.append((tr_global, va_global))
    return folds


def make_loaders_for_cut(
    data_root: Path,
    cut: Cut,
    train_idx: Iterable[int],
    val_idx: Iterable[int],
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = IMG_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for a single cut + index split."""
    train_tf = make_train_transform(img_size)
    eval_tf = make_eval_transform(img_size)

    train_base = load_imagefolder(data_root, transform=train_tf)
    val_base = load_imagefolder(data_root, transform=eval_tf)

    train_ds = BinaryCutDataset(train_base, cut)
    val_ds = BinaryCutDataset(val_base, cut)

    train_loader = DataLoader(
        Subset(train_ds, list(train_idx)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch_generator(),
    )
    val_loader = DataLoader(
        Subset(val_ds, list(val_idx)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def make_eval_loader_for_cut(
    data_root: Path,
    cut: Cut,
    idx: Iterable[int],
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = IMG_SIZE,
) -> DataLoader:
    eval_tf = make_eval_transform(img_size)
    base = load_imagefolder(data_root, transform=eval_tf)
    ds = BinaryCutDataset(base, cut)
    return DataLoader(
        Subset(ds, list(idx)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def make_4class_eval_loader(
    data_root: Path,
    idx: Iterable[int],
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = IMG_SIZE,
) -> DataLoader:
    """Eval loader that yields (image, original 4-class label 0..3)."""
    eval_tf = make_eval_transform(img_size)
    base = load_imagefolder(data_root, transform=eval_tf)
    return DataLoader(
        Subset(base, list(idx)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
