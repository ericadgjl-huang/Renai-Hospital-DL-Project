"""Top-3 cross-family ensemble for one cut — OOF-based selection (v2.1).

All decisions in this module use **out-of-fold (OOF) predictions only**.  The
20% outer test set is NEVER touched here; that is reserved for the final
report produced by `hierarchy.search_best_topology` on the winning topology.

Pipeline per cut
----------------
1. For every backbone, run the 5 fold ckpts on the full train+val pool and
   build an "OOF prediction matrix" (N_train_val, 2) per backbone:
     - Samples inside the cut's stages: use the fold model that did NOT see
       this sample as training data (true OOF).
     - Samples outside the cut's stages (e.g. stage-1 samples for cut
       `2_vs_3`): use the mean of all 5 fold models (none of them saw the
       sample, so any choice is fair; averaging is the most stable).
2. **Single OOF**: per-backbone OOF macro-F1 on in-cut samples.  Best one
   becomes the `single` candidate.  Top-3 across families are kept for the
   ensemble candidates.
3. **Voting OOF**: average of top-3 OOF softmax → argmax → macro-F1.
4. **Stacking OOF (unbiased)**: 5-fold CV of LogisticRegression on the
   stacked OOF matrix — the meta is evaluated on samples its training fold
   never saw.  A "deployment" meta trained on the full OOF matrix is also
   saved for inference time.
5. **Decision**: max(voting OOF, stacking OOF) wins iff strictly greater
   than single OOF macro-F1, else single (matches the user's rule
   "沒變好就用單一模型").

Saved artefacts per cut
-----------------------
- `outputs/cuts/<cut>/oof/all_train_val.npz`  per-backbone OOF probs +
  4-class labels + cut_mask + indices (consumed by `hierarchy.py`).
- `outputs/cuts/<cut>/oof/oof_p_class1.npy`   P(class=1) per train_val
  sample using the cut's WINNING strategy (already unbiased).
- `outputs/cuts/<cut>/ensemble/winner.json`   decision + candidates
  (OOF metrics only, no test).
- `outputs/cuts/<cut>/ensemble/meta_logreg.pkl`  deployment meta (only
  meaningful when stacking is the winner, but always saved).
- `outputs/cuts/<cut>/ensemble/confusion_matrix_oof.png`
- `outputs/cuts/<cut>/ensemble/classification_report_oof.txt`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .data import (
    Cut,
    filter_indices_for_cut,
    get_4class_labels,
    make_4class_eval_loader,
    make_cv_folds,
    make_outer_split,
)
from .eval import (
    binary_metrics,
    dump_json,
    save_classification_report,
    save_confusion_matrix,
)
from .models import DEFAULT_BACKBONES, FAMILY_OF, create_model
from .seed import SEED, set_seed


@dataclass
class EnsembleDecision:
    cut: str
    chosen: str                # "single" | "voting" | "stacking"
    chosen_backbone: str | None
    chosen_members: list[str]
    oof_macro_f1: float
    oof_accuracy: float
    oof_auc: float
    artifacts_dir: str


# ---------------------------------------------------------------------------
# OOF computation
# ---------------------------------------------------------------------------

def _compute_full_oof(
    cut_root: Path,
    cut: Cut,
    data_root: Path,
    backbones: Sequence[str],
    all_train_val_idx: Sequence[int],
    device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list]:
    """Compute per-backbone OOF softmax probabilities for ALL train_val samples.

    For samples whose 4-class stage is one the cut distinguishes, we use the
    fold model that did NOT have them in its training data (true OOF).  For
    samples outside the cut's stages, we average the 5 fold models — none of
    them saw the sample, so any one is fair; averaging is most stable.

    Returns:
        probs_by_bb: {backbone: (N, 2)} float32 in train_val order
        y_4class:    (N,) int64, 4-class labels 0..3
        cut_mask:    (N,) bool, True where the sample is in cut's stages
        folds_filtered: list of (train_idx, val_idx) tuples in global indices
                        (for in-cut samples), used by stacking CV
    """
    labels_global = get_4class_labels(data_root).numpy()
    in_cut_stage_set = set(cut.positives_zero) | set(cut.positives_one)

    cut_train_val_idx = filter_indices_for_cut(data_root, all_train_val_idx, cut)
    folds_filtered = make_cv_folds(data_root, cut_train_val_idx)

    fold_of: dict[int, int] = {}
    for fi, (_tr, va) in enumerate(folds_filtered):
        for i in va:
            fold_of[int(i)] = fi

    cut_mask = np.array([
        (int(labels_global[i]) + 1) in in_cut_stage_set
        for i in all_train_val_idx
    ], dtype=bool)
    y_4class = np.array([int(labels_global[i]) for i in all_train_val_idx], dtype=np.int64)

    loader = make_4class_eval_loader(data_root, list(all_train_val_idx), batch_size=batch_size)

    probs_by_bb: dict[str, np.ndarray] = {}
    n_folds = len(folds_filtered)

    for bb in backbones:
        fold_models = []
        for fi in range(n_folds):
            ckpt = cut_root / "cv" / f"fold_{fi}" / bb / f"best_{bb}.pth"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"missing fold ckpt {ckpt} — rerun scripts/04_train_all_cuts.py first"
                )
            m = create_model(bb, num_classes=2).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            fold_models.append(m)

        probs = np.zeros((len(all_train_val_idx), 2), dtype=np.float32)
        cursor = 0
        with torch.no_grad():
            for imgs, _labels4 in loader:
                imgs = imgs.to(device)
                fold_probs = np.stack([
                    F.softmax(fm(imgs), dim=1).cpu().numpy() for fm in fold_models
                ])  # (n_folds, B, 2)
                for bi in range(imgs.size(0)):
                    global_idx = all_train_val_idx[cursor + bi]
                    fi = fold_of.get(int(global_idx), -1)
                    if fi >= 0:
                        probs[cursor + bi] = fold_probs[fi, bi]
                    else:
                        probs[cursor + bi] = fold_probs.mean(axis=0)[bi]
                cursor += imgs.size(0)

        probs_by_bb[bb] = probs
        del fold_models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return probs_by_bb, y_4class, cut_mask, folds_filtered


def _pick_top3_cross_family(single_oof_f1: dict[str, float]) -> list[str]:
    """Pick the best backbone per family by OOF macro-F1, up to 3."""
    rows = [(bb, FAMILY_OF[bb], f1) for bb, f1 in single_oof_f1.items() if bb in FAMILY_OF]
    rows.sort(key=lambda r: r[2], reverse=True)
    seen_family: set[str] = set()
    picked: list[str] = []
    for bb, fam, _f in rows:
        if fam in seen_family:
            continue
        seen_family.add(fam)
        picked.append(bb)
        if len(picked) == 3:
            break
    return picked


def _stacking_oof_cv(
    X_stack: np.ndarray,
    y_binary: np.ndarray,
    n_splits: int = 5,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    """5-fold CV of LogisticRegression on the stacked OOF matrix.

    Returns:
        oof_pred:   (N,)   predictions from meta-on-other-folds
        oof_probs:  (N, 2) softmax of those predictions
        final_meta: LogisticRegression trained on the FULL X_stack (used at
                    inference time for samples never seen during meta CV).
    """
    n = len(y_binary)
    oof_pred = np.zeros(n, dtype=np.int64)
    oof_probs = np.zeros((n, 2), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X_stack, y_binary):
        meta = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed,
        )
        meta.fit(X_stack[tr_idx], y_binary[tr_idx])
        oof_pred[va_idx] = meta.predict(X_stack[va_idx])
        oof_probs[va_idx] = meta.predict_proba(X_stack[va_idx]).astype(np.float32)

    final_meta = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed,
    )
    final_meta.fit(X_stack, y_binary)
    return oof_pred, oof_probs, final_meta


def build_ensemble_for_cut(
    cut: Cut,
    data_root: Path,
    out_root: Path,
    splits_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
) -> EnsembleDecision:
    set_seed(SEED)
    cut_root = out_root / "cuts" / cut.name
    ens_dir = cut_root / "ensemble"
    oof_dir = cut_root / "oof"
    ens_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    outer = make_outer_split(data_root, splits_dir / "outer_split.json")
    all_train_val_idx = list(outer["train_val_idx"])

    backbones = list(DEFAULT_BACKBONES)

    # 1. OOF probs for ALL train_val samples, per backbone
    probs_by_bb, y_4class, cut_mask, _folds = _compute_full_oof(
        cut_root, cut, data_root, backbones, all_train_val_idx, device, batch_size,
    )

    np.savez(
        oof_dir / "all_train_val.npz",
        all_train_val_idx=np.asarray(all_train_val_idx, dtype=np.int64),
        y_4class=y_4class,
        cut_mask=cut_mask,
        **{f"probs_{bb}": p for bb, p in probs_by_bb.items()},
    )

    # 2. Single OOF macro-F1 per backbone (in-cut only)
    in_cut_pos = np.where(cut_mask)[0]
    y_binary = np.array(
        [cut.relabel(int(y_4class[i])) for i in in_cut_pos], dtype=np.int64,
    )

    single_oof_f1: dict[str, float] = {}
    single_metrics_by_bb: dict[str, dict] = {}
    for bb, probs in probs_by_bb.items():
        p_in_cut = probs[in_cut_pos]
        pred = p_in_cut.argmax(axis=1)
        m = binary_metrics(y_binary, pred, p_in_cut)
        single_oof_f1[bb] = m["macro_f1"]
        single_metrics_by_bb[bb] = m

    best_single_bb = max(single_oof_f1, key=lambda k: single_oof_f1[k])
    best_single_metrics = single_metrics_by_bb[best_single_bb]

    # 3. Top-3 cross-family
    top3 = _pick_top3_cross_family(single_oof_f1)

    # 4. Voting OOF on in-cut samples
    voting_probs_in_cut = np.mean(
        np.stack([probs_by_bb[bb][in_cut_pos] for bb in top3]), axis=0,
    )
    voting_pred = voting_probs_in_cut.argmax(axis=1)
    voting_metrics = binary_metrics(y_binary, voting_pred, voting_probs_in_cut)

    # 5. Stacking OOF (unbiased CV) on in-cut samples
    X_stack_in_cut = np.concatenate(
        [probs_by_bb[bb][in_cut_pos] for bb in top3], axis=1,
    ).astype(np.float32)
    stack_pred, stack_probs, final_meta = _stacking_oof_cv(X_stack_in_cut, y_binary)
    stack_metrics = binary_metrics(y_binary, stack_pred, stack_probs)

    joblib.dump(final_meta, ens_dir / "meta_logreg.pkl")

    # 6. Decision (strict improvement over single)
    candidates = {
        "single":   {"backbone": best_single_bb, "members": [best_single_bb], **best_single_metrics},
        "voting":   {"backbone": None,           "members": top3, **voting_metrics},
        "stacking": {"backbone": None,           "members": top3, **stack_metrics},
    }
    best_ens = max(("voting", "stacking"), key=lambda k: candidates[k]["macro_f1"])
    chosen = best_ens if candidates[best_ens]["macro_f1"] > candidates["single"]["macro_f1"] else "single"

    # 7. P(class=1) per train_val sample under the chosen strategy (unbiased)
    n_all = len(all_train_val_idx)
    oof_p_class1 = np.zeros(n_all, dtype=np.float32)
    if chosen == "single":
        oof_p_class1 = probs_by_bb[best_single_bb][:, 1].astype(np.float32)
    elif chosen == "voting":
        avg = np.mean(np.stack([probs_by_bb[bb] for bb in top3]), axis=0)
        oof_p_class1 = avg[:, 1].astype(np.float32)
    elif chosen == "stacking":
        oof_p_class1[in_cut_pos] = stack_probs[:, 1]
        out_of_cut_pos = np.where(~cut_mask)[0]
        if len(out_of_cut_pos) > 0:
            X_oc = np.concatenate(
                [probs_by_bb[bb][out_of_cut_pos] for bb in top3], axis=1,
            ).astype(np.float32)
            oof_p_class1[out_of_cut_pos] = final_meta.predict_proba(X_oc)[:, 1].astype(np.float32)

    np.save(oof_dir / "oof_p_class1.npy", oof_p_class1)

    decision = EnsembleDecision(
        cut=cut.name,
        chosen=chosen,
        chosen_backbone=best_single_bb if chosen == "single" else None,
        chosen_members=[best_single_bb] if chosen == "single" else top3,
        oof_macro_f1=float(candidates[chosen]["macro_f1"]),
        oof_accuracy=float(candidates[chosen]["accuracy"]),
        oof_auc=float(candidates[chosen].get("auc", float("nan"))),
        artifacts_dir=str(ens_dir),
    )
    dump_json({
        "cut": cut.name,
        "selection_metric": "5-fold OOF macro-F1 (no test set used)",
        "n_train_val_total": int(n_all),
        "n_in_cut": int(cut_mask.sum()),
        "n_outside_cut": int((~cut_mask).sum()),
        "single_oof_f1_per_backbone": {bb: float(v) for bb, v in single_oof_f1.items()},
        "top3_cross_family": top3,
        "candidates": candidates,
        "decision": asdict(decision),
    }, ens_dir / "winner.json")

    if chosen == "single":
        chosen_pred = probs_by_bb[best_single_bb][in_cut_pos].argmax(axis=1)
    elif chosen == "voting":
        chosen_pred = voting_pred
    else:
        chosen_pred = stack_pred

    save_confusion_matrix(
        y_binary, chosen_pred, list(cut.class_names),
        ens_dir / "confusion_matrix_oof.png",
        title=f"{cut.name} | {chosen} | OOF",
    )
    save_classification_report(
        y_binary, chosen_pred, list(cut.class_names),
        ens_dir / "classification_report_oof.txt",
    )

    return decision
