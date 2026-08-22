"""Stage 17 — Finalize the CORN ensemble: OOF-weighted averaging + decode
tuning + optional test-time augmentation (TTA). Squeezes extra accuracy out of
already-trained backbones without more training.

Levers (all selected/tuned on OOF; the held-out test is report-only):
  1. OOF-weighted ensemble — weight each backbone's cumulative probs by its own
     OOF QWK (a weak backbone contributes less) instead of a plain average.
  2. Decode tuning — pick rank-count vs expected-value by OOF QWK.
  3. TTA (optional, --tta) — average each fold model's cumulative probs over a
     few no-flip augmented views (small rotations + brightness). No horizontal
     flip: the pipeline already flips right hips to the left orientation.

Reads every backbone under <out-root>/corn_cw[_radimagenet]/<backbone>/ that has
fold_*.pth. Reports the held-out test with QWK/accuracy CIs and per-stage recall,
and writes <out-root>/FINAL_ENSEMBLE.json.

Usage:
    python scripts/17_finalize_ensemble.py --out-root outputs_merged_patient \
        --data-root stage_cls_merged --groups-csv outputs_merged/patient_groups.csv --device 0
    ... add --tta for test-time augmentation.
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
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from renai.corn import corn_cumulative_probs, create_corn_model
from renai.data import (
    IMG_SIZE, NORM_MEAN, NORM_STD, get_4class_labels, make_cv_folds,
    make_eval_transform, make_outer_split, patient_groups,
)
from renai.eval import save_classification_report, save_confusion_matrix
from renai.ordinal import (
    OrdinalLogisticMeta,
    bootstrap_ci, decode_expected, decode_rank_count, decode_rank_count_thresh,
    ordinal_metrics, tune_decode_thresholds, _qwk,
)
from renai.seed import SEED, set_seed

RADIMAGENET_DIR = ROOT / "weights" / "radimagenet"


def discover_backbones(out_root: Path):
    """(sub, backbone, is_radimagenet) for every trained backbone found. Matches
    any corn* subdir (corn_cw, corn_cw_radimagenet, corn_cw_s3w2.5, ...)."""
    found = []
    for d in sorted(out_root.glob("corn*")):
        if not d.is_dir():
            continue
        is_rin = "radimagenet" in d.name
        for bb in sorted(p.name for p in d.iterdir() if p.is_dir()):
            if list((d / bb).glob("fold_*.pth")):
                found.append((d.name, bb, is_rin))
    return found


@torch.no_grad()
def _cumprobs_views(model, x, device, tta):
    """Cumulative probs for a batch, averaged over TTA views (no h-flip)."""
    if not tta:
        return corn_cumulative_probs(model(x.to(device))).cpu().numpy()
    acc = None
    # deterministic no-flip views: identity + small rotations + brightness
    views = [lambda t: t,
             lambda t: TF.rotate(t, 7), lambda t: TF.rotate(t, -7),
             lambda t: TF.adjust_brightness(t, 0.9),
             lambda t: TF.adjust_brightness(t, 1.1)]
    for v in views:
        p = corn_cumulative_probs(model(v(x).to(device))).cpu().numpy()
        acc = p if acc is None else acc + p
    return acc / len(views)


@torch.no_grad()
def cumprobs_on(model, data_root, idx, device, tta, bs=16):
    ds = ImageFolder(data_root, transform=make_eval_transform(IMG_SIZE))
    loader = DataLoader(Subset(ds, list(idx)), batch_size=bs, shuffle=False)
    out = [_cumprobs_views(model, x, device, tta) for x, _ in loader]
    return np.concatenate(out) if out else np.zeros((0, 3))


def per_backbone_probs(sub, bb, is_rin, out_root, data_root, splits_dir, groups,
                       tv, test_idx, device, tta):
    """OOF + test cumulative probs for one backbone. Uses saved .npy when no TTA
    (fast); reloads fold checkpoints for TTA."""
    bdir = out_root / sub / bb
    if not tta and (bdir / "oof_cumulative.npy").exists():
        return np.load(bdir / "oof_cumulative.npy"), np.load(bdir / "test_cumulative.npy")

    pos = {g: i for i, g in enumerate(tv)}
    oof = np.full((len(tv), 3), np.nan)
    test_sum, n = None, 0
    folds = make_cv_folds(data_root, tv, groups=groups)
    rin_dir = str(RADIMAGENET_DIR) if is_rin else None
    for fi, (_tr, va) in enumerate(folds):
        ckpt = bdir / f"fold_{fi}.pth"
        if not ckpt.exists():
            continue
        m = create_corn_model(bb, radimagenet_dir=None).to(device)  # arch only
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        va = [int(g) for g in va]
        pva = cumprobs_on(m, data_root, va, device, tta)
        for j, g in enumerate(va):
            oof[pos[g]] = pva[j]
        pte = cumprobs_on(m, data_root, test_idx, device, tta)
        test_sum = pte if test_sum is None else test_sum + pte
        n += 1
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof, test_sum / max(n, 1)


STAGE_NAMES = ["stage_1", "stage_2", "stage_3", "stage_4"]


def save_eval_artifacts(out_dir: Path, y_test, y_pred, test_paths, title: str):
    """Confusion matrix + classification report + per-crop misclassification log
    (folders true{T}_pred{P}/ + csv). Reused for both voting and stacking."""
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(y_test - 1, y_pred - 1, STAGE_NAMES,
                          out_dir / "confusion_matrix_test.png", title=title)
    save_classification_report(y_test - 1, y_pred - 1, STAGE_NAMES,
                               out_dir / "classification_report_test.txt")
    mis_root = out_dir / "misclassified"
    if mis_root.exists():
        shutil.rmtree(mis_root)
    rows = []
    for i in range(len(y_test)):
        t, p = int(y_test[i]), int(y_pred[i])
        if t == p:
            continue
        folder = mis_root / f"true{t}_pred{p}"
        folder.mkdir(parents=True, exist_ok=True)
        src = Path(test_paths[i])
        shutil.copy2(src, folder / f"true{t}_as_pred{p}__{src.name}")
        rows.append({"crop_filename": src.name, "true_stage": t, "pred_stage": p,
                     "note": f"stage {t} -> misclassified as stage {p}"})
    pd.DataFrame(rows).to_csv(out_dir / "misclassified.csv", index=False, encoding="utf-8-sig")
    return len(rows)


def candidate_metas():
    """Low-capacity meta-learners for STACKING over the backbones' cumulative
    probs. Deliberately low-capacity: a high-capacity meta (RandomForest) was
    shown to overfit this small dataset, so we prefer linear / ordinal models."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    metas = {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0),
        "svm_linear": SVC(kernel="linear", class_weight="balanced"),
    }
    for C in (0.3, 1.0, 3.0):
        metas[f"ordinal_logit(C={C})"] = OrdinalLogisticMeta(C=C, decode="expected")
    return metas


def run_stacking(entries, y_oof, y_test):
    """STACKING: concat the 9 backbones' cumulative probs (9x3=27 features) and
    let a meta-learner predict the 4-class stage. Selection by OOF QWK
    (cross_val_predict); the test set is report-only. Returns
    (oof_pred, test_pred, best_name, rows)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X_oof = np.nan_to_num(np.hstack([e["oof"] for e in entries]))     # (n_tv, 27)
    X_test = np.nan_to_num(np.hstack([e["test"] for e in entries]))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, best = [], None
    for name, clf in candidate_metas().items():
        oof_pred = cross_val_predict(clf, X_oof, y_oof, cv=cv)
        q = _qwk(y_oof, oof_pred)
        rows.append({"meta": name, "oof_qwk": round(float(q), 4)})
        print(f"  [stacking] {name:20s} OOF-QWK={q:.4f}", flush=True)
        if best is None or q > best[0]:
            best = (q, name, clf)
    _, best_name, best_clf = best
    best_clf.fit(X_oof, y_oof)
    return (cross_val_predict(best_clf, X_oof, y_oof, cv=cv),
            best_clf.predict(X_test), best_name, rows)


def main():
    ap = argparse.ArgumentParser(description="Finalize CORN ensemble (OOF-weighted + decode + TTA).")
    ap.add_argument("--out-root", type=Path, default=Path("outputs_merged_patient"))
    ap.add_argument("--data-root", type=Path, default=Path("stage_cls_merged"))
    ap.add_argument("--splits-dir", type=Path, default=None)
    ap.add_argument("--groups-csv", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (reloads checkpoints)")
    ap.add_argument("--tune-thresholds", action="store_true",
                    help="tune per-threshold decode cutoffs on OOF (macro-F1) to lift stage-3 recall")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    if args.splits_dir is None:
        args.splits_dir = args.out_root / "splits"
    set_seed(SEED)

    groups = patient_groups(args.data_root, args.groups_csv) if args.groups_csv else None
    outer = make_outer_split(args.data_root, args.splits_dir / "outer_split.json", groups=groups)
    tv = sorted(int(i) for i in outer["train_val_idx"])
    test_idx = sorted(int(i) for i in outer["test_idx"])
    labels = get_4class_labels(args.data_root).numpy()
    y_oof, y_test = labels[tv] + 1, labels[test_idx] + 1

    backbones = discover_backbones(args.out_root)
    print(f"[final] {len(backbones)} backbones: {[b for _,b,_ in backbones]}  tta={args.tta}", flush=True)

    entries = []
    for sub, bb, is_rin in backbones:
        oof, test = per_backbone_probs(sub, bb, is_rin, args.out_root, args.data_root,
                                       args.splits_dir, groups, tv, test_idx, args.device, args.tta)
        w = max(0.0, _qwk(y_oof, decode_rank_count(oof)))   # OOF-QWK weight
        entries.append({"name": f"{bb}{'(RIN)' if is_rin else ''}", "oof": oof, "test": test, "w": w})
        print(f"  [final] {entries[-1]['name']:22s} OOF-QWK weight={w:.3f}", flush=True)

    W = np.array([e["w"] for e in entries])
    W = W / W.sum() if W.sum() > 0 else np.ones(len(entries)) / len(entries)

    def blend(key, weighted):
        stack = np.stack([e[key] for e in entries])          # (B, N, 3)
        ww = W if weighted else np.ones(len(entries)) / len(entries)
        return np.tensordot(ww, stack, axes=(0, 0))

    # pick weighting + decode on OOF
    best = None
    for weighted in (False, True):
        oof_blend = blend("oof", weighted)
        for dec_name, dec in (("rank_count", decode_rank_count), ("expected", decode_expected)):
            q = _qwk(y_oof, dec(oof_blend))
            if best is None or q > best[0]:
                best = (q, weighted, dec_name, dec)
    _, weighted, dec_name, dec = best

    oof_blend = blend("oof", weighted)
    test_blend = blend("test", weighted)
    thr = None
    if args.tune_thresholds:
        # tune per-threshold decode cutoffs on OOF to lift the weak stage-3 class
        thr, s = tune_decode_thresholds(oof_blend, y_oof, objective="macro_f1")
        dec_name = f"rank_count_thresh{thr}"
        y_pred = decode_rank_count_thresh(test_blend, thr)
        print(f"[final] tuned decode thresholds on OOF (macro-F1={s:.3f}): "
              f"[>=2,>=3,>=4]={thr}", flush=True)
    else:
        y_pred = dec(test_blend)
    print(f"[final] selected on OOF: weighting={'OOF-QWK' if weighted else 'equal'}, "
          f"decode={dec_name}", flush=True)
    vote_oof_pred = (decode_rank_count_thresh(oof_blend, thr) if args.tune_thresholds
                     else dec(oof_blend))
    split = "patient-level" if groups else "image-level"
    samples = ImageFolder(args.data_root).samples
    test_paths = [samples[i][0] for i in test_idx]

    def build_result(name, oof_pred, test_pred, extra=None):
        mo, mt = ordinal_metrics(y_oof, oof_pred), ordinal_metrics(y_test, test_pred)
        _, qlo, qhi = bootstrap_ci(y_test, test_pred, "qwk", args.n_boot)
        _, alo, ahi = bootstrap_ci(y_test, test_pred, "accuracy", args.n_boot)
        r = recall_score(y_test, test_pred, labels=[1, 2, 3, 4], average=None, zero_division=0)
        res = {"method": name, "split": split, "tta": args.tta, "n_test": int(mt["n"]),
               "oof_qwk": round(mo["qwk"], 3), "oof_accuracy": round(mo["accuracy"], 3),
               "qwk": round(mt["qwk"], 3), "qwk_ci": [round(qlo, 3), round(qhi, 3)],
               "accuracy": round(mt["accuracy"], 3), "accuracy_ci": [round(alo, 3), round(ahi, 3)],
               "macro_f1": round(mt["macro_f1"], 3), "off_by_one": round(mt["off_by_one"], 3),
               "per_stage_recall": {f"stage_{k+1}": round(float(r[k]), 3) for k in range(4)}}
        if extra:
            res.update(extra)
        return res

    # === VOTING: OOF-weighted soft-vote of the backbones' cumulative probs ===
    vote_res = build_result(
        f"voting({'OOF-QWK' if weighted else 'equal'},{dec_name})", vote_oof_pred, y_pred,
        {"backbones": [e["name"] for e in entries],
         "weights": {e["name"]: round(float(w), 3) for e, w in zip(entries, W)}})
    n_mis_v = save_eval_artifacts(args.out_root, y_test, y_pred, test_paths,
                                  f"voting (test n={len(y_test)}, {split})")
    (args.out_root / "FINAL_ENSEMBLE.json").write_text(
        json.dumps(vote_res, indent=2, ensure_ascii=False), encoding="utf-8")

    # === STACKING: meta-learner over the 9x3 cumulative-prob features ===
    print("\n[stacking] training meta-learners over backbone probs (OOF-selected) ...", flush=True)
    stack_oof, stack_test, best_meta, meta_rows = run_stacking(entries, y_oof, y_test)
    stack_res = build_result(f"stacking({best_meta})", stack_oof, stack_test,
                             {"meta_selection": meta_rows})
    stack_dir = args.out_root / "stacking"
    n_mis_s = save_eval_artifacts(stack_dir, y_test, stack_test, test_paths,
                                  f"stacking[{best_meta}] (test n={len(y_test)}, {split})")
    (stack_dir / "best.json").write_text(
        json.dumps(stack_res, indent=2, ensure_ascii=False), encoding="utf-8")

    # === COMPARISON (winner chosen by OOF QWK; test is report-only) ===
    cols = ["method", "oof_qwk", "qwk", "qwk_ci", "accuracy", "macro_f1", "off_by_one"]
    cmp = pd.DataFrame([{k: r[k] for k in cols} for r in (vote_res, stack_res)])
    cmp.to_csv(args.out_root / "voting_vs_stacking.csv", index=False, encoding="utf-8-sig")
    win_is_stack = stack_res["oof_qwk"] > vote_res["oof_qwk"]
    winner_res = stack_res if win_is_stack else vote_res
    winner = "stacking" if win_is_stack else "voting"

    # Top-level headline = the OOF-selected winner (unambiguous best model).
    (args.out_root / "BEST_MODEL.json").write_text(
        json.dumps({"selected_by": "OOF QWK", "winner": winner, **winner_res},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ VOTING vs STACKING (selection by OOF QWK) ================")
    print(cmp.to_string(index=False))
    print(f"\n[final] voting misclassified {n_mis_v}/{len(y_test)} -> {args.out_root/'misclassified'}")
    print(f"[final] stacking misclassified {n_mis_s}/{len(y_test)} -> {stack_dir/'misclassified'}")
    print(f"[final] *** OOF-selected BEST = {winner.upper()} *** -> BEST_MODEL.json")
    print(f"        (voting OOF-QWK={vote_res['oof_qwk']}, stacking OOF-QWK={stack_res['oof_qwk']})")
    print(f"[final] files: BEST_MODEL.json (winner), voting=FINAL_ENSEMBLE.json, "
          f"stacking/best.json, voting_vs_stacking.csv", flush=True)


if __name__ == "__main__":
    main()
