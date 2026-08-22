"""Stage 15 — Train a rank-consistent ordinal CORN model (Tier-2 lever).

One shared backbone, a 3-logit CORN head, trained end-to-end with the CORN
conditional loss on all 4 stages at once — the principled ordinal counterpart of
the 10 independent binary cuts. Uses the SAME frozen outer 80/20 split and the
same 5-fold CV, so its OOF / internal-test / external numbers are directly
comparable to scripts 06/09/14.

Honesty: OOF cumulative probs are leak-free (each val sample scored by the fold
model that didn't train on it). Model/epoch selection uses val QWK. The internal
test and external AVNFH cohort are report-only.

Usage (repo root, conda env unet_labeling):
    python scripts/15_train_corn.py --backbone efficientnet_b0 --device 0
    python scripts/15_train_corn.py --backbone efficientnet_b0 --device 0 --smoke
    python scripts/15_train_corn.py --backbones efficientnet_b0,convnext_tiny,densenet121 --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from renai.corn import (
    corn_cumulative_probs, corn_loss, corn_pos_weights, corn_predict, create_corn_model,
)
from renai.data import (
    get_4class_labels,
    make_4class_eval_loader,
    make_cv_folds,
    make_eval_transform,
    make_outer_split,
    make_train_transform,
    patient_groups,
)
from renai.ordinal import decode_rank_count, ordinal_metrics, bootstrap_ci
from renai.seed import SEED, set_seed, torch_generator

STAGE_NAMES = ["stage_1", "stage_2", "stage_3", "stage_4"]


def make_4class_train_loader(data_root, idx, batch_size, img_size=384):
    base = ImageFolder(data_root, transform=make_train_transform(img_size))
    return DataLoader(Subset(base, list(idx)), batch_size=batch_size, shuffle=True,
                      num_workers=0, generator=torch_generator())


@torch.no_grad()
def cumulative_on_loader(model, loader, device):
    model.eval()
    probs, ys = [], []
    for imgs, lbls in loader:
        logits = model(imgs.to(device))
        probs.append(corn_cumulative_probs(logits).cpu().numpy())
        ys.extend(int(l) + 1 for l in lbls.numpy().tolist())
    return (np.concatenate(probs) if probs else np.zeros((0, 3)),
            np.asarray(ys, dtype=np.int64))


def train_corn_fold(backbone, tr_idx, va_idx, data_root, device, epochs, lr,
                    weight_decay, patience, batch_size, ckpt_path, class_weight=False,
                    radimagenet_dir=None, stage3_weight=1.0):
    """Train one fold; keep best-by-val-QWK weights. Returns best val QWK."""
    model = create_corn_model(backbone, radimagenet_dir=radimagenet_dir).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tr_loader = make_4class_train_loader(data_root, tr_idx, batch_size)
    va_loader = make_4class_eval_loader(data_root, va_idx, batch_size=batch_size)

    pos_w = None
    if class_weight:
        tr_stages = get_4class_labels(data_root).numpy()[list(tr_idx)] + 1
        pos_w = corn_pos_weights(tr_stages, device)
        print(f"    [{backbone}] class-weighted CORN pos_weights={pos_w.tolist()}", flush=True)
    stage_w = None
    if stage3_weight and stage3_weight != 1.0:
        stage_w = torch.tensor([1.0, 1.0, float(stage3_weight), 1.0], device=device)
        print(f"    [{backbone}] stage-3 weighting: stage_weights={stage_w.tolist()}", flush=True)

    best_qwk, best_state, since = -2.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for imgs, lbls in tr_loader:
            imgs = imgs.to(device)
            stages = (lbls + 1).to(device)          # 1..4
            opt.zero_grad()
            loss = corn_loss(model(imgs), stages, pos_weights=pos_w, stage_weights=stage_w)
            loss.backward()
            opt.step()
            tot += loss.item() * imgs.size(0)
        P, yv = cumulative_on_loader(model, va_loader, device)
        pred = decode_rank_count(P)
        qwk = ordinal_metrics(yv, pred)["qwk"]
        improved = qwk > best_qwk
        print(f"    [{backbone}] ep{ep:02d}/{epochs} loss={tot/max(len(tr_idx),1):.4f} "
              f"val_qwk={qwk:.4f}{'  <- best' if improved else ''}", flush=True)
        if improved:
            best_qwk, since = qwk, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if patience and since >= patience:
                print(f"    [{backbone}] early stop @ep{ep} (best val_qwk={best_qwk:.4f})", flush=True)
                break
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, ckpt_path)
    return best_qwk


def run_backbone(backbone, args):
    data_root, out_root, device = args.data_root, args.out_root, args.device
    set_seed(SEED)
    groups = patient_groups(data_root, args.groups_csv) if args.groups_csv else None
    outer = make_outer_split(data_root, args.splits_dir / "outer_split.json", groups=groups)
    tv = sorted(int(i) for i in outer["train_val_idx"])
    test_idx = sorted(int(i) for i in outer["test_idx"])
    labels = get_4class_labels(data_root).numpy()
    y_tv = labels[tv] + 1
    pos = {g: i for i, g in enumerate(tv)}

    folds = make_cv_folds(data_root, tv, groups=groups)
    if args.smoke:
        folds = folds[:1]
    sub = "corn_cw" if args.class_weight else "corn"
    if args.radimagenet:
        sub += "_radimagenet"
    if args.stage3_weight and args.stage3_weight != 1.0:
        sub += f"_s3w{args.stage3_weight:g}"
    bdir = out_root / sub / backbone
    bdir.mkdir(parents=True, exist_ok=True)

    epochs = 2 if args.smoke else args.epochs
    oof_P = np.full((len(tv), 3), np.nan)
    test_P_sum, ext_P_sum, n_models = None, None, 0
    y_test = y_ext = None

    test_loader = make_4class_eval_loader(data_root, test_idx, batch_size=args.batch_size)
    ext_loader = (make_4class_eval_loader(args.ext_root, list(range(len(ImageFolder(args.ext_root).samples))),
                                          batch_size=args.batch_size)
                  if args.ext_root and Path(args.ext_root).exists() else None)

    for fi, (tr_i, va_i) in enumerate(folds):
        t0 = time.time()
        ckpt = bdir / f"fold_{fi}.pth"
        print(f"\n[corn] {backbone} fold {fi} - train={len(tr_i)} val={len(va_i)}", flush=True)
        train_corn_fold(backbone, tr_i, va_i, data_root, device, epochs, args.lr,
                        args.weight_decay, args.patience, args.batch_size, ckpt,
                        class_weight=args.class_weight, radimagenet_dir=args.radimagenet,
                        stage3_weight=args.stage3_weight)
        model = create_corn_model(backbone).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        # OOF cumulative probs for this fold's val samples (leak-free)
        va_loader = make_4class_eval_loader(data_root, [int(g) for g in va_i], batch_size=args.batch_size)
        Pva, _ = cumulative_on_loader(model, va_loader, device)
        for j, g in enumerate(va_i):
            oof_P[pos[int(g)]] = Pva[j]
        # accumulate test / external cumulative probs (fold-averaged)
        Pte, y_test = cumulative_on_loader(model, test_loader, device)
        test_P_sum = Pte if test_P_sum is None else test_P_sum + Pte
        if ext_loader is not None:
            Pex, y_ext = cumulative_on_loader(model, ext_loader, device)
            ext_P_sum = Pex if ext_P_sum is None else ext_P_sum + Pex
        n_models += 1
        print(f"[corn] {backbone} fold {fi} done in {time.time()-t0:.0f}s", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {"backbone": backbone, "n_folds": n_models, "smoke": args.smoke}
    if not args.smoke:
        oof_pred = decode_rank_count(oof_P)
        result["oof"] = ordinal_metrics(y_tv, oof_pred)
        test_pred = decode_rank_count(test_P_sum / n_models)
        q, lo, hi = bootstrap_ci(y_test, test_pred, "qwk", args.n_boot)
        result["test"] = {**ordinal_metrics(y_test, test_pred),
                          "qwk_ci": [round(lo, 3), round(hi, 3)]}
        np.save(bdir / "oof_cumulative.npy", oof_P)
        np.save(bdir / "test_cumulative.npy", test_P_sum / n_models)
        if ext_P_sum is not None:
            ext_pred = decode_rank_count(ext_P_sum / n_models)
            qe, loe, hie = bootstrap_ci(y_ext, ext_pred, "qwk", args.n_boot)
            result["external"] = {**ordinal_metrics(y_ext, ext_pred),
                                 "qwk_ci": [round(loe, 3), round(hie, 3)]}
            np.save(bdir / "ext_cumulative.npy", ext_P_sum / n_models)
        (bdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[corn] {backbone} RESULT:\n{json.dumps(result, indent=2)}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="Train rank-consistent CORN ordinal model.")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--backbones", default=None, help="comma list; overrides --backbone")
    ap.add_argument("--data-root", type=Path, default=Path("stage_cls_dataset"))
    ap.add_argument("--ext-root", type=Path, default=Path("stage_cls_avnfh"))
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--splits-dir", type=Path, default=Path("outputs/splits"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--class-weight", action="store_true",
                    help="class-weighted CORN loss (up-weight rare high stages)")
    ap.add_argument("--radimagenet", default=None,
                    help="dir with RadImageNet weights (ResNet50.pt/DenseNet121.pt); "
                         "medical-pretrained backbone instead of ImageNet")
    ap.add_argument("--groups-csv", default=None,
                    help="CSV with filename,patient columns -> patient-level "
                         "(leak-free) outer split + CV instead of image-level")
    ap.add_argument("--stage3-weight", type=float, default=1.0,
                    help="per-sample loss weight for TRUE stage-3 images "
                         "(>1 emphasises the hardest, weakest-recall class)")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs — wiring check")
    args = ap.parse_args()
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    set_seed(SEED)

    backbones = ([b.strip() for b in args.backbones.split(",")] if args.backbones
                 else [args.backbone])
    summary = [run_backbone(b, args) for b in backbones]

    if not args.smoke:
        sub = "corn_cw" if args.class_weight else "corn"
        if args.radimagenet:
            sub += "_radimagenet"
        (args.out_root / sub).mkdir(parents=True, exist_ok=True)
        rows = []
        for r in summary:
            rows.append({
                "backbone": r["backbone"],
                "oof_qwk": round(r["oof"]["qwk"], 3),
                "test_qwk": round(r["test"]["qwk"], 3),
                "test_qwk_ci": "-".join(str(x) for x in r["test"]["qwk_ci"]),
                "test_acc": round(r["test"]["accuracy"], 3),
                "test_off1": round(r["test"]["off_by_one"], 3),
                "ext_qwk": round(r.get("external", {}).get("qwk", float("nan")), 3),
                "ext_acc": round(r.get("external", {}).get("accuracy", float("nan")), 3),
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(args.out_root / sub / "summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n================ CORN SUMMARY ({sub}) ================")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
