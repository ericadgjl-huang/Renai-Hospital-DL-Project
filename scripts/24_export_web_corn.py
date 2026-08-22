"""Stage 24 — Export the CORN + stacking model for the web app.

Fits the OOF-selected stacking meta (OrdinalLogisticMeta C=3) on the ensemble's
OOF cumulative features and writes a small runtime the Flask app can load:

    web_app/corn_runtime/meta.joblib     — fitted stacking meta-learner
    web_app/corn_runtime/manifest.json   — backbone order + fold checkpoint paths
                                           + YOLO weights + Grad-CAM backbone

The backbone ORDER in the manifest is exactly the order the meta was fit on, so
the web builds its 9x3 feature vector identically (no train/serve skew).

Usage:
    python scripts/24_export_web_corn.py --out-root outputs_merged_patient_rebox \
        --data-root stage_cls_merged_rebox --yolo weights/yolo_best_rebox.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from torchvision.datasets import ImageFolder

from renai.data import get_4class_labels, make_outer_split, patient_groups
from renai.ordinal import OrdinalLogisticMeta
from renai.seed import SEED, set_seed


def discover(out_root: Path):
    """(sub, backbone, is_rin) in a fixed order; matches scripts/17."""
    out = []
    for d in sorted(out_root.glob("corn*")):
        if not d.is_dir():
            continue
        is_rin = "radimagenet" in d.name
        for bb in sorted(p.name for p in d.iterdir() if p.is_dir()):
            if list((d / bb).glob("fold_*.pth")):
                out.append((d.name, bb, is_rin))
    return out


def main():
    ap = argparse.ArgumentParser(description="Export CORN+stacking for the web app.")
    ap.add_argument("--out-root", type=Path, default=Path("outputs_merged_patient_rebox"))
    ap.add_argument("--data-root", type=Path, default=Path("stage_cls_merged_rebox"))
    ap.add_argument("--groups-csv", default="outputs_merged/patient_groups.csv")
    ap.add_argument("--yolo", type=Path, default=Path("weights/yolo_best_rebox.pt"))
    ap.add_argument("--gradcam-backbone", default="densenet121")
    ap.add_argument("--C", type=float, default=3.0)
    args = ap.parse_args()
    set_seed(SEED)

    groups = patient_groups(args.data_root, args.groups_csv) if args.groups_csv else None
    outer = make_outer_split(args.data_root, args.out_root / "splits" / "outer_split.json",
                             groups=groups)
    tv = sorted(int(i) for i in outer["train_val_idx"])
    y_oof = get_4class_labels(args.data_root).numpy()[tv] + 1

    backbones = discover(args.out_root)
    feats, manifest_bb = [], []
    for sub, bb, is_rin in backbones:
        bdir = args.out_root / sub / bb
        feats.append(np.load(bdir / "oof_cumulative.npy"))
        manifest_bb.append({
            "sub": sub, "backbone": bb, "is_radimagenet": is_rin,
            "ckpts": [os.path.relpath(p.resolve(), ROOT).replace("\\", "/")
                      for p in sorted(bdir.glob("fold_*.pth"))],
        })
    X_oof = np.nan_to_num(np.hstack(feats))
    print(f"[web-export] {len(backbones)} backbones, OOF features {X_oof.shape}")

    meta = OrdinalLogisticMeta(C=args.C, decode="expected").fit(X_oof, y_oof)

    out_dir = ROOT / "web_app" / "corn_runtime"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta, out_dir / "meta.joblib")
    manifest = {
        "model": "CORN + stacking (OrdinalLogisticMeta C=%.1f)" % args.C,
        "yolo_weights": str(args.yolo).replace("\\", "/"),
        "gradcam_backbone": args.gradcam_backbone,
        "img_size": 384,
        "backbones": manifest_bb,   # ORDER == meta feature order
        "n_features": int(X_oof.shape[1]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"[web-export] wrote {out_dir/'meta.joblib'} and manifest.json")
    print(f"[web-export] gradcam backbone = {args.gradcam_backbone}; "
          f"restart the Flask app to pick it up.")


if __name__ == "__main__":
    main()
