"""Stage 28 — CLAHE + (optional) Canny preprocessing (professor's request).

Two preprocessed copies of a stage_* dataset, keeping folder names + filenames
so patient_groups.csv still matches:

  --clahe-out : CLAHE contrast-enhanced grayscale (clearer femoral-head contour),
                saved as 3-channel so it drops into the classifier unchanged.
  --canny-out : Canny edges computed ON the CLAHE image (the "outline" version),
                also 3-channel.

Usage:
    python scripts/28_preprocess_clahe_canny.py \
        --src stage_cls_merged_rebox \
        --clahe-out stage_cls_merged_rebox_clahe \
        --canny-out stage_cls_merged_rebox_canny
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser(description="CLAHE + Canny preprocessing.")
    ap.add_argument("--src", type=Path, required=True, help="dataset root with stage_* subfolders")
    ap.add_argument("--clahe-out", type=Path, default=None)
    ap.add_argument("--canny-out", type=Path, default=None)
    ap.add_argument("--clip", type=float, default=2.0, help="CLAHE clipLimit")
    ap.add_argument("--tile", type=int, default=8, help="CLAHE tileGridSize")
    ap.add_argument("--canny-low", type=int, default=50)
    ap.add_argument("--canny-high", type=int, default=150)
    ap.add_argument("--canny-blur", type=int, default=3, help="Gaussian blur ksize before Canny (odd)")
    args = ap.parse_args()

    clahe = cv2.createCLAHE(clipLimit=args.clip, tileGridSize=(args.tile, args.tile))
    stages = sorted(d for d in args.src.iterdir() if d.is_dir() and d.name.startswith("stage"))
    n_cl = n_ca = 0
    for sd in stages:
        for out in (args.clahe_out, args.canny_out):
            if out is not None:
                (out / sd.name).mkdir(parents=True, exist_ok=True)
        for img_path in sorted(sd.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            cl = clahe.apply(gray)
            if args.clahe_out is not None:
                cv2.imwrite(str(args.clahe_out / sd.name / img_path.name),
                            cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR))
                n_cl += 1
            if args.canny_out is not None:
                blur = cv2.GaussianBlur(cl, (args.canny_blur, args.canny_blur), 0) if args.canny_blur else cl
                edges = cv2.Canny(blur, args.canny_low, args.canny_high)
                cv2.imwrite(str(args.canny_out / sd.name / img_path.name),
                            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
                n_ca += 1
        print(f"  [{sd.name}] done", flush=True)
    print(f"[preproc] CLAHE {n_cl} imgs -> {args.clahe_out} | Canny {n_ca} imgs -> {args.canny_out}")


if __name__ == "__main__":
    main()
