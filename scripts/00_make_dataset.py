"""Stage 00 — Build stage_cls_dataset/ from YOLO ROIs (mirrors STEP 0 of the
old 03 notebooks, but as a pure script).

Historical flow that produced the artefacts currently in this repo:

    drive-download-<...>/Ficat stage {1..4}/<L|R><n>.jpg   (raw X-rays)
              │
              ▼ (YOLOv8 detection — separate step, not in this repo)
    yolo_dataset_process/yolo_dataset/images/<filename>    (per-knee crops)
    roi_all.csv                                            (per-knee ROI rows)
              │
              ▼ (this script)
    stage_cls_dataset/stage_{1..4}/<filename>              (side-unified crops)

`stage_cls_dataset/` is the only directory the new v3 pipeline reads from.
If you already have it populated (which is the case in this checkout), this
script is purely archival — you do not need to rerun it.

Reads `roi_all.csv` at the project root, crops each ROI from the YOLO image
folder, flips R-side knees so every saved crop looks like a left knee, and
writes them to `stage_cls_dataset/stage_<n>/<filename>`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proj-root", type=Path, default=Path("."))
    p.add_argument("--unify-side", choices=("L", "R"), default="L")
    p.add_argument("--roi-csv", type=Path, default=None)
    p.add_argument("--yolo-images", type=Path, default=None,
                   help="Folder of per-knee YOLO crops. Default: "
                        "<proj-root>/yolo_dataset_process/yolo_dataset/images")
    p.add_argument("--out", type=Path, default=Path("stage_cls_dataset"))
    args = p.parse_args()

    roi_csv = args.roi_csv if args.roi_csv else (args.proj_root / "roi_all.csv")
    img_root = (
        args.yolo_images
        if args.yolo_images
        else (args.proj_root / "yolo_dataset_process" / "yolo_dataset" / "images")
    )

    # Friendly skip when the YOLO intermediate folder is absent — that folder
    # is git-ignored and is not part of this checkout.  If stage_cls_dataset/
    # already has crops, the new pipeline can run without rerunning Step 0.
    if not img_root.exists():
        msg = (
            f"[skip] YOLO intermediate folder missing: {img_root}\n"
            f"       This is the per-knee crop folder produced by the historical\n"
            f"       YOLOv8 step (drive-download-* -> YOLO -> {img_root}).\n"
            f"       It is git-ignored on purpose.\n"
        )
        if args.out.exists() and any(args.out.iterdir()):
            msg += (
                f"       {args.out}/ already exists and is non-empty -- the new\n"
                f"       v3 pipeline reads from there, so you can safely skip Step 0.\n"
            )
        else:
            msg += (
                f"       Re-run the YOLOv8 detection step on\n"
                f"       drive-download-<...>/Ficat stage {{1..4}}/ to recreate\n"
                f"       this folder, or pass --yolo-images <path> manually.\n"
            )
        print(msg)
        sys.exit(0)

    if not roi_csv.exists():
        print(f"[error] roi_all.csv missing at {roi_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(roi_csv)
    n_written = 0
    for _, r in df.iterrows():
        if pd.isna(r["x1"]):
            continue
        stage = int(r["stage"])
        side = str(r["side"]).upper()
        out_dir = args.out / f"stage_{stage}"
        out_dir.mkdir(parents=True, exist_ok=True)

        img_path = img_root / r["filename"]
        x1, y1, x2, y2 = map(int, [r.x1, r.y1, r.x2, r.y2])
        with Image.open(img_path) as im:
            crop = im.crop((x1, y1, x2, y2))
            if args.unify_side == "L" and side == "R":
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            elif args.unify_side == "R" and side == "L":
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            crop.save(out_dir / r["filename"])
            n_written += 1

    print(f"wrote {n_written} crops to {args.out}")


if __name__ == "__main__":
    main()
