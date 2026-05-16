"""Stage 00 — Build stage_cls_dataset/ from YOLO ROIs (mirrors STEP 0 of the
old 03 notebooks, but as a pure script).

Reads roi_all.csv at the project root, crops each ROI from the YOLO image
folder, flips R-side knees so every saved crop looks like a left knee, and
writes them to stage_cls_dataset/stage_<n>/<filename>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proj-root", type=Path, default=Path("."))
    p.add_argument("--unify-side", choices=("L", "R"), default="L")
    p.add_argument("--roi-csv", type=Path, default=None)
    p.add_argument("--yolo-images", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("stage_cls_dataset"))
    args = p.parse_args()

    roi_csv = args.roi_csv if args.roi_csv else (args.proj_root / "roi_all.csv")
    img_root = (
        args.yolo_images
        if args.yolo_images
        else (args.proj_root / "yolo_dataset_process" / "yolo_dataset" / "images")
    )

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

    print(f"✅ wrote {n_written} crops to {args.out}")


if __name__ == "__main__":
    main()
