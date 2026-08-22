"""Stage 22 — Re-crop the classification dataset DIRECTLY from labelme boxes.

After re-drawing tighter boxes in labelme (professor's request: focus on the
femoral head, drop background noise), this crops each X-ray using the EXACT
boxes you drew — no YOLO, so the crop is exactly as tight as your box.

Box-selection + flip follow scripts/01_prepare_stage_dataset.py so the outputs
line up with the rest of the pipeline:
  * each X-ray has both hips boxed; the target hip is chosen by the L/R in the
    filename — L-file uses the RIGHTMOST box, R-file uses the LEFTMOST box;
  * right-side crops are flipped so every classifier input faces the same way;
  * output name is S{stage}_{prefix}{original}, identical to script 01, so the
    existing patient_groups.csv / splits still match by filename.

Usage (conda env unet_labeling):
    python scripts/22_recrop_from_labelme.py \
        --raw-root drive-download-20251023T113302Z-1-001 \
        --out stage_cls_dataset_rebox --roi-csv roi_all_rebox.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"]


def _side(path: Path) -> str:
    f = path.stem[:1].upper()
    return f if f in {"L", "R"} else "U"


def _bbox(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _boxes_from_json(js: Path):
    data = json.loads(js.read_text(encoding="utf-8"))
    boxes = []
    for sh in data.get("shapes", []):
        if sh.get("shape_type") in ("rectangle", "polygon"):
            boxes.append(_bbox(sh["points"]))
    return boxes


def _find_image(js: Path) -> Path | None:
    for ext in IMG_EXTS:
        p = js.with_suffix(ext)
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Crop dataset directly from labelme boxes.")
    ap.add_argument("--raw-root", type=Path,
                    default=Path("drive-download-20251023T113302Z-1-001"),
                    help="Folder with 'Ficat stage 1..4' subfolders (image + .json).")
    ap.add_argument("--out", type=Path, default=Path("stage_cls_dataset_rebox"))
    ap.add_argument("--roi-csv", type=Path, default=Path("roi_all_rebox.csv"))
    ap.add_argument("--source-prefix", default="",
                    help="e.g. A_ for the AVNFH cohort -> S1_A_L1.jpg")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.rglob("*")) and not args.overwrite:
        raise SystemExit(f"{args.out} not empty; add --overwrite to regenerate.")

    rows, failures, n = [], [], 0
    for stage in range(1, 5):
        sd = args.raw_root / f"Ficat stage {stage}"
        if not sd.exists():
            raise SystemExit(f"missing stage folder: {sd}")
        (args.out / f"stage_{stage}").mkdir(parents=True, exist_ok=True)
        jsons = sorted(sd.glob("*.json"))
        print(f"[stage {stage}] {len(jsons)} labelme json files")
        for js in jsons:
            img_path = _find_image(js)
            if img_path is None:
                failures.append(f"{js}: no image"); continue
            boxes = _boxes_from_json(js)
            if not boxes:
                failures.append(f"{js}: no boxes"); continue
            side = _side(img_path)
            # same selection rule as script 01
            if side == "L":
                k = max(range(len(boxes)), key=lambda i: boxes[i][2])   # rightmost
            elif side == "R":
                k = min(range(len(boxes)), key=lambda i: boxes[i][0])   # leftmost
            else:
                k = 0
            x1, y1, x2, y2 = boxes[k]
            out_name = f"S{stage}_{args.source_prefix}{img_path.name}"
            with Image.open(img_path).convert("RGB") as im:
                crop = im.crop((int(x1), int(y1), int(x2), int(y2)))
                if side == "R":
                    crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                crop.save(args.out / f"stage_{stage}" / out_name)
            rows.append({"filename": out_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "side": side, "stage": stage, "source": str(img_path)})
            n += 1

    pd.DataFrame(rows).to_csv(args.roi_csv, index=False, encoding="utf-8-sig")
    print(f"\nwrote {n} crops -> {args.out}\nwrote ROI table -> {args.roi_csv}")
    if failures:
        fp = args.roi_csv.with_name(args.roi_csv.stem + "_failures.txt")
        fp.write_text("\n".join(failures), encoding="utf-8")
        print(f"warning: {len(failures)} images had no image/box; see {fp}", file=sys.stderr)


if __name__ == "__main__":
    main()
