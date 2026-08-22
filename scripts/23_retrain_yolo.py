"""Stage 23 — Retrain the YOLO ROI detector on the NEW (tighter) labelme boxes.

After re-drawing tighter femoral-head boxes in labelme, this trains a fresh YOLO
detector on them, so the AVNFH cohort (which has no manual boxes) can be
auto-cropped just as tightly. The old cohort keeps its exact hand-drawn crops
(scripts/22); this detector is used for AVNFH (scripts/01 --weights the new pt).

Pipeline (self-contained, mirrors notebooks/01_yolo_train):
  labelme *.json  ->  YOLO images+labels  ->  80/20 split  ->  train YOLOv8
  ->  copy best.pt to weights/yolo_best_rebox.pt

Usage (conda env unet_labeling):
    python scripts/23_retrain_yolo.py --device 0 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bbox(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def to_yolo(x1, y1, x2, y2, w, h):
    return (f"0 {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
            f"{abs(x2-x1)/w:.6f} {abs(y2-y1)/h:.6f}")


def build_yolo_dataset(raw_root: Path, work: Path):
    """labelme json -> images/ + labels/ (YOLO txt)."""
    from PIL import Image
    img_dir, lbl_dir = work / "images", work / "labels"
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for stage in range(1, 5):
        sd = raw_root / f"Ficat stage {stage}"
        for js in sorted(sd.glob("*.json")):
            data = json.loads(js.read_text(encoding="utf-8"))
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                p = js.with_suffix(ext)
                if p.exists():
                    img = p; break
            if img is None:
                continue
            w, h = data.get("imageWidth"), data.get("imageHeight")
            if not (isinstance(w, (int, float)) and isinstance(h, (int, float))):
                with Image.open(img) as im:
                    w, h = im.size
            lines = [to_yolo(*bbox(sh["points"]), w, h) for sh in data.get("shapes", [])
                     if sh.get("shape_type") in ("rectangle", "polygon")]
            if not lines:
                continue
            name = f"S{stage}_{img.name}"
            dst = img_dir / name
            if not dst.exists():
                try:
                    os.link(img, dst)
                except Exception:
                    shutil.copy2(img, dst)
            (lbl_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            n_img += 1; n_box += len(lines)
    print(f"[yolo] built {n_img} images / {n_box} boxes -> {work}")
    return img_dir, lbl_dir


def split_and_yaml(img_dir: Path, lbl_dir: Path, work: Path, seed=42):
    random.seed(seed)
    imgs = sorted(p for p in img_dir.glob("*.*") if p.suffix.lower() in
                  {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    random.shuffle(imgs)
    cut = int(0.8 * len(imgs))
    split = {"train": imgs[:cut], "val": imgs[cut:]}
    for name, files in split.items():
        di, dl = work / "split/images" / name, work / "split/labels" / name
        di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
        for im in files:
            (di / im.name).write_bytes(im.read_bytes())
            lt = lbl_dir / (im.stem + ".txt")
            (dl / (im.stem + ".txt")).write_text(lt.read_text(encoding="utf-8") if lt.exists() else "",
                                                 encoding="utf-8")
    yaml = work / "data.yaml"
    yaml.write_text(
        f"train: {(work/'split/images/train').as_posix()}\n"
        f"val:   {(work/'split/images/val').as_posix()}\n"
        f"nc: 1\nnames: [hip_target]\n", encoding="utf-8")
    print(f"[yolo] split train={len(split['train'])} val={len(split['val'])}; wrote {yaml}")
    return yaml


def main():
    ap = argparse.ArgumentParser(description="Retrain YOLO ROI detector on new labelme boxes.")
    ap.add_argument("--raw-root", type=Path, default=ROOT / "drive-download-20251023T113302Z-1-001")
    ap.add_argument("--work", type=Path, default=ROOT / "yolo_rebox")
    ap.add_argument("--base", type=Path, default=ROOT / "weights" / "yolov8n.pt")
    ap.add_argument("--out-weight", type=Path, default=ROOT / "weights" / "yolo_best_rebox.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from ultralytics import YOLO
    img_dir, lbl_dir = build_yolo_dataset(args.raw_root, args.work)
    yaml = split_and_yaml(img_dir, lbl_dir, args.work)

    model = YOLO(str(args.base))
    res = model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz,
                      device=args.device, name="rebox", project=str(args.work / "runs"),
                      exist_ok=True, batch=args.batch, workers=0, cache=False, amp=False)
    best = Path(res.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training finished but {best} missing")
    shutil.copy2(best, args.out_weight)
    print(f"\n[yolo] DONE. best weight -> {args.out_weight}")
    print(f"[yolo] next: crop AVNFH with it:\n"
          f"  python scripts/01_prepare_stage_dataset.py --raw-root AVNFH_staged "
          f"--weights weights/yolo_best_rebox.pt --out stage_cls_avnfh_rebox "
          f"--source-prefix A_ --roi-csv roi_avnfh_rebox.csv --device 0 --overwrite")


if __name__ == "__main__":
    main()
