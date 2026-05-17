"""Build stage_cls_dataset/ directly from the raw Drive download folder.

Input layout:
    drive-download-20251023T113302Z-1-001/
      Ficat stage 1/L1.jpg, R1.jpg, ...
      Ficat stage 2/...

For each raw X-ray, YOLO detects candidate hip/knee ROI boxes. The selected
box follows the same rule as the web app: left-side images use the rightmost
box, right-side images use the leftmost box. Right-side crops are flipped so
all classifier inputs face the same direction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _stage_dir(raw_root: Path, stage: int) -> Path:
    return raw_root / f"Ficat stage {stage}"


def _side_from_name(path: Path) -> str:
    first = path.stem[:1].upper()
    if first in {"L", "R"}:
        return first
    return "U"


def _output_name(stage: int, raw_path: Path) -> str:
    return f"S{stage}_{raw_path.name}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="YOLO-crop raw Ficat images into stage_cls_dataset/."
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("drive-download-20251023T113302Z-1-001"),
        help="Folder containing Ficat stage 1..4 subfolders.",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("weights") / "yolo_best.pt",
        help="YOLO detector weights.",
    )
    p.add_argument("--out", type=Path, default=Path("stage_cls_dataset"))
    p.add_argument("--roi-csv", type=Path, default=Path("roi_all.csv"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="Example: 0, cuda, or cpu.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    args = p.parse_args()

    if not args.raw_root.exists():
        raise SystemExit(f"Raw root not found: {args.raw_root}")
    if not args.weights.exists():
        raise SystemExit(f"YOLO weights not found: {args.weights}")
    if args.out.exists() and any(args.out.rglob("*")) and not args.overwrite:
        raise SystemExit(
            f"{args.out} is not empty. Add --overwrite if you really want to "
            "regenerate crops in place."
        )

    for stage in range(1, 5):
        sd = _stage_dir(args.raw_root, stage)
        if not sd.exists():
            raise SystemExit(f"Missing stage folder: {sd}")
        (args.out / f"stage_{stage}").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    rows: list[dict] = []
    failures: list[str] = []
    n_written = 0

    for stage in range(1, 5):
        sd = _stage_dir(args.raw_root, stage)
        images = sorted(p for p in sd.iterdir() if p.suffix.lower() in IMG_EXTS)
        print(f"[stage {stage}] {len(images)} images")
        for img_path in images:
            side = _side_from_name(img_path)
            res = model.predict(
                source=str(img_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            if res.boxes is None or len(res.boxes) == 0:
                failures.append(f"{img_path}: no boxes")
                continue

            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            if side == "L":
                k = boxes[:, 2].argmax()
            elif side == "R":
                k = boxes[:, 0].argmin()
            else:
                k = scores.argmax()

            x1, y1, x2, y2 = [float(v) for v in boxes[k]]
            out_name = _output_name(stage, img_path)
            out_path = args.out / f"stage_{stage}" / out_name

            with Image.open(img_path).convert("RGB") as im:
                crop = im.crop((int(x1), int(y1), int(x2), int(y2)))
                if side == "R":
                    crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                crop.save(out_path)

            rows.append(
                {
                    "filename": out_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": float(scores[k]),
                    "side": side,
                    "stage": stage,
                    "source": str(img_path),
                }
            )
            n_written += 1

    pd.DataFrame(rows).to_csv(args.roi_csv, index=False, encoding="utf-8-sig")
    print(f"wrote {n_written} crops to {args.out}")
    print(f"wrote ROI table to {args.roi_csv}")

    if failures:
        fail_path = args.roi_csv.with_name(args.roi_csv.stem + "_failures.txt")
        fail_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"warning: {len(failures)} images failed; see {fail_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
