"""Stage 18 — OCR patient IDs off the AVNFH X-ray corners (easyocr).

The AVNFH cohort's staged images (`AVNFH_staged/Ficat stage N/*.jpg`) keep the
scanner's corner overlay: PID, sex/age, birthday and a patient ID that repeats
in several corners (e.g. `S223541253`). This recovers those IDs so the merged
patient-level split can cover AVNFH too (instead of the conservative
one-group-per-image fallback).

Method (follows the user's 表格製作1.ipynb, but with easyocr instead of
Tesseract — no system binary needed, uses the existing PyTorch):
  * crop the 4 corners, OCR each, collect strings;
  * ID regex broadened to `[A-Z]\\d{8,}` (AVNFH IDs start with S; the old cohort
    used L/B/A/K), majority-vote across corners for robustness;
  * map staged file `Ficat stage N/<base>` -> dataset name `S{N}_A_<base>`.

Output: `outputs_merged/avnfh_patient_ids.csv` with columns
    filename,patient,n_votes,sex,age,birthday
(filename is the MERGED dataset basename, so it can be concatenated with the
old-cohort mapping in patient_groups.csv).

Usage (conda env unet_labeling):
    python scripts/18_ocr_avnfh_patient_ids.py --device 0
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import pandas as pd

ID_RE = re.compile(r"[A-Z]\d{8,}")
SEX_RE = re.compile(r"\b([FM])\b")
AGE_RE = re.compile(r"(\d{1,3})\s*Y", re.I)
BDAY_RE = re.compile(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})")
STAGE_DIRS = {1: "Ficat stage 1", 2: "Ficat stage 2", 3: "Ficat stage 3", 4: "Ficat stage 4"}


def corners(img, p=0.24):
    h, w = img.shape[:2]
    dx, dy = int(w * p), int(h * p)
    return [img[0:dy, 0:dx], img[0:dy, w - dx:w],
            img[h - dy:h, 0:dx], img[h - dy:h, w - dx:w]]


def read_one(reader, path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return {"patient": "", "n_votes": 0, "sex": "", "age": "", "birthday": ""}
    ids, texts = [], []
    for c in corners(img):
        for t in reader.readtext(c, detail=0):
            texts.append(t)
    joined = " ".join(texts)
    compact = joined.replace(" ", "")
    ids = ID_RE.findall(compact)
    pid, votes = "", 0
    if ids:
        pid, votes = Counter(ids).most_common(1)[0]
    sex = (SEX_RE.search(joined).group(1) if SEX_RE.search(joined) else "")
    age = (AGE_RE.search(joined).group(1) if AGE_RE.search(joined) else "")
    bday = (BDAY_RE.search(joined).group(1) if BDAY_RE.search(joined) else "")
    return {"patient": pid, "n_votes": int(votes), "sex": sex, "age": age, "birthday": bday}


def main():
    ap = argparse.ArgumentParser(description="OCR AVNFH patient IDs from corners.")
    ap.add_argument("--staged-root", type=Path, default=Path("AVNFH_staged"))
    ap.add_argument("--out-csv", type=Path, default=Path("outputs_merged/avnfh_patient_ids.csv"))
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=(args.device != "cpu"), verbose=False)

    rows = []
    for stage, sdir in STAGE_DIRS.items():
        folder = args.staged_root / sdir
        if not folder.exists():
            print(f"[ocr] missing {folder}", flush=True)
            continue
        for f in sorted(folder.glob("*.jpg")):
            info = read_one(reader, f)
            dataset_name = f"S{stage}_A_{f.name}"     # merged ImageFolder basename
            rows.append({"filename": dataset_name, "stage": stage,
                         "staged_source": str(f), **info})
            print(f"  [ocr] {dataset_name}: id={info['patient'] or '(none)'} "
                  f"votes={info['n_votes']}", flush=True)

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    n = len(df)
    got = int((df["patient"] != "").sum())
    uniq = df.loc[df["patient"] != "", "patient"].nunique()
    multi = (df.loc[df["patient"] != ""].groupby("patient").size() > 1).sum()
    print(f"\n[ocr] {n} AVNFH images: {got} with a patient ID ({100*got/max(n,1):.0f}%), "
          f"{uniq} unique patients, {multi} patients with >1 image.", flush=True)
    print(f"[ocr] wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
