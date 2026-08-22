"""Stage 16 — Build a filename -> patient-group table for the MERGED dataset,
so training can use patient-level (leak-free) splits instead of image-level.

Source of patient IDs: OCR metadata extracted from the OLD cohort's X-ray
corners (`20251103開會前/X光,病患統計表/raw_ocr_metadata_Ficat stage N.csv`),
which has 95% coverage and reveals 33 patients whose images span >1 Ficat stage
— the real leakage an image-level split cannot prevent.

Mapping:
  * OLD cohort merged file `S{stage}_{base}`  -> OCR[stage][base] -> patient ID.
  * AVNFH file `S{stage}_A_{base}` (no OCR) and any OLD file with a missing ID
    -> its own unique group (conservative: cannot fake leakage removal, and the
    only KNOWN leakage — old-cohort multi-stage patients — is still removed).

Output: `outputs_merged/patient_groups.csv` with columns filename,patient.
This is consumed by scripts/15_train_corn.py --groups-csv to switch the outer
split + CV to GroupShuffleSplit / StratifiedGroupKFold.

Usage:
    python scripts/16_build_patient_groups.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from torchvision.datasets import ImageFolder

STAGE_NAMES = ["stage_1", "stage_2", "stage_3", "stage_4"]


def load_ocr_map(ocr_dir: Path) -> dict[tuple[int, str], str]:
    """(stage, base_filename) -> patient ID, from the per-stage OCR CSVs."""
    m: dict[tuple[int, str], str] = {}
    for f in glob.glob(str(ocr_dir / "raw_ocr_metadata_Ficat stage *.csv")):
        stage = int(re.search(r"stage (\d)", os.path.basename(f)).group(1))
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            pid = str(r.get("ID", "nan"))
            if pid and pid != "nan":
                m[(stage, str(r["filename"]))] = pid
    return m


def base_from_merged(name: str) -> tuple[int, str, bool]:
    """`S1_L2.jpg` -> (1, 'L2.jpg', is_avnfh=False);
       `S1_A_L2.jpg` -> (1, 'L2.jpg', is_avnfh=True)."""
    mobj = re.match(r"S(\d)_(A_)?(.+)$", name)
    if not mobj:
        return (0, name, False)
    return int(mobj.group(1)), mobj.group(3), bool(mobj.group(2))


def _norm_bday(b) -> str:
    s = str(b).strip()
    m = re.match(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    return f"{int(m.group(1))}-{int(m.group(2))}-{int(m.group(3))}" if m else ""


def build_canonical_map(ocr_dir: Path, avnfh_csv: Path) -> dict[str, str]:
    """id -> canonical id. OCR can misread a patient's ID string (append/drop
    digits), splitting one patient into several IDs and MISSING cross-cohort
    duplicates. Birthday+sex is far more OCR-stable, so we merge every ID sharing
    the same (birthday, sex) to a single canonical id (the shortest — OCR tends
    to add characters). This repairs both within-cohort fragmentation and missed
    cross-cohort matches. Prints exactly what it merges (auditable)."""
    by_key: dict[tuple[str, str], set[str]] = {}
    def add(pid, bday, sex):
        b = _norm_bday(bday)
        if pid and str(pid) != "nan" and b:
            by_key.setdefault((b, str(sex).strip()), set()).add(str(pid))
    for f in glob.glob(str(ocr_dir / "raw_ocr_metadata_Ficat stage *.csv")):
        for _, r in pd.read_csv(f).iterrows():
            add(r.get("ID"), r.get("birthday"), r.get("patient_sex"))
    if avnfh_csv.exists():
        for _, r in pd.read_csv(avnfh_csv).iterrows():
            add(r.get("patient"), r.get("birthday"), r.get("sex"))
    canon: dict[str, str] = {}
    for (b, sx), ids in by_key.items():
        if len(ids) > 1:
            target = min(ids, key=lambda s: (len(s), s))   # shortest = likely correct
            for i in ids:
                canon[i] = target
            print(f"[canon] birthday {b} {sx}: merge {sorted(ids)} -> {target}")
    return canon


def main():
    data_root = ROOT / "stage_cls_merged"
    ocr_dir = ROOT / "20251103開會前" / "X光,病患統計表"
    avnfh_csv = ROOT / "outputs_merged" / "avnfh_patient_ids.csv"
    out_csv = ROOT / "outputs_merged" / "patient_groups.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ocr = load_ocr_map(ocr_dir)
    print(f"[groups] OLD-cohort OCR patient IDs: {len(ocr)}")

    # AVNFH IDs from scripts/18 (keyed by the merged dataset basename S{N}_A_*)
    avnfh_map: dict[str, str] = {}
    if avnfh_csv.exists():
        a = pd.read_csv(avnfh_csv)
        a = a[a["patient"].notna() & (a["patient"].astype(str) != "")]
        avnfh_map = dict(zip(a["filename"].astype(str), a["patient"].astype(str)))
        print(f"[groups] AVNFH OCR patient IDs: {len(avnfh_map)}")
    else:
        print("[groups] no AVNFH OCR csv -> AVNFH stays conservative (run scripts/18 first).")

    # birthday+sex canonicalisation: repair OCR-split IDs (both within-cohort
    # fragmentation and missed cross-cohort duplicates).
    canon = build_canonical_map(ocr_dir, avnfh_csv)
    print(f"[groups] canonicalised {len(canon)} OCR-variant IDs via birthday+sex")

    base = ImageFolder(str(data_root))
    rows, n_ocr, n_avnfh, n_conserv = [], 0, 0, 0
    for path, _ in base.samples:
        fn = Path(path).name
        stage, b, is_avnfh = base_from_merged(fn)
        if is_avnfh:
            pid = avnfh_map.get(fn)
            if pid is not None:
                n_avnfh += 1
        else:
            pid = ocr.get((stage, b))
            if pid is not None:
                n_ocr += 1
        if pid is None:
            pid = f"__self__{fn}"   # conservative: unique group per image
            n_conserv += 1
        else:
            pid = canon.get(pid, pid)              # collapse OCR-variant IDs
        rows.append({"filename": fn, "patient": pid})
    print(f"[groups] IDs: old-cohort={n_ocr}, AVNFH={n_avnfh}, conservative self-groups={n_conserv}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    n_real = df[~df["patient"].str.startswith("__self__")]["patient"].nunique()
    print(f"[groups] {len(df)} images: {n_ocr + n_avnfh} with real patient ID "
          f"({n_real} unique patients), {n_conserv} conservative self-groups.")
    print(f"[groups] total groups: {df['patient'].nunique()} "
          f"(fewer than {len(df)} images -> leakage-preventing split is active)")
    print(f"[groups] wrote {out_csv}")


if __name__ == "__main__":
    main()
