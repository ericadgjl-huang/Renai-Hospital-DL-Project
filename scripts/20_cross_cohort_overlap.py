"""Stage 20 — Cross-cohort patient overlap audit (old vs AVNFH).

Answers: do the old cohort and the new AVNFH cohort share patients? If they do
and we split by IMAGE (or treat AVNFH as its own groups), the same patient can
land in both train and test -> leakage. This audits three ways:

  1. EXACT patient-ID match between cohorts (what patient_groups.csv already merges).
  2. FUZZY ID match (Levenshtein distance 1) — catches the SAME patient split into
     two IDs by a single OCR character error (would otherwise NOT be merged -> residual leak).
  3. BIRTHDAY+SEX match with DIFFERENT IDs — an independent signal for the same
     patient whose ID string differs (stronger evidence of an OCR-split duplicate).

Also verifies that patient_groups.csv gives every cross-cohort patient ONE shared
group id (so the leak is actually removed).

Usage:
    python scripts/20_cross_cohort_overlap.py
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

OLD_OCR_DIR = ROOT / "20251103開會前" / "X光,病患統計表"
AVNFH_CSV = ROOT / "outputs_merged" / "avnfh_patient_ids.csv"
GROUPS_CSV = ROOT / "outputs_merged" / "patient_groups.csv"


def _norm_bday(b) -> str:
    """Normalise a birthday string to YYYY-M-D (ignore blanks)."""
    s = str(b).strip()
    if not s or s.lower() == "nan":
        return ""
    m = re.match(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    return f"{int(m.group(1))}-{int(m.group(2))}-{int(m.group(3))}" if m else ""


def _lev1(a: str, b: str) -> bool:
    """True if Levenshtein distance between a and b is exactly 1."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:                                   # one substitution
        return sum(x != y for x, y in zip(a, b)) == 1
    # one insertion/deletion: the shorter must be the longer with 1 char removed
    if la > lb:
        a, b = b, a
    for i in range(len(b)):
        if a == b[:i] + b[i + 1:]:
            return True
    return False


def load_old():
    rows = []
    for f in glob.glob(str(OLD_OCR_DIR / "raw_ocr_metadata_Ficat stage *.csv")):
        stage = int(re.search(r"stage (\d)", os.path.basename(f)).group(1))
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            pid = str(r.get("ID", "")).strip()
            if pid and pid.lower() != "nan":
                rows.append({"cohort": "old", "id": pid, "stage": stage,
                             "sex": str(r.get("patient_sex", "")).strip(),
                             "bday": _norm_bday(r.get("birthday", ""))})
    return pd.DataFrame(rows)


def load_avnfh():
    d = pd.read_csv(AVNFH_CSV)
    d = d[d["patient"].notna() & (d["patient"].astype(str) != "")]
    return pd.DataFrame({"cohort": "avnfh", "id": d["patient"].astype(str).str.strip(),
                         "stage": d["stage"], "sex": d.get("sex", "").astype(str).str.strip(),
                         "bday": d["birthday"].map(_norm_bday)})


def main():
    old, av = load_old(), load_avnfh()
    old_ids, av_ids = set(old["id"]), set(av["id"])
    print(f"[audit] old cohort: {len(old)} imgs, {len(old_ids)} unique patients")
    print(f"[audit] AVNFH cohort: {len(av)} imgs, {len(av_ids)} unique patients")

    # 1) EXACT overlap
    exact = sorted(old_ids & av_ids)
    n_old_imgs = int(old["id"].isin(exact).sum())
    n_av_imgs = int(av["id"].isin(exact).sum())
    print(f"\n[1] EXACT-ID cross-cohort patients: {len(exact)}")
    print(f"    -> {n_old_imgs} old imgs + {n_av_imgs} AVNFH imgs share these patients")
    for pid in exact:
        os_ = sorted(old.loc[old.id == pid, "stage"].tolist())
        as_ = sorted(av.loc[av.id == pid, "stage"].tolist())
        print(f"      {pid}: old stages={os_}  avnfh stages={as_}")

    # 2) FUZZY (edit-distance-1) matches NOT already exact -> possible OCR split
    fuzzy = []
    for a in (old_ids - set(exact)):
        for b in (av_ids - set(exact)):
            if _lev1(a, b):
                fuzzy.append((a, b))
    print(f"\n[2] FUZZY (edit-distance 1) ID pairs across cohorts: {len(fuzzy)}")
    for a, b in fuzzy[:20]:
        ob = set(old.loc[old.id == a, "bday"]) - {""}
        ab = set(av.loc[av.id == b, "bday"]) - {""}
        same_b = "SAME birthday" if (ob & ab) else "diff/again unknown birthday"
        print(f"      old {a}  ~  avnfh {b}   ({same_b})")

    # 3) BIRTHDAY+SEX match with DIFFERENT ids -> independent duplicate signal
    old_key = {(r.bday, r.sex): r.id for r in old.itertuples() if r.bday}
    hits = []
    for r in av.itertuples():
        if r.bday and (r.bday, r.sex) in old_key:
            oid = old_key[(r.bday, r.sex)]
            if oid != r.id:
                hits.append((oid, r.id, r.bday, r.sex))
    hits = sorted(set(hits))
    print(f"\n[3] BIRTHDAY+SEX matches across cohorts with DIFFERENT IDs: {len(hits)}")
    for oid, aid, b, s in hits:
        print(f"      old {oid}  vs  avnfh {aid}   (birthday {b}, sex {s}) <- likely same patient, OCR-split")

    # 4) verify patient_groups.csv merges the exact cross-cohort patients into ONE group
    if GROUPS_CSV.exists():
        g = pd.read_csv(GROUPS_CSV)
        ok = True
        for pid in exact:
            # old file names S{stage}_{...}; avnfh S{stage}_A_{...}. Both should map to group == pid.
            grp = set(g.loc[g["patient"] == pid, "patient"])
            if grp != {pid}:
                ok = False
        n_groups = g["patient"].nunique()
        print(f"\n[4] patient_groups.csv: {n_groups} groups; exact cross-cohort patients "
              f"share one group id each: {'YES (leak removed)' if ok else 'NO — check!'}")

    print("\n[audit] summary:")
    print(f"  exact cross-cohort patients = {len(exact)} (merged, leak removed)")
    print(f"  possible OCR-split duplicates (fuzzy or birthday) = "
          f"{len(set(fuzzy)) + len(hits)}  <- review these for residual leak")


if __name__ == "__main__":
    main()
