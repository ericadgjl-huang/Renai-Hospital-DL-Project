"""Stage 21 — Grad-CAM heatmaps for the final CORN model (professor's request).

Overlays a Grad-CAM "where the model looked" heatmap on each X-ray crop, using
one representative backbone (DenseNet121 by default — a standard choice for
radiograph Grad-CAM). For the CORN head we target the cumulative-threshold logit
matching the predicted severity (stage S -> threshold index clamp(S-2,0,2)),
i.e. "the evidence that pushed it to at least stage S". CAMs are averaged over
the 5 CV fold models for robustness.

Produces (into <out-root>/gradcam/):
  misclassified/true{T}_pred{P}__<name>.png  — every wrong test crop + heatmap
  correct/stage{S}__<name>.png               — a few correct crops per stage (contrast)
Each PNG is a side-by-side: original crop | Grad-CAM overlay (title = true/pred stage).

Usage (conda env unet_labeling):
    python scripts/21_gradcam.py --out-root outputs_merged_patient_v2 --device 0
    python scripts/21_gradcam.py --backbone convnext_tiny --device 0   # different backbone
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from renai.corn import corn_cumulative_probs, create_corn_model
from renai.data import IMG_SIZE, make_eval_transform, make_outer_split, patient_groups, get_4class_labels
from renai.gradcam import GradCAM
from renai.models import get_target_layer
from renai.ordinal import decode_rank_count
from renai.seed import SEED, set_seed

STAGE_DIR = "stage_cls_merged"


def load_folds(backbone, bdir: Path, device):
    models = []
    for f in sorted(bdir.glob("fold_*.pth")):
        m = create_corn_model(backbone).to(device)
        m.load_state_dict(torch.load(f, map_location=device))
        m.eval()
        models.append(m)
    return models


def cam_for_image(models, backbone, x, device):
    """Averaged Grad-CAM over folds. Target = threshold logit for predicted stage."""
    # ensemble-of-folds cumulative probs -> predicted stage on this backbone
    with torch.no_grad():
        probs = torch.stack([corn_cumulative_probs(m(x)) for m in models]).mean(0)
    stage = int(decode_rank_count(probs.cpu().numpy())[0])
    tgt = min(max(stage - 2, 0), 2)
    cams = []
    for m in models:
        gc = GradCAM(m, get_target_layer(m, backbone))
        cams.append(gc(x, class_idx=tgt))
        gc.close()
    cam = np.mean(cams, axis=0)
    return cam, stage


def overlay_png(crop_path: Path, cam: np.ndarray, title: str, out_path: Path):
    img = Image.open(crop_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img) / 255.0
    cam_r = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heat = cm.jet(cam_r)[:, :, :3]
    blend = 0.55 * arr + 0.45 * heat
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.7))
    ax[0].imshow(arr); ax[0].set_title("X-ray crop", fontsize=10); ax[0].axis("off")
    ax[1].imshow(blend); ax[1].set_title("Grad-CAM (where the model looked)", fontsize=10); ax[1].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM heatmaps for the final CORN model.")
    ap.add_argument("--out-root", type=Path, default=Path("outputs_merged_patient_v2"))
    ap.add_argument("--data-root", type=Path, default=Path(STAGE_DIR))
    ap.add_argument("--groups-csv", default="outputs_merged/patient_groups.csv")
    ap.add_argument("--splits-dir", type=Path, default=None)
    ap.add_argument("--backbone", default="densenet121")
    ap.add_argument("--device", default="0")
    ap.add_argument("--mis-source", default="stacking",
                    help="which misclassified.csv to visualise: 'stacking' or 'voting'")
    ap.add_argument("--n-correct", type=int, default=2, help="correct examples per stage")
    args = ap.parse_args()
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    if args.splits_dir is None:
        args.splits_dir = args.out_root / "splits"
    set_seed(SEED)

    bdir = args.out_root / "corn_cw" / args.backbone
    if not bdir.exists():
        raise SystemExit(f"no trained {args.backbone} at {bdir}")
    models = load_folds(args.backbone, bdir, args.device)
    tf = make_eval_transform(IMG_SIZE)
    samples = {Path(p).name: p for p, _ in ImageFolder(args.data_root).samples}
    gdir = args.out_root / "gradcam"

    # --- misclassified test crops ---
    mis_csv = (args.out_root / ("stacking/misclassified.csv" if args.mis_source == "stacking"
                                else "misclassified.csv"))
    mis = pd.read_csv(mis_csv)
    print(f"[gradcam] {len(mis)} misclassified crops from {mis_csv.name} "
          f"using {args.backbone}", flush=True)
    for _, r in mis.iterrows():
        fn = r["crop_filename"]
        if fn not in samples:
            continue
        x = tf(Image.open(samples[fn]).convert("RGB")).unsqueeze(0).to(args.device)
        cam, _ = cam_for_image(models, args.backbone, x, args.device)
        t, p = int(r["true_stage"]), int(r["pred_stage"])
        overlay_png(Path(samples[fn]), cam,
                    f"{fn}   TRUE stage {t}  ->  PRED stage {p}  (WRONG)",
                    gdir / "misclassified" / f"true{t}_pred{p}__{Path(fn).stem}.png")

    # --- a few correctly-classified crops per stage (contrast) ---
    groups = patient_groups(args.data_root, args.groups_csv) if args.groups_csv else None
    outer = make_outer_split(args.data_root, args.splits_dir / "outer_split.json", groups=groups)
    test_idx = sorted(int(i) for i in outer["test_idx"])
    labels = get_4class_labels(args.data_root).numpy()
    all_samples = ImageFolder(args.data_root).samples
    wrong = set(mis["crop_filename"])
    per_stage: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    for i in test_idx:
        path = all_samples[i][0]; fn = Path(path).name
        stg = int(labels[i]) + 1
        if fn in wrong or per_stage[stg] >= args.n_correct:
            continue
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(args.device)
        cam, pred = cam_for_image(models, args.backbone, x, args.device)
        if pred != stg:
            continue
        per_stage[stg] += 1
        overlay_png(Path(path), cam, f"{fn}   stage {stg}  (CORRECT)",
                    gdir / "correct" / f"stage{stg}__{Path(fn).stem}.png")

    n_mis = len(list((gdir / "misclassified").glob("*.png")))
    n_ok = len(list((gdir / "correct").glob("*.png")))
    print(f"[gradcam] wrote {n_mis} misclassified + {n_ok} correct heatmaps -> {gdir}", flush=True)


if __name__ == "__main__":
    main()
