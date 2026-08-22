"""Flask app — Ficat 4-stage classifier, now served by the CORN + stacking model.

Loads `web_app/corn_runtime/manifest.json` + `meta.joblib` (written by
scripts/24_export_web_corn.py): a 9-backbone class-weighted CORN ensemble whose
per-backbone cumulative probabilities are combined by a stacking meta-learner
(the OOF-selected best model, QWK ~0.80). On upload:

    X-ray -> YOLO ROI crop (right hip flipped to left) -> 9 CORN backbones
    (5 folds each, averaged) -> 9x3 cumulative features -> stacking meta -> stage
    + a Grad-CAM heatmap (DenseNet121) of where the model looked.

If the runtime is missing the app still boots but /predict returns a clear error.
Legacy hierarchy runtime (`_runtime.json`) is no longer used.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

WEB_APP_DIR = Path(__file__).resolve().parent
PROJ_ROOT = WEB_APP_DIR.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from renai.corn import corn_cumulative_probs, create_corn_model  # noqa: E402
from renai.gradcam import GradCAM  # noqa: E402
from renai.models import get_target_layer  # noqa: E402
from renai.ordinal import decode_rank_count  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
app = Flask(__name__)

CORN_DIR = WEB_APP_DIR / "corn_runtime"
MANIFEST_PATH = CORN_DIR / "manifest.json"
META_PATH = CORN_DIR / "meta.joblib"


# ---------------------------------------------------------------------------
# CORN + stacking model
# ---------------------------------------------------------------------------

class CornStackingModel:
    """9-backbone CORN ensemble + stacking meta. Feature order == manifest order
    (== the order the meta was fitted on), so there is no train/serve skew."""

    def __init__(self, manifest: dict, meta):
        self.manifest = manifest
        self.meta = meta
        self.backbones: list[dict] = manifest["backbones"]
        self.img_size = int(manifest.get("img_size", 384))

        # load every fold checkpoint, grouped by backbone (in manifest order)
        self.models: list[list[torch.nn.Module]] = []
        gc_bb = manifest.get("gradcam_backbone", "densenet121")
        self._gc_model = None
        self._gc_backbone = gc_bb
        for entry in self.backbones:
            bb = entry["backbone"]
            fold_models = []
            for rel in entry["ckpts"]:
                m = create_corn_model(bb).to(device)
                m.load_state_dict(torch.load(PROJ_ROOT / rel, map_location=device))
                m.eval()
                fold_models.append(m)
            self.models.append(fold_models)
            if bb == gc_bb and self._gc_model is None and fold_models:
                self._gc_model = fold_models[0]
        if self._gc_model is not None:
            self._gradcam = GradCAM(self._gc_model, get_target_layer(self._gc_model, gc_bb))
        else:
            self._gradcam = None
        n_loaded = sum(len(f) for f in self.models)
        print(f"[boot] CORN: {len(self.backbones)} backbones / {n_loaded} fold models "
              f"on {device}; gradcam={gc_bb}", flush=True)

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> np.ndarray:
        """9x3 cumulative-prob feature vector (avg over each backbone's folds)."""
        cols = []
        for fold_models in self.models:
            probs = torch.stack([corn_cumulative_probs(m(x)) for m in fold_models]).mean(0)
            cols.append(probs.cpu().numpy())          # (1,3)
        return np.nan_to_num(np.hstack(cols))          # (1, 27)

    def predict(self, x: torch.Tensor) -> dict:
        x = x.to(device)
        feat = self._features(x)
        stage = int(self.meta.predict(feat)[0])        # thesis decode (expected value)
        proba = self.meta.predict_proba(feat)[0]       # 4-class simplex
        classes = [int(c) for c in self.meta.classes_]
        per_stage = {s: 0.0 for s in (1, 2, 3, 4)}
        for c, p in zip(classes, proba):
            per_stage[c] = float(p)
        return {"final_stage": stage, "per_stage": per_stage, "method": self.manifest["model"]}

    def gradcam(self, x: torch.Tensor):
        """Grad-CAM on the representative backbone, targeting the cumulative
        threshold matching the predicted severity."""
        if self._gradcam is None:
            return None, None
        with torch.no_grad():
            probs = corn_cumulative_probs(self._gc_model(x.to(device)))
        stage = int(decode_rank_count(probs.cpu().numpy())[0])
        tgt = min(max(stage - 2, 0), 2)
        xg = x.to(device).clone().requires_grad_(True)
        cam = self._gradcam(xg, class_idx=tgt)
        return self._gc_backbone, cam


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

yolo_model: YOLO | None = None
model: CornStackingModel | None = None
val_tf: T.Compose | None = None


def init_models():
    global yolo_model, model, val_tf
    if not MANIFEST_PATH.exists() or not META_PATH.exists():
        print(f"[boot] Missing {CORN_DIR}. Run scripts/24_export_web_corn.py first. "
              "The app boots but /predict will error until then.", flush=True)
        return

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    meta = joblib.load(META_PATH)

    yolo_w = os.environ.get("RENAI_YOLO_WEIGHTS") or manifest.get("yolo_weights") or "weights/yolo_best_rebox.pt"
    yolo_path = Path(yolo_w)
    if not yolo_path.is_absolute():
        yolo_path = PROJ_ROOT / yolo_path
    if not yolo_path.exists():
        yolo_path = PROJ_ROOT / "weights" / "yolo_best.pt"
    yolo_model = YOLO(str(yolo_path))

    model = CornStackingModel(manifest, meta)
    val_tf = T.Compose([
        T.Resize((model.img_size, model.img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    print(f"[boot] YOLO = {yolo_path.name}; model = {manifest['model']}", flush=True)


try:
    init_models()
except Exception as e:
    print(f"[boot] init_models failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


def _overlay_b64(cam_map: np.ndarray, original_img: Image.Image, side: str) -> str:
    if side == "R":
        cam_map = np.fliplr(cam_map)
    img_np = np.array(original_img.resize((384, 384))) / 255.0
    h, w, _ = img_np.shape
    cam_resized = cv2.resize(cam_map, (w, h))
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.4 * heat + 0.6 * img_np, 0, 1)
    pil = Image.fromarray((overlay * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_image(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or yolo_model is None or val_tf is None:
        return jsonify({"error": "Models not initialized. Run scripts/24_export_web_corn.py first."}), 503
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    side = request.form.get("side", "L")
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    tmp_path = WEB_APP_DIR / "temp.jpg"
    img.save(tmp_path)

    try:
        res = yolo_model.predict(source=str(tmp_path), conf=0.25, device=device, imgsz=640, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return jsonify({"error": "未偵測到髖關節！請更換圖片或降低信心閾值。"}), 400

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        if side == "L":
            k = boxes[:, 2].argmax()
        elif side == "R":
            k = boxes[:, 0].argmin()
        else:
            k = scores.argmax()
        x1, y1, x2, y2 = map(int, boxes[k])
        cropped_img = img.crop((x1, y1, x2, y2))

        annotated_b64 = _b64_image(Image.fromarray(res.plot()[..., ::-1]))
        cropped_b64 = _b64_image(cropped_img)

        inference_img = cropped_img if side != "R" else cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
        x = val_tf(inference_img).unsqueeze(0).to(device)

        result = model.predict(x)
        gc_backbone, cam = model.gradcam(x)
        cam_overlays = []
        if cam is not None:
            cam_overlays.append({
                "cut": "CORN", "backbone": gc_backbone,
                "image": "data:image/jpeg;base64," + _overlay_b64(cam, cropped_img, side),
            })

        per_stage = result["per_stage"]
        sorted_probs = sorted(
            ({"label": f"第{i}類", "value": float(per_stage[i])} for i in (1, 2, 3, 4)),
            key=lambda d: d["value"], reverse=True,
        )

        return jsonify({
            "annotated_image": "data:image/jpeg;base64," + annotated_b64,
            "cropped_image":   "data:image/jpeg;base64," + cropped_b64,
            "stage": f"Stage {result['final_stage']}",
            "sorted_probs": sorted_probs,
            "method": result.get("method", ""),
            "topology": result.get("method", ""),
            "cam_overlays": cam_overlays,
            "gradcam_m1": cam_overlays[0]["image"] if cam_overlays else None,
            "gradcam_m2": None,
            "gradcam_m3": None,
        })
    except Exception as e:
        return jsonify({"error": f"預測時發生錯誤: {e}"}), 500
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
