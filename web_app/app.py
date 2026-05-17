"""Flask app for the hierarchical 4-stage knee classifier.

Reads `web_app/_runtime.json` (written by scripts/07_update_web.py) at startup
and builds a tree of cut predictors that match the topology chosen by the
hierarchy search. Each cut predictor is a soft-vote of every per-fold best
checkpoint (5 backbones x 5 folds = up to 25 models per cut).

If `_runtime.json` is missing the app still boots but every /predict call
returns a clear "models not configured yet" error.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

WEB_APP_DIR = Path(__file__).resolve().parent
PROJ_ROOT = WEB_APP_DIR.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from renai.gradcam import GradCAM  # noqa: E402
from renai.models import create_model, get_target_layer  # noqa: E402

device = "cpu"
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

RUNTIME_PATH = WEB_APP_DIR / "_runtime.json"


def load_runtime() -> dict | None:
    if not RUNTIME_PATH.exists():
        return None
    return json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Cut predictor — soft-vote of every per-fold best ckpt for this cut.
# ---------------------------------------------------------------------------

class CutPredictor:
    def __init__(self, cut_name: str, info: dict):
        self.cut_name = cut_name
        self.members: list[dict] = list(info["members"])

        self._models: list[tuple[str, torch.nn.Module]] = []
        for entry in self.members:
            bb = entry["backbone"]
            ckpt = Path(entry["ckpt"])
            m = create_model(bb, num_classes=2).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            self._models.append((bb, m))

        if self._models:
            bb0, m0 = self._models[0]
            self._gradcam_backbone = bb0
            self._gradcam = GradCAM(m0, get_target_layer(m0, bb0))
        else:
            self._gradcam_backbone = None
            self._gradcam = None

    @torch.no_grad()
    def prob_class1(self, x: torch.Tensor) -> float:
        sums = None
        for _, m in self._models:
            p = F.softmax(m(x.to(device)), dim=1).cpu().numpy()
            sums = p if sums is None else sums + p
        if sums is None:
            return 0.0
        avg = sums / float(len(self._models))
        return float(avg[0, 1])

    def gradcam(self, x: torch.Tensor) -> tuple[str, np.ndarray]:
        if self._gradcam is None or self._gradcam_backbone is None:
            raise RuntimeError(f"No Grad-CAM available for cut {self.cut_name}")
        x = x.clone().requires_grad_(True)
        cam = self._gradcam(x)
        return self._gradcam_backbone, cam


# ---------------------------------------------------------------------------
# Hierarchy router
# ---------------------------------------------------------------------------

class HierarchyRouter:
    def __init__(self, runtime: dict):
        self.runtime = runtime
        self.topology = runtime["topology"]
        self.cuts: dict[str, CutPredictor] = {
            cn: CutPredictor(cn, info) for cn, info in runtime["cuts"].items()
        }

    def predict(self, x: torch.Tensor) -> dict:
        rules = list(self.topology["rules"])
        per_stage_prob: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

        cam_data: list[dict] = []
        for r in rules:
            backbone, cam = self.cuts[r["cut"]].gradcam(x)
            cam_data.append({"cut": r["cut"], "backbone": backbone, "cam": cam})

        prob_cache = {r["cut"]: self.cuts[r["cut"]].prob_class1(x) for r in rules}

        def walk(subset: list[int], parent_prob: float, remaining: list[dict]):
            if len(subset) == 1:
                per_stage_prob[subset[0]] += parent_prob
                return
            for r in remaining:
                if sorted(r["left"] + r["right"]) == sorted(subset):
                    p_right = prob_cache[r["cut"]]
                    rest = [rr for rr in remaining if rr is not r]
                    walk(r["left"],  parent_prob * (1.0 - p_right), rest)
                    walk(r["right"], parent_prob * p_right,         rest)
                    return
            per_stage_prob[subset[0]] += parent_prob

        walk([1, 2, 3, 4], 1.0, rules)

        final_stage = max(per_stage_prob.items(), key=lambda kv: kv[1])[0]
        return {
            "final_stage": final_stage,
            "per_stage": per_stage_prob,
            "cams": cam_data,
            "prob_cache": prob_cache,
            "rules": rules,
        }


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

yolo_model: YOLO | None = None
router: HierarchyRouter | None = None
val_tf: T.Compose | None = None


def init_models():
    global yolo_model, router, val_tf

    runtime = load_runtime()
    if runtime is None:
        print(
            f"[boot] Missing {RUNTIME_PATH}. Run scripts/07_update_web.py after "
            "hierarchy search to populate it. The app will boot, but /predict "
            "will return an error until that file exists.",
            flush=True,
        )
        return

    yolo_weights = Path(os.environ.get("RENAI_YOLO_WEIGHTS", "")).expanduser() if os.environ.get("RENAI_YOLO_WEIGHTS") else PROJ_ROOT / "weights" / "yolo_best.pt"
    if not yolo_weights.exists():
        yolo_weights = PROJ_ROOT / "weights" / "yolov8n.pt"
    yolo_model = YOLO(str(yolo_weights))

    router = HierarchyRouter(runtime)

    val_tf = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    print(f"[boot] topology = {router.topology['name']} {router.topology['description']}", flush=True)
    print(f"[boot] cuts: {sorted(router.cuts.keys())}", flush=True)


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
    if router is None or yolo_model is None or val_tf is None:
        return jsonify({"error": "Models not initialized. Run scripts/07_update_web.py first."}), 503
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

        annotated = res.plot()[..., ::-1]
        annotated_b64 = _b64_image(Image.fromarray(annotated))
        cropped_b64 = _b64_image(cropped_img)

        inference_img = cropped_img if side != "R" else cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
        x = val_tf(inference_img).unsqueeze(0).to(device)

        result = router.predict(x)
        cam_overlays = [
            {
                "cut": entry["cut"],
                "backbone": entry["backbone"],
                "image": "data:image/jpeg;base64," + _overlay_b64(entry["cam"], cropped_img, side),
            }
            for entry in result["cams"]
        ]

        per_stage = result["per_stage"]
        sorted_probs = sorted(
            ({"label": f"第{i}類", "value": float(per_stage[i])} for i in (1, 2, 3, 4)),
            key=lambda d: d["value"],
            reverse=True,
        )

        return jsonify({
            "annotated_image": "data:image/jpeg;base64," + annotated_b64,
            "cropped_image":   "data:image/jpeg;base64," + cropped_b64,
            "stage": f"Stage {result['final_stage']}",
            "sorted_probs": sorted_probs,
            "topology": router.topology["name"] + " " + router.topology["description"],
            "cam_overlays": cam_overlays,
            "gradcam_m1": cam_overlays[0]["image"] if len(cam_overlays) > 0 else None,
            "gradcam_m2": cam_overlays[1]["image"] if len(cam_overlays) > 1 else None,
            "gradcam_m3": cam_overlays[2]["image"] if len(cam_overlays) > 2 else None,
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
