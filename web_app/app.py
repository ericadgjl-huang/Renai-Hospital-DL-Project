import os
import io
import base64
import json
import shutil
import urllib.request
import joblib
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, render_template
from pathlib import Path
from ultralytics import YOLO

WEB_APP_DIR = Path(__file__).resolve().parent
PROJ_ROOT = WEB_APP_DIR.parent
LOCAL_MODEL_ROOT = PROJ_ROOT
DEPLOY_MODEL_ROOT = WEB_APP_DIR / "_model_cache"

MODEL_FILES = {
    "YOLO_WEIGHTS": Path("yolo_dataset_process") / "runs" / "detect" / "train_nb" / "weights" / "best.pt",
    "M1_CKPT": Path("outputs_bin(1,2,3),(4)") / "_best_model" / "best_densenet121.pth",
    "M3_CKPT": Path("outputs_bin(2),(3)") / "_best_model" / "best_densenet121.pth",
    "CFG_PATH": Path("03.5_combination") / "03.25_m2_stacking_top3" / "config.json",
    "M2_GCAM_CKPT": Path("outputs_bin(1),(2,3,4)") / "_best_model" / "best_efficientnet_b0.pth",
    "M2_BASE_EFFICIENTNET": Path("03.5_combination") / "03.25_m2_stacking_top3" / "m2_base_ckpts" / "efficientnet_b0__best_efficientnet_b0.pth",
    "M2_BASE_RESNET50": Path("03.5_combination") / "03.25_m2_stacking_top3" / "m2_base_ckpts" / "resnet50__best_resnet50.pth",
    "M2_BASE_CONVNEXT": Path("03.5_combination") / "03.25_m2_stacking_top3" / "m2_base_ckpts" / "convnext_tiny__best_convnext_tiny.pth",
    "M2_META": Path("03.5_combination") / "03.25_m2_stacking_top3" / "meta" / "m2_meta_logreg.pkl",
}

MODEL_ASSET_FILENAMES = {
    "YOLO_WEIGHTS": "best.pt",
    "M1_CKPT": "m1_best_densenet121.pth",
    "M3_CKPT": "m3_best_densenet121.pth",
    "CFG_PATH": "config.json",
    "M2_GCAM_CKPT": "m2_best_efficientnet_b0.pth",
    "M2_BASE_EFFICIENTNET": "m2_best_efficientnet_b0.pth",
    "M2_BASE_RESNET50": "m2_best_resnet50.pth",
    "M2_BASE_CONVNEXT": "m2_best_convnext_tiny.pth",
    "M2_META": "m2_meta_logreg.pkl",
}


def running_on_render() -> bool:
    return bool(os.environ.get("RENDER"))


def get_model_root() -> Path:
    override = os.environ.get("MODEL_ROOT", "").strip()
    if override:
        return Path(override)
    if os.environ.get("MODEL_ASSET_BASE_URL", "").strip():
        return DEPLOY_MODEL_ROOT
    return LOCAL_MODEL_ROOT


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def download_model_assets(model_root: Path):
    base_url = os.environ.get("MODEL_ASSET_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return

    print("MODEL_ASSET_BASE_URL detected, ensuring model files are downloaded...", flush=True)
    for key, rel_path in MODEL_FILES.items():
        target = model_root / rel_path
        if target.exists():
            continue
        ensure_parent(target)
        asset_name = MODEL_ASSET_FILENAMES[key]
        source_url = f"{base_url}/{asset_name}"
        print(f"Downloading {source_url} -> {target}", flush=True)
        with urllib.request.urlopen(source_url) as response, target.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)


def resolve_model_path(model_root: Path, key: str) -> Path:
    rel_path = MODEL_FILES[key]
    primary = model_root / rel_path
    if primary.exists():
        return primary

    fallback = LOCAL_MODEL_ROOT / rel_path
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Missing required model file: {rel_path}")


def resolve_optional_model_path(model_root: Path, primary_key: str, fallback_key: str) -> Path:
    try:
        return resolve_model_path(model_root, primary_key)
    except FileNotFoundError:
        return resolve_model_path(model_root, fallback_key)


MODEL_ROOT = get_model_root()

HAS_REMOTE_MODEL_ASSETS = bool(os.environ.get("MODEL_ASSET_BASE_URL", "").strip())

YOLO_WEIGHTS = resolve_model_path(MODEL_ROOT, "YOLO_WEIGHTS") if not HAS_REMOTE_MODEL_ASSETS else MODEL_ROOT / MODEL_FILES["YOLO_WEIGHTS"]
M1_CKPT = resolve_model_path(MODEL_ROOT, "M1_CKPT") if not HAS_REMOTE_MODEL_ASSETS else MODEL_ROOT / MODEL_FILES["M1_CKPT"]
M3_CKPT = resolve_model_path(MODEL_ROOT, "M3_CKPT") if not HAS_REMOTE_MODEL_ASSETS else MODEL_ROOT / MODEL_FILES["M3_CKPT"]
CFG_PATH = resolve_model_path(MODEL_ROOT, "CFG_PATH") if not HAS_REMOTE_MODEL_ASSETS else MODEL_ROOT / MODEL_FILES["CFG_PATH"]

app = Flask(__name__)
device = "cpu"  # 強制使用 CPU 避免 CUDA kernel image mismatch 錯誤

def create_model(model_name: str, num_classes: int = 2):
    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif model_name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif model_name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        model = models.convnext_base(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == "densenet169":
        weights = models.DenseNet169_Weights.DEFAULT
        model = models.densenet169(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model

class M2Stacker:
    def __init__(self, models, meta_clf, device="cuda"):
        self.models = models
        self.meta = meta_clf
        self.device = device

    @torch.no_grad()
    def _make_meta_features(self, x):
        x = x.to(self.device)
        feats = []
        for m in self.models:
            logits = m(x)                 
            p = F.softmax(logits, dim=1)  
            feats.append(p)
        feats = torch.cat(feats, dim=1)   
        return feats.detach().cpu().numpy()

    @torch.no_grad()
    def p_stage1(self, x):
        X_meta = self._make_meta_features(x)
        prob_stage1 = self.meta.predict_proba(X_meta)[:, 0]  
        return prob_stage1[0] 

# ----------------------------------------
# Grad-CAM Utilities
# ----------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        
        # We need gradients to flow here
        score = out[0, class_idx]
        score.backward()
        
        acts = self.activations
        grads = self.gradients
        weights = grads.mean(dim=(2, 3))[0]
        
        gcam = torch.zeros(acts.shape[2:], dtype=torch.float32).to(acts.device)
        for i, w in enumerate(weights):
            gcam += w * acts[0, i, :, :]
            
        gcam = F.relu(gcam)
        gcam -= gcam.min()
        if gcam.max() > 0:
            gcam /= gcam.max()
        return gcam.detach().cpu().numpy()

def get_target_layer(model, model_name: str):
    model_name = model_name.lower()
    if model_name in ["densenet121", "densenet169"]:
        return model.features
    elif model_name in ["efficientnet_b0", "efficientnet_b1"]:
        return model.features[-1]
    return model.features

# Globals for models
yolo_model = None
m1_model = None
m2_model = None
m3_model = None
val_tf = None
thr2_val = 0.5 

# Globals for Grad-CAM
gcam_m1 = None
m2_gcam_model = None # Standalone EfficientNet-B0 since M2 is stacked
gcam_m2 = None
gcam_m3 = None

def load_runtime_config(model_root: Path):
    cfg_path = resolve_model_path(model_root, "CFG_PATH")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    meta_path = model_root / MODEL_FILES["M2_META"]
    if not meta_path.exists():
        meta_path = LOCAL_MODEL_ROOT / MODEL_FILES["M2_META"]
    cfg["meta_clf_path"] = str(meta_path)

    base_ckpt_overrides = {
        "efficientnet_b0": model_root / MODEL_FILES["M2_BASE_EFFICIENTNET"],
        "resnet50": model_root / MODEL_FILES["M2_BASE_RESNET50"],
        "convnext_tiny": model_root / MODEL_FILES["M2_BASE_CONVNEXT"],
    }

    for item in cfg["m2_base_models"]:
        model_name = item["model_name"]
        runtime_path = base_ckpt_overrides[model_name]
        if not runtime_path.exists():
            runtime_path = LOCAL_MODEL_ROOT / MODEL_FILES[
                {
                    "efficientnet_b0": "M2_BASE_EFFICIENTNET",
                    "resnet50": "M2_BASE_RESNET50",
                    "convnext_tiny": "M2_BASE_CONVNEXT",
                }[model_name]
            ]
        item["ckpt_in_comb_dir"] = str(runtime_path)

    return cfg


def init_models():
    global yolo_model, m1_model, m2_model, m3_model, val_tf, thr2_val
    global gcam_m1, m2_gcam_model, gcam_m2, gcam_m3
    
    print("Loading Models...", flush=True)
    download_model_assets(MODEL_ROOT)
    yolo_weights = resolve_model_path(MODEL_ROOT, "YOLO_WEIGHTS")
    m1_ckpt = resolve_model_path(MODEL_ROOT, "M1_CKPT")
    m3_ckpt = resolve_model_path(MODEL_ROOT, "M3_CKPT")
    cfg = load_runtime_config(MODEL_ROOT)
    m2_gcam_ckpt = resolve_optional_model_path(MODEL_ROOT, "M2_GCAM_CKPT", "M2_BASE_EFFICIENTNET")

    yolo_model = YOLO(str(yolo_weights))
    
    # --- M1 (DenseNet121) ---
    m1_model = create_model("densenet121", num_classes=2).to(device)
    m1_model.load_state_dict(torch.load(m1_ckpt, map_location=device))
    m1_model.eval()
    gcam_m1 = GradCAM(m1_model, get_target_layer(m1_model, "densenet121"))

    # --- M2 (Stacking Meta Classifier for inference) ---
    meta_clf = joblib.load(cfg["meta_clf_path"])
    thr2_val = float(cfg.get("thr2_default", 0.5))
    
    m2_base_models = []
    for item in cfg["m2_base_models"]:
        name = item["model_name"]
        ckpt = Path(item["ckpt_in_comb_dir"])
        m = create_model(name, num_classes=2).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        m2_base_models.append(m)
        
    m2_model = M2Stacker(m2_base_models, meta_clf, device=device)
    # --- M2 (Standalone EfficientNet-B0 for Grad-CAM) ---
    # The user specifically requested this model for M2 heatmaps
    m2_gcam_model = create_model("efficientnet_b0", num_classes=2).to(device)
    if m2_gcam_ckpt.exists():
        m2_gcam_model.load_state_dict(torch.load(m2_gcam_ckpt, map_location=device))
    m2_gcam_model.eval()
    gcam_m2 = GradCAM(m2_gcam_model, get_target_layer(m2_gcam_model, "efficientnet_b0"))

    # --- M3 (DenseNet121) ---
    m3_model = create_model("densenet121", num_classes=2).to(device)
    m3_model.load_state_dict(torch.load(m3_ckpt, map_location=device))
    m3_model.eval()
    gcam_m3 = GradCAM(m3_model, get_target_layer(m3_model, "densenet121"))
    
    # Load config for thr2_val
    # (Removed duplicate load since it's now handled smoothly in M2 setup)
    
    val_tf = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    print("Models Initialized Successfully!", flush=True)

try:
    init_models()
except Exception as e:
    print(f"Failed to intialize models. Please check if the weights exist: {e}")

@app.route("/")
def index():
    return render_template("index.html")

def generate_overlay_base64(gcam_map, original_img, side):
    """Helper function to cleanly generate the overlay base64 string"""
    if side == "R":
        gcam_map = np.fliplr(gcam_map)
    
    img_np = np.array(original_img.resize((384, 384))) / 255.0
    h, w, _ = img_np.shape
    gcam_resized = cv2.resize(gcam_map, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * gcam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.4 * heatmap + 0.6 * img_np, 0, 1)
    
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    buffered = io.BytesIO()
    overlay_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    side = request.form.get("side", "L")
    file = request.files["image"]
    img = Image.open(file.stream).convert('RGB')
    
    # Save a temp image for YOLO 
    img_path = WEB_APP_DIR / "temp.jpg"
    img.save(img_path)
    
    try:
        # 1. YOLO inference
        res = yolo_model.predict(source=str(img_path), conf=0.25, device=device, imgsz=640, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return jsonify({"error": "未偵測到髖關節！請更換圖片或降低信心閾值。"}), 400
        
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        
        if side == "L" and boxes.shape[0] > 0:
            k = boxes[:, 2].argmax()
        elif side == "R" and boxes.shape[0] > 0:
            k = boxes[:, 0].argmin()
        else:
            k = scores.argmax()
            
        x1, y1, x2, y2 = map(int, boxes[k])
        
        cropped_img = img.crop((x1, y1, x2, y2))
        
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="JPEG")
        cropped_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        annotated_img_numpy = res.plot()[..., ::-1] # BGR to RGB
        annotated_img = Image.fromarray(annotated_img_numpy)
        buffered_anno = io.BytesIO()
        annotated_img.save(buffered_anno, format="JPEG")
        annotated_base64 = base64.b64encode(buffered_anno.getvalue()).decode("utf-8")
        
        # ---------------------------------------------------------
        # 2. Hierarchical Classification (Unify orientation to 'L')
        # ---------------------------------------------------------
        inference_img = cropped_img
        if side == "R":
            inference_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
            
        img_tensor = val_tf(inference_img).unsqueeze(0).to(device)
        
        # --- Node-1: 123 vs 4 (Stage 4 check) ---
        m1_logits = m1_model(img_tensor)
        p_stage4 = F.softmax(m1_logits, dim=1)[0, 1].item()
        
        img_tensor_gcam_m1 = img_tensor.clone().requires_grad_(True)
        pred_idx_m1 = m1_logits.argmax(1).item()
        gcam_map_m1 = gcam_m1(img_tensor_gcam_m1, class_idx=pred_idx_m1)
        overlay_m1 = generate_overlay_base64(gcam_map_m1, cropped_img, side)
        
        # --- Node-2: 1 vs 23 (Stage 1 check) ---
        p_stage1_nodal = m2_model.p_stage1(img_tensor).item()
        
        # GCAM for M2 (using standalone efficientnet_b0)
        m2_logits = m2_gcam_model(img_tensor)
        img_tensor_gcam_m2 = img_tensor.clone().requires_grad_(True)
        pred_idx_m2 = 0 if p_stage1_nodal >= thr2_val else 1
        gcam_map_m2 = gcam_m2(img_tensor_gcam_m2, class_idx=pred_idx_m2)
        overlay_m2 = generate_overlay_base64(gcam_map_m2, cropped_img, side)
        
        # --- Node-3: 2 vs 3 ---
        m3_logits = m3_model(img_tensor)
        p_stage3_nodal = F.softmax(m3_logits, dim=1)[0, 1].item()
        
        img_tensor_gcam_m3 = img_tensor.clone().requires_grad_(True)
        pred_idx_m3 = m3_logits.argmax(1).item()
        gcam_map_m3 = gcam_m3(img_tensor_gcam_m3, class_idx=pred_idx_m3)
        overlay_m3 = generate_overlay_base64(gcam_map_m3, cropped_img, side)
        
        # ---------------------------------------------------------
        # 3. Final Logic and Probabilities
        # ---------------------------------------------------------
            
        final_stage = None
        if p_stage4 >= 0.5:
            final_stage = "Stage 4"
        elif p_stage1_nodal >= thr2_val:
            final_stage = "Stage 1"
        elif p_stage3_nodal >= 0.5:
            final_stage = "Stage 3"
        else:
            final_stage = "Stage 2"
            
        # Calculate overall 4 stage probabilities
        p_not_4 = 1.0 - p_stage4
        prob_1 = p_not_4 * p_stage1_nodal
        prob_23 = p_not_4 * (1.0 - p_stage1_nodal)
        prob_3 = prob_23 * p_stage3_nodal
        prob_2 = prob_23 * (1.0 - p_stage3_nodal)
        
        probs = [
            {"label": "第1類", "value": prob_1},
            {"label": "第2類", "value": prob_2},
            {"label": "第3類", "value": prob_3},
            {"label": "第4類", "value": p_stage4}
        ]
        
        # Sort highest to lowest
        probs = sorted(probs, key=lambda x: x["value"], reverse=True)
            
        return jsonify({
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "cropped_image": f"data:image/jpeg;base64,{cropped_base64}",
            "gradcam_m1": f"data:image/jpeg;base64,{overlay_m1}",
            "gradcam_m2": f"data:image/jpeg;base64,{overlay_m2}",
            "gradcam_m3": f"data:image/jpeg;base64,{overlay_m3}",
            "stage": final_stage,
            "sorted_probs": probs
        })
    except Exception as e:
        return jsonify({"error": f"預測時發生錯誤: {str(e)}"}), 500
    finally:
        # Cleanup temp file
        if img_path.exists():
            img_path.unlink()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
