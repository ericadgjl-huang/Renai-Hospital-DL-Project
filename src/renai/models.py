"""Backbone factory.

The pool intentionally spans four families (EfficientNet, ResNet, ConvNeXt,
DenseNet) so that the fold-voting ensemble averages across diverse inductive
biases. DenseNet121/169 were the workhorses of the earlier "調整8" experiments,
so they are kept in the pool."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

# RadImageNet (Mei et al., Radiology:AI 2022) provides medical-image pretrained
# weights for ResNet50 and DenseNet121 (+ Inception variants). On small
# radiology datasets it beats ImageNet transfer by ~0.9-9.4% AUC. Weights are
# stored with a Sequential "backbone." prefix; these maps translate them onto
# torchvision's named modules.
RADIMAGENET_FILES = {"resnet50": "ResNet50.pt", "densenet121": "DenseNet121.pt"}
_RESNET_IDX_TO_NAME = {0: "conv1", 1: "bn1", 4: "layer1",
                       5: "layer2", 6: "layer3", 7: "layer4"}


def load_radimagenet_weights(model: nn.Module, backbone: str, weights_dir) -> int:
    """Load RadImageNet medical-pretrained weights into a torchvision backbone,
    IN PLACE, before its classifier head is swapped. Returns the number of
    tensors successfully loaded (0 = nothing matched -> stays on ImageNet).

    Only resnet50 / densenet121 are covered (the two RadImageNet models in this
    project's pool); other backbones are left on their ImageNet weights."""
    name = backbone.lower()
    fn = RADIMAGENET_FILES.get(name)
    if fn is None:
        print(f"  [radimagenet] {backbone}: no RadImageNet weights available "
              f"-> keeping ImageNet.", flush=True)
        return 0
    path = Path(weights_dir) / fn
    if not path.exists():
        print(f"  [radimagenet] {backbone}: {path} not found -> keeping ImageNet.", flush=True)
        return 0

    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    remapped = {}
    for k, v in raw.items():
        if not k.startswith("backbone."):
            continue
        rest = k[len("backbone."):]
        if name == "resnet50":
            head, tail = rest.split(".", 1)
            nm = _RESNET_IDX_TO_NAME.get(int(head))
            if nm is None:
                continue
            remapped[f"{nm}.{tail}"] = v
        else:  # densenet121: backbone.0.* -> features.*
            if rest.startswith("0."):
                remapped[f"features.{rest[2:]}"] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    loaded = len(remapped) - len(set(remapped) & set(unexpected))
    print(f"  [radimagenet] {backbone}: loaded {loaded}/{len(remapped)} tensors "
          f"(unexpected={len(unexpected)}); classifier head stays random.", flush=True)
    return loaded


DEFAULT_BACKBONES: tuple[str, ...] = (
    "efficientnet_b0",
    "efficientnet_b1",
    "resnet50",
    "convnext_tiny",
    "convnext_small",
    "densenet121",
    "densenet169",
)


def create_model(model_name: str, num_classes: int = 2) -> nn.Module:
    name = model_name.lower()

    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "efficientnet_b1":
        m = tvm.efficientnet_b1(weights=tvm.EfficientNet_B1_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if name == "convnext_small":
        m = tvm.convnext_small(weights=tvm.ConvNeXt_Small_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    if name == "densenet169":
        m = tvm.densenet169(weights=tvm.DenseNet169_Weights.DEFAULT)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    raise ValueError(f"Unknown backbone: {model_name}")


def get_target_layer(model: nn.Module, model_name: str):
    """Layer to hook for Grad-CAM."""
    name = model_name.lower()
    if name in {"efficientnet_b0", "efficientnet_b1"}:
        return model.features[-1]
    if name == "resnet50":
        return model.layer4
    if name in {"convnext_tiny", "convnext_small"}:
        return model.features[-1]
    if name in {"densenet121", "densenet169"}:
        # NOT norm5: DenseNet.forward applies an in-place ReLU to the features
        # output, which clashes with the backward hook. denseblock4's output is
        # consumed by norm5 (a fresh tensor), so it is safe to hook.
        return model.features.denseblock4
    raise ValueError(f"No Grad-CAM target layer rule for {model_name}")
