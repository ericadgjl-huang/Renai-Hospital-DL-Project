"""Backbone factory.

The pool intentionally spans four families (EfficientNet, ResNet, ConvNeXt,
DenseNet) so that the fold-voting ensemble averages across diverse inductive
biases. DenseNet121/169 were the workhorses of the earlier "調整8" experiments,
so they are kept in the pool."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

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
