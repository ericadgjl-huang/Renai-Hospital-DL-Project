"""Grad-CAM utility, ported from the original notebooks."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._forward_hook)
        self._bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inp, out):
        self.activations = out.detach()

    def _backward_hook(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())
        score = out[0, class_idx]
        score.backward()

        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = grads.mean(dim=(2, 3))[0]
        cam = torch.zeros(acts.shape[2:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[0, i]
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()
