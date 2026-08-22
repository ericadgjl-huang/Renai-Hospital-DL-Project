"""CORN: rank-consistent ordinal head for 4-stage Ficat (idea: idea #ordinal).

Why this is the strongest single lever
--------------------------------------
The current pipeline decomposes 4-class into 10 *independent* binary CNNs. That
throws away two things an ordinal model keeps:
  * a SHARED representation (one backbone learns features useful for every
    threshold, instead of 10 backbones each re-learning from ~230 images), and
  * RANK CONSISTENCY — P(stage>=2) >= P(stage>=3) >= P(stage>=4) by construction,
    which the independent cuts routinely violate.

CORN (Shi, Cao & Raschka, 2021/2023 — "Deep NNs for rank-consistent ordinal
regression based on conditional probabilities") trains K-1 = 3 thresholds via
*conditional* binary cross-entropy and recovers unconditional cumulative
probabilities with the chain rule, giving guaranteed rank consistency without
CORAL's restrictive weight sharing. Decoding is the standard rank count.

This module implements the loss/decoding directly (no extra dependency) and
reuses the project's backbone factory, so a CORN model is a drop-in single
backbone with a 3-logit head.

Labels here are 1-based stages (1..4); internally rank r = stage - 1 in {0..3}
and the K-1=3 thresholds are r>0, r>1, r>2  (== stage>=2, >=3, >=4).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_THRESH = 3        # K-1 for K=4 stages


def corn_loss(logits: torch.Tensor, stages_1based: torch.Tensor,
              pos_weights: torch.Tensor | None = None,
              stage_weights: torch.Tensor | None = None) -> torch.Tensor:
    """CORN conditional-BCE loss.

    logits: (B, 3) — one logit per threshold (stage>=2, >=3, >=4).
    stages_1based: (B,) integer stages in 1..4.
    pos_weights: optional (3,) tensor; pos_weights[t] up-weights the POSITIVE
        class of threshold t (== BCEWithLogits `pos_weight`). Counters class
        imbalance across thresholds.
    stage_weights: optional (4,) tensor indexed by (stage-1); multiplies each
        SAMPLE's loss by its true stage's weight. Use e.g. [1,1,W,1] to make the
        model care more about getting stage-3 samples right (stage 3 is the
        hardest adjacent boundary and the weakest recall). Applies at every
        threshold the sample participates in.

    Threshold t (0-based) predicts stage >= t+2, but is trained ONLY on the
    conditional subset that "reached" the previous threshold:
      t=0: all samples,
      t=1: samples with stage >= 2,
      t=2: samples with stage >= 3.
    Empty subsets (possible in a tiny imbalanced batch) contribute 0.
    """
    y = stages_1based.long()
    total = logits.new_zeros(())
    n_terms = 0
    for t in range(NUM_THRESH):
        need_stage = t + 2                     # this threshold asks "stage >= need_stage"
        if t == 0:
            mask = torch.ones_like(y, dtype=torch.bool)
        else:
            mask = y >= (t + 1)                # reached previous threshold (stage >= t+1)
        if mask.sum() == 0:
            continue
        ym = y[mask]
        target = (ym >= need_stage).float()
        pw = None if pos_weights is None else pos_weights[t]
        if stage_weights is None:
            loss_t = F.binary_cross_entropy_with_logits(
                logits[mask, t], target, reduction="mean", pos_weight=pw)
        else:
            per = F.binary_cross_entropy_with_logits(
                logits[mask, t], target, reduction="none", pos_weight=pw)
            sw = stage_weights[ym - 1]         # per-sample weight by true stage
            loss_t = (per * sw).sum() / sw.sum().clamp_min(1e-8)
        total = total + loss_t
        n_terms += 1
    return total / max(n_terms, 1)


def corn_pos_weights(stages_1based, device, cap: float = 8.0) -> torch.Tensor:
    """Per-threshold pos_weight = (#neg / #pos) on the conditional subset, from
    the FULL training labels (stable, unlike per-batch). Capped to avoid blow-up
    when a positive class is extremely rare."""
    y = torch.as_tensor(stages_1based).long()
    w = []
    for t in range(NUM_THRESH):
        need = t + 2
        sub = y if t == 0 else y[y >= (t + 1)]
        pos = int((sub >= need).sum())
        neg = int(len(sub) - pos)
        w.append(min(cap, neg / max(pos, 1)))
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def corn_cumulative_probs(logits: torch.Tensor) -> torch.Tensor:
    """Unconditional cumulative probs via the chain rule.

    P(stage>=2)          = sigmoid(f0)
    P(stage>=3)          = sigmoid(f0) * sigmoid(f1)
    P(stage>=4)          = sigmoid(f0) * sigmoid(f1) * sigmoid(f2)
    Returns (B, 3), automatically non-increasing across columns -> rank
    consistent."""
    s = torch.sigmoid(logits)
    return torch.cumprod(s, dim=1)


@torch.no_grad()
def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """Predicted 1-based stage = 1 + #{cumulative prob > 0.5}."""
    probs = corn_cumulative_probs(logits)
    return 1 + (probs > 0.5).sum(dim=1)


def create_corn_model(backbone: str, radimagenet_dir=None):
    """A backbone whose final linear outputs NUM_THRESH=3 CORN logits.

    If ``radimagenet_dir`` is given, RadImageNet medical-pretrained weights are
    loaded into the backbone first (resnet50 / densenet121 only); the CORN head
    is then attached fresh."""
    from .models import create_model, load_radimagenet_weights
    model = create_model(backbone, num_classes=NUM_THRESH)
    if radimagenet_dir:
        load_radimagenet_weights(model, backbone, radimagenet_dir)
    return model
