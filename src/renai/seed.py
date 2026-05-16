"""Single source of truth for seeding.

The whole project pins seed=42 so that every cut / fold / backbone / topology
result is directly comparable across runs.  Call `set_seed(SEED)` at the very
top of every script and at the start of every training run."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def torch_generator(seed: int = SEED) -> torch.Generator:
    return torch.Generator().manual_seed(seed)
