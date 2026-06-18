"""Centralized RNG seeding for reproducible runs.

The simulation draws randomness from three sources: Python's global ``random``
module (combat resolution, comms noise, threat spawning), NumPy (terrain,
spawn positions), and PyTorch (network init, action sampling). Seeding only
one of them — as the original code did — leaves runs non-deterministic.
``set_seed`` seeds all three from a single call.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
