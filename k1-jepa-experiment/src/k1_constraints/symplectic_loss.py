"""J_G consistency losses for exploratory models."""

from __future__ import annotations

import numpy as np


def symplectic_consistency_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(predicted, dtype=float) - np.asarray(target, dtype=float)
    return float(np.mean(diff * diff))
