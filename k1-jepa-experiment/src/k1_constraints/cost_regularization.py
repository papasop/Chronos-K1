"""Cost regularization terms motivated by Axioms R, E, T."""

from __future__ import annotations

import numpy as np


def k1_penalty(x: np.ndarray, G: np.ndarray) -> float:
    K = np.einsum("...i,ij,...j->...", np.asarray(x, dtype=float), np.asarray(G, dtype=float), np.asarray(x, dtype=float))
    return float(np.mean((K - 1.0) ** 2))
