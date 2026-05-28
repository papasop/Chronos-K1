"""Projection helpers for K=1 toy latent spaces."""

from __future__ import annotations

import numpy as np


def project_to_k1(x: np.ndarray, G: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Scale vectors to satisfy ``x.T G x = 1`` when the sign permits it."""

    x = np.asarray(x, dtype=float)
    G = np.asarray(G, dtype=float)
    K = np.einsum("...i,ij,...j->...", x, G, x)
    scale = np.sqrt(np.maximum(K, eps))
    return x / scale[..., None]
