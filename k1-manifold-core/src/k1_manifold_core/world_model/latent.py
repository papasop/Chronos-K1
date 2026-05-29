"""Latent-state helpers for K=1 world-model experiments."""

from __future__ import annotations

import numpy as np


Array = np.ndarray
DEFAULT_G = np.diag([1.0, -1.0])


def as_latent_batch(z: Array) -> Array:
    """Return ``z`` as an ``(n, 2)`` latent-state batch."""

    z = np.asarray(z, dtype=float)
    if z.shape == (2,):
        return z.reshape(1, 2)
    if z.ndim == 2 and z.shape[1] == 2:
        return z
    raise ValueError("latent states must have shape (2,) or (n, 2)")


def k_values(z: Array, G: Array = DEFAULT_G) -> Array:
    """Return ``K(z)=z.T @ G @ z`` for each latent state."""

    z_batch = as_latent_batch(z)
    G = np.asarray(G, dtype=float)
    return np.einsum("bi,ij,bj->b", z_batch, G, z_batch)


def k_constraint_error(z: Array, G: Array = DEFAULT_G) -> Array:
    """Return ``K(z)-1`` for each latent state."""

    return k_values(z, G) - 1.0


def project_to_k1(z: Array, G: Array = DEFAULT_G, *, eps: float = 1e-12) -> Array:
    """Project positive-timelike latent states onto ``K=1`` by radial scaling.

    This projection is intentionally minimal and only applies to states with
    ``K(z)>0``. It is a numerical regularizer for toy latent benchmarks, not a
    general geometric projection theorem.
    """

    z_array = np.asarray(z, dtype=float)
    z_batch = as_latent_batch(z_array)
    K = k_values(z_batch, G)
    if np.any(K <= eps):
        raise ValueError("project_to_k1 requires K(z) > eps for all states")
    projected = z_batch / np.sqrt(K)[:, None]
    if z_array.shape == (2,):
        return projected[0]
    return projected
