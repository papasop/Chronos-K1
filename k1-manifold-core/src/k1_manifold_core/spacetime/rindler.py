"""Rindler wedge formulas."""

from __future__ import annotations

import numpy as np

from k1_manifold_core.geometry.symplectic import spectral_threshold
from k1_manifold_core.thermodynamics.temperature import hawking_temperature, tolman_temperature


def rindler_metric(kappa: float, ell: float) -> np.ndarray:
    """Return spacetime-sign Rindler metric ``diag(-(kappa ell)^2, 1)``."""

    if kappa <= 0.0 or ell <= 0.0:
        raise ValueError("kappa and ell must be positive")
    return np.diag([-(kappa * ell) ** 2, 1.0])


def rindler_cost_form(kappa: float, ell: float) -> np.ndarray:
    """Return cost-sign form ``diag((kappa ell)^2, -1)``."""

    if kappa <= 0.0 or ell <= 0.0:
        raise ValueError("kappa and ell must be positive")
    return np.diag([(kappa * ell) ** 2, -1.0])


def rindler_dc(kappa: float, ell: float, *, alpha: float = 1.0) -> float:
    return spectral_threshold(rindler_cost_form(kappa, ell), alpha=alpha)


def tolman_identity_product(kappa: float, ell: float, *, alpha: float = 1.0, hbar: float = 1.0) -> float:
    """Return ``d_c * T_H`` for the Rindler wedge."""

    return rindler_dc(kappa, ell, alpha=alpha) * hawking_temperature(kappa, hbar=hbar)


def expected_tolman(kappa: float, ell: float, *, alpha: float = 1.0, hbar: float = 1.0) -> float:
    """Return the Tolman temperature; kappa is accepted for API symmetry."""

    _ = kappa
    return tolman_temperature(ell, alpha=alpha, hbar=hbar)
