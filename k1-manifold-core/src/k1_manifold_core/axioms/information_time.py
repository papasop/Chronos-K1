"""Information-time helper formulas."""

from __future__ import annotations

from typing import Any

import numpy as np


def information_time(d_phi: Any, H: Any) -> Any:
    """Return ``dt_info = dPhi / H`` for positive scalar cost ``H``."""

    H_array = np.asarray(H, dtype=float)
    if np.any(H_array <= 0.0):
        raise ValueError("H must be positive")
    return np.asarray(d_phi, dtype=float) / H_array
