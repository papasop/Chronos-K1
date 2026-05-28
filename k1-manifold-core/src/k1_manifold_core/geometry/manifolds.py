"""Lightweight manifold abstractions used by examples and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


Array = np.ndarray


class Metric(Protocol):
    def __call__(self, x: Array) -> Array: ...


@dataclass(frozen=True)
class LocalChart:
    name: str
    dimension: int


@dataclass(frozen=True)
class PseudoRiemannianManifold:
    chart: LocalChart
    metric: Metric

    def metric_at(self, x: Array) -> Array:
        G = np.asarray(self.metric(np.asarray(x, dtype=float)), dtype=float)
        if G.shape != (self.chart.dimension, self.chart.dimension):
            raise ValueError("metric shape does not match chart dimension")
        return G
