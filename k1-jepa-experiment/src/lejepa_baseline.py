"""Minimal baseline hooks for future LeJEPA experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    latent_dim: int = 128
    context_dim: int = 128
    learning_rate: float = 1e-3
    epochs: int = 10


def describe_baseline(config: BaselineConfig = BaselineConfig()) -> dict[str, float | int]:
    return {
        "latent_dim": config.latent_dim,
        "context_dim": config.context_dim,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
    }
