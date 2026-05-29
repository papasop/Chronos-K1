"""OOD light-cone classification benchmark.

Train on event differences sampled from a small box and test on larger boxes.
This is a research benchmark, not a unit test. It compares an explicit
Lorentzian inductive bias against Euclidean baselines on a synthetic
timelike/spacelike classification task.

Run from ``k1-manifold-core``:

    python benchmarks/ood_extrapolation.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

N_SPACE = 2
K = 8
WIDTH = 64
EPOCHS = 120
LEARNING_RATE = 0.01

OOD_TEST_BOXES = (2.0, 4.0, 8.0, 12.0)
OOD_N_TRAIN = 3000
OOD_N_TEST = 3000
OOD_SEEDS = 10


def make_events(n: int, n_space: int, rng: np.random.Generator, *, box: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample event differences and label them by the Lorentz interval."""

    dim = n_space + 1
    X = rng.uniform(-box, box, size=(n, dim)).astype(np.float32)
    interval = X[:, 0] * X[:, 0] - np.sum(X[:, 1:] * X[:, 1:], axis=1)
    y = (interval >= 0.0).astype(np.float32)
    return X, y, dim


def roc_auc_score_np(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return binary ROC AUC using average ranks for ties."""

    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(y) & np.isfinite(s)
    y = y[finite]
    s = s[finite]
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == 0.0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = float(np.sum(original_ranks[y == 1.0]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Return Wilcoxon p-value with finite-value and zero-gap guards."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    gap = a[mask] - b[mask]
    if np.allclose(gap, 0.0):
        return 1.0
    _stat, p = wilcoxon(a[mask], b[mask])
    return float(p)


class LorentzMetric(nn.Module):
    """Explicit Lorentzian score ``scale * (dt^2 - ||dx||^2) + bias``."""

    def __init__(self, dim: int, _k: int, _width: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        interval = x[:, 0] * x[:, 0] - torch.sum(x[:, 1:dim] * x[:, 1:dim], dim=1)
        return torch.exp(self.log_scale) * interval + self.bias


class EuclideanMahalanobis(nn.Module):
    """Euclidean radial baseline with positive quadratic weights."""

    def __init__(self, dim: int, _k: int, _width: int) -> None:
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        weights = torch.nn.functional.softplus(self.raw_weights[:dim])
        return self.bias - torch.sum(weights * x[:, :dim] * x[:, :dim], dim=1)


class EuclideanMLP(nn.Module):
    """Unstructured Euclidean MLP baseline on raw event coordinates."""

    def __init__(self, dim: int, _k: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return self.net(x[:, :dim]).squeeze(-1)


MODELS = {
    "lorentz": LorentzMetric,
    "euclid_mahalanobis": EuclideanMahalanobis,
    "euclid_mlp": EuclideanMLP,
}


def summarize_ood_results(results: dict[str, dict[str, list[float]]]) -> dict[str, object]:
    """Summarize AUCs and Lorentz-vs-MLP gaps."""

    summary: dict[str, object] = {}
    for box in OOD_TEST_BOXES:
        key = str(box)
        summary[key] = {}
        for name in MODELS:
            arr = np.asarray(results[name][key], dtype=float)
            summary[key][name] = {
                "mean_auc": float(np.nanmean(arr)),
                "std_auc": float(np.nanstd(arr)),
                "n_valid": int(np.isfinite(arr).sum()),
            }

        lo = np.asarray(results["lorentz"][key], dtype=float)
        mlp = np.asarray(results["euclid_mlp"][key], dtype=float)
        gap = lo - mlp
        summary[key]["lorentz_minus_euclid_mlp"] = {
            "mean_gap": float(np.nanmean(gap)),
            "median_gap": float(np.nanmedian(gap)),
            "wilcoxon_p": safe_wilcoxon(lo, mlp),
        }
    return summary


def train_model(model: nn.Module, Xtr_t: torch.Tensor, ytr_t: torch.Tensor, *, dim: int) -> nn.Module:
    """Train one classifier with full-batch Adam."""

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t, dim).squeeze(-1)
        loss = bce(logits, ytr_t)
        loss.backward()
        opt.step()
    return model


def run_ood_extrapolation() -> dict[str, object]:
    """Run the OOD extrapolation benchmark and write JSON/PNG artifacts."""

    print("\n=== EXPERIMENT 2: OOD EXTRAPOLATION ===")
    print(f"Train box = 2.0; Test boxes = {OOD_TEST_BOXES}")
    results = {name: {str(box): [] for box in OOD_TEST_BOXES} for name in MODELS}

    for seed in range(OOD_SEEDS):
        rng_train = np.random.default_rng(9000 + seed)
        X_tr, y_tr, dim = make_events(OOD_N_TRAIN, N_SPACE, rng_train, box=2.0)
        Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
        ytr_t = torch.tensor(y_tr, dtype=torch.float32)

        trained_models = {}
        for name, model_type in MODELS.items():
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_type(dim, K, WIDTH)
            trained_models[name] = train_model(model, Xtr_t, ytr_t, dim=dim)

        for test_box in OOD_TEST_BOXES:
            rng_test = np.random.default_rng(10000 + seed + int(test_box * 100))
            X_te, y_te, _dim = make_events(OOD_N_TEST, N_SPACE, rng_test, box=test_box)
            Xte_t = torch.tensor(X_te, dtype=torch.float32)

            for name, model in trained_models.items():
                model.eval()
                with torch.no_grad():
                    scores = model(Xte_t, dim).squeeze(-1).detach().cpu().numpy()
                auc = roc_auc_score_np(y_te, scores)
                results[name][str(test_box)].append(float(auc))
        print(f"seed {seed:2d} done")

    summary = summarize_ood_results(results)

    print("\n=== OOD AUC mean +/- std ===")
    for box in OOD_TEST_BOXES:
        key = str(box)
        print(f"\nTest box = {box:g}")
        for name in MODELS:
            s = summary[key][name]
            print(f"  {name:18s}: {s['mean_auc']:.4f} +/- {s['std_auc']:.4f} (n={s['n_valid']})")
        gap = summary[key]["lorentz_minus_euclid_mlp"]
        print(
            "  Lorentz - Euclid_MLP: "
            f"mean_gap={gap['mean_gap']:+.4f} "
            f"median_gap={gap['median_gap']:+.4f} "
            f"p={gap['wilcoxon_p']:.4f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "ood_extrapolation_lightcone_classification",
        "description": "Train on event differences from box=2 and test on larger boxes for timelike/spacelike classification.",
        "scope": "Research benchmark; not a unit test and not a broad world-model claim.",
        "config": {
            "train_box": 2.0,
            "test_boxes": list(OOD_TEST_BOXES),
            "n_train": OOD_N_TRAIN,
            "n_test": OOD_N_TEST,
            "seeds": OOD_SEEDS,
            "n_space": N_SPACE,
            "k": K,
            "width": WIDTH,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        "raw": results,
        "summary": summary,
    }

    json_path = RESULTS_DIR / "ood_extrapolation.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(7, 4))
    for name in MODELS:
        means = [summary[str(box)][name]["mean_auc"] for box in OOD_TEST_BOXES]
        stds = [summary[str(box)][name]["std_auc"] for box in OOD_TEST_BOXES]
        plt.errorbar(OOD_TEST_BOXES, means, yerr=stds, marker="o", label=name)

    plt.xlabel("test box size")
    plt.ylabel("test AUC")
    plt.title("OOD extrapolation: train box=2, test on larger boxes")
    plt.xscale("log")
    plt.ylim(0.5, 1.02)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    fig_path = RESULTS_DIR / "ood_extrapolation_auc.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print(f"\nsaved results = {json_path}")
    print(f"saved figure = {fig_path}")
    return payload


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    run_ood_extrapolation()
