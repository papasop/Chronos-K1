"""Experiment 5b: causal mechanism ablation.

This is a research benchmark, not a unit test. It decomposes the Chronos
latent predictor used in Experiment 5 into mechanism variants:

- Euclidean latent predictor
- Chronos geometry-only latent predictor
- Chronos causal-only latent predictor
- Chronos interval-only latent predictor
- Chronos full latent predictor

The goal is to probe which pieces of the Chronos constraint stack contribute
to decoded causal consistency under OOD rollout stress.

The models are JEPA-style in the narrow sense that they predict future
embeddings. They are not implementations of Meta/LeCun JEPA.

Run from ``k1-manifold-core``:

    python benchmarks/experiment_5b_causal_mechanism_ablation.py
    python benchmarks/experiment_5b_causal_mechanism_ablation.py --smoke
    python benchmarks/experiment_5b_causal_mechanism_ablation.py --full
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chronos-K1 Experiment 5b mechanism ablation.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny CPU-friendly smoke configuration.")
    parser.add_argument("--full", action="store_true", help="Run the larger benchmark configuration.")
    args = parser.parse_args()
    if args.smoke and args.full:
        parser.error("--smoke and --full cannot be used together")
    return args


ARGS = parse_args()
RUN_MODE = "smoke" if ARGS.smoke else "full" if ARGS.full else "default"

if RUN_MODE == "smoke":
    N_SEEDS = 1
    N_TRAIN = 120
    N_TEST = 64
    EPOCHS = 8
elif RUN_MODE == "default":
    N_SEEDS = 5
    N_TRAIN = 1000
    N_TEST = 256
    EPOCHS = 120
else:
    N_SEEDS = 10
    N_TRAIN = 3000
    N_TEST = 512
    EPOCHS = 250

N_SPACE = 3
DIM = 1 + N_SPACE
LATENT_DIM = 16
WIDTH = 64
LR = 1e-3

T_TOTAL = 80
T_OBS = 10
ROLL_STEPS = 50

TRAIN_BOX = 2.0
TEST_BOXES = (2.0, 32.0) if RUN_MODE == "smoke" else (2.0, 8.0, 32.0)

REQUESTED_DEVICE = os.environ.get("CHRONOS_DEVICE", "cpu")
if REQUESTED_DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CHRONOS_DEVICE=cuda requested, but CUDA is not available")
DEVICE = REQUESTED_DEVICE

torch.set_num_threads(2)


ABLATIONS: dict[str, dict[str, float | str]] = {
    "euclidean_latent_predictor": {
        "type": "euclidean",
        "lambda_interval": 0.0,
        "lambda_causal": 0.0,
        "lambda_k1": 0.0,
    },
    "chronos_geometry_only": {
        "type": "chronos",
        "lambda_interval": 0.0,
        "lambda_causal": 0.0,
        "lambda_k1": 0.0,
    },
    "chronos_causal_only": {
        "type": "chronos",
        "lambda_interval": 0.0,
        "lambda_causal": 0.1,
        "lambda_k1": 0.0,
    },
    "chronos_interval_only": {
        "type": "chronos",
        "lambda_interval": 0.1,
        "lambda_causal": 0.0,
        "lambda_k1": 0.0,
    },
    "chronos_full": {
        "type": "chronos",
        "lambda_interval": 0.1,
        "lambda_causal": 0.1,
        "lambda_k1": 0.02,
    },
}


def make_lorentz_oscillator_trajs(
    n_seq: int,
    t_total: int,
    dim: int,
    rng: np.random.Generator,
    *,
    box: float = 2.0,
    dt: float = 0.08,
    noise: float = 0.002,
) -> np.ndarray:
    """Generate synthetic causal-like Lorentzian oscillator trajectories."""

    x = rng.uniform(-box, box, size=(n_seq, dim)).astype(np.float32)

    v = rng.normal(size=(n_seq, dim)).astype(np.float32)
    v[:, 0] = np.abs(v[:, 0]) + 1.5 * np.linalg.norm(v[:, 1:], axis=1)
    v *= 0.04

    omega = rng.uniform(0.6, 1.4, size=(n_seq, 1)).astype(np.float32)
    traj = [x.copy()]

    for _ in range(t_total - 1):
        a = np.zeros_like(x)
        a[:, 1:] = -omega * omega * x[:, 1:]

        v[:, 1:] = 0.995 * v[:, 1:] + dt * a[:, 1:]
        spatial_speed = np.linalg.norm(v[:, 1:], axis=1, keepdims=True)
        v[:, 0:1] = np.abs(v[:, 0:1]) + 0.03 + 1.05 * spatial_speed

        x = x + dt * v + noise * rng.normal(size=x.shape).astype(np.float32)
        traj.append(x.copy())

    return np.stack(traj, axis=1).astype(np.float32)


def make_pairs(trajs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one-step prediction pairs ``x_t -> x_{t+1}``."""

    x_t = trajs[:, :-1, :]
    x_next = trajs[:, 1:, :]
    return (
        x_t.reshape(-1, DIM).astype(np.float32),
        x_next.reshape(-1, DIM).astype(np.float32),
    )


def torch_lorentz_interval(dz: torch.Tensor) -> torch.Tensor:
    return dz[:, 0] ** 2 - torch.sum(dz[:, 1:] ** 2, dim=1)


def np_lorentz_interval(dz: np.ndarray) -> np.ndarray:
    return dz[..., 0] ** 2 - np.sum(dz[..., 1:] ** 2, axis=-1)


def causal_violation_by_t_np(traj: np.ndarray) -> np.ndarray:
    delta = traj[:, 1:, :] - traj[:, :-1, :]
    interval = np_lorentz_interval(delta)
    return np.mean(interval < 0.0, axis=0)


def interval_drift_np(pred: np.ndarray, target: np.ndarray) -> float:
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dt = target[:, 1:, :] - target[:, :-1, :]
    return float(np.mean(np.abs(np_lorentz_interval(dp) - np_lorentz_interval(dt))))


def effective_rank_np(z: np.ndarray, eps: float = 1e-12) -> float:
    z = np.asarray(z, dtype=float)
    z = z - z.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(z, compute_uv=False)
    weights = singular_values / (singular_values.sum() + eps)
    entropy = -np.sum(weights * np.log(weights + eps))
    return float(np.exp(entropy))


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return None
    diff = a[mask] - b[mask]
    if np.allclose(diff, 0.0):
        return 1.0
    _stat, p = wilcoxon(a[mask], b[mask])
    return float(p)


class Encoder(nn.Module):
    def __init__(self, dim: int, k: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, k: int, dim: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EuclideanLatentPredictor(nn.Module):
    def __init__(self, dim: int, k: int, width: int) -> None:
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)
        self.pred = nn.Sequential(
            nn.Linear(k, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, k),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def predict_z(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        dz = self.pred(z)
        return z + dz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.predict_z(x))

    def rollout(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        xs = [x0]
        x = x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)


class ChronosLatentPredictor(nn.Module):
    def __init__(self, dim: int, k: int, width: int) -> None:
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)

        g = torch.ones(k)
        g[1:] = -1.0
        self.register_buffer("g", g)

        self.pred = nn.Sequential(
            nn.Linear(k, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, k),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def latent_interval(self, dz: torch.Tensor) -> torch.Tensor:
        return torch.sum(dz * dz * self.g, dim=1)

    def k_value(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sum(z * z * self.g, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def predict_z(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        raw = self.pred(z)
        q = self.latent_interval(raw).unsqueeze(1)
        denom = torch.sqrt(torch.clamp(torch.abs(q), min=1e-3))
        dz = self.scale * raw / denom
        return z + dz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.predict_z(x))

    def rollout(self, x0: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [x0]
        zs = []

        x = x0
        for _ in range(steps + 1):
            z = self.encode(x)
            zs.append(z)
            if len(xs) < steps + 1:
                x = self.forward(x)
                xs.append(x)

        return torch.stack(xs, dim=1), torch.stack(zs, dim=1)


def train_model(
    model: nn.Module,
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    config: dict[str, float | str],
) -> tuple[nn.Module, dict[str, list[float | None]]]:
    model.to(DEVICE)

    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    history: dict[str, list[float | None]] = {"loss": [], "mean_abs_k_minus_1": []}

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()

        z_t = model.encode(x_train)
        z_next_true = model.encode(y_train).detach()
        z_next_pred = model.predict_z(x_train)
        x_next_pred = model.dec(z_next_pred)

        loss = mse(x_next_pred, y_train) + mse(z_next_pred, z_next_true)
        mean_abs_k: float | None = None

        if isinstance(model, ChronosLatentPredictor):
            dz_pred = z_next_pred - z_t
            interval_pred = model.latent_interval(dz_pred)
            interval_true = torch_lorentz_interval(y_train - x_train).detach()

            lambda_interval = float(config["lambda_interval"])
            lambda_causal = float(config["lambda_causal"])
            lambda_k1 = float(config["lambda_k1"])

            if lambda_interval > 0:
                loss = loss + lambda_interval * mse(interval_pred, interval_true)

            if lambda_causal > 0:
                loss = loss + lambda_causal * torch.relu(-interval_pred).mean()

            k_next = model.k_value(z_next_pred)
            if lambda_k1 > 0:
                loss = loss + lambda_k1 * torch.mean((k_next - 1.0) ** 2)

            with torch.no_grad():
                mean_abs_k = float(torch.mean(torch.abs(k_next - 1.0)).detach().cpu().item())

        loss.backward()
        opt.step()

        if epoch % max(1, EPOCHS // 20) == 0 or epoch == EPOCHS - 1:
            history["loss"].append(float(loss.detach().cpu().item()))
            history["mean_abs_k_minus_1"].append(mean_abs_k)

    return model, history


def eval_model(model: nn.Module, trajs: np.ndarray, rank_sample_np: np.ndarray) -> dict[str, Any]:
    model.eval()

    x0_np = trajs[:, T_OBS, :]
    target_np = trajs[:, T_OBS : T_OBS + ROLL_STEPS + 1, :]

    x0 = torch.tensor(x0_np, dtype=torch.float32, device=DEVICE)
    rank_sample = torch.tensor(rank_sample_np, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        if isinstance(model, ChronosLatentPredictor):
            pred, z_roll = model.rollout(x0, ROLL_STEPS)
            pred_np = pred.detach().cpu().numpy()
            z_roll_np = z_roll.detach().cpu().numpy()
            k_by_t = z_roll_np[..., 0] ** 2 - np.sum(z_roll_np[..., 1:] ** 2, axis=-1)
            mean_abs_k_minus_1_by_t: list[float | None] = np.mean(
                np.abs(k_by_t - 1.0), axis=0
            ).tolist()
        else:
            pred = model.rollout(x0, ROLL_STEPS)
            pred_np = pred.detach().cpu().numpy()
            mean_abs_k_minus_1_by_t = [None] * (ROLL_STEPS + 1)

        z_sample = model.encode(rank_sample).detach().cpu().numpy()
        encoder_effective_rank = effective_rank_np(z_sample)

    mse_by_t = np.mean((pred_np - target_np) ** 2, axis=(0, 2))
    violation_by_t = causal_violation_by_t_np(pred_np)

    return {
        "mean_rollout_mse": float(np.mean(mse_by_t)),
        "final_rollout_mse": float(mse_by_t[-1]),
        "causal_violation_rate": float(np.mean(violation_by_t)),
        "interval_drift": interval_drift_np(pred_np, target_np),
        "encoder_effective_rank": encoder_effective_rank,
        "mse_by_t": mse_by_t.tolist(),
        "violation_by_t": violation_by_t.tolist(),
        "mean_abs_k_minus_1_by_t": mean_abs_k_minus_1_by_t,
    }


def metric_array(raw: dict[str, dict[str, list[dict[str, Any]]]], name: str, box: float, metric: str) -> np.ndarray:
    return np.array([record[metric] for record in raw[name][str(box)]], dtype=float)


def compare_against_euclidean(
    raw: dict[str, dict[str, list[dict[str, Any]]]],
    name: str,
    box: float,
    metric: str,
) -> dict[str, float]:
    baseline = metric_array(raw, "euclidean_latent_predictor", box, metric)
    candidate = metric_array(raw, name, box, metric)
    improvement = (baseline - candidate) / (baseline + 1e-12)
    return {
        "relative_improvement_mean": float(np.nanmean(improvement)),
        "relative_improvement_median": float(np.nanmedian(improvement)),
        "wilcoxon_p": safe_wilcoxon(baseline, candidate),
    }


def run_experiment_5b() -> tuple[pd.DataFrame, dict[str, Any]]:
    rollout_raw: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {str(box): [] for box in TEST_BOXES} for name in ABLATIONS
    }
    history_raw: dict[str, list[dict[str, list[float | None]]]] = {name: [] for name in ABLATIONS}

    for seed in range(N_SEEDS):
        print(f"\nseed {seed}", flush=True)

        rng_train = np.random.default_rng(70000 + seed)
        train_trajs = make_lorentz_oscillator_trajs(
            N_TRAIN,
            T_TOTAL,
            DIM,
            rng_train,
            box=TRAIN_BOX,
        )
        x_train, y_train = make_pairs(train_trajs)

        trained: dict[str, nn.Module] = {}
        for name, config in ABLATIONS.items():
            torch.manual_seed(seed)
            np.random.seed(seed)

            if config["type"] == "euclidean":
                model: nn.Module = EuclideanLatentPredictor(DIM, LATENT_DIM, WIDTH)
            else:
                model = ChronosLatentPredictor(DIM, LATENT_DIM, WIDTH)

            model, history = train_model(model, x_train, y_train, config)
            trained[name] = model
            history_raw[name].append(history)

        for box in TEST_BOXES:
            rng_test = np.random.default_rng(80000 + seed + int(box * 100))
            test_trajs = make_lorentz_oscillator_trajs(
                N_TEST,
                T_TOTAL,
                DIM,
                rng_test,
                box=box,
            )
            rank_sample = test_trajs[:, T_OBS, :]

            for name, model in trained.items():
                rollout_raw[name][str(box)].append(eval_model(model, test_trajs, rank_sample))

    rows: list[dict[str, Any]] = []
    for box in TEST_BOXES:
        for name in ABLATIONS:
            row: dict[str, Any] = {"box": box, "ablation": name}

            for metric in (
                "final_rollout_mse",
                "causal_violation_rate",
                "interval_drift",
                "encoder_effective_rank",
            ):
                values = metric_array(rollout_raw, name, box, metric)
                row[f"{metric}_mean"] = float(np.nanmean(values))
                row[f"{metric}_std"] = float(np.nanstd(values))

            rows.append(row)

        for name in ABLATIONS:
            if name == "euclidean_latent_predictor":
                continue

            for metric in ("final_rollout_mse", "causal_violation_rate", "interval_drift"):
                comparison = compare_against_euclidean(rollout_raw, name, box, metric)
                rows.append(
                    {
                        "box": box,
                        "ablation": f"{name}_vs_euclidean",
                        "metric": metric,
                        **comparison,
                    }
                )

    df = pd.DataFrame(rows)

    payload: dict[str, Any] = {
        "benchmark": "experiment_5b_causal_mechanism_ablation",
        "scope": "research benchmark, not unit test",
        "description": (
            "Mechanism ablation for Chronos latent predictors. It separates "
            "Lorentzian latent-step geometry from causal, interval, and K=1 "
            "regularization losses on synthetic Lorentzian oscillator rollouts."
        ),
        "config": {
            "run_mode": RUN_MODE,
            "n_seeds": N_SEEDS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "epochs": EPOCHS,
            "train_box": TRAIN_BOX,
            "test_boxes": list(TEST_BOXES),
            "t_total": T_TOTAL,
            "t_obs": T_OBS,
            "roll_steps": ROLL_STEPS,
            "dim": DIM,
            "latent_dim": LATENT_DIM,
            "width": WIDTH,
            "lr": LR,
            "device": DEVICE,
        },
        "ablation_configs": ABLATIONS,
        "summary": rows,
        "raw": {
            "rollouts": rollout_raw,
            "history": history_raw,
        },
    }

    csv_path = RESULTS_DIR / "experiment_5b_causal_mechanism_ablation_summary.csv"
    json_path = RESULTS_DIR / "experiment_5b_causal_mechanism_ablation_raw.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    print("\n=== Experiment 5b summary comparisons ===")
    comparison_rows = df[df["ablation"].astype(str).str.contains("_vs_euclidean", na=False)]
    if comparison_rows.empty:
        print(df.to_string(index=False))
    else:
        print(comparison_rows.to_string(index=False))

    plot_violation_vs_box(df)
    plot_mse_vs_box(df)
    plot_effective_rank(df)
    plot_violation_by_step(rollout_raw)
    plot_k_drift_by_step(rollout_raw)

    print("\nsaved:")
    for path in (
        csv_path,
        json_path,
        RESULTS_DIR / "experiment_5b_violation_vs_box.png",
        RESULTS_DIR / "experiment_5b_mse_vs_box.png",
        RESULTS_DIR / "experiment_5b_effective_rank.png",
        RESULTS_DIR / "experiment_5b_violation_by_step.png",
        RESULTS_DIR / "experiment_5b_K_drift_by_step.png",
    ):
        print(path)

    return df, payload


def plot_violation_vs_box(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    for name in ABLATIONS:
        subset = df[(df["ablation"] == name) & df["causal_violation_rate_mean"].notna()]
        plt.plot(subset["box"], subset["causal_violation_rate_mean"], marker="o", label=name)

    plt.xscale("log")
    plt.xlabel("test box")
    plt.ylabel("causal violation rate")
    plt.title("Experiment 5b: violation vs OOD box")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5b_violation_vs_box.png", dpi=160)
    plt.close()


def plot_mse_vs_box(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    for name in ABLATIONS:
        subset = df[(df["ablation"] == name) & df["final_rollout_mse_mean"].notna()]
        plt.plot(subset["box"], subset["final_rollout_mse_mean"], marker="o", label=name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("test box")
    plt.ylabel("final rollout MSE")
    plt.title("Experiment 5b: final rollout MSE")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5b_mse_vs_box.png", dpi=160)
    plt.close()


def plot_effective_rank(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    for name in ABLATIONS:
        subset = df[(df["ablation"] == name) & df["encoder_effective_rank_mean"].notna()]
        plt.plot(subset["box"], subset["encoder_effective_rank_mean"], marker="o", label=name)

    plt.xscale("log")
    plt.xlabel("test box")
    plt.ylabel("encoder effective rank")
    plt.title("Experiment 5b: encoder effective rank")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5b_effective_rank.png", dpi=160)
    plt.close()


def plot_violation_by_step(raw: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    max_box = str(max(TEST_BOXES))
    plt.figure(figsize=(8, 5))

    for name in ABLATIONS:
        curves = np.array([record["violation_by_t"] for record in raw[name][max_box]], dtype=float)
        plt.plot(np.arange(curves.shape[1]), curves.mean(axis=0), marker="o", label=name)

    plt.xlabel("rollout step")
    plt.ylabel("causal violation rate")
    plt.title(f"Experiment 5b: violation by step, box={max_box}")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5b_violation_by_step.png", dpi=160)
    plt.close()


def plot_k_drift_by_step(raw: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    max_box = str(max(TEST_BOXES))
    plt.figure(figsize=(8, 5))

    for name in ABLATIONS:
        if name == "euclidean_latent_predictor":
            continue
        curves = np.array([record["mean_abs_k_minus_1_by_t"] for record in raw[name][max_box]], dtype=float)
        plt.plot(np.arange(curves.shape[1]), curves.mean(axis=0), marker="o", label=name)

    plt.xlabel("rollout step")
    plt.ylabel("mean |K(z)-1|")
    plt.title(f"Experiment 5b: latent K drift, box={max_box}")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5b_K_drift_by_step.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    run_experiment_5b()
