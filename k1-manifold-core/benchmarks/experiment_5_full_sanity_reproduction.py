"""Experiment 5 full sanity reproduction.

This is a research benchmark, not a unit test. It reproduces the original
Experiment 5 two-model design with a larger seed count and records whether the
Chronos latent predictor reduces decoded causal violations while keeping
rollout MSE comparable.

The models are JEPA-style in the narrow sense that they predict future
embeddings. They are not implementations of Meta/LeCun JEPA.

Run from ``k1-manifold-core``:

    python benchmarks/experiment_5_full_sanity_reproduction.py --smoke
    CHRONOS_DEVICE=cuda python benchmarks/experiment_5_full_sanity_reproduction.py --full
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
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
DEFAULT_RESULTS_DIR = ROOT / "results"

N_SPACE = 3
DIM = 1 + N_SPACE
LATENT_DIM = 16
WIDTH = 64
LR = 1e-3

T_TOTAL = 80
T_OBS = 10
ROLL_STEPS = 50

TRAIN_BOX = 2.0
TEST_BOXES = (2.0, 4.0, 8.0, 16.0, 32.0)
LAMBDA_GRID = (0.0, 0.1, 0.2, 0.5)
LAMBDA_K1 = 0.02

TRAIN_SEED_OFFSET = 50000
TEST_SEED_OFFSET = 60000
TEST_BOX_MULTIPLIER = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 5 full sanity reproduction.")
    parser.add_argument("--full", action="store_true", help="Run the N=10 reproduction configuration.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny CPU-friendly smoke configuration.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for CSV/JSON/PNG artifacts.",
    )
    args = parser.parse_args()
    if args.full and args.smoke:
        parser.error("--full and --smoke cannot be used together")
    return args


ARGS = parse_args()
RUN_MODE = "full" if ARGS.full else "smoke"

if RUN_MODE == "full":
    N_SEEDS = 10
    N_TRAIN = 3000
    N_TEST = 512
    EPOCHS = 250
else:
    N_SEEDS = 2
    N_TRAIN = 250
    N_TEST = 96
    EPOCHS = 12

RESULTS_DIR = ARGS.output_dir.resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REQUESTED_DEVICE = os.environ.get("CHRONOS_DEVICE", "cpu")
if REQUESTED_DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CHRONOS_DEVICE=cuda requested, but CUDA is not available")
DEVICE = REQUESTED_DEVICE

torch.set_num_threads(2)


def train_seed(seed: int) -> int:
    return TRAIN_SEED_OFFSET + seed


def test_seed(seed: int, box: float) -> int:
    return TEST_SEED_OFFSET + seed + int(box * TEST_BOX_MULTIPLIER)


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


def interval_drift_by_t_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dt = target[:, 1:, :] - target[:, :-1, :]
    return np.mean(np.abs(np_lorentz_interval(dp) - np_lorentz_interval(dt)), axis=0)


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


MODELS: dict[str, type[nn.Module]] = {
    "euclidean_latent_predictor": EuclideanLatentPredictor,
    "chronos_latent_predictor": ChronosLatentPredictor,
}


def train_model(
    model: nn.Module,
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    lambda_interval: float,
    lambda_causal: float,
) -> tuple[nn.Module, dict[str, list[float | None]]]:
    model.to(DEVICE)
    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    history: dict[str, list[float | None]] = {"loss": [], "mean_abs_k_minus_1": [], "scale": []}

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()

        z_t = model.encode(x_train)
        z_next_true = model.encode(y_train).detach()
        z_next_pred = model.predict_z(x_train)
        x_next_pred = model.dec(z_next_pred)

        loss = mse(x_next_pred, y_train) + mse(z_next_pred, z_next_true)
        mean_abs_k: float | None = None
        scale_value: float | None = None

        if isinstance(model, ChronosLatentPredictor):
            dz_pred = z_next_pred - z_t
            interval_true = torch_lorentz_interval(y_train - x_train).detach()
            interval_pred = model.latent_interval(dz_pred)

            loss_interval = mse(interval_pred, interval_true)
            loss_causal = torch.relu(-interval_pred).mean()
            k_next = model.k_value(z_next_pred)
            loss_k1 = torch.mean((k_next - 1.0) ** 2)

            loss = (
                loss
                + lambda_interval * loss_interval
                + lambda_causal * loss_causal
                + LAMBDA_K1 * loss_k1
            )

            with torch.no_grad():
                mean_abs_k = float(torch.mean(torch.abs(k_next - 1.0)).detach().cpu().item())
                scale_value = float(model.scale.detach().cpu().item())

        loss.backward()
        opt.step()

        if epoch % max(1, EPOCHS // 20) == 0 or epoch == EPOCHS - 1:
            history["loss"].append(float(loss.detach().cpu().item()))
            history["mean_abs_k_minus_1"].append(mean_abs_k)
            history["scale"].append(scale_value)

    return model, history


def eval_model(model: nn.Module, trajs: np.ndarray) -> dict[str, Any]:
    model.eval()

    x0_np = trajs[:, T_OBS, :]
    target_np = trajs[:, T_OBS : T_OBS + ROLL_STEPS + 1, :]
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        if isinstance(model, ChronosLatentPredictor):
            pred, z_roll = model.rollout(x0, ROLL_STEPS)
            pred_np = pred.detach().cpu().numpy()
            z_np = z_roll.detach().cpu().numpy()
            k_by_t = z_np[..., 0] ** 2 - np.sum(z_np[..., 1:] ** 2, axis=-1)
            mean_abs_k_minus_1_by_t: list[float | None] = np.mean(
                np.abs(k_by_t - 1.0), axis=0
            ).tolist()
        else:
            pred = model.rollout(x0, ROLL_STEPS)
            pred_np = pred.detach().cpu().numpy()
            mean_abs_k_minus_1_by_t = [None] * (ROLL_STEPS + 1)

    mse_by_t = np.mean((pred_np - target_np) ** 2, axis=(0, 2))
    violation_by_t = causal_violation_by_t_np(pred_np)
    drift_by_t = interval_drift_by_t_np(pred_np, target_np)

    return {
        "mean_rollout_mse": float(np.mean(mse_by_t)),
        "final_rollout_mse": float(mse_by_t[-1]),
        "causal_violation_rate": float(np.mean(violation_by_t)),
        "interval_drift": float(np.mean(drift_by_t)),
        "mse_by_t": mse_by_t.tolist(),
        "violation_by_t": violation_by_t.tolist(),
        "interval_drift_by_t": drift_by_t.tolist(),
        "mean_abs_k_minus_1_by_t": mean_abs_k_minus_1_by_t,
    }


def rng_documentation() -> dict[str, Any]:
    examples: dict[str, dict[str, Any]] = {}
    for seed in range(min(3, N_SEEDS)):
        examples[f"seed_{seed}"] = {
            "train": train_seed(seed),
            "test_by_box": {str(box): test_seed(seed, box) for box in TEST_BOXES},
        }
    return {
        "strategy": "offset-based seeding with box-dependent test offset",
        "train_seed_offset": TRAIN_SEED_OFFSET,
        "test_seed_offset": TEST_SEED_OFFSET,
        "test_box_multiplier": TEST_BOX_MULTIPLIER,
        "examples": examples,
    }


def paired_summary(
    raw: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    lambda_value: float,
    box: float,
) -> dict[str, Any]:
    key_lambda = str(lambda_value)
    key_box = str(box)
    euclid = raw[key_lambda]["rollouts"]["euclidean_latent_predictor"][key_box]
    chronos = raw[key_lambda]["rollouts"]["chronos_latent_predictor"][key_box]

    e_violation = np.array([record["causal_violation_rate"] for record in euclid], dtype=float)
    c_violation = np.array([record["causal_violation_rate"] for record in chronos], dtype=float)
    e_final_mse = np.array([record["final_rollout_mse"] for record in euclid], dtype=float)
    c_final_mse = np.array([record["final_rollout_mse"] for record in chronos], dtype=float)
    e_mean_mse = np.array([record["mean_rollout_mse"] for record in euclid], dtype=float)
    c_mean_mse = np.array([record["mean_rollout_mse"] for record in chronos], dtype=float)
    e_drift = np.array([record["interval_drift"] for record in euclid], dtype=float)
    c_drift = np.array([record["interval_drift"] for record in chronos], dtype=float)

    return {
        "lambda": lambda_value,
        "box": box,
        "euclid_violation_mean": float(np.mean(e_violation)),
        "euclid_violation_std": float(np.std(e_violation)),
        "chronos_violation_mean": float(np.mean(c_violation)),
        "chronos_violation_std": float(np.std(c_violation)),
        "violation_reduction": float((np.mean(e_violation) - np.mean(c_violation)) / (np.mean(e_violation) + 1e-12)),
        "p_violation": safe_wilcoxon(e_violation, c_violation),
        "euclid_final_mse_mean": float(np.mean(e_final_mse)),
        "chronos_final_mse_mean": float(np.mean(c_final_mse)),
        "final_mse_relative_change": float((np.mean(c_final_mse) - np.mean(e_final_mse)) / (np.mean(e_final_mse) + 1e-12)),
        "p_final_mse": safe_wilcoxon(e_final_mse, c_final_mse),
        "euclid_mean_mse_mean": float(np.mean(e_mean_mse)),
        "chronos_mean_mse_mean": float(np.mean(c_mean_mse)),
        "p_mean_mse": safe_wilcoxon(e_mean_mse, c_mean_mse),
        "euclid_interval_drift_mean": float(np.mean(e_drift)),
        "chronos_interval_drift_mean": float(np.mean(c_drift)),
        "p_interval_drift": safe_wilcoxon(e_drift, c_drift),
    }


def run_experiment() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw: dict[str, dict[str, Any]] = {}

    print("=" * 80)
    print(f"Experiment 5 full sanity reproduction ({RUN_MODE})")
    print("=" * 80)
    print(f"device={DEVICE}")
    print(f"n_seeds={N_SEEDS}, n_train={N_TRAIN}, n_test={N_TEST}, epochs={EPOCHS}")
    print(f"started={datetime.now().isoformat(timespec='seconds')}")

    for lambda_value in LAMBDA_GRID:
        print(f"\nlambda={lambda_value}", flush=True)
        raw[str(lambda_value)] = {
            "rollouts": {name: {str(box): [] for box in TEST_BOXES} for name in MODELS},
            "history": {name: [] for name in MODELS},
        }

        for seed in range(N_SEEDS):
            print(f"  seed {seed}", flush=True)

            train_rng = np.random.default_rng(train_seed(seed))
            train_trajs = make_lorentz_oscillator_trajs(
                N_TRAIN,
                T_TOTAL,
                DIM,
                train_rng,
                box=TRAIN_BOX,
            )
            x_train, y_train = make_pairs(train_trajs)

            trained: dict[str, nn.Module] = {}
            for name, cls in MODELS.items():
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = cls(DIM, LATENT_DIM, WIDTH)
                model, history = train_model(
                    model,
                    x_train,
                    y_train,
                    lambda_interval=lambda_value,
                    lambda_causal=lambda_value,
                )
                trained[name] = model
                raw[str(lambda_value)]["history"][name].append(history)

            for box in TEST_BOXES:
                test_rng = np.random.default_rng(test_seed(seed, box))
                test_trajs = make_lorentz_oscillator_trajs(
                    N_TEST,
                    T_TOTAL,
                    DIM,
                    test_rng,
                    box=box,
                )

                for name, model in trained.items():
                    raw[str(lambda_value)]["rollouts"][name][str(box)].append(eval_model(model, test_trajs))

    rows = [paired_summary(raw, lambda_value, box) for lambda_value in LAMBDA_GRID for box in TEST_BOXES]
    df = pd.DataFrame(rows)

    payload = {
        "benchmark": "experiment_5_full_sanity_reproduction",
        "scope": "research benchmark, not unit test",
        "description": (
            "Reproduction of the original Experiment 5 two-model lambda scan. "
            "Records mean/final rollout MSE, decoded causal violation, interval "
            "drift, paired Wilcoxon p-values, and RNG seeding."
        ),
        "config": {
            "run_mode": RUN_MODE,
            "n_seeds": N_SEEDS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "epochs": EPOCHS,
            "dim": DIM,
            "latent_dim": LATENT_DIM,
            "width": WIDTH,
            "lr": LR,
            "t_total": T_TOTAL,
            "t_obs": T_OBS,
            "roll_steps": ROLL_STEPS,
            "train_box": TRAIN_BOX,
            "test_boxes": list(TEST_BOXES),
            "lambda_grid": list(LAMBDA_GRID),
            "lambda_k1": LAMBDA_K1,
            "device": DEVICE,
        },
        "rng": rng_documentation(),
        "summary": rows,
        "raw": raw,
    }

    write_artifacts(df, payload, raw)
    return df, payload


def write_artifacts(df: pd.DataFrame, payload: dict[str, Any], raw: dict[str, Any]) -> None:
    csv_path = RESULTS_DIR / "experiment_5_full_sanity_summary.csv"
    json_path = RESULTS_DIR / "experiment_5_full_sanity_payload.json"
    rng_path = RESULTS_DIR / "rng_seeding_documentation.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    with open(rng_path, "w", encoding="utf-8") as handle:
        json.dump(payload["rng"], handle, indent=2, allow_nan=False)

    plot_violation_vs_box(df)
    plot_mse_vs_box(df)
    plot_violation_by_step(raw)

    print("\n=== summary ===")
    print(df.to_string(index=False))
    print("\nsaved:")
    for path in (
        csv_path,
        json_path,
        rng_path,
        RESULTS_DIR / "experiment_5_full_sanity_violation_vs_box.png",
        RESULTS_DIR / "experiment_5_full_sanity_mse_vs_box.png",
        RESULTS_DIR / "experiment_5_full_sanity_violation_by_step.png",
    ):
        print(path)


def plot_violation_vs_box(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    for lambda_value in LAMBDA_GRID:
        subset = df[df["lambda"] == lambda_value]
        plt.plot(
            subset["box"],
            subset["chronos_violation_mean"],
            marker="o",
            label=f"Chronos lambda={lambda_value}",
        )

    euclid = df[df["lambda"] == LAMBDA_GRID[0]]
    plt.plot(
        euclid["box"],
        euclid["euclid_violation_mean"],
        marker="s",
        linestyle="--",
        label="Euclidean",
    )

    plt.xscale("log")
    plt.xlabel("test box")
    plt.ylabel("causal violation rate")
    plt.title("Experiment 5 full sanity: causal violation")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5_full_sanity_violation_vs_box.png", dpi=160)
    plt.close()


def plot_mse_vs_box(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    for lambda_value in LAMBDA_GRID:
        subset = df[df["lambda"] == lambda_value]
        plt.plot(
            subset["box"],
            subset["chronos_final_mse_mean"],
            marker="o",
            label=f"Chronos lambda={lambda_value}",
        )

    euclid = df[df["lambda"] == LAMBDA_GRID[0]]
    plt.plot(
        euclid["box"],
        euclid["euclid_final_mse_mean"],
        marker="s",
        linestyle="--",
        label="Euclidean",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("test box")
    plt.ylabel("final rollout MSE")
    plt.title("Experiment 5 full sanity: final MSE")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5_full_sanity_mse_vs_box.png", dpi=160)
    plt.close()


def plot_violation_by_step(raw: dict[str, Any]) -> None:
    lambda_key = "0.1" if "0.1" in raw else str(LAMBDA_GRID[0])
    box_key = str(max(TEST_BOXES))

    euclid_curves = np.array(
        [
            record["violation_by_t"]
            for record in raw[lambda_key]["rollouts"]["euclidean_latent_predictor"][box_key]
        ],
        dtype=float,
    )
    chronos_curves = np.array(
        [
            record["violation_by_t"]
            for record in raw[lambda_key]["rollouts"]["chronos_latent_predictor"][box_key]
        ],
        dtype=float,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(euclid_curves.shape[1]),
        euclid_curves.mean(axis=0),
        marker="s",
        label="Euclidean",
    )
    plt.plot(
        np.arange(chronos_curves.shape[1]),
        chronos_curves.mean(axis=0),
        marker="o",
        label=f"Chronos lambda={lambda_key}",
    )
    plt.xlabel("rollout step")
    plt.ylabel("causal violation rate")
    plt.title(f"Experiment 5 full sanity: violation by step, box={box_key}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5_full_sanity_violation_by_step.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    run_experiment()
