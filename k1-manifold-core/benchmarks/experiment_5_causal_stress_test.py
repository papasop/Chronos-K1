"""Experiment 5: constraint ablation and causal stress test.

This is a research benchmark, not a unit test. It scans Chronos latent
predictor causal regularization strength on synthetic Lorentzian oscillator
trajectories and measures whether constraints reduce decoded causal violations
under OOD box extrapolation while keeping rollout MSE comparable.

The models are JEPA-style in the narrow sense that they predict future
embeddings. They are not implementations of Meta/LeCun JEPA.

Run from ``k1-manifold-core``:

    python benchmarks/experiment_5_causal_stress_test.py
    python benchmarks/experiment_5_causal_stress_test.py --smoke
    python benchmarks/experiment_5_causal_stress_test.py --full
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Run Chronos-K1 Experiment 5 causal stress benchmark.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny CPU-friendly smoke configuration.")
    parser.add_argument("--full", action="store_true", help="Run the larger benchmark configuration.")
    args = parser.parse_args()
    if args.smoke and args.full:
        parser.error("--smoke and --full cannot be used together")
    return args


ARGS = parse_args()
RUN_MODE = "smoke" if ARGS.smoke else "full" if ARGS.full else "default"

if RUN_MODE == "smoke":
    N_SEEDS = 2
    N_TRAIN = 250
    N_TEST = 96
    EPOCHS = 20
elif RUN_MODE == "default":
    N_SEEDS = 5
    N_TRAIN = 1000
    N_TEST = 256
    EPOCHS = 100
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
TEST_BOXES = (2.0, 8.0, 32.0) if RUN_MODE == "smoke" else (2.0, 4.0, 8.0, 16.0, 32.0)

LAMBDA_GRID = (0.0, 0.5) if RUN_MODE == "smoke" else (0.0, 0.1, 0.2, 0.5)
LAMBDA_K1 = 0.02

REQUESTED_DEVICE = os.environ.get("CHRONOS_DEVICE", "cpu")
if REQUESTED_DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CHRONOS_DEVICE=cuda requested, but CUDA is not available")
DEVICE = REQUESTED_DEVICE

torch.set_num_threads(2)


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
    """Generate causal-like oscillator trajectories."""

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


def causal_violation_rate_np(traj: np.ndarray) -> float:
    return float(np.mean(causal_violation_by_t_np(traj)))


def interval_drift_np(pred: np.ndarray, target: np.ndarray) -> float:
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dt = target[:, 1:, :] - target[:, :-1, :]
    return float(np.mean(np.abs(np_lorentz_interval(dp) - np_lorentz_interval(dt))))


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
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

    def K_value(self, z: torch.Tensor) -> torch.Tensor:
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


MODELS = {
    "euclidean_latent_predictor": EuclideanLatentPredictor,
    "chronos_latent_predictor": ChronosLatentPredictor,
}


def train_model(
    model: nn.Module,
    Xtr_np: np.ndarray,
    Ytr_np: np.ndarray,
    *,
    lambda_interval: float,
    lambda_causal: float,
) -> tuple[nn.Module, dict[str, list[float]]]:
    model.to(DEVICE)

    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=DEVICE)
    Ytr = torch.tensor(Ytr_np, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    history = {"loss": [], "mean_abs_K_minus_1": []}

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()

        z_t = model.encode(Xtr)
        z_next_true = model.encode(Ytr).detach()
        z_next_pred = model.predict_z(Xtr)
        x_next_pred = model.dec(z_next_pred)

        loss_recon = mse(x_next_pred, Ytr)
        loss_latent = mse(z_next_pred, z_next_true)

        if hasattr(model, "latent_interval"):
            dz_pred = z_next_pred - z_t
            dx_true = Ytr - Xtr
            interval_true = torch_lorentz_interval(dx_true).detach()
            interval_pred = model.latent_interval(dz_pred)

            loss_interval = mse(interval_pred, interval_true)
            loss_causal = torch.relu(-interval_pred).mean()

            Kz = model.K_value(z_next_pred)
            loss_k1 = torch.mean((Kz - 1.0) ** 2)

            loss = (
                loss_recon
                + loss_latent
                + lambda_interval * loss_interval
                + lambda_causal * loss_causal
                + LAMBDA_K1 * loss_k1
            )
            with torch.no_grad():
                mean_abs_k = torch.mean(torch.abs(Kz - 1.0)).item()
        else:
            loss = loss_recon + loss_latent
            mean_abs_k = float("nan")

        loss.backward()
        opt.step()

        if epoch % max(1, EPOCHS // 20) == 0 or epoch == EPOCHS - 1:
            history["loss"].append(float(loss.detach().cpu().item()))
            history["mean_abs_K_minus_1"].append(mean_abs_k)

    return model, history


def eval_model(model: nn.Module, trajs: np.ndarray) -> dict[str, object]:
    model.eval()

    x0_np = trajs[:, T_OBS, :]
    target_np = trajs[:, T_OBS : T_OBS + ROLL_STEPS + 1, :]
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        if isinstance(model, ChronosLatentPredictor):
            pred, z_roll = model.rollout(x0, ROLL_STEPS)
            pred_np = pred.detach().cpu().numpy()
            z_np = z_roll.detach().cpu().numpy()
            K_by_t = z_np[..., 0] ** 2 - np.sum(z_np[..., 1:] ** 2, axis=-1)
            mean_abs_K_minus_1_by_t = np.mean(np.abs(K_by_t - 1.0), axis=0)
        else:
            pred = model.rollout(x0, ROLL_STEPS)
            pred_np = pred.detach().cpu().numpy()
            mean_abs_K_minus_1_by_t = np.full(ROLL_STEPS + 1, np.nan)

    mse_by_t = np.mean((pred_np - target_np) ** 2, axis=(0, 2))
    violation_by_t = causal_violation_by_t_np(pred_np)

    return {
        "mean_rollout_mse": float(np.mean(mse_by_t)),
        "final_rollout_mse": float(mse_by_t[-1]),
        "causal_violation_rate": float(np.mean(violation_by_t)),
        "interval_drift": interval_drift_np(pred_np, target_np),
        "mse_by_t": mse_by_t.tolist(),
        "violation_by_t": violation_by_t.tolist(),
        "mean_abs_K_minus_1_by_t": mean_abs_K_minus_1_by_t.tolist(),
    }


def run_experiment_5() -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    rows = []
    raw: dict[str, object] = {}

    for lam in LAMBDA_GRID:
        print(f"\n=== lambda = {lam} ===")
        raw[str(lam)] = {name: {str(box): [] for box in TEST_BOXES} for name in MODELS}

        for seed in range(N_SEEDS):
            print(f"seed {seed}", flush=True)

            rng_train = np.random.default_rng(50000 + seed)
            train_trajs = make_lorentz_oscillator_trajs(N_TRAIN, T_TOTAL, DIM, rng_train, box=TRAIN_BOX)
            Xtr, Ytr = make_pairs(train_trajs)

            trained = {}
            for name, cls in MODELS.items():
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = cls(DIM, LATENT_DIM, WIDTH)
                model, history = train_model(model, Xtr, Ytr, lambda_interval=lam, lambda_causal=lam)
                trained[name] = model

                raw[str(lam)].setdefault("history", {})
                raw[str(lam)]["history"].setdefault(name, [])
                raw[str(lam)]["history"][name].append(history)

            for box in TEST_BOXES:
                rng_test = np.random.default_rng(60000 + seed + int(box * 100))
                test_trajs = make_lorentz_oscillator_trajs(N_TEST, T_TOTAL, DIM, rng_test, box=box)

                for name, model in trained.items():
                    raw[str(lam)][name][str(box)].append(eval_model(model, test_trajs))

    for lam in LAMBDA_GRID:
        for box in TEST_BOXES:
            key = str(box)
            for name in MODELS:
                records = raw[str(lam)][name][key]
                final_mse = np.array([r["final_rollout_mse"] for r in records])
                violation = np.array([r["causal_violation_rate"] for r in records])
                drift = np.array([r["interval_drift"] for r in records])

                rows.append(
                    {
                        "lambda": lam,
                        "box": box,
                        "model": name,
                        "final_mse_mean": float(np.mean(final_mse)),
                        "final_mse_std": float(np.std(final_mse)),
                        "violation_mean": float(np.mean(violation)),
                        "violation_std": float(np.std(violation)),
                        "interval_drift_mean": float(np.mean(drift)),
                        "interval_drift_std": float(np.std(drift)),
                    }
                )

            euclid = raw[str(lam)]["euclidean_latent_predictor"][key]
            chronos = raw[str(lam)]["chronos_latent_predictor"][key]

            e_v = np.array([r["causal_violation_rate"] for r in euclid])
            c_v = np.array([r["causal_violation_rate"] for r in chronos])
            e_m = np.array([r["final_rollout_mse"] for r in euclid])
            c_m = np.array([r["final_rollout_mse"] for r in chronos])
            e_d = np.array([r["interval_drift"] for r in euclid])
            c_d = np.array([r["interval_drift"] for r in chronos])

            rows.append(
                {
                    "lambda": lam,
                    "box": box,
                    "model": "chronos_vs_euclid",
                    "final_mse_mean": float(np.mean(c_m)),
                    "final_mse_std": float(np.std(c_m)),
                    "violation_mean": float(np.mean(c_v)),
                    "violation_std": float(np.std(c_v)),
                    "interval_drift_mean": float(np.mean(c_d)),
                    "interval_drift_std": float(np.std(c_d)),
                    "mse_improvement": float(np.mean((e_m - c_m) / (e_m + 1e-12))),
                    "violation_improvement": float(np.mean((e_v - c_v) / (e_v + 1e-12))),
                    "interval_drift_improvement": float(np.mean((e_d - c_d) / (e_d + 1e-12))),
                    "p_mse": safe_wilcoxon(e_m, c_m),
                    "p_violation": safe_wilcoxon(e_v, c_v),
                    "p_interval_drift": safe_wilcoxon(e_d, c_d),
                }
            )

    df = pd.DataFrame(rows)
    payload = {
        "benchmark": "experiment_5_causal_stress_test",
        "scope": "research benchmark, not unit test",
        "description": (
            "Constraint ablation and OOD causal stress test for Chronos latent predictor. "
            "Lambda scans interval/causal losses only; K1 regularization and "
            "Lorentz-normalized latent steps remain active for Chronos latent predictor."
        ),
        "config": {
            "run_mode": RUN_MODE,
            "n_seeds": N_SEEDS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "epochs": EPOCHS,
            "train_box": TRAIN_BOX,
            "test_boxes": list(TEST_BOXES),
            "lambda_grid": list(LAMBDA_GRID),
            "lambda_k1": LAMBDA_K1,
            "t_total": T_TOTAL,
            "t_obs": T_OBS,
            "roll_steps": ROLL_STEPS,
            "dim": DIM,
            "latent_dim": LATENT_DIM,
            "width": WIDTH,
            "lr": LR,
            "device": DEVICE,
        },
        "summary": rows,
        "raw": raw,
    }
    return df, raw, payload


def write_outputs(df: pd.DataFrame, raw: dict[str, object], payload: dict[str, object]) -> None:
    csv_path = RESULTS_DIR / "experiment_5_ablation_stress_summary.csv"
    json_path = RESULTS_DIR / "experiment_5_ablation_stress_raw.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== SUMMARY TABLE ===")
    print(df[df["model"] == "chronos_vs_euclid"].to_string(index=False), flush=True)

    plt.figure(figsize=(8, 5))
    for lam in LAMBDA_GRID:
        sub = df[(df["lambda"] == lam) & (df["model"] == "chronos_latent_predictor")]
        plt.plot(sub["box"], sub["violation_mean"], marker="o", label=f"Chronos lambda={lam}")
    euclid = df[(df["lambda"] == LAMBDA_GRID[0]) & (df["model"] == "euclidean_latent_predictor")]
    plt.plot(euclid["box"], euclid["violation_mean"], marker="s", linestyle="--", label="ELP")
    plt.xscale("log")
    plt.xlabel("test box")
    plt.ylabel("causal violation rate")
    plt.title("Causal violation under OOD stress")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5_violation_vs_box.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for lam in LAMBDA_GRID:
        sub = df[(df["lambda"] == lam) & (df["model"] == "chronos_latent_predictor")]
        plt.plot(sub["box"], sub["final_mse_mean"], marker="o", label=f"Chronos lambda={lam}")
    plt.plot(euclid["box"], euclid["final_mse_mean"], marker="s", linestyle="--", label="ELP")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("test box")
    plt.ylabel("final rollout MSE")
    plt.title("Final rollout MSE under OOD stress")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5_mse_vs_box.png", dpi=160)
    plt.close()

    max_box = str(max(TEST_BOXES))
    plt.figure(figsize=(8, 5))
    for lam in LAMBDA_GRID:
        records = raw[str(lam)]["chronos_latent_predictor"][max_box]
        curves = np.array([r["violation_by_t"] for r in records])
        plt.plot(np.arange(curves.shape[1]), curves.mean(axis=0), marker="o", label=f"Chronos lambda={lam}")
    records = raw[str(LAMBDA_GRID[0])]["euclidean_latent_predictor"][max_box]
    curves = np.array([r["violation_by_t"] for r in records])
    plt.plot(np.arange(curves.shape[1]), curves.mean(axis=0), marker="s", linestyle="--", label="ELP")
    plt.xlabel("rollout step")
    plt.ylabel("causal violation rate")
    plt.title(f"Causal violation by step, box={max_box}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5_violation_by_step.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for lam in LAMBDA_GRID:
        records = raw[str(lam)]["chronos_latent_predictor"][max_box]
        curves = np.array([r["mean_abs_K_minus_1_by_t"] for r in records])
        plt.plot(np.arange(curves.shape[1]), curves.mean(axis=0), marker="o", label=f"lambda={lam}")
    plt.xlabel("rollout step")
    plt.ylabel("mean |K(z)-1|")
    plt.title(f"Latent K drift by step, box={max_box}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_5_K_drift_by_step.png", dpi=160)
    plt.close()

    print("\nsaved:")
    print(csv_path)
    print(json_path)
    print(RESULTS_DIR / "experiment_5_violation_vs_box.png")
    print(RESULTS_DIR / "experiment_5_mse_vs_box.png")
    print(RESULTS_DIR / "experiment_5_violation_by_step.png")
    print(RESULTS_DIR / "experiment_5_K_drift_by_step.png")


if __name__ == "__main__":
    df5, raw5, payload5 = run_experiment_5()
    write_outputs(df5, raw5, payload5)
