"""Experiment K3.0-D: periodic Sine-Gordon winding-density regime validation.

Stage: K3.0-D, baseline-only VPSL regime validation.

This experiment tests whether periodic Sine-Gordon, represented as
``[sin(u), cos(u), u_t]``, yields a graceful-failure regime for a continuous
winding-density / local-topological-structure diagnostic.

It intentionally does not certify integer topological charge. The primary
structure metric is the local winding density

    rho(x) = (1 / 2 pi) d_x theta, theta = atan2(sin u, cos u)

computed with wrapped phase differences. Integer winding is tracked only as a
secondary near-floor hard check.

Prior K3.0 negatives motivating this representation:
- K3.0-A phi^4 single kink: no graceful band.
- K3.0-B phi^4 kink-antikink: integer sector readout was too abrupt.
- K3.0-C lifted-field periodic Sine-Gordon: representation artifact from a
  non-equivariant lifted angle seam.

Verdict classes:
- VALID_WINDING_DENSITY_REGIME_H{H}
- VALID_WINDING_SECTOR_REGIME_H{H}
- ALL_STABLE_NEARFLOOR
- NO_GRACEFUL_BAND
- MIXED
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


EPS = 1e-8
TWO_PI = 2.0 * np.pi


@dataclass
class Config:
    run_mode: str = "FULL"
    deterministic: bool = True
    results_dir: Path = Path("/content/exp_k3_0d_sg")

    nx: int = 256
    length: float = 50.0
    n_wind: int = 1
    s_fixed: int = 1
    h_scan: tuple[int, ...] = (80, 120, 160, 200)
    ref_horizon: int = 30

    kink_x0: float = 25.0
    c_lo: float = 0.1
    c_hi: float = 0.4
    ic_x0_jitter: float = 6.0
    ic_rad_noise: float = 0.02

    hidden: int = 64
    n_layers: int = 3
    radius: int = 2
    lr: float = 3e-4
    epochs: int = 150
    n_seeds: int = 30
    n_train_traj: int = 40
    train_steps: int = 80
    grad_clip: float = 1.0

    div_ceil_fac: float = 50.0
    div_abs: float = 100.0
    stable_fac: float = 2.0
    div_frac: float = 0.5
    stable_frac: float = 0.1
    func_div_thr: float = 10.0

    wdens_elev_fac: float = 2.0
    clean_hard_div: float = 0.2
    clean_grad_fac: float = 4.0
    wind_intact_frac: float = 0.6

    def smoke(self) -> "Config":
        self.run_mode = "SMOKE"
        self.n_seeds = 3
        self.epochs = 12
        self.n_train_traj = 6
        self.train_steps = 30
        self.h_scan = (40, 80)
        return self

    def scout(self) -> "Config":
        self.run_mode = "SCOUT"
        self.n_seeds = 5
        return self

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dt_phys(self) -> float:
        return 0.4 * self.dx


def make_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny invalid run for wiring checks.")
    parser.add_argument("--scout", action="store_true", help="Five-seed directional sanity run.")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.smoke()
    elif args.scout:
        cfg.scout()
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    return cfg


def setup_runtime(cfg: Config) -> str:
    if cfg.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    torch.set_num_threads(2)
    return "cuda" if torch.cuda.is_available() else "cpu"


def grid(cfg: Config) -> np.ndarray:
    return np.linspace(0.0, cfg.length, cfg.nx, endpoint=False)


def sg_kink(cfg: Config, x0: float, c: float) -> tuple[np.ndarray, np.ndarray]:
    x = grid(cfg)
    gamma = 1.0 / np.sqrt(1.0 - c**2) if abs(c) < 0.999 else 1.0
    arg = gamma * (x - x0)
    exp_arg = np.exp(arg)
    u = 4.0 * np.arctan(exp_arg)
    ut = -c * gamma * 4.0 * exp_arg / (1.0 + exp_arg**2)
    return u, ut


def random_kink_ic(cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x0 = (cfg.kink_x0 + rng.uniform(-cfg.ic_x0_jitter, cfg.ic_x0_jitter)) % cfg.length
    c = rng.uniform(cfg.c_lo, cfg.c_hi)
    u, ut = sg_kink(cfg, x0, c)
    u = u + rng.normal(0.0, cfg.ic_rad_noise, cfg.nx)
    ut = ut + rng.normal(0.0, cfg.ic_rad_noise, cfg.nx)
    return u, ut


def sg_lap(cfg: Config, u: np.ndarray) -> np.ndarray:
    up = np.roll(u, -1).copy()
    um = np.roll(u, 1).copy()
    up[-1] += TWO_PI * cfg.n_wind
    um[0] -= TWO_PI * cfg.n_wind
    return (up - 2.0 * u + um) / cfg.dx**2


def sg_accel(cfg: Config, u: np.ndarray) -> np.ndarray:
    return sg_lap(cfg, u) - np.sin(u)


def verlet_step(cfg: Config, u: np.ndarray, ut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = sg_accel(cfg, u)
    ut = ut + 0.5 * cfg.dt_phys * a
    u = u + cfg.dt_phys * ut
    a = sg_accel(cfg, u)
    ut = ut + 0.5 * cfg.dt_phys * a
    return u, ut


def sg_simulate(
    cfg: Config, u0: np.ndarray, ut0: np.ndarray, n_model_steps: int, s_fixed: int
) -> np.ndarray:
    u = u0.copy()
    ut = ut0.copy()
    traj = [np.stack([np.sin(u), np.cos(u), ut])]
    for _ in range(n_model_steps):
        for _ in range(s_fixed):
            u, ut = verlet_step(cfg, u, ut)
        traj.append(np.stack([np.sin(u), np.cos(u), ut]))
    return np.array(traj)


def wrap_phase(dtheta: np.ndarray) -> np.ndarray:
    return (dtheta + np.pi) % TWO_PI - np.pi


def phase(sc: np.ndarray) -> np.ndarray:
    return np.arctan2(sc[0], sc[1])


def winding_density(cfg: Config, sc: np.ndarray) -> np.ndarray:
    theta = phase(sc)
    diffs = np.diff(theta, append=theta[0])
    return wrap_phase(diffs) / (TWO_PI * cfg.dx)


def winding_number(sc: np.ndarray) -> float:
    theta = phase(sc)
    diffs = np.diff(theta, append=theta[0])
    return float(np.sum(wrap_phase(diffs)) / TWO_PI)


def max_grad(cfg: Config, sc: np.ndarray) -> float:
    theta = phase(sc)
    diffs = np.diff(theta, append=theta[0])
    return float(np.max(np.abs(wrap_phase(diffs) / cfg.dx)))


def fit_norm(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(999)
    channels: list[list[np.ndarray]] = [[], [], []]
    for _ in range(20):
        u0, ut0 = random_kink_ic(cfg, rng)
        traj = sg_simulate(cfg, u0, ut0, cfg.train_steps, cfg.s_fixed)
        for channel in range(3):
            channels[channel].append(traj[:, channel, :])
    means, stds = [], []
    for channel in range(3):
        values = np.concatenate(channels[channel])
        means.append(values.mean())
        stds.append(values.std() + EPS)
    return (
        np.array(means, dtype=np.float32).reshape(1, 3, 1),
        np.array(stds, dtype=np.float32).reshape(1, 3, 1),
    )


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class SGConvNet(nn.Module):
    def __init__(self, cfg: Config, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        channels = [3] + [cfg.hidden] * (cfg.n_layers - 1) + [3]
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    2 * cfg.radius + 1,
                    padding=cfg.radius,
                    padding_mode="circular",
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.act = nn.GELU()
        self.register_buffer("mean", torch.tensor(mean.reshape(1, 3, 1), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std.reshape(1, 3, 1), dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = self.act(x)
        y = state + x
        phys = y * self.std + self.mean
        s = phys[:, 0:1, :]
        c = phys[:, 1:2, :]
        ut = phys[:, 2:3, :]
        radius = torch.sqrt(s * s + c * c) + 1e-6
        phys = torch.cat([s / radius, c / radius, ut], dim=1)
        return (phys - self.mean) / self.std

    def rollout(self, initial: torch.Tensor, horizon: int) -> torch.Tensor:
        outputs = [initial]
        state = initial
        for _ in range(horizon):
            state = self.forward(state)
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def k_for_epoch(cfg: Config, epoch: int) -> int:
    if epoch < cfg.epochs * 0.2:
        return 1
    if epoch < cfg.epochs * 0.55:
        return 3
    return 5


def build_training_trajectories(cfg: Config, norm: Normalizer, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(10000 + seed)
    return [
        norm.normalize(sg_simulate(cfg, *random_kink_ic(cfg, rng), cfg.train_steps, cfg.s_fixed))
        for _ in range(cfg.n_train_traj)
    ]


def train_model(cfg: Config, norm: Normalizer, seed: int, device: str) -> SGConvNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = SGConvNet(cfg, norm.mean, norm.std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()
    trajectories = build_training_trajectories(cfg, norm, seed)
    data = torch.tensor(np.stack(trajectories), dtype=torch.float32, device=device)
    n_traj, t_total = data.shape[0], data.shape[1]
    rng_local = np.random.default_rng(50000 + seed)
    for epoch in range(cfg.epochs):
        rollout_steps = k_for_epoch(cfg, epoch)
        max_start = max(0, t_total - 1 - rollout_steps)
        permutation = torch.randperm(n_traj)
        for batch_index in range(n_traj):
            traj_index = permutation[batch_index].item()
            start = int(rng_local.integers(0, max_start + 1))
            current = data[traj_index, start].unsqueeze(0)
            optimizer.zero_grad()
            loss = 0.0
            for offset in range(1, rollout_steps + 1):
                current = model(current)
                loss = loss + mse(current, data[traj_index, start + offset].unsqueeze(0))
            loss = loss / rollout_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
    return model


def evaluate(
    cfg: Config, norm: Normalizer, model: SGConvNet, seed: int, horizon: int, device: str
) -> dict[str, float | int | bool]:
    model.eval()
    rng = np.random.default_rng(30000 + seed + 777777)
    u0, ut0 = random_kink_ic(cfg, rng)
    truth = sg_simulate(cfg, u0, ut0, horizon, cfg.s_fixed)
    init_grad = max_grad(cfg, truth[0, 0:2, :])
    initial = torch.tensor(norm.normalize(truth[0:1]), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model.rollout(initial, horizon).squeeze(0).cpu().numpy()
    pred = norm.denormalize(pred)
    if not (np.all(np.isfinite(pred)) and np.max(np.abs(pred[:, 0:2, :])) < 1e4):
        return {
            "roll_mse": 1e4,
            "wdens_err": np.inf,
            "wind_err": 1,
            "wind_pred": np.nan,
            "wind_true": np.nan,
            "max_grad": np.inf,
            "hard_diverged": True,
            "functional_diverged": True,
        }
    t_steps = min(len(pred), len(truth))
    roll_mse = float(np.mean((pred[:t_steps] - truth[:t_steps]) ** 2))
    rho_pred = winding_density(cfg, pred[t_steps - 1, 0:2, :])
    rho_true = winding_density(cfg, truth[t_steps - 1, 0:2, :])
    wdens_err = float(np.sqrt(np.mean((rho_pred - rho_true) ** 2)))
    wind_pred = winding_number(pred[t_steps - 1, 0:2, :])
    wind_true = winding_number(truth[t_steps - 1, 0:2, :])
    max_pred_grad = max_grad(cfg, pred[t_steps - 1, 0:2, :])
    per_step = np.mean((pred[:t_steps] - truth[:t_steps]) ** 2, axis=(1, 2))
    over = per_step > cfg.func_div_thr
    sustained = bool(over[-1] and over[max(0, t_steps - 10) :].mean() > 0.8)
    grad_blow = bool(np.isfinite(max_pred_grad) and max_pred_grad > cfg.clean_grad_fac * max(init_grad, EPS))
    return {
        "roll_mse": roll_mse,
        "wdens_err": wdens_err,
        "wind_err": int(round(wind_pred) != round(wind_true)),
        "wind_pred": round(wind_pred),
        "wind_true": round(wind_true),
        "max_grad": max_pred_grad,
        "hard_diverged": bool(sustained or grad_blow),
        "functional_diverged": bool(over.any()),
    }


def classify_cell(cfg: Config, med_mse: float, frac_div: float, ref: float, med_wdens: float, ref_wdens: float) -> str:
    if frac_div > cfg.div_frac or med_mse > max(cfg.div_ceil_fac * ref, cfg.div_abs):
        return "DIVERGENT"
    if med_mse < cfg.stable_fac * ref and frac_div < cfg.stable_frac and med_wdens < cfg.wdens_elev_fac * ref_wdens:
        return "STABLE"
    return "GRACEFUL_FAIL"


def analyze(cfg: Config, df: pd.DataFrame) -> tuple[dict[int, str], dict[int, bool], dict[int, bool], str]:
    ref = float(df["ref_mse"].median())
    ref_wdens = float(df["ref_wdens"].median())
    cells: dict[int, str] = {}
    wdens_ok: dict[int, bool] = {}
    wind_moves: dict[int, bool] = {}
    for horizon in cfg.h_scan:
        group = df[df["horizon"] == horizon]
        med = group["roll_mse"].median()
        hard_div = group["hard_diverged"].mean()
        wdens = group["wdens_err"].median()
        wind_err = group["wind_err"].mean()
        cls = classify_cell(cfg, med, hard_div, ref, wdens, ref_wdens)
        cells[horizon] = cls
        wind_intact_frac = 1.0 - wind_err
        wdens_ok[horizon] = (
            wdens > cfg.wdens_elev_fac * ref_wdens
            and hard_div < cfg.clean_hard_div
            and wind_intact_frac >= cfg.wind_intact_frac
        )
        wind_moves[horizon] = wind_err > 0.0 and wind_intact_frac >= cfg.wind_intact_frac
    verdict_value = verdict(cfg, cells, wdens_ok, wind_moves)
    return cells, wdens_ok, wind_moves, verdict_value


def verdict(cfg: Config, cells: dict[int, str], wdens_ok: dict[int, bool], wind_moves: dict[int, bool]) -> str:
    graceful = [h for h, cls in cells.items() if cls == "GRACEFUL_FAIL"]
    density_windows = [h for h, ok in wdens_ok.items() if ok]
    sector_windows = [h for h in density_windows if wind_moves.get(h, False)]
    stable = [h for h, cls in cells.items() if cls == "STABLE"]
    divergent = [h for h, cls in cells.items() if cls == "DIVERGENT"]
    if sector_windows:
        return f"VALID_WINDING_SECTOR_REGIME_H{max(sector_windows)}"
    if density_windows:
        return f"VALID_WINDING_DENSITY_REGIME_H{max(density_windows)}"
    if stable and not graceful and not divergent:
        return "ALL_STABLE_NEARFLOOR"
    if divergent and not density_windows:
        return "NO_GRACEFUL_BAND"
    return "MIXED"


def run(cfg: Config) -> tuple[pd.DataFrame, dict[int, str], str]:
    device = setup_runtime(cfg)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    mean, std = fit_norm(cfg)
    norm = Normalizer(mean, std)
    rows: list[dict[str, float | int | bool]] = []
    for seed in range(cfg.n_seeds):
        model = train_model(cfg, norm, seed, device)
        ref = evaluate(cfg, norm, model, seed, cfg.ref_horizon, device)
        for horizon in cfg.h_scan:
            metrics = evaluate(cfg, norm, model, seed, horizon, device)
            metrics.update(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "ref_mse": ref["roll_mse"],
                    "ref_wdens": ref["wdens_err"],
                }
            )
            rows.append(metrics)
    df = pd.DataFrame(rows)
    cells, wdens_ok, wind_moves, verdict_value = analyze(cfg, df)
    df.to_csv(cfg.results_dir / "exp_k3_0d_results.csv", index=False)
    summary = {
        "experiment": "k3_0d_sine_gordon_angle_winding_density",
        "verdict": verdict_value,
        **{f"class_H{h}": cells.get(h, "n/a") for h in cfg.h_scan},
        **{f"wdens_ok_H{h}": wdens_ok.get(h, False) for h in cfg.h_scan},
        **{f"wind_moves_H{h}": wind_moves.get(h, False) for h in cfg.h_scan},
    }
    pd.DataFrame([summary]).to_csv(cfg.results_dir / "exp_k3_0d_summary.csv", index=False)
    with open(cfg.results_dir / "config.json", "w") as handle:
        json.dump(
            {
                "experiment": "k3_0d_sine_gordon_angle_winding_density",
                "mode": cfg.run_mode,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "h_scan": list(cfg.h_scan),
                "target_structure": "winding-density / local topological structure, not integer-charge certification",
            },
            handle,
            indent=2,
        )
    return df, cells, verdict_value


def main() -> None:
    cfg = make_config()
    _, cells, verdict_value = run(cfg)
    print("K3.0-D cells:", cells)
    print("FINAL RESULT:", verdict_value)


if __name__ == "__main__":
    main()
