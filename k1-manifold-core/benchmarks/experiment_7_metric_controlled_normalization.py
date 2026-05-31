"""
EXPERIMENT 7: METRIC-CONTROLLED NORMALIZATION TEST
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from k1_manifold_core.benchmark_datasets import (
    make_spacelike_dataset,
    make_timelike_dataset,
)
RESULTS_DIR = ROOT / "results" / "experiment_7_metric_controlled_normalization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DIM = 4
LATENT_DIM = 16
WIDTH = 64
LR = 1e-3

T_TOTAL, T_OBS, ROLL_STEPS = 80, 10, 50
N_TRAIN, N_TEST = 250, 96
TEST_BOX = 2.0

N_SEEDS = 30
EPOCHS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(2)


class Encoder(nn.Module):
    def __init__(self, dim, k, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, k, dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, dim),
        )

    def forward(self, z):
        return self.net(z)


class ChronosLorentz(nn.Module):
    def __init__(self, dim, k, width):
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)
        g = torch.ones(k)
        g[1:] = -1.0
        self.register_buffer("g", g)
        self.pred = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def metric(self, dz):
        return torch.sum(dz * dz * self.g, dim=1)

    def encode(self, x):
        return self.enc(x)

    def predict_z(self, x):
        z = self.enc(x)
        raw = self.pred(z)
        q = self.metric(raw).unsqueeze(1)
        denom = torch.sqrt(torch.clamp(torch.abs(q), min=1e-3))
        dz = self.scale * raw / denom
        return z + dz

    def forward(self, x):
        return self.dec(self.predict_z(x))

    def rollout(self, x0, steps):
        xs, x = [x0], x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)


class ChronosEuclidean(nn.Module):
    def __init__(self, dim, k, width):
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)
        self.pred = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def metric(self, dz):
        return torch.sum(dz * dz, dim=1)

    def encode(self, x):
        return self.enc(x)

    def predict_z(self, x):
        z = self.enc(x)
        raw = self.pred(z)
        q = self.metric(raw).unsqueeze(1)
        denom = torch.sqrt(torch.clamp(q, min=1e-3))
        dz = self.scale * raw / denom
        return z + dz

    def forward(self, x):
        return self.dec(self.predict_z(x))

    def rollout(self, x0, steps):
        xs, x = [x0], x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)


class ChronosRandom(nn.Module):
    def __init__(self, dim, k, width, seed=42):
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)
        self.pred = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))
        torch.manual_seed(seed)
        a = torch.randn(k, k)
        m = 0.5 * (a + a.T)
        self.register_buffer("M", m)

    def metric(self, dz):
        return torch.sum((dz @ self.M) * dz, dim=1)

    def encode(self, x):
        return self.enc(x)

    def predict_z(self, x):
        z = self.enc(x)
        raw = self.pred(z)
        q = self.metric(raw).unsqueeze(1)
        denom = torch.sqrt(torch.clamp(torch.abs(q), min=1e-3))
        dz = self.scale * raw / denom
        return z + dz

    def forward(self, x):
        return self.dec(self.predict_z(x))

    def rollout(self, x0, steps):
        xs, x = [x0], x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)


class EuclideanBaseline(nn.Module):
    def __init__(self, dim, k, width):
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)
        self.pred = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def encode(self, x):
        return self.enc(x)

    def predict_z(self, x):
        z = self.enc(x)
        return z + self.pred(z) * self.scale

    def forward(self, x):
        return self.dec(self.predict_z(x))

    def rollout(self, x0, steps):
        xs, x = [x0], x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)


def np_lorentz_interval(dz):
    return dz[..., 0] ** 2 - np.sum(dz[..., 1:] ** 2, axis=-1)


def make_pairs(trajs):
    x_t = trajs[:, :-1, :]
    x_next = trajs[:, 1:, :]
    return x_t.reshape(-1, DIM).astype(np.float32), x_next.reshape(-1, DIM).astype(np.float32)


def causal_violation_rate(traj):
    delta = traj[:, 1:, :] - traj[:, :-1, :]
    return np.mean(np_lorentz_interval(delta) < 0.0)


def train_model(model, x_train_np, y_train_np, epochs):
    model.to(DEVICE)
    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        x_next_pred = model(x_train)
        loss = mse_loss(x_next_pred, y_train)
        z_next_pred = model.predict_z(x_train)
        z_next_true = model.encode(y_train).detach()
        loss = loss + mse_loss(z_next_pred, z_next_true)
        loss.backward()
        opt.step()

    return model


def eval_model(model, trajs):
    model.eval()
    x0 = torch.tensor(trajs[:, T_OBS, :], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred = model.rollout(x0, ROLL_STEPS).detach().cpu().numpy()
    return causal_violation_rate(pred)


def run_experiment_7(n_seeds, epochs):
    datasets = {"timelike": make_timelike_dataset, "spacelike": make_spacelike_dataset}
    models = {
        "euclidean_baseline": EuclideanBaseline,
        "chronos_lorentz": ChronosLorentz,
        "chronos_euclidean": ChronosEuclidean,
        "chronos_random": ChronosRandom,
    }
    raw = []
    for dname, dgen in datasets.items():
        for mname, mcls in models.items():
            for seed in range(n_seeds):
                train_rng = np.random.default_rng(50000 + seed)
                train_trajs = dgen(N_TRAIN, T_TOTAL, DIM, train_rng, box=TEST_BOX)
                x_train, y_train = make_pairs(train_trajs)

                torch.manual_seed(seed)
                np.random.seed(seed)
                model = mcls(DIM, LATENT_DIM, WIDTH, seed=seed) if mname == "chronos_random" else mcls(DIM, LATENT_DIM, WIDTH)
                model = train_model(model, x_train, y_train, epochs=epochs)

                test_rng = np.random.default_rng(60000 + seed)
                test_trajs = dgen(N_TEST, T_TOTAL, DIM, test_rng, box=TEST_BOX)
                raw.append({
                    "dataset": dname,
                    "model": mname,
                    "seed": seed,
                    "violation": eval_model(model, test_trajs),
                })
    return pd.DataFrame(raw)


def analyze_exp7(raw_df):
    baselines = {}
    for dataset in raw_df["dataset"].unique():
        b = raw_df[(raw_df["dataset"] == dataset) & (raw_df["model"] == "euclidean_baseline")]["violation"].values
        baselines[dataset] = np.mean(b)

    out = raw_df.copy()
    out["improvement"] = out.apply(
        lambda r: (baselines[r["dataset"]] - r["violation"]) / baselines[r["dataset"]] * 100,
        axis=1,
    )

    rows = []
    for metric in ["chronos_lorentz", "chronos_euclidean", "chronos_random"]:
        t = out[(out["dataset"] == "timelike") & (out["model"] == metric)]["improvement"].values
        s = out[(out["dataset"] == "spacelike") & (out["model"] == metric)]["improvement"].values
        _, p = stats.wilcoxon(t, s, alternative="greater")
        d = t - s
        rows.append({
            "metric": metric,
            "timelike_improvement_mean_pct": np.mean(t),
            "spacelike_improvement_mean_pct": np.mean(s),
            "difference_tl_minus_sl_pct": np.mean(d),
            "wilcoxon_one_sided_p": p,
        })
    return out, pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = 4 if args.smoke else N_SEEDS
    epochs = 3 if args.smoke else EPOCHS

    print(f"EXPERIMENT 7: device={DEVICE} seeds={seeds} epochs={epochs}")
    raw_df = run_experiment_7(seeds, epochs)
    raw_df.to_csv(RESULTS_DIR / "experiment_7_raw_results.csv", index=False)

    out_df, interaction_df = analyze_exp7(raw_df)
    out_df.to_csv(RESULTS_DIR / "experiment_7_raw_results_with_improvement.csv", index=False)
    interaction_df.to_csv(RESULTS_DIR / "experiment_7_metric_dataset_interaction.csv", index=False)
    print(interaction_df)
