# Chronos-K1 Experiment 5 Extended Fixed v2 Colab Guide

This is the extended Colab guide corresponding to the supplied 2026-05-29
Experiment 5 extended fixed-v2 run.

Status of the supplied run:

- The rollout sweep completed: 800 results.
- The headline `box=2.0`, `lambda=0.1` result reproduced the earlier `N=10`
  causal-violation reduction.
- The mechanism-analysis block in the supplied console output failed for every
  lambda with a CUDA tensor to NumPy conversion error.

This guide keeps the extended lambda grid and includes the corrected mechanism
analysis pattern: convert both latent tensors to CPU/NumPy only after computing
latent differences in torch.

## Run Summary To Expect

```text
n_seeds: 10
n_train: 3000
n_test: 512
epochs: 250
lambda_grid: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
results: 800 rows
```

Headline result from the supplied Colab run:

```text
box=2.0, lambda=0.1
Euclidean violation: 0.2866 +/- 0.2321
Chronos violation: 0.1475 +/- 0.1452
Relative reduction: 48.5%
Wilcoxon p-value: 0.083984375
```

Interpretation boundary: this is a substantial effect size and OOD-persistent
world-model phenomenon result. It is not a statistical significance claim
because `p=0.0840` does not meet `p<0.05`.

## Cell 1: Setup

```python
!pip install -q torch numpy scipy pandas matplotlib seaborn

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, pearsonr

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 5)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path("/content")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {torch.device(DEVICE)}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Results dir: {RESULTS_DIR}")
```

## Cell 2: Config And Data

```python
CONFIG = {
    "n_seeds": 10,
    "n_train": 3000,
    "n_test": 512,
    "epochs": 250,
    "dim": 4,
    "latent_dim": 16,
    "width": 64,
    "lr": 1e-3,
    "t_total": 80,
    "t_obs": 10,
    "roll_steps": 50,
    "train_box": 2.0,
    "test_boxes": [2.0, 4.0, 8.0, 16.0, 32.0],
    "lambda_grid": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    "lambda_k1": 0.02,
    "device": DEVICE,
}

def make_lorentz_oscillator_trajs(n_seq, t_total, dim, rng, box=2.0, dt=0.08, noise=0.002):
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

def make_pairs(trajs, dim):
    x_t = trajs[:, :-1, :]
    x_next = trajs[:, 1:, :]
    return x_t.reshape(-1, dim).astype(np.float32), x_next.reshape(-1, dim).astype(np.float32)

def torch_lorentz_interval(dz):
    return dz[:, 0] ** 2 - torch.sum(dz[:, 1:] ** 2, dim=1)

def np_lorentz_interval(dz):
    return dz[..., 0] ** 2 - np.sum(dz[..., 1:] ** 2, axis=-1)

def causal_violation_rate_np(traj):
    delta = traj[:, 1:, :] - traj[:, :-1, :]
    return float(np.mean(np_lorentz_interval(delta) < 0.0))
```

## Cell 3: Models

```python
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

class EuclidJEPA(nn.Module):
    def __init__(self, dim, k, width):
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)
        self.pred = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k),
        )

    def encode(self, x):
        return self.enc(x)

    def predict_z(self, x):
        z = self.enc(x)
        return z + self.pred(z)

    def forward(self, x):
        return self.dec(self.predict_z(x))

    def rollout(self, x0, steps):
        xs = [x0]
        x = x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)

class ChronosJEPA(nn.Module):
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

    def latent_interval(self, dz):
        return torch.sum(dz * dz * self.g, dim=1)

    def K_value(self, z):
        return torch.sum(z * z * self.g, dim=1)

    def encode(self, x):
        return self.enc(x)

    def predict_z(self, x):
        z = self.enc(x)
        raw = self.pred(z)
        q = self.latent_interval(raw).unsqueeze(1)
        denom = torch.sqrt(torch.clamp(torch.abs(q), min=1e-3))
        dz = self.scale * raw / denom
        return z + dz

    def forward(self, x):
        return self.dec(self.predict_z(x))

    def rollout(self, x0, steps):
        xs = [x0]
        x = x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)
```

## Cell 4: Training And Evaluation

```python
def train_model(model, Xtr_np, Ytr_np, lambda_interval, lambda_causal):
    model.to(CONFIG["device"])
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=CONFIG["device"])
    Ytr = torch.tensor(Ytr_np, dtype=torch.float32, device=CONFIG["device"])

    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    mse_loss = nn.MSELoss()

    for _ in range(CONFIG["epochs"]):
        model.train()
        opt.zero_grad()

        z_t = model.encode(Xtr)
        z_next_true = model.encode(Ytr).detach()
        z_next_pred = model.predict_z(Xtr)
        x_next_pred = model.dec(z_next_pred)

        loss = mse_loss(x_next_pred, Ytr) + mse_loss(z_next_pred, z_next_true)

        if hasattr(model, "latent_interval"):
            dz_pred = z_next_pred - z_t
            interval_true = torch_lorentz_interval(Ytr - Xtr).detach()
            interval_pred = model.latent_interval(dz_pred)
            loss_interval = mse_loss(interval_pred, interval_true)
            loss_causal = torch.relu(-interval_pred).mean()
            loss_k1 = torch.mean((model.K_value(z_next_pred) - 1.0) ** 2)
            loss = loss + lambda_interval * loss_interval + lambda_causal * loss_causal + CONFIG["lambda_k1"] * loss_k1

        loss.backward()
        opt.step()

    return model

def eval_model(model, test_trajs):
    model.eval()
    x0_np = test_trajs[:, CONFIG["t_obs"], :]
    target_np = test_trajs[:, CONFIG["t_obs"]:CONFIG["t_obs"] + CONFIG["roll_steps"] + 1, :]
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=CONFIG["device"])

    with torch.no_grad():
        pred_np = model.rollout(x0, CONFIG["roll_steps"]).detach().cpu().numpy()

    return {
        "mse": float(np.mean((pred_np - target_np) ** 2)),
        "violation": causal_violation_rate_np(pred_np),
    }
```

## Cell 5: Corrected Mechanism Analysis

```python
def analyze_interval_preservation(model, test_trajs):
    model.eval()
    device = next(model.parameters()).device
    x_t = test_trajs[:, :-1, :]
    x_next = test_trajs[:, 1:, :]
    x_t_flat = x_t.reshape(-1, CONFIG["dim"])
    x_next_flat = x_next.reshape(-1, CONFIG["dim"])

    with torch.no_grad():
        x_t_tensor = torch.tensor(x_t_flat, dtype=torch.float32, device=device)
        x_next_tensor = torch.tensor(x_next_flat, dtype=torch.float32, device=device)
        z_t = model.encode(x_t_tensor)
        z_next = model.encode(x_next_tensor)
        dz = z_next - z_t
        interval_pred = model.latent_interval(dz).detach().cpu().numpy()

    interval_true = np_lorentz_interval(x_next_flat - x_t_flat)

    try:
        corr, p_corr = pearsonr(interval_pred, interval_true)
    except Exception:
        corr, p_corr = np.nan, np.nan

    sign_pred = np.sign(interval_pred)
    sign_true = np.sign(interval_true)
    sign_match = sign_pred == sign_true
    mask_timelike = interval_true > 0
    mask_spacelike = interval_true <= 0

    return {
        "correlation": float(corr),
        "p_correlation": float(p_corr),
        "sign_accuracy": float(np.mean(sign_match)),
        "timelike_accuracy": float(np.mean(sign_match[mask_timelike])) if mask_timelike.any() else float("nan"),
        "spacelike_accuracy": float(np.mean(sign_match[mask_spacelike])) if mask_spacelike.any() else float("nan"),
        "mse": float(np.mean((interval_pred - interval_true) ** 2)),
    }

def analyze_latent_geometry(model, test_trajs):
    model.eval()
    device = next(model.parameters()).device
    x_t = test_trajs[:, :-1, :].reshape(-1, CONFIG["dim"])
    x_next = test_trajs[:, 1:, :].reshape(-1, CONFIG["dim"])

    with torch.no_grad():
        x_t_tensor = torch.tensor(x_t, dtype=torch.float32, device=device)
        x_next_tensor = torch.tensor(x_next, dtype=torch.float32, device=device)
        z = model.encode(x_t_tensor)
        z_next = model.encode(x_next_tensor)
        dz = z_next - z
        intervals = model.latent_interval(dz).detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()

    g = np.ones(CONFIG["latent_dim"])
    g[1:] = -1.0
    K_values = np.sum(z_np * z_np * g, axis=1)
    interval_true = np_lorentz_interval(x_next - x_t)

    return {
        "K_mean": float(np.mean(K_values)),
        "K_std": float(np.std(K_values)),
        "K_min": float(np.min(K_values)),
        "K_max": float(np.max(K_values)),
        "K_within_threshold": float(np.mean(np.abs(K_values - 1.0) < 0.2)),
        "interval_positive_rate": float(np.mean(intervals > 0)),
        "interval_true_positive_rate": float(np.mean(interval_true > 0)),
        "interval_mean_pred": float(np.mean(intervals)),
        "interval_mean_true": float(np.mean(interval_true)),
    }
```

## Cell 6: Main Run

```python
print("=" * 80)
print("CHRONOS-K1 EXP 5 EXTENDED: MECHANISM VERIFICATION")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

results = []
mechanism_results = {}

for lam in CONFIG["lambda_grid"]:
    print(f"\nlambda = {lam:.4f}")
    mechanism_results[str(lam)] = {"chronos": {}}

    for seed in range(CONFIG["n_seeds"]):
        print(f"  seed {seed}", end=" ", flush=True)
        rng_train = np.random.default_rng(50000 + seed)
        train_trajs = make_lorentz_oscillator_trajs(
            CONFIG["n_train"], CONFIG["t_total"], CONFIG["dim"], rng_train, box=CONFIG["train_box"]
        )
        Xtr, Ytr = make_pairs(train_trajs, CONFIG["dim"])

        for model_name, model_class in [("euclid", EuclidJEPA), ("chronos", ChronosJEPA)]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_class(CONFIG["dim"], CONFIG["latent_dim"], CONFIG["width"])
            model = train_model(model, Xtr, Ytr, lambda_interval=lam, lambda_causal=lam)

            for box in CONFIG["test_boxes"]:
                rng_test = np.random.default_rng(60000 + seed + int(box * 100))
                test_trajs = make_lorentz_oscillator_trajs(
                    CONFIG["n_test"], CONFIG["t_total"], CONFIG["dim"], rng_test, box=box
                )
                metrics = eval_model(model, test_trajs)
                results.append({
                    "seed": seed,
                    "lambda": lam,
                    "model": model_name,
                    "box": box,
                    "violation_rate": metrics["violation"],
                    "mse": metrics["mse"],
                })

            if seed == 0 and model_name == "chronos":
                rng_test = np.random.default_rng(60000 + int(2.0 * 100))
                test_trajs_mech = make_lorentz_oscillator_trajs(
                    CONFIG["n_test"], CONFIG["t_total"], CONFIG["dim"], rng_test, box=2.0
                )
                mechanism_results[str(lam)]["chronos"] = {
                    "interval": analyze_interval_preservation(model, test_trajs_mech),
                    "geometry": analyze_latent_geometry(model, test_trajs_mech),
                }
        print("done")

print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total results collected: {len(results)}")
```

## Cell 7: Analyze And Save

```python
df = pd.DataFrame(results)
subset = df[(df["box"] == 2.0) & (df["lambda"] == 0.1)]
euclid = subset[subset["model"] == "euclid"]["violation_rate"].values
chronos = subset[subset["model"] == "chronos"]["violation_rate"].values
improvement = (np.mean(euclid) - np.mean(chronos)) / np.mean(euclid) * 100
_, p_val = wilcoxon(euclid, chronos)

print("SUMMARY: box=2.0, lambda=0.1")
print(f"Euclid:  {np.mean(euclid):.4f} +/- {np.std(euclid):.4f}")
print(f"Chronos: {np.mean(chronos):.4f} +/- {np.std(chronos):.4f}")
print(f"Reduction: {improvement:.1f}%")
print(f"Wilcoxon p-value: {p_val:.4f}")
print("Interpretation: substantial effect size; not p<0.05 unless future runs establish it.")

csv_path = RESULTS_DIR / "exp5_extended_fixed_v2_results.csv"
json_path = RESULTS_DIR / "exp5_mechanism_analysis_v2.json"
summary_path = RESULTS_DIR / "exp5_extended_fixed_v2_summary.json"

df.to_csv(csv_path, index=False)
with open(json_path, "w") as f:
    json.dump(mechanism_results, f, indent=2)
with open(summary_path, "w") as f:
    json.dump({
        "config": CONFIG,
        "main_result": {
            "box": 2.0,
            "lambda": 0.1,
            "euclid_mean": float(np.mean(euclid)),
            "euclid_std": float(np.std(euclid)),
            "chronos_mean": float(np.mean(chronos)),
            "chronos_std": float(np.std(chronos)),
            "reduction_percent": float(improvement),
            "wilcoxon_p": float(p_val),
            "interpretation": "substantial effect size; statistical significance not established at p<0.05",
        },
    }, f, indent=2)

plt.figure(figsize=(14, 5))
subset_01 = df[df["lambda"] == 0.1]
for model in ["euclid", "chronos"]:
    grouped = subset_01[subset_01["model"] == model].groupby("box")["violation_rate"].mean()
    plt.plot(grouped.index, grouped.values, marker="o", label=model)
plt.xscale("log")
plt.xlabel("Box")
plt.ylabel("Causal violation rate")
plt.title("Experiment 5 Extended: OOD robustness, lambda=0.1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "exp5_extended_fixed_v2_results.png", dpi=150)
plt.show()

print(f"CSV saved: {csv_path}")
print(f"Mechanism JSON saved: {json_path}")
print(f"Summary JSON saved: {summary_path}")
```
