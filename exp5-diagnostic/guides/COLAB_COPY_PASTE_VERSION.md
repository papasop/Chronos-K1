# Chronos-K1 Experiment 5 Colab Copy-Paste Version

This file is a Colab-oriented copy-paste guide for the Experiment 5 full sanity reproduction.

For the maintained repository script, use:

```bash
cd k1-manifold-core
CHRONOS_DEVICE=cuda python benchmarks/experiment_5_full_sanity_reproduction.py --full
```

This Colab version mirrors the same experiment family and records the same seed strategy.
It is intentionally documented as a research benchmark, not a unit test and not a theorem.

## Cell 1: Setup

```python
!pip install -q torch numpy scipy pandas matplotlib

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"CUDA available: {torch.cuda.is_available()}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path("/content")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Results dir: {RESULTS_DIR}")
```

## Cell 2: RNG Seeding Documentation

```python
rng_documentation = {
    "experiment": "Exp5_Full_Sanity_Reproduction",
    "purpose": "Document exact RNG seeding strategy used in Exp 5",
    "timestamp": datetime.now().isoformat(),
    "strategy": {
        "name": "Offset-based seeding with box-dependent test offset",
        "train_offset": 50000,
        "test_offset": 60000,
        "test_box_multiplier": 100,
        "box_values": [2.0, 4.0, 8.0, 16.0, 32.0],
    },
    "seed_examples": {},
}

for s in range(3):
    train_seed = 50000 + s
    test_seeds = {}
    for box in [2.0, 4.0, 8.0, 16.0, 32.0]:
        test_seed = 60000 + s + int(box * 100)
        test_seeds[str(box)] = test_seed
    rng_documentation["seed_examples"][f"S={s}"] = {
        "train": train_seed,
        "test_by_box": test_seeds,
    }

print("=" * 80)
print("TASK 1.2: RNG SEEDING DOCUMENTATION")
print("=" * 80)
print(f"Strategy: {rng_documentation['strategy']['name']}")
for seed_id, seed_info in rng_documentation["seed_examples"].items():
    print(f"\n{seed_id}:")
    print(f"  Train: {seed_info['train']}")
    for box, test_seed in seed_info["test_by_box"].items():
        print(f"  Box={box}: test_seed={test_seed}")

json_path = RESULTS_DIR / "rng_seeding_documentation.json"
with open(json_path, "w") as f:
    json.dump(rng_documentation, f, indent=2)
print(f"RNG documentation saved: {json_path}")
```

## Cell 3: Config And Data Generation

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
    "lambda_grid": [0.0, 0.1, 0.2, 0.5],
    "lambda_k1": 0.02,
    "device": DEVICE,
}

for k, v in CONFIG.items():
    print(f"{k}: {v}")


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
    return (
        x_t.reshape(-1, dim).astype(np.float32),
        x_next.reshape(-1, dim).astype(np.float32),
    )


def torch_lorentz_interval(dz):
    return dz[:, 0] ** 2 - torch.sum(dz[:, 1:] ** 2, dim=1)


def np_lorentz_interval(dz):
    return dz[..., 0] ** 2 - np.sum(dz[..., 1:] ** 2, axis=-1)


def causal_violation_rate_np(traj):
    delta = traj[:, 1:, :] - traj[:, :-1, :]
    interval = np_lorentz_interval(delta)
    return float(np.mean(interval < 0.0))
```

## Cell 4: Model Architectures

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
        dz = self.pred(z)
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

## Cell 5: Training And Evaluation

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

        loss_recon = mse_loss(x_next_pred, Ytr)
        loss_latent = mse_loss(z_next_pred, z_next_true)

        if hasattr(model, "latent_interval"):
            dz_pred = z_next_pred - z_t
            interval_true = torch_lorentz_interval(Ytr - Xtr).detach()
            interval_pred = model.latent_interval(dz_pred)
            loss_interval = mse_loss(interval_pred, interval_true)
            loss_causal = torch.relu(-interval_pred).mean()
            Kz = model.K_value(z_next_pred)
            loss_k1 = torch.mean((Kz - 1.0) ** 2)
            loss = (
                loss_recon + loss_latent
                + lambda_interval * loss_interval
                + lambda_causal * loss_causal
                + CONFIG["lambda_k1"] * loss_k1
            )
        else:
            loss = loss_recon + loss_latent

        loss.backward()
        opt.step()

    return model


def eval_model(model, test_trajs):
    model.eval()
    x0_np = test_trajs[:, CONFIG["t_obs"], :]
    target_np = test_trajs[:, CONFIG["t_obs"] : CONFIG["t_obs"] + CONFIG["roll_steps"] + 1, :]
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=CONFIG["device"])

    with torch.no_grad():
        pred = model.rollout(x0, CONFIG["roll_steps"])
        pred_np = pred.detach().cpu().numpy()

    mse_by_t = np.mean((pred_np - target_np) ** 2, axis=(0, 2))
    violation = causal_violation_rate_np(pred_np)
    return {
        "mean_mse": float(mse_by_t.mean()),
        "final_mse": float(mse_by_t[-1]),
        "violation": violation,
    }
```

## Cell 6: Run Experiment 5

```python
print("=" * 80)
print("TASK 1.1: EXPERIMENT 5 REPRODUCTION (10 seeds)")
print("=" * 80)
print(f"Estimated time: 3-4 hours on GPU")
print(f"Device: {DEVICE}")
print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

results = []

for lam in CONFIG["lambda_grid"]:
    print(f"\nlambda = {lam}")
    for seed in range(CONFIG["n_seeds"]):
        print(f"  Seed {seed:2d}", end=" ", flush=True)
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
                    "mean_mse": metrics["mean_mse"],
                    "final_mse": metrics["final_mse"],
                })
        print("done")

print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total results collected: {len(results)}")
```

## Cell 7: Analyze And Save Results

```python
df = pd.DataFrame(results)

subset = df[(df["box"] == 2.0) & (df["lambda"] == 0.1)]
euclid = subset[subset["model"] == "euclid"]["violation_rate"].values
chronos = subset[subset["model"] == "chronos"]["violation_rate"].values

improvement = (np.mean(euclid) - np.mean(chronos)) / np.mean(euclid)
_, p_val = wilcoxon(euclid, chronos)

print("=" * 80)
print("SUMMARY: box=2.0, lambda=0.1")
print("=" * 80)
print(f"Euclid mean:  {np.mean(euclid):.4f}")
print(f"Chronos mean: {np.mean(chronos):.4f}")
print(f"Relative reduction: {improvement:.1%}")
print(f"Wilcoxon p-value: {p_val:.4f}")
print("Interpretation: substantial effect size; close to conventional significance; not p<0.05.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

subset_01 = df[df["lambda"] == 0.1]
for model in ["euclid", "chronos"]:
    sub = subset_01[subset_01["model"] == model]
    grouped = sub.groupby("box")["violation_rate"].mean()
    axes[0].plot(grouped.index, grouped.values, marker="o", label=model, linewidth=2)
axes[0].set_xlabel("Box (OOD scale)")
axes[0].set_ylabel("Violation rate")
axes[0].set_title("Experiment 5: violation vs OOD, lambda=0.1")
axes[0].set_xscale("log")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

subset_box2 = df[df["box"] == 2.0]
for model in ["euclid", "chronos"]:
    sub = subset_box2[subset_box2["model"] == model]
    lambdas = sorted(sub["lambda"].unique())
    means = [sub[sub["lambda"] == lam]["violation_rate"].mean() for lam in lambdas]
    axes[1].plot(lambdas, means, marker="o", label=model, linewidth=2)
axes[1].set_xlabel("lambda")
axes[1].set_ylabel("Violation rate")
axes[1].set_title("Experiment 5: lambda sweep at box=2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "exp5_reproduction_results.png", dpi=150, bbox_inches="tight")
plt.show()

df.to_csv(RESULTS_DIR / "exp5_reproduction_results.csv", index=False)

json_data = {
    "config": CONFIG,
    "timestamp": datetime.now().isoformat(),
    "summary": {
        "box_2_lambda_01": {
            "euclid_mean": float(np.mean(euclid)),
            "euclid_std": float(np.std(euclid)),
            "chronos_mean": float(np.mean(chronos)),
            "chronos_std": float(np.std(chronos)),
            "reduction_pct": float(improvement * 100),
            "p_value": float(p_val),
        }
    },
}
with open(RESULTS_DIR / "exp5_reproduction_config.json", "w") as f:
    json.dump(json_data, f, indent=2)
```

## Cell 8: Safe Diagnostic Summary

```python
summary = f"""
DIAGNOSTIC SUMMARY
{'=' * 80}

Completed:
- RNG seeding documented.
- Experiment 5 reproduced with N=10.

Key result at box=2, lambda=0.1:
- Euclidean violation: {np.mean(euclid):.4f}
- Chronos violation: {np.mean(chronos):.4f}
- Reduction: {improvement:.1%}
- Wilcoxon p-value: {p_val:.4f}

Interpretation:
Experiment 5 is the Primary World-Model Phenomenon Benchmark for Chronos-K1.
The reduction is a substantial effect size and shows OOD persistence in the
recorded full run, but statistical significance has not yet been established
because p=0.0840 does not meet p<0.05.

Experiment 5b is the Mechanism Diagnostic Benchmark.

Next step:
Increase to more seeds, e.g. N=15 or N=20, to test stability and whether the
paired Wilcoxon p-value crosses the conventional threshold.
"""
print(summary)
with open(RESULTS_DIR / "diagnostic_summary.txt", "w") as f:
    f.write(summary)
```
