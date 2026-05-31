"""
EXPERIMENT 7: METRIC-CONTROLLED NORMALIZATION TEST
Fixed version for Colab - handles both script and interactive execution
"""

import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

# ============================================================================
# COLAB DETECTION AND PATH SETUP
# ============================================================================

def is_colab():
    """Detect if running in Colab environment."""
    try:
        from google.colab import drive
        return True
    except ImportError:
        return False

def get_root_dir():
    """Get root directory, handling both script and interactive execution."""
    try:
        # In a script
        return Path(__file__).resolve().parents[1]
    except NameError:
        # In Jupyter/Colab/IPython interactive environment
        current_dir = Path.cwd()
        
        # Look for k1-manifold-core directory
        if (current_dir / "k1-manifold-core").exists():
            return current_dir / "k1-manifold-core"
        elif (current_dir.parent / "k1-manifold-core").exists():
            return current_dir.parent / "k1-manifold-core"
        elif "k1-manifold-core" in str(current_dir):
            # Already inside k1-manifold-core
            return current_dir
        else:
            # Default: assume we're in the right place
            print(f"⚠️  Warning: Could not determine root directory. Using current directory: {current_dir}")
            return current_dir


ROOT = get_root_dir()
SRC = ROOT / "src"

# Add to path
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

print(f"✓ ROOT: {ROOT}")
print(f"✓ SRC:  {SRC}")
print(f"✓ Path setup complete")
if is_colab():
    print(f"✓ Running in Colab environment\n")
else:
    print(f"✓ Running in local environment\n")

# ============================================================================
# IMPORT BENCHMARK DATASETS
# ============================================================================

try:
    from k1_manifold_core.benchmark_datasets import (
        make_spacelike_dataset,
        make_timelike_dataset,
    )
    print("✓ Successfully imported benchmark_datasets from k1_manifold_core\n")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure you have:")
    print("  1. Cloned the Chronos-K1 repository")
    print("  2. Run: pip install -e 'k1-manifold-core/[dev]'")
    print("  3. Set current directory to k1-manifold-core")
    sys.exit(1)

# ============================================================================
# SETUP RESULTS DIRECTORY
# ============================================================================

RESULTS_DIR = ROOT / "results" / "experiment_7_metric_controlled_normalization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Results directory: {RESULTS_DIR}\n")

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

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

print(f"Configuration:")
print(f"  DIM={DIM}, LATENT_DIM={LATENT_DIM}, WIDTH={WIDTH}")
print(f"  N_TRAIN={N_TRAIN}, N_TEST={N_TEST}")
print(f"  N_SEEDS={N_SEEDS}, EPOCHS={EPOCHS}")
print(f"  DEVICE={DEVICE}\n")

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

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
    """Chronos with Lorentz metric normalization."""
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
    """Chronos with Euclidean metric normalization (control)."""
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
    """Chronos with random metric (control)."""
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
    """Baseline: Euclidean prediction without metric normalization."""
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def np_lorentz_interval(dz):
    """Lorentz interval: η(v,v) = v₀² - ||v_s||²"""
    return dz[..., 0] ** 2 - np.sum(dz[..., 1:] ** 2, axis=-1)


def make_pairs(trajs):
    """Convert trajectories into (x_t, x_{t+1}) pairs for training."""
    x_t = trajs[:, :-1, :]
    x_next = trajs[:, 1:, :]
    return x_t.reshape(-1, DIM).astype(np.float32), x_next.reshape(-1, DIM).astype(np.float32)


def causal_violation_rate(traj):
    """Fraction of steps where Lorentz interval is negative (causal violation)."""
    delta = traj[:, 1:, :] - traj[:, :-1, :]
    return np.mean(np_lorentz_interval(delta) < 0.0)


def train_model(model, x_train_np, y_train_np, epochs):
    """Train a model on (x, y) prediction task."""
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
    """Evaluate model on test trajectories."""
    model.eval()
    x0 = torch.tensor(trajs[:, T_OBS, :], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred = model.rollout(x0, ROLL_STEPS).detach().cpu().numpy()
    return causal_violation_rate(pred)


# ============================================================================
# EXPERIMENT 7: MAIN TEST
# ============================================================================

def run_experiment_7(n_seeds, epochs, verbose=True):
    """Run metric-controlled normalization test across all models and datasets."""
    datasets = {"timelike": make_timelike_dataset, "spacelike": make_spacelike_dataset}
    models = {
        "euclidean_baseline": EuclideanBaseline,
        "chronos_lorentz": ChronosLorentz,
        "chronos_euclidean": ChronosEuclidean,
        "chronos_random": ChronosRandom,
    }
    raw = []
    
    total_runs = len(datasets) * len(models) * n_seeds
    run_count = 0
    
    for dname, dgen in datasets.items():
        for mname, mcls in models.items():
            for seed in range(n_seeds):
                run_count += 1
                if verbose and run_count % 10 == 0:
                    print(f"  Progress: {run_count}/{total_runs}")
                
                # Generate training data
                train_rng = np.random.default_rng(50000 + seed)
                train_trajs = dgen(N_TRAIN, T_TOTAL, DIM, train_rng, box=TEST_BOX)
                x_train, y_train = make_pairs(train_trajs)

                # Initialize and train model
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = (mcls(DIM, LATENT_DIM, WIDTH, seed=seed) 
                        if mname == "chronos_random" 
                        else mcls(DIM, LATENT_DIM, WIDTH))
                model = train_model(model, x_train, y_train, epochs=epochs)

                # Evaluate on test data
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
    """Analyze results: compute improvements and test for interaction."""
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


# ============================================================================
# MAIN ENTRY POINT - HANDLES BOTH SCRIPT AND INTERACTIVE
# ============================================================================

def main(n_seeds=N_SEEDS, epochs=EPOCHS, verbose=True):
    """Main function - can be called from Colab or as a script."""
    print(f"{'='*70}")
    print(f"EXPERIMENT 7: METRIC-CONTROLLED NORMALIZATION TEST")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Seeds:  {n_seeds}")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}\n")

    # Run experiment
    if verbose:
        print("Running experiment...")
    raw_df = run_experiment_7(n_seeds, epochs, verbose=verbose)
    raw_df.to_csv(RESULTS_DIR / "experiment_7_raw_results.csv", index=False)
    if verbose:
        print(f"✓ Saved: {RESULTS_DIR / 'experiment_7_raw_results.csv'}\n")

    # Analyze results
    if verbose:
        print("Analyzing results...")
    out_df, interaction_df = analyze_exp7(raw_df)
    out_df.to_csv(RESULTS_DIR / "experiment_7_raw_results_with_improvement.csv", index=False)
    interaction_df.to_csv(RESULTS_DIR / "experiment_7_metric_dataset_interaction.csv", index=False)
    
    if verbose:
        print(f"✓ Saved: {RESULTS_DIR / 'experiment_7_raw_results_with_improvement.csv'}")
        print(f"✓ Saved: {RESULTS_DIR / 'experiment_7_metric_dataset_interaction.csv'}\n")
        
        print("RESULTS:")
        print(interaction_df.to_string(index=False))
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
    
    return raw_df, out_df, interaction_df


# ============================================================================
# SCRIPT EXECUTION (when run as a script)
# ============================================================================

if __name__ == "__main__":
    # For Colab compatibility: ignore unknown arguments
    if is_colab():
        # In Colab, just run with defaults
        main(n_seeds=N_SEEDS, epochs=EPOCHS, verbose=True)
    else:
        # In script mode, try to parse arguments
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--smoke", action="store_true", help="Quick smoke test (4 seeds, 3 epochs)")
        ap.add_argument("--seeds", type=int, default=N_SEEDS, help=f"Number of seeds (default: {N_SEEDS})")
        ap.add_argument("--epochs", type=int, default=EPOCHS, help=f"Number of epochs (default: {EPOCHS})")
        
        # Handle unknown args for Colab (ignore them)
        args, unknown = ap.parse_known_args()
        
        seeds = 4 if args.smoke else args.seeds
        epochs = 3 if args.smoke else args.epochs
        
        main(n_seeds=seeds, epochs=epochs, verbose=True)
