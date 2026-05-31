"""
EXPERIMENT 7: METRIC-CONTROLLED NORMALIZATION TEST

Core claims:
1. Metric-controlled normalization (not general geometry superiority)
2. Metric x Dataset interaction test
3. Diagnosis on improvement deltas, not absolute values alone
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats


# ============================================================================
# SETUP
# ============================================================================

IN_COLAB = "google.colab" in sys.modules
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "experiment_7_metric_controlled_normalization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
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


# ============================================================================
# DATASET GENERATORS
# ============================================================================

def make_timelike_dataset(n_seq, t_total, dim, rng, box=2.0, dt=0.08, noise=0.0):
    """Timelike geodesics: eta(v,v) > 0."""
    x = rng.uniform(-box, box, size=(n_seq, dim)).astype(np.float32)
    x[:, 0] = 0.0

    v = rng.normal(size=(n_seq, dim)).astype(np.float32) * 0.1
    spatial_norm = np.linalg.norm(v[:, 1:], axis=1, keepdims=True)
    v[:, 0] = 1.5 * spatial_norm[:, 0]

    traj = [x.copy()]
    for _ in range(t_total - 1):
        x = x + v * dt
        if noise > 0:
            x += noise * rng.normal(size=x.shape).astype(np.float32)
        traj.append(x.copy())

    return np.stack(traj, axis=1).astype(np.float32)


def make_spacelike_dataset(n_seq, t_total, dim, rng, box=2.0, dt=0.08, noise=0.0):
    """Spacelike geodesics: eta(v,v) < 0."""
    x = rng.uniform(-box, box, size=(n_seq, dim)).astype(np.float32)
    x[:, 0] = 0.0

    v = rng.normal(size=(n_seq, dim)).astype(np.float32) * 0.1
    spatial_norm = np.linalg.norm(v[:, 1:], axis=1, keepdims=True)
    v[:, 0] = 0.3 * spatial_norm[:, 0]

    traj = [x.copy()]
    for _ in range(t_total - 1):
        x = x + v * dt
        if noise > 0:
            x += noise * rng.normal(size=x.shape).astype(np.float32)
        traj.append(x.copy())

    return np.stack(traj, axis=1).astype(np.float32)


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, dim, k, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, k),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, k, dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, dim),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================================
# METRIC VARIANTS (NORMALIZATION METRICS)
# ============================================================================

class ChronosLorentz(nn.Module):
    """Lorentz normalization metric."""

    def __init__(self, dim, k, width):
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
        xs = [x0]
        x = x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)


class ChronosEuclidean(nn.Module):
    """Euclidean normalization metric."""

    def __init__(self, dim, k, width):
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
        xs = [x0]
        x = x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)


class ChronosRandom(nn.Module):
    """Random symmetric normalization metric."""

    def __init__(self, dim, k, width, seed=42):
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
        xs = [x0]
        x = x0
        for _ in range(steps):
            x = self.forward(x)
            xs.append(x)
        return torch.stack(xs, dim=1)


class EuclideanBaseline(nn.Module):
    """Baseline with no metric normalization."""

    def __init__(self, dim, k, width):
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
        self.scale = nn.Parameter(torch.tensor(0.05))

    def encode(self, x):
        return self.enc(x)

    def predict_z(self, x):
        z = self.enc(x)
        dz = self.pred(z) * self.scale
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


# ============================================================================
# EVALUATION
# ============================================================================

def np_lorentz_interval(dz):
    return dz[..., 0] ** 2 - np.sum(dz[..., 1:] ** 2, axis=-1)


def make_pairs(trajs):
    x_t = trajs[:, :-1, :]
    x_next = trajs[:, 1:, :]
    return x_t.reshape(-1, DIM).astype(np.float32), x_next.reshape(-1, DIM).astype(np.float32)


def causal_violation_rate(traj):
    delta = traj[:, 1:, :] - traj[:, :-1, :]
    interval = np_lorentz_interval(delta)
    return np.mean(interval < 0.0)


def train_model(model, x_train_np, y_train_np):
    model.to(DEVICE)
    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()

        x_next_pred = model(x_train)
        loss = mse_loss(x_next_pred, y_train)

        if hasattr(model, "encode"):
            z_next_pred = model.predict_z(x_train)
            z_next_true = model.encode(y_train).detach()
            loss = loss + mse_loss(z_next_pred, z_next_true)

        loss.backward()
        opt.step()

    return model


def eval_model(model, trajs):
    model.eval()
    x0_np = trajs[:, T_OBS, :]
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred = model.rollout(x0, ROLL_STEPS)
        pred_np = pred.detach().cpu().numpy()

    return causal_violation_rate(pred_np)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment_7():
    print("=" * 80)
    print("EXPERIMENT 7: METRIC-CONTROLLED NORMALIZATION TEST")
    print("=" * 80)
    print(f"device={DEVICE} seeds={N_SEEDS} results_dir={RESULTS_DIR}")

    datasets = {
        "timelike": make_timelike_dataset,
        "spacelike": make_spacelike_dataset,
    }

    models = {
        "euclidean_baseline": EuclideanBaseline,
        "chronos_lorentz": ChronosLorentz,
        "chronos_euclidean": ChronosEuclidean,
        "chronos_random": ChronosRandom,
    }

    raw_results = []

    for dataset_name, dataset_generator in datasets.items():
        print(f"\n[dataset={dataset_name}]")
        for model_name, model_class in models.items():
            violations = []
            for seed in range(N_SEEDS):
                train_rng = np.random.default_rng(50000 + seed)
                train_trajs = dataset_generator(N_TRAIN, T_TOTAL, DIM, train_rng, box=TEST_BOX)
                x_train, y_train = make_pairs(train_trajs)

                torch.manual_seed(seed)
                np.random.seed(seed)

                if model_name == "chronos_random":
                    model = model_class(DIM, LATENT_DIM, WIDTH, seed=seed)
                else:
                    model = model_class(DIM, LATENT_DIM, WIDTH)

                model = train_model(model, x_train, y_train)

                test_rng = np.random.default_rng(60000 + seed)
                test_trajs = dataset_generator(N_TEST, T_TOTAL, DIM, test_rng, box=TEST_BOX)
                violation = eval_model(model, test_trajs)
                violations.append(violation)

                raw_results.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "violation": violation,
                    }
                )

            print(f"  {model_name:24s} -> {np.mean(violations):.4f} +/- {np.std(violations):.4f}")

    return pd.DataFrame(raw_results)


# ============================================================================
# ANALYSIS: METRIC x DATASET INTERACTION
# ============================================================================

def analyze_exp7(raw_df):
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: METRIC x DATASET INTERACTION ANALYSIS")
    print("=" * 80)

    baselines = {}
    for dataset in raw_df["dataset"].unique():
        baseline_violation = raw_df[
            (raw_df["dataset"] == dataset) & (raw_df["model"] == "euclidean_baseline")
        ]["violation"].values
        baselines[dataset] = np.mean(baseline_violation)

    raw_df = raw_df.copy()
    raw_df["improvement"] = raw_df.apply(
        lambda row: (baselines[row["dataset"]] - row["violation"]) / baselines[row["dataset"]] * 100,
        axis=1,
    )

    interaction_rows = []
    metrics = ["chronos_lorentz", "chronos_euclidean", "chronos_random"]

    for metric_name in metrics:
        timelike_imp = raw_df[(raw_df["dataset"] == "timelike") & (raw_df["model"] == metric_name)][
            "improvement"
        ].values
        spacelike_imp = raw_df[(raw_df["dataset"] == "spacelike") & (raw_df["model"] == metric_name)][
            "improvement"
        ].values

        stat, pval = stats.wilcoxon(timelike_imp, spacelike_imp, alternative="greater")
        paired_diff = timelike_imp - spacelike_imp
        cohens_d = np.mean(paired_diff) / np.std(paired_diff, ddof=1)

        n_boot = 10000
        boot_diffs = []
        for _ in range(n_boot):
            boot_sample = np.random.choice(paired_diff, size=len(paired_diff), replace=True)
            boot_diffs.append(np.mean(boot_sample))
        ci_lower = np.percentile(boot_diffs, 2.5)
        ci_upper = np.percentile(boot_diffs, 97.5)

        interaction_rows.append(
            {
                "metric": metric_name,
                "timelike_improvement_mean_pct": np.mean(timelike_imp),
                "spacelike_improvement_mean_pct": np.mean(spacelike_imp),
                "difference_tl_minus_sl_pct": np.mean(paired_diff),
                "wilcoxon_one_sided_p": pval,
                "cohens_d": cohens_d,
                "bootstrap_ci95_low_pct": ci_lower,
                "bootstrap_ci95_high_pct": ci_upper,
            }
        )

        print(f"\n{metric_name}")
        print(f"  timelike improvement:  {np.mean(timelike_imp):+.2f}%")
        print(f"  spacelike improvement: {np.mean(spacelike_imp):+.2f}%")
        print(f"  diff (TL-SL):          {np.mean(paired_diff):+.2f}%")
        print(f"  wilcoxon (TL>SL):      p={pval:.6f}")
        print(f"  cohen's d:             {cohens_d:.3f}")
        print(f"  bootstrap 95% CI:      [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")

    interaction_df = pd.DataFrame(interaction_rows)
    return raw_df, interaction_df


def summarize_pass(interaction_df):
    row_l = interaction_df[interaction_df["metric"] == "chronos_lorentz"].iloc[0]
    row_e = interaction_df[interaction_df["metric"] == "chronos_euclidean"].iloc[0]
    row_r = interaction_df[interaction_df["metric"] == "chronos_random"].iloc[0]

    lorentz_sig = row_l["wilcoxon_one_sided_p"] < 0.05
    euclid_sig = row_e["wilcoxon_one_sided_p"] < 0.05
    random_sig = row_r["wilcoxon_one_sided_p"] < 0.05

    if lorentz_sig and (not euclid_sig) and (not random_sig):
        return "passed"
    if row_l["wilcoxon_one_sided_p"] < 0.1:
        return "inconclusive"
    return "failed"


if __name__ == "__main__":
    raw_df = run_experiment_7()

    raw_csv = RESULTS_DIR / "experiment_7_raw_results.csv"
    raw_df.to_csv(raw_csv, index=False)

    raw_with_imp_df, interaction_df = analyze_exp7(raw_df)
    imp_csv = RESULTS_DIR / "experiment_7_raw_results_with_improvement.csv"
    interaction_csv = RESULTS_DIR / "experiment_7_metric_dataset_interaction.csv"
    raw_with_imp_df.to_csv(imp_csv, index=False)
    interaction_df.to_csv(interaction_csv, index=False)

    outcome = summarize_pass(interaction_df)

    print("\n" + "=" * 80)
    if outcome == "passed":
        print("EXPERIMENT 7 PASSED: Lorentz-specific Metric x Dataset interaction confirmed")
    elif outcome == "inconclusive":
        print("EXPERIMENT 7 INCONCLUSIVE: borderline Lorentz interaction, increase seeds")
    else:
        print("EXPERIMENT 7 FAILED: no Lorentz-specific interaction detected")
    print("=" * 80)

    print(f"saved: {raw_csv}")
    print(f"saved: {imp_csv}")
    print(f"saved: {interaction_csv}")
