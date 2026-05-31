"""
EXPERIMENT 7: METRIC-CONTROLLED NORMALIZATION TEST
最终版本 - 包含正确的交互作用检验和精确的论文表述

核心修正：
1. ✓ 精确的措辞：Metric-controlled NORMALIZATION（不是 geometry）
2. ✓ 完整的交互作用：Metric × Dataset interaction test
3. ✓ 正确的诊断：基于改进的差异，而不是绝对值
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    RESULTS_DIR = Path("/content/experiment_7_metric_control")
else:
    RESULTS_DIR = Path.cwd() / "experiment_7_metric_control"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ Running in {'COLAB' if IN_COLAB else 'LOCAL'}")
print(f"✓ Results directory: {RESULTS_DIR}")
print()

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

N_SEEDS = 30  # 增加到 30 做稳健性验证
EPOCHS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✓ Device: {DEVICE}")
print(f"✓ N_SEEDS: {N_SEEDS}")
print()

torch.set_num_threads(2)

# ============================================================================
# DATASET GENERATORS
# ============================================================================

def make_timelike_dataset(n_seq, t_total, dim, rng, box=2.0, dt=0.08, noise=0.0):
    """Timelike geodesics: η(v,v) > 0"""
    x = rng.uniform(-box, box, size=(n_seq, dim)).astype(np.float32)
    x[:, 0] = 0.0
    
    v = rng.normal(size=(n_seq, dim)).astype(np.float32) * 0.1
    spatial_norm = np.linalg.norm(v[:, 1:], axis=1, keepdims=True)
    v[:, 0] = 1.5 * spatial_norm[:, 0]
    
    traj = [x.copy()]
    for t in range(t_total - 1):
        x = x + v * dt
        if noise > 0:
            x += noise * rng.normal(size=x.shape).astype(np.float32)
        traj.append(x.copy())
    
    return np.stack(traj, axis=1).astype(np.float32)


def make_spacelike_dataset(n_seq, t_total, dim, rng, box=2.0, dt=0.08, noise=0.0):
    """Spacelike geodesics: η(v,v) < 0"""
    x = rng.uniform(-box, box, size=(n_seq, dim)).astype(np.float32)
    x[:, 0] = 0.0
    
    v = rng.normal(size=(n_seq, dim)).astype(np.float32) * 0.1
    spatial_norm = np.linalg.norm(v[:, 1:], axis=1, keepdims=True)
    v[:, 0] = 0.3 * spatial_norm[:, 0]
    
    traj = [x.copy()]
    for t in range(t_total - 1):
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
            nn.Linear(dim, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k)
        )
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, k, dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, dim)
        )
    def forward(self, z):
        return self.net(z)


# ============================================================================
# METRIC VARIANTS (Normalization Metrics)
# ============================================================================

class ChronosLorentz(nn.Module):
    """Lorentz normalization metric"""
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
            nn.Linear(width, k)
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
    """Euclidean normalization metric"""
    def __init__(self, dim, k, width):
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)
        self.pred = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k)
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
    """Random normalization metric"""
    def __init__(self, dim, k, width, seed=42):
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)
        self.pred = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k)
        )
        self.scale = nn.Parameter(torch.tensor(0.05))
        
        torch.manual_seed(seed)
        A = torch.randn(k, k)
        M = 0.5 * (A + A.T)
        self.register_buffer("M", M)
    
    def metric(self, dz):
        quad_form = torch.sum((dz @ self.M) * dz, dim=1)
        return quad_form
    
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
    """Baseline: No normalization"""
    def __init__(self, dim, k, width):
        super().__init__()
        self.enc = Encoder(dim, k, width)
        self.dec = Decoder(k, dim, width)
        self.pred = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, k)
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
    
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        
        x_next_pred = model(x_train)
        loss = mse_loss(x_next_pred, y_train)
        
        if hasattr(model, 'encode'):
            z_t = model.encode(x_train)
            z_next_pred = model.predict_z(x_train)
            z_next_true = model.encode(y_train).detach()
            loss = loss + mse_loss(z_next_pred, z_next_true)
        
        loss.backward()
        opt.step()
    
    return model


def eval_model(model, trajs):
    model.eval()
    x0_np = trajs[:, T_OBS, :]
    target_np = trajs[:, T_OBS : T_OBS + ROLL_STEPS + 1, :]
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=DEVICE)
    
    with torch.no_grad():
        pred = model.rollout(x0, ROLL_STEPS)
        pred_np = pred.detach().cpu().numpy()
    
    violation_rate = causal_violation_rate(pred_np)
    return violation_rate


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment_7():
    """Run Metric-Controlled Normalization Test"""
    
    print("=" * 80)
    print("EXPERIMENT 7: METRIC-CONTROLLED NORMALIZATION TEST")
    print("=" * 80)
    print()
    
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
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        for model_name, model_class in models.items():
            violations = []
            
            for seed in range(N_SEEDS):
                if seed % 2 == 0:
                    print(f"  {model_name}: seed {seed}")
                
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
                
                raw_results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "violation": violation,
                })
            
            mean_violation = np.mean(violations)
            std_violation = np.std(violations)
            
            print(f"    → {mean_violation:.4f} ± {std_violation:.4f}")
    
    return pd.DataFrame(raw_results)


# ============================================================================
# ANALYSIS WITH METRIC × DATASET INTERACTION
# ============================================================================

def analyze_exp7(raw_df):
    """
    Complete analysis with proper Metric × Dataset interaction test.
    
    Key insight: We test whether the IMPROVEMENT differs between datasets.
    This is the true test of whether a metric is structure-specific.
    """
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: METRIC × DATASET INTERACTION ANALYSIS")
    print("=" * 80)
    print()
    
    # Dynamic baseline
    baselines = {}
    for dataset in raw_df['dataset'].unique():
        baseline_violation = raw_df[
            (raw_df['dataset'] == dataset) & 
            (raw_df['model'] == 'euclidean_baseline')
        ]['violation'].values
        baselines[dataset] = np.mean(baseline_violation)
    
    raw_df['improvement'] = raw_df.apply(
        lambda row: (baselines[row['dataset']] - row['violation']) / baselines[row['dataset']] * 100,
        axis=1
    )
    
    # Display results
    print("Results by Dataset and Model:\n")
    for dataset in ['timelike', 'spacelike']:
        print(f"{dataset.upper()}:")
        dataset_df = raw_df[raw_df['dataset'] == dataset]
        for model in ['euclidean_baseline', 'chronos_lorentz', 'chronos_euclidean', 'chronos_random']:
            violations = dataset_df[dataset_df['model'] == model]['violation'].values
            improvements = dataset_df[dataset_df['model'] == model]['improvement'].values
            if len(violations) > 0:
                print(f"  {model:25s}: {np.mean(violations):.4f} ± {np.std(violations):.4f}  "
                      f"({np.mean(improvements):+.1f}%)")
        print()
    
    # FIX B: Compute metric gains and test interaction
    print("=" * 80)
    print("METRIC × DATASET INTERACTION TEST")
    print("(Does each normalization metric affect timelike and spacelike differently?)")
    print("=" * 80)
    print()
    
    # For each metric, compute: metric_gain = metric_improvement - baseline_improvement
    metrics = ['chronos_lorentz', 'chronos_euclidean', 'chronos_random']
    
    results_by_metric = {}
    
    for metric_name in metrics:
        print(f"\n{metric_name.upper()}")
        print("-" * 80)
        
        timelike_imp = raw_df[(raw_df['dataset'] == 'timelike') & (raw_df['model'] == metric_name)]['improvement'].values
        spacelike_imp = raw_df[(raw_df['dataset'] == 'spacelike') & (raw_df['model'] == metric_name)]['improvement'].values
        
        print(f"  Timelike improvement:   {np.mean(timelike_imp):+.1f}% ± {np.std(timelike_imp):.1f}%")
        print(f"  Spacelike improvement:  {np.mean(spacelike_imp):+.1f}% ± {np.std(spacelike_imp):.1f}%")
        print(f"  Difference (TL - SL):   {np.mean(timelike_imp - spacelike_imp):+.1f}%")
        
        # Wilcoxon test: does this metric help timelike MORE than spacelike?
        # H0: improvement(timelike) = improvement(spacelike)
        # H1: improvement(timelike) > improvement(spacelike)
        stat, pval = stats.wilcoxon(timelike_imp, spacelike_imp, alternative='greater')
        
        print(f"\n  Wilcoxon Test (Timelike > Spacelike improvement):")
        print(f"    p-value: {pval:.6f}")
        
        if pval < 0.05:
            print(f"    ✅ SIGNIFICANT: This metric helps timelike MORE than spacelike")
            metric_result = True
        elif pval < 0.1:
            print(f"    ⚠️ BORDERLINE (p < 0.1)")
            metric_result = None
        else:
            print(f"    ❌ NOT SIGNIFICANT: This metric doesn't discriminate")
            metric_result = False
        
        # Effect size
        paired_diff = timelike_imp - spacelike_imp
        cohens_d = np.mean(paired_diff) / np.std(paired_diff, ddof=1)
        print(f"    Cohen's d: {cohens_d:.3f}")
        
        # Bootstrap CI
        n_boot = 10000
        boot_diffs = []
        for _ in range(n_boot):
            boot_sample = np.random.choice(paired_diff, size=len(paired_diff), replace=True)
            boot_diffs.append(np.mean(boot_sample))
        
        ci_lower = np.percentile(boot_diffs, 2.5)
        ci_upper = np.percentile(boot_diffs, 97.5)
        print(f"    Bootstrap 95% CI: [{ci_lower:+.1f}%, {ci_upper:+.1f}%]")
        
        results_by_metric[metric_name] = {
            'pval': pval,
            'result': metric_result,
            'cohens_d': cohens_d,
            'mean_diff': np.mean(paired_diff)
        }
    
    # Final diagnosis
    print()
    print("=" * 80)
    print("DIAGNOSIS: METRIC × DATASET INTERACTION")
    print("=" * 80)
    print()
    
    lorentz_result = results_by_metric['chronos_lorentz']['result']
    euclidean_result = results_by_metric['chronos_euclidean']['result']
    random_result = results_by_metric['chronos_random']['result']
    
    lorentz_pval = results_by_metric['chronos_lorentz']['pval']
    euclidean_pval = results_by_metric['chronos_euclidean']['pval']
    
    passed = False
    
    if lorentz_result is True:
        print("✅ LORENTZ metric shows SIGNIFICANT interaction:")
        print(f"   Helps timelike MORE than spacelike (p = {lorentz_pval:.6f})")
        
        if euclidean_result is not True:
            print("✅ EUCLIDEAN metric does NOT show this interaction")
            if random_result is not True:
                print("✅ RANDOM metric does NOT show this interaction")
                print()
                print("✅✅✅ EXPERIMENT 7 PASSED")
                print("Lorentz metric is SPECIFICALLY beneficial for timelike data")
                print("(Metric × Dataset interaction confirmed)")
                passed = True
            else:
                print("⚠️ But RANDOM also shows interaction (unexpected)")
                print("PARTIAL EVIDENCE")
                passed = None
        else:
            print("❌ But EUCLIDEAN also shows similar interaction")
            print("INCONCLUSIVE: Interaction not metric-specific")
            passed = None
    elif lorentz_result is None:
        print("⚠️ LORENTZ shows borderline interaction (p < 0.1)")
        print("INCONCLUSIVE: Suggest increasing N_SEEDS")
        passed = None
    else:
        print("❌ LORENTZ does NOT show significant interaction")
        print("Metric-specific benefits NOT confirmed")
        passed = False
    
    return passed


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀" * 20)
    print("Starting Experiment 7: Metric-Controlled Normalization Test")
    print("🚀" * 20 + "\n")
    
    raw_df = run_experiment_7()
    
    csv_path = RESULTS_DIR / "experiment_7_raw_results.csv"
    raw_df.to_csv(csv_path, index=False)
    print(f"\n✓ Raw results saved to {csv_path}")
    
    success = analyze_exp7(raw_df)
    
    print()
    print("=" * 80)
    if success is True:
        print("✅ EXPERIMENT 7 PASSED: Metric × Dataset interaction confirmed")
        print("   Lorentz normalization has structure-specific benefits")
    elif success is None:
        print("⚠️ EXPERIMENT 7 INCONCLUSIVE")
        print("   Suggest increasing N_SEEDS or data size")
    else:
        print("❌ EXPERIMENT 7 FAILED: No metric-specific benefits detected")
    print("=" * 80)
    
    print("\n" + "✅" * 20)
    print("Experiment 7 Complete!")
    print("✅" * 20)
