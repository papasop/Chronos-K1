"""
EXPERIMENT 8-B: LORENZ ATTRACTOR BENCHMARK  —  COLAB EDITION
Architectural-prior experiment on real nonlinear dynamics.
"""

# CONFIG
RUN_MODE = "FULL"   # "FULL" or "SMOKE"
DETERMINISTIC = True
OVERRIDE_SEEDS = None
OVERRIDE_EPOCHS = None

import sys, subprocess
try:
    import statsmodels  # noqa
except ImportError:
    print("Installing statsmodels ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "statsmodels"], check=False)

import os, numpy as np, torch, torch.nn as nn
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.stats import wasserstein_distance
from statsmodels.stats.multitest import multipletests
import json
import warnings
warnings.filterwarnings('ignore')

IN_COLAB = 'google.colab' in sys.modules
RESULTS_DIR = Path("/content/exp8b_lorenz") if IN_COLAB else Path.cwd() / "exp8b_lorenz"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Results: {RESULTS_DIR}\n")

SMOKE = (RUN_MODE.upper() == "SMOKE")

DATA_DIM = 3
LATENT_DIM = 16
WIDTH = 64
LR = 1e-3
EPOCHS = 60
N_SEEDS = 30

DT = 0.01
LYAPUNOV_TIME = 1.0 / 0.906
TRAIN_T = 12.0
N_TRAIN_TRAJ = 40
ROLL_STEPS = 600
LONG_ROLL_STEPS = 3000
VPT_THRESHOLD = 0.4
VPT_THRESHOLDS_SENSITIVITY = [0.3, 0.4, 0.5]

if SMOKE:
    N_SEEDS, EPOCHS, N_TRAIN_TRAJ = 3, 5, 5
    TRAIN_T, ROLL_STEPS, LONG_ROLL_STEPS = 5.0, 200, 500
    print("⚠️  SMOKE MODE — tiny config, results NOT valid. Set RUN_MODE='FULL' for real run.\n")

if OVERRIDE_SEEDS is not None:
    N_SEEDS = OVERRIDE_SEEDS
if OVERRIDE_EPOCHS is not None:
    EPOCHS = OVERRIDE_EPOCHS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)

if DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("✓ cudnn deterministic mode ENABLED (slower, fully reproducible)")

if DEVICE == "cpu":
    print("⚠️  DEVICE=cpu — no GPU detected. FULL run will be VERY slow.")

print("Exp8-B: Lorenz Attractor Benchmark (architecture-prior experiment)")
print(f"  mode={'SMOKE' if SMOKE else 'FULL'}, N_SEEDS={N_SEEDS}, EPOCHS={EPOCHS}, DEVICE={DEVICE}, deterministic={DETERMINISTIC}")


def lorenz_rhs(t, s, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = s
    return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]


def make_lorenz_trajectory(T, dt, ic, transient=5.0):
    t_total = T + transient
    t_eval = np.arange(0, t_total, dt)
    sol = solve_ivp(lorenz_rhs, [0, t_total], ic, t_eval=t_eval, rtol=1e-9, atol=1e-9, method='RK45')
    traj = sol.y.T
    n_transient = int(transient / dt)
    return traj[n_transient:].astype(np.float32)


def sample_ic(rng):
    return [rng.uniform(-15, 15), rng.uniform(-20, 20), rng.uniform(5, 40)]


_ref = make_lorenz_trajectory(50.0, DT, [1.0, 1.0, 1.0])
DATA_MEAN = _ref.mean(axis=0)
DATA_STD = _ref.std(axis=0)


def normalize(traj):
    return (traj - DATA_MEAN) / DATA_STD


def make_pairs(traj_norm):
    return traj_norm[:-1].astype(np.float32), traj_norm[1:].astype(np.float32)


class Encoder(nn.Module):
    def __init__(s, d, k, w):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d,w), nn.Tanh(), nn.Linear(w,w), nn.Tanh(), nn.Linear(w,k))
    def forward(s, x): return s.net(x)


class Decoder(nn.Module):
    def __init__(s, k, d, w):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(k,w), nn.Tanh(), nn.Linear(w,w), nn.Tanh(), nn.Linear(w,d))
    def forward(s, z): return s.net(z)


class _Base(nn.Module):
    def forward(s, x): return s.dec(s.predict_z(x))
    def rollout(s, x0, steps):
        xs=[x0]; x=x0
        for _ in range(steps):
            x=s.forward(x); xs.append(x)
        return torch.stack(xs, dim=1)
    def encode(s, x): return s.enc(x)


class Baseline(_Base):
    def __init__(s, d=DATA_DIM, k=LATENT_DIM, w=WIDTH):
        super().__init__()
        s.enc=Encoder(d,k,w); s.dec=Decoder(k,d,w)
        s.pred=nn.Sequential(nn.Linear(k,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh(),nn.Linear(w,k))
        s.scale=nn.Parameter(torch.tensor(0.05))
    def predict_z(s, x):
        z=s.enc(x); return z + s.pred(z)*s.scale


class LorentzNormalized(_Base):
    def __init__(s, d=DATA_DIM, k=LATENT_DIM, w=WIDTH):
        super().__init__()
        s.enc=Encoder(d,k,w); s.dec=Decoder(k,d,w)
        g=torch.ones(k); g[1:]=-1.0; s.register_buffer("g",g)
        s.pred=nn.Sequential(nn.Linear(k,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh(),nn.Linear(w,k))
        s.scale=nn.Parameter(torch.tensor(0.05))
    def metric(s,dz): return torch.sum(dz*dz*s.g,dim=1)
    def predict_z(s,x):
        z=s.enc(x); raw=s.pred(z)
        q=s.metric(raw).unsqueeze(1)
        denom=torch.sqrt(torch.clamp(torch.abs(q),min=1e-3))
        return z + s.scale*raw/denom


class EuclideanNormalized(_Base):
    def __init__(s, d=DATA_DIM, k=LATENT_DIM, w=WIDTH):
        super().__init__()
        s.enc=Encoder(d,k,w); s.dec=Decoder(k,d,w)
        s.pred=nn.Sequential(nn.Linear(k,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh(),nn.Linear(w,k))
        s.scale=nn.Parameter(torch.tensor(0.05))
    def metric(s,dz): return torch.sum(dz*dz,dim=1)
    def predict_z(s,x):
        z=s.enc(x); raw=s.pred(z)
        q=s.metric(raw).unsqueeze(1)
        denom=torch.sqrt(torch.clamp(q,min=1e-3))
        return z + s.scale*raw/denom


_FIXED_M = {}
def _get_M(mid, k=LATENT_DIM):
    if mid not in _FIXED_M:
        g=torch.Generator().manual_seed(mid*1000)
        A=torch.randn(k,k,generator=g)
        M=0.5*(A+A.T)
        M = M / (torch.linalg.matrix_norm(M, ord=2) + 1e-9)
        _FIXED_M[mid]=M
    return _FIXED_M[mid]


class RandomNormalized(_Base):
    def __init__(s, d=DATA_DIM, k=LATENT_DIM, w=WIDTH, metric_id=1):
        super().__init__()
        s.enc=Encoder(d,k,w); s.dec=Decoder(k,d,w)
        s.pred=nn.Sequential(nn.Linear(k,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh(),nn.Linear(w,k))
        s.scale=nn.Parameter(torch.tensor(0.05))
        s.register_buffer("M", _get_M(metric_id,k))
    def metric(s,dz): return torch.sum((dz@s.M)*dz,dim=1)
    def predict_z(s,x):
        z=s.enc(x); raw=s.pred(z)
        q=s.metric(raw).unsqueeze(1)
        denom=torch.sqrt(torch.clamp(torch.abs(q),min=1e-3))
        return z + s.scale*raw/denom


def build_training_set(seed):
    rng = np.random.default_rng(10000 + seed)
    X, Y = [], []
    for _ in range(N_TRAIN_TRAJ):
        traj = make_lorenz_trajectory(TRAIN_T, DT, sample_ic(rng))
        x, y = make_pairs(normalize(traj))
        X.append(x); Y.append(y)
    return np.concatenate(X), np.concatenate(Y)


def train_model(model, X, Y):
    model.to(DEVICE)
    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    n = xt.shape[0]; bs = 4096
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs].to(DEVICE)
            xb, yb = xt[idx], yt[idx]
            opt.zero_grad()
            loss = mse(model(xb), yb)
            zb = model.predict_z(xb); zt = model.encode(yb).detach()
            loss = loss + mse(zb, zt)
            loss.backward(); opt.step()
    return model


def compute_vpt_multi(pred, true, thresholds):
    err = np.sqrt(np.mean((pred - true)**2, axis=1))
    out = {}
    for th in thresholds:
        exceed = np.where(err > th)[0]
        steps = len(true) if len(exceed)==0 else exceed[0]
        out[th] = steps * DT / LYAPUNOV_TIME
    return out


def attractor_distance(pred_long, true_long):
    return np.mean([wasserstein_distance(pred_long[:,i], true_long[:,i])/(true_long[:,i].std()+1e-9) for i in range(3)])


def short_horizon_mse(pred, true, horizon=100):
    h = min(horizon, len(true))
    return np.mean((pred[:h]-true[:h])**2)


def evaluate(model, seed):
    model.eval()
    rng = np.random.default_rng(20000 + seed)
    test_traj = normalize(make_lorenz_trajectory(ROLL_STEPS*DT + 1.0, DT, sample_ic(rng)))
    x0 = torch.tensor(test_traj[0:1], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred = model.rollout(x0, ROLL_STEPS).squeeze(0).cpu().numpy()
    true = test_traj[:ROLL_STEPS+1]
    m = min(len(pred), len(true))
    vpt_multi = compute_vpt_multi(pred[:m], true[:m], VPT_THRESHOLDS_SENSITIVITY)
    vpt = vpt_multi[VPT_THRESHOLD]
    smse = short_horizon_mse(pred[:m], true[:m])

    long_true = normalize(make_lorenz_trajectory(LONG_ROLL_STEPS*DT + 1.0, DT, sample_ic(rng)))
    x0L = torch.tensor(long_true[0:1], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        long_pred = model.rollout(x0L, LONG_ROLL_STEPS).squeeze(0).cpu().numpy()
    diverged = (not np.all(np.isfinite(long_pred))) or (np.max(np.abs(long_pred)) > 100)
    if diverged:
        attr = 10.0
    else:
        attr = min(attractor_distance(long_pred, long_true[:len(long_pred)]), 10.0)
    return vpt, attr, smse, vpt_multi, bool(diverged)


def run():
    import time
    models = {
        'Baseline': lambda: Baseline(),
        'LorentzNormalized': lambda: LorentzNormalized(),
        'EuclideanNormalized': lambda: EuclideanNormalized(),
        'RandomMetric_1': lambda: RandomNormalized(metric_id=1),
        'RandomMetric_2': lambda: RandomNormalized(metric_id=2),
        'RandomMetric_3': lambda: RandomNormalized(metric_id=3),
    }
    rows = []
    for seed in range(N_SEEDS):
        X, Y = build_training_set(seed)
        for name, ctor in models.items():
            torch.manual_seed(seed); np.random.seed(seed)
            t0 = time.time()
            model = train_model(ctor(), X, Y)
            train_time = time.time() - t0
            vpt, attr, smse, vpt_multi, diverged = evaluate(model, seed)
            row = {'seed':seed,'model':name,'vpt':vpt,'attractor':attr,'mse':smse,
                   'train_time_s':round(train_time,2),'diverged':diverged}
            for th, v in vpt_multi.items():
                row[f'vpt_th{th}'] = v
            rows.append(row)
        if (seed+1) % 5 == 0:
            print(f"  seed {seed+1}/{N_SEEDS} done")
    return pd.DataFrame(rows)


def safe_wilcoxon_annotated(a, b, alternative):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0, "all-zero-diff fallback"
    try:
        return stats.wilcoxon(a, b, alternative=alternative).pvalue, ""
    except ValueError as e:
        return 1.0, f"wilcoxon error fallback ({e})"


def analyze(df):
    model_order = ['Baseline','LorentzNormalized','EuclideanNormalized','RandomMetric_1','RandomMetric_2','RandomMetric_3']
    controls = ['EuclideanNormalized','RandomMetric_1','RandomMetric_2','RandomMetric_3']
    def seeded(name): return df[df['model']==name].sort_values('seed')
    L = seeded('LorentzNormalized')

    primary_p, primary_lab, primary_note = [], [], []
    for ctrl in controls:
        C = seeded(ctrl)
        p, note = safe_wilcoxon_annotated(L['vpt'].values, C['vpt'].values, 'greater')
        primary_p.append(p); primary_lab.append(f"VPT: Lorentz>{ctrl}"); primary_note.append(note)
    reject_p, corr_p, _, _ = multipletests(primary_p, method='holm')

    beats_euclidean = reject_p[0]
    beats_all_random = all(reject_p[1:])

    E = seeded('EuclideanNormalized')
    sens = {}
    for th in VPT_THRESHOLDS_SENSITIVITY:
        col = f'vpt_th{th}'
        if col in df.columns:
            p, _ = safe_wilcoxon_annotated(L[col].values, E[col].values, 'greater')
            sens[th] = {'p':p, 'dir_consistent': L[col].mean() > E[col].mean()}
    consistent = all(s['dir_consistent'] for s in sens.values()) if sens else False

    summary = pd.DataFrame({
        'comparison':[f'Lorentz_vs_{c}' for c in controls],
        'mean_delta_vpt':[L['vpt'].mean()-seeded(c)['vpt'].mean() for c in controls],
        'holm_p':list(corr_p), 'significant':list(reject_p),
    })
    summary.to_csv(RESULTS_DIR / "exp8b_summary_table.csv", index=False)

    info = {
        'vpt': {'L':L['vpt'].mean(),'E':E['vpt'].mean(),'beats_euclidean':beats_euclidean,
                'beats_all_random':beats_all_random,'holm_p_vs_euclidean':corr_p[0],
                'holm_p_vs_random':list(corr_p[1:])},
        'threshold_consistent': consistent,
    }
    return info


def verdict(info):
    vpt = info['vpt']
    beats_euc = vpt['beats_euclidean']
    beats_all_rand = vpt['beats_all_random']
    primary_pass = beats_euc and beats_all_rand
    if primary_pass:
        return "PRIOR_HELPS"
    if beats_euc:
        return "PARTIAL_VS_EUCLIDEAN_ONLY"
    return "NO_BENEFIT"


def main():
    print("\n" + "🚀"*20)
    print("EXPERIMENT 8-B: LORENZ ATTRACTOR BENCHMARK")
    print("🚀"*20 + "\n")

    config = {
        "mode": "SMOKE" if SMOKE else "FULL", "deterministic": DETERMINISTIC,
        "DATA_DIM": DATA_DIM, "LATENT_DIM": LATENT_DIM, "WIDTH": WIDTH,
        "LR": LR, "EPOCHS": EPOCHS, "N_SEEDS": N_SEEDS,
        "DT": DT, "LYAPUNOV_TIME": LYAPUNOV_TIME, "TRAIN_T": TRAIN_T,
        "N_TRAIN_TRAJ": N_TRAIN_TRAJ, "ROLL_STEPS": ROLL_STEPS,
        "LONG_ROLL_STEPS": LONG_ROLL_STEPS, "VPT_THRESHOLD": VPT_THRESHOLD,
        "VPT_THRESHOLDS_SENSITIVITY": VPT_THRESHOLDS_SENSITIVITY,
        "DATA_MEAN": DATA_MEAN.tolist(), "DATA_STD": DATA_STD.tolist(),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    df = run()
    df.to_csv(RESULTS_DIR / "exp8b_results.csv", index=False)
    info = analyze(df)
    v = verdict(info)
    print(f"\nFINAL RESULT: {v}")
    return df, info, v


if __name__ == "__main__":
    df, info, v = main()
