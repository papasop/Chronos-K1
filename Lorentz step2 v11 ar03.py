"""
Lorentzian Backbone — Step 2 (v11): MLP backbone + sign-aware head
==================================================================

走法 A from the post-mortem:
  v6/v7 confirmed mq differential between classes existed in MLP
  backbone (seed 43 v7 BC=0.460, AUC=0.943). But all centroid sign-
  splits were 0/N because the original LorentzianHead is BILINEAR in
  h and mathematically cannot use the sign of mq = ‖h_s‖² - ‖h_t‖²
  for classification (mq is QUADRATIC in h).

  v11 isolates this: keeps the v7 MLP backbone (SplitNorm + causal
  residual + ReLU body) but REPLACES the head with one that has
  explicit access to mq.

  Three head types via --head-type:
    'linear'        — original LorentzianHead (control / regression test)
    'mq-mlp'        — MLP over [h, mq], DEFAULT
    'quad-features' — linear over [h, ‖h_t‖², ‖h_s‖²]

  Hypothesis: with mq-mlp head, sign-split should EMERGE naturally if
  the task has any sign-aware optimal solution. The architecture
  doesn't force class 0 = timelike — it just makes that decision
  REPRESENTABLE. If the model spontaneously chooses sign-split, the
  paper claim is justified. If it doesn't, sign-split isn't the
  natural representation for this task.

CLI:
    python lorentz_step2_causal_seq_v11.py --n-seeds 3 --head-type mq-mlp --ar-rho 0.9
    python lorentz_step2_causal_seq_v11.py --n-seeds 3 --head-type linear --ar-rho 0.9  # control
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Device: {device}')


def _json_safe(o):
    """Convert numpy scalar / tensor / ndarray types to JSON-safe Python.
    Used as the `default=` callback in json.dump so that stray numpy
    objects don't blow up serialization."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, 'item'):     # torch tensors with single element
        try:
            return o.item()
        except Exception:
            pass
    raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable')

# ─── Hyperparameters ──────────────────────────────────────────────
HIDDEN_DIM   = 128
D_T_RATIO    = 0.5
N_LAYERS     = 3
N_CLASSES    = 2          # serial-dependent vs iid (NOT 'causal' — that
                          # would require a more careful task design;
                          # see Step-2 limitations / Issue #4 in review)
SEQ_LEN      = 32         # sequence length
N_TRAIN      = 8000
N_TEST       = 2000
BATCH_SIZE   = 64
EPOCHS       = 15
LR           = 3e-4
SEED         = 42
AR_RHO       = 0.3        # AR(1) coefficient — HARD task (preconfigured)
NOISE_STD    = 0.5        # AR noise scale, chosen so marginal var ≈ 1
                          # Var(AR(1)) = σ²/(1-ρ²) = 0.25 / 0.19 ≈ 1.32

D_T = max(1, int(HIDDEN_DIM * D_T_RATIO))
D_S = HIDDEN_DIM - D_T
print(f'Hidden dim={HIDDEN_DIM}  D_t={D_T}  D_s={D_S}  '
      f'(symmetric: init post-norm mq ≈ 0)')


# ══════════════════════════════════════════════════════════════════
# Pillar 1: Lorentzian utilities (unchanged from Step 1)
# ══════════════════════════════════════════════════════════════════
def lorentz_inner(u, v, d_t):
    t_part = -(u[..., :d_t] * v[..., :d_t]).sum(-1)
    s_part =  (u[..., d_t:] * v[..., d_t:]).sum(-1)
    return t_part + s_part


def compute_mq(x, d_t):
    t_norm_sq = (x[..., :d_t] ** 2).sum(-1)
    s_norm_sq = (x[..., d_t:] ** 2).sum(-1)
    return s_norm_sq - t_norm_sq


# ══════════════════════════════════════════════════════════════════
# Pillar 3: Split-Norm (unchanged from Step 1, beta OFF)
# ══════════════════════════════════════════════════════════════════
class SplitNorm(nn.Module):
    def __init__(self, dim, d_t, eps=1e-6, use_beta=False):
        super().__init__()
        assert 0 < d_t < dim
        self.dim, self.d_t, self.d_s, self.eps = dim, d_t, dim - d_t, eps
        self.gamma_t = nn.Parameter(torch.ones(d_t))
        self.gamma_s = nn.Parameter(torch.ones(self.d_s))
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('beta', None)

    def forward(self, x):
        t = x[..., :self.d_t]
        s = x[..., self.d_t:]
        rms_t = torch.sqrt((t ** 2).mean(-1, keepdim=True) + self.eps)
        rms_s = torch.sqrt((s ** 2).mean(-1, keepdim=True) + self.eps)
        t_n = (t / rms_t) * self.gamma_t
        s_n = (s / rms_s) * self.gamma_s
        out = torch.cat([t_n, s_n], dim=-1)
        if self.beta is not None:
            out = out + self.beta
        return out


# ══════════════════════════════════════════════════════════════════
# Lorentzian inner-product head — bounded, geometry-preserving
# ══════════════════════════════════════════════════════════════════
class LorentzianHead(nn.Module):
    """
    Replaces nn.Linear(hidden, n_classes) with a head whose logits are
    Lorentzian INNER PRODUCTS to learnable class centroids:

        logit_k = η(h, c_k) / τ
                = ( -h_t · c_k_t  +  h_s · c_k_s ) / τ

    Why inner product, not "-d²_M":
        The naive head logit_k = -d²_M(h, c_k) / τ with the Minkowski
        SQUARED INTERVAL  d²_M = -‖h_t - c_k_t‖² + ‖h_s - c_k_s‖² is
        NOT a real distance and is UNBOUNDED BELOW: pushing the time
        block of h far from c_k_t makes d²_M → -∞, hence -d²/τ → +∞.
        That rewards runaway in time direction, not "Minkowski-near."
        The objective has no lower bound and the classification logic
        is undefined.

        The inner-product form avoids this incorrect distance-runaway
        path. It's bilinear with a clean geometric meaning (Lorentzian
        alignment). Logits are NOT globally bounded — they still scale
        with ‖h‖·‖c_k‖, which is controlled by weight decay, SplitNorm
        gammas, and τ — but they no longer have the inverted-sign
        runaway optimum that v2 had.

    Why this still forces the geometry to matter:
        η(h, c_k) = -h_t·c_k_t + h_s·c_k_s
        If both centroids end up in the same mq region (both timelike
        or both spacelike), the discrimination has to come from the
        *direction* within that region. But if the two classes have
        opposite-sign mq centroids, the η inner product can use the
        SIGN to separate them — which is the Lorentzian-specific signal.

    tau is parameterized via softplus(log_tau) to stay strictly
    positive and bounded away from 0 (no escape route via τ → 0
    making logits explode).

    Centroids are NOT projected to any shell. They can drift to be
    timelike, spacelike, or null — whatever minimizes loss.
    """
    def __init__(self, dim, d_t, n_classes, init_tau=1.0):
        super().__init__()
        self.dim   = dim
        self.d_t   = d_t
        self.n_cls = n_classes
        # softplus(log_tau) gives strictly positive tau ≥ ~init_tau
        # log_tau initialized so softplus(log_tau) = init_tau
        log_tau_init = torch.log(torch.expm1(torch.tensor(float(init_tau))))
        self.log_tau = nn.Parameter(log_tau_init)
        # Centroids initialized small, no shell constraint
        self.centroids = nn.Parameter(torch.randn(n_classes, dim) * 0.1)

    @property
    def tau(self):
        return F.softplus(self.log_tau) + 1e-3

    def forward(self, h):
        """
        h:           (B, D)
        centroids:   (K, D)
        Returns logits of shape (B, K):
            logit[i, k] = ( -<h_t_i, c_t_k> + <h_s_i, c_s_k> ) / τ
        """
        # (B, K, D) = (B, 1, D) * (1, K, D)
        t_score = -(h[:, None, :self.d_t] * self.centroids[None, :, :self.d_t]).sum(-1)
        s_score =  (h[:, None, self.d_t:] * self.centroids[None, :, self.d_t:]).sum(-1)
        return (t_score + s_score) / self.tau

    def centroid_mq(self):
        """Diagnostic: mq of each learned class centroid."""
        c = self.centroids
        return ((c[:, self.d_t:] ** 2).sum(-1) -
                (c[:, :self.d_t] ** 2).sum(-1)).detach()


# ══════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════
class LorentzianMLP(nn.Module):
    """
    Step-2 v7 architecture: SplitNorm + Causal Residual + Lorentzian Head.

    Differences from v6:
      v6 used a SYMMETRIC residual `h = h + GELU(fc(norm(h)))` over the
      whole h. That symmetric residual was identified by 3-seed analysis
      as the architectural reason mq could degenerate into "signed scalar
      feature": every seed found a solution where both classes ended up
      on the SAME side of the lightcone, classified by mq magnitude
      rather than by mq sign.

      v7 introduces an ASYMMETRIC ("causal") residual that breaks the
      time/space symmetry at the architecture level:

        h_t_new = h_t + update_t                          (full residual)
        h_s_new = spacelike_residual * h_s + update_s     (configurable)

      With spacelike_residual=0 (default), the timelike block accumulates
      across layers (information integrates over depth, like time
      "flowing forward"), while the spacelike block is a stateless
      function of the previous layer (no integration, no history).

      Motivation (Theorem 5, T-axiom): in a Lorentzian metric, the time
      direction has a privileged orientation — it's the one along which
      causal influence propagates. The spacelike directions are
      exchangeable and don't carry time-arrow structure. Asymmetric
      residual operationalizes this distinction.

      Hypothesis: this asymmetry should make sign(mq) carry semantic
      content (timelike = "deeply integrated past", spacelike = "no
      integration history"), and therefore make sign-split solutions
      lower-loss than same-side solutions.
    """
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, d_t=D_T,
                 n_layers=N_LAYERS, n_classes=N_CLASSES,
                 spacelike_residual=0.0, head_type='linear'):
        super().__init__()
        self.d_t = d_t
        self.spacelike_residual = float(spacelike_residual)
        self.head_type = head_type
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': SplitNorm(hidden_dim, d_t, use_beta=False),
                'fc':   nn.Linear(hidden_dim, hidden_dim),
            }) for _ in range(n_layers)
        ])
        self.final_norm = SplitNorm(hidden_dim, d_t, use_beta=False)
        if head_type == 'linear':
            self.head = LorentzianHead(hidden_dim, d_t, n_classes)
        elif head_type == 'mq-mlp':
            self.head = MqMLPHead(hidden_dim, d_t, n_classes)
        elif head_type == 'quad-features':
            self.head = HeadWithQuadFeatures(hidden_dim, d_t, n_classes)
        else:
            raise ValueError(f'unknown head_type: {head_type!r}. '
                             f'Choose from: linear, mq-mlp, quad-features')

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.embed(x)
        layer_mqs = []
        for blk in self.blocks:
            h_n = blk['norm'](h)
            layer_mqs.append(compute_mq(h_n, self.d_t).detach())

            update = F.gelu(blk['fc'](h_n))
            # Asymmetric (causal) residual — breaks time/space symmetry.
            # timelike: full residual accumulation
            # spacelike: configurable decay (0 = no accumulation, 1 = symmetric/v6)
            h_t = h[..., :self.d_t] + update[..., :self.d_t]
            h_s = self.spacelike_residual * h[..., self.d_t:] + update[..., self.d_t:]
            h = torch.cat([h_t, h_s], dim=-1)

        h_n = self.final_norm(h)
        layer_mqs.append(compute_mq(h_n, self.d_t).detach())
        logits = self.head(h_n)
        return logits, h_n, layer_mqs

    def centroid_mq(self):
        # only meaningful when head has class centroids in mq space
        if hasattr(self.head, 'centroid_mq'):
            return self.head.centroid_mq()
        return None


class EuclideanMLP(nn.Module):
    """
    Baseline: same backbone (SplitNorm replaced with LayerNorm) AND a
    structurally matched Euclidean inner-product head:
        logit_k = <h, c_k> / τ
    This isolates the GEOMETRY (Lorentzian vs Euclidean inner product)
    rather than the head FORM (dot product vs negative squared distance).
    """
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, d_t=D_T,
                 n_layers=N_LAYERS, n_classes=N_CLASSES):
        super().__init__()
        self.d_t = d_t
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'fc':   nn.Linear(hidden_dim, hidden_dim),
            }) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        log_tau_init = torch.log(torch.expm1(torch.tensor(1.0)))
        self.log_tau   = nn.Parameter(log_tau_init)
        self.centroids = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.1)

    @property
    def tau(self):
        return F.softplus(self.log_tau) + 1e-3

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.embed(x)
        layer_mqs = []
        for blk in self.blocks:
            h_n = blk['norm'](h)
            layer_mqs.append(compute_mq(h_n, self.d_t).detach())
            h = h + F.gelu(blk['fc'](h_n))
        h_n = self.final_norm(h)
        layer_mqs.append(compute_mq(h_n, self.d_t).detach())
        # Euclidean inner-product logits (standard, structurally matched)
        logits = (h_n @ self.centroids.T) / self.tau
        return logits, h_n, layer_mqs

    def centroid_mq(self):
        c = self.centroids
        return ((c[:, self.d_t:] ** 2).sum(-1) -
                (c[:, :self.d_t] ** 2).sum(-1)).detach()


class MqMLPHead(nn.Module):
    """
    走法 A — Direction 2 head.

    MLP head that takes h CONCATENATED WITH mq(h) as input. This gives
    the head explicit access to the quadratic feature mq = ‖h_s‖² - ‖h_t‖²
    that the original LorentzianHead is mathematically incapable of using
    (since LorentzianHead is bilinear in h).

    Key design: head can use mq if it wants, OR ignore it and use h
    directly. The architecture does NOT force sign-aware encoding —
    instead it ENABLES the model to discover sign-split as the natural
    representation IF that's what's optimal for the task.

    If sign-split emerges with this head, the paper claim is:
      "When the head can use sign(mq), the model spontaneously evolves
       a Lorentzian sign-aware representation."
    """
    def __init__(self, dim, d_t, n_classes, hidden=64):
        super().__init__()
        self.d_t = d_t
        self.mlp = nn.Sequential(
            nn.Linear(dim + 1, hidden),  # +1 for the mq scalar
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, h):
        mq = compute_mq(h, self.d_t).unsqueeze(-1)        # (B, 1)
        feats = torch.cat([h, mq], dim=-1)                # (B, D+1)
        return self.mlp(feats)

    # no centroid_mq() — this head has no class centroids in mq space


class HeadWithQuadFeatures(nn.Module):
    """
    Reference head from the synthetic sanity test (Test D).
    Single linear layer over [h, ‖h_t‖², ‖h_s‖²].

    Compared to MqMLPHead:
      - Linear (no MLP), so simpler and more interpretable
      - Exposes ‖h_t‖² and ‖h_s‖² SEPARATELY rather than just their
        difference (mq). This gives the head MORE information than
        mq alone — it can learn 'I prefer time-dominated' regardless
        of overall norm.
      - Faster to train, but less flexible.
    """
    def __init__(self, dim, d_t, n_classes):
        super().__init__()
        self.d_t = d_t
        self.linear = nn.Linear(dim + 2, n_classes)
        nn.init.normal_(self.linear.weight, std=0.05)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h):
        t_norm_sq = (h[..., :self.d_t] ** 2).sum(-1, keepdim=True)
        s_norm_sq = (h[..., self.d_t:] ** 2).sum(-1, keepdim=True)
        feats = torch.cat([h, t_norm_sq, s_norm_sq], dim=-1)
        return self.linear(feats)

    def quad_weights(self):
        # diagnostic: (n_classes, 2) weights on [‖h_t‖², ‖h_s‖²]
        return self.linear.weight[:, -2:].detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════════
# Causal sequence dataset
# ══════════════════════════════════════════════════════════════════
def generate_serial_sequences(n_per_class, seq_len, ar_rho, noise_std, seed):
    """
    Class 0 (DEPENDENT): x_t = rho * x_{t-1} + noise_t,  noise ~ N(0, noise_std)
    Class 1 (IID):       x_t ~ N(0, sqrt(noise_std^2 / (1-rho^2))) — i.i.d.

    Note (issue #4 from review): we say "serial dependence" not "causal"
    because AR(1) vs i.i.d. can be solved by a trivial lag-1
    autocorrelation classifier; this is not the strict causal-direction
    test. See lag1_autocorr_baseline() for the floor.

    The acausal noise scale is chosen so its marginal variance equals
    the AR(1) stationary variance. This way the two classes have the same
    pointwise distribution and differ ONLY in their autocovariance.
    """
    rng = np.random.default_rng(seed)

    # Serial-dependent class
    x_dep = np.zeros((n_per_class, seq_len), dtype=np.float32)
    x_dep[:, 0] = rng.normal(0, noise_std / np.sqrt(1 - ar_rho ** 2),
                             size=n_per_class)
    for t in range(1, seq_len):
        x_dep[:, t] = (
            ar_rho * x_dep[:, t - 1]
            + rng.normal(0, noise_std, size=n_per_class)
        )

    # IID class — match marginal variance
    marginal_std = noise_std / np.sqrt(1 - ar_rho ** 2)
    x_iid = rng.normal(0, marginal_std,
                       size=(n_per_class, seq_len)).astype(np.float32)

    X = np.concatenate([x_dep, x_iid], axis=0)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)

    perm = rng.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


# Keep old name as alias for any external callers
generate_causal_sequences = generate_serial_sequences


def verify_dataset(X, y):
    """Sanity check: class 0 should have high lag-1 autocorr, class 1 ~0."""
    print('  Dataset sanity check:')
    for cls in [0, 1]:
        mask = (y == cls)
        seqs = X[mask].numpy()
        ac1 = []
        for seq in seqs:
            if seq.std() > 1e-6:
                c = np.corrcoef(seq[:-1], seq[1:])[0, 1]
                ac1.append(c)
        ac1 = np.array(ac1)
        cls_name = 'DEPENDENT' if cls == 0 else 'IID      '
        print(f'    {cls_name} (cls {cls}): n={mask.sum()}  '
              f'mean={seqs.mean():+.3f}  std={seqs.std():.3f}  '
              f'lag-1 autocorr={ac1.mean():+.3f}±{ac1.std():.3f}')
    print('  → marginals matched, autocorrelation differs by class (good).')


# ══════════════════════════════════════════════════════════════════
# Bimodality indicators (same as Step 1)
# ══════════════════════════════════════════════════════════════════
def bimodality_indicators(samples):
    if len(samples) < 20:
        return {'bc': float('nan'), 'kde_minima': -1, 'sign_balance': float('nan')}
    samples = np.asarray(samples)
    n = len(samples)
    mean = samples.mean()
    std = samples.std() + 1e-12
    z = (samples - mean) / std
    skew = (z ** 3).mean()
    kurt_excess = (z ** 4).mean() - 3.0
    if n > 3:
        denom = kurt_excess + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    else:
        denom = kurt_excess + 3.0
    bc = (skew ** 2 + 1.0) / max(denom, 1e-12)

    lo, hi = np.percentile(samples, [1, 99])
    if hi - lo < 1e-9:
        kde_minima = 0
    else:
        grid = np.linspace(lo, hi, 200)
        h = 1.06 * std * (n ** (-0.2))
        d = (grid[:, None] - samples[None, :]) / max(h, 1e-9)
        kde = np.exp(-0.5 * d ** 2).mean(1) / max(h, 1e-9)
        is_min = (kde[1:-1] < kde[:-2]) & (kde[1:-1] < kde[2:])
        cand_idx = np.where(is_min)[0] + 1
        PROM_FRAC = 0.05
        kde_max = kde.max()
        kept = 0
        for i in cand_idx:
            left_max  = kde[:i].max() if i > 0 else 0.0
            right_max = kde[i+1:].max() if i < len(kde) - 1 else 0.0
            min_val   = kde[i]
            if (left_max - min_val >= PROM_FRAC * kde_max and
                right_max - min_val >= PROM_FRAC * kde_max):
                kept += 1
        kde_minima = int(kept)

    pos = float((samples > 0).mean())
    sign_balance = 1.0 - 2.0 * abs(pos - 0.5)
    return {'bc': float(bc), 'kde_minima': kde_minima,
            'sign_balance': float(sign_balance)}


def lag1_autocorr_baseline(X_train, y_train, X_test, y_test):
    """
    Sanity baseline: a logistic regression on a single feature — the
    lag-1 autocorrelation of each sequence — should already achieve
    most of the classification accuracy on the AR-vs-iid task.

    This pins the FLOOR for "trivial statistical classifier."  If the
    Lorentzian model just hits this number, we haven't done anything
    special; we want to see it well above this floor while ALSO having
    class-aligned mq.

    Returns (acc, weight, bias). Since class 0 = high autocorr,
    class 1 = low autocorr, a correctly-fit logistic regression should
    have w < 0 (high autocorr lowers logit, predicts class 0).
    """
    def feat(X):
        X = X.numpy()
        feats = []
        for seq in X:
            if seq.std() > 1e-6:
                ac1 = np.corrcoef(seq[:-1], seq[1:])[0, 1]
            else:
                ac1 = 0.0
            feats.append(ac1)
        return np.array(feats, dtype=np.float32).reshape(-1, 1)

    f_tr = feat(X_train); f_te = feat(X_test)
    y_tr = y_train.numpy(); y_te = y_test.numpy()

    w = np.zeros(1); b = 0.0
    lr = 0.5
    for _ in range(500):
        z = (f_tr @ w + b)
        p = 1 / (1 + np.exp(-z))
        grad_w = ((p - y_tr).reshape(-1, 1) * f_tr).mean(0)
        grad_b = (p - y_tr).mean()
        w -= lr * grad_w
        b -= lr * grad_b

    p_te = 1 / (1 + np.exp(-(f_te @ w + b)))
    pred = (p_te > 0.5).astype(int)
    acc = float((pred == y_te).mean())
    return acc, float(w[0]), float(b)


# ══════════════════════════════════════════════════════════════════
# NEW for Step 2: class-alignment of mq
# ══════════════════════════════════════════════════════════════════
def mq_class_alignment(mqs, labels):
    """
    The Step-2 key metric. Measures whether mq SIGN predicts CLASS.

    Returns:
      auc:               ROC AUC of mq → class label, computed via
                         rank-sum. AUC > 0.5 means class 1 (IID) tends
                         to have HIGHER mq than class 0 (DEPENDENT).
                         Hypothesis-consistent direction: AUC > 0.5
                         (acausal/iid more spacelike → higher mq).
      accuracy_bestthr:  best classification accuracy using only sign(mq - threshold)
      best_threshold:    threshold giving best accuracy
      mean_mq_class0:    average mq for the DEPENDENT (serial) class
      mean_mq_class1:    average mq for the IID class
      gap:               mean_mq_class1 - mean_mq_class0
                         (positive = dependent more timelike than iid,
                          hypothesis-consistent)
      ttest_p:           p-value of two-sample t-test for mq across classes
    """
    mqs = np.asarray(mqs)
    labels = np.asarray(labels)
    m0 = mqs[labels == 0]
    m1 = mqs[labels == 1]

    # Mean per class and gap
    mean_mq_class0 = float(m0.mean())
    mean_mq_class1 = float(m1.mean())
    gap = mean_mq_class1 - mean_mq_class0

    # Welch's t-test by hand (no scipy dependency)
    n0, n1 = len(m0), len(m1)
    var0 = m0.var(ddof=1) if n0 > 1 else 0.0
    var1 = m1.var(ddof=1) if n1 > 1 else 0.0
    se = np.sqrt(var0 / max(n0, 1) + var1 / max(n1, 1)) + 1e-12
    t_stat = (mean_mq_class1 - mean_mq_class0) / se
    # Approximate p-value via normal (sample sizes are large here)
    from math import erf, sqrt
    z = abs(t_stat)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))

    # ROC AUC: rank-based, no sklearn
    sort_idx = np.argsort(mqs)
    ranks = np.empty_like(sort_idx, dtype=float)
    ranks[sort_idx] = np.arange(1, len(mqs) + 1)
    sum_ranks_class1 = ranks[labels == 1].sum()
    auc = (sum_ranks_class1 - n1 * (n1 + 1) / 2) / max(n0 * n1, 1)
    # Convention: AUC > 0.5 means class 1 has higher mq (acausal more spacelike)
    # If hypothesis holds, this is exactly what we want.

    # Best-threshold accuracy on mq
    sorted_mq = np.sort(mqs)
    candidates = np.concatenate([[sorted_mq[0] - 1.0],
                                 (sorted_mq[:-1] + sorted_mq[1:]) / 2,
                                 [sorted_mq[-1] + 1.0]])
    best_acc = 0.0
    best_thr = 0.0
    for thr in candidates[::max(1, len(candidates) // 200)]:  # sample 200 thresholds
        # Predict class 1 (acausal) if mq > thr, else class 0
        pred = (mqs > thr).astype(int)
        acc = (pred == labels).mean()
        # Also try the opposite direction
        acc_inv = ((mqs <= thr).astype(int) == labels).mean()
        if max(acc, acc_inv) > best_acc:
            best_acc = max(acc, acc_inv)
            best_thr = float(thr)

    return {
        'auc':               float(auc),
        'accuracy_bestthr':  float(best_acc),
        'best_threshold':    best_thr,
        'mean_mq_class0':    mean_mq_class0,
        'mean_mq_class1':    mean_mq_class1,
        'gap':               float(gap),
        'ttest_p':           float(p),
    }


# ══════════════════════════════════════════════════════════════════
# Training & evaluation (now tracks per-class mq)
# ══════════════════════════════════════════════════════════════════
def train_one(model, train_loader, test_loader, name=''):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    history = {
        'train_loss': [], 'test_acc': [],
        'mq_mean': [], 'mq_std': [],
        'mq_mean_class0': [], 'mq_mean_class1': [],
        'mq_gap': [],                     # class1 - class0
        'mq_class_auc': [],               # AUC for sign(mq) → class
        'mq_class_acc': [],               # best-threshold class accuracy via mq alone
        'tl_ratio': [], 'sl_ratio': [],
        'nan_seen': [],
        'final_mq_dist': [],
        'final_mq_class0': [],
        'final_mq_class1': [],
        'final_labels': [],
    }

    all_mqs = None
    all_labels = None
    nan_seen = False

    for ep in range(EPOCHS):
        model.train()
        total = 0.; n_batches = 0; nan_seen = False
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                nan_seen = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); n_batches += 1

        history['nan_seen'].append(nan_seen)
        if nan_seen:
            for k in ('train_loss','test_acc','mq_mean','mq_std',
                      'mq_mean_class0','mq_mean_class1','mq_gap',
                      'mq_class_auc','mq_class_acc','tl_ratio','sl_ratio'):
                history[k].append(float('nan'))
            print(f'  [{name:8s}] ep {ep+1:2d}  [NaN/Inf — aborting]')
            break

        # Evaluation
        model.eval()
        correct, count = 0, 0
        all_mqs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb_d, yb_d = xb.to(device), yb.to(device)
                logits, h_n, _ = model(xb_d)
                correct += (logits.argmax(-1) == yb_d).sum().item()
                count += yb_d.size(0)
                all_mqs.append(compute_mq(h_n, model.d_t).cpu())
                all_labels.append(yb)
        all_mqs = torch.cat(all_mqs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        cls_align = mq_class_alignment(all_mqs, all_labels)

        # Centroid mq: where do the learned class centroids sit?
        # If hypothesis holds, centroid_0 (causal) → timelike (mq<0)
        #                       centroid_1 (acausal) → spacelike (mq>0)
        try:
            cm = model.centroid_mq()
            centroid_mqs = cm.cpu().numpy().tolist() if cm is not None else []
        except Exception:
            centroid_mqs = []

        history['train_loss'].append(total / max(n_batches, 1))
        history['test_acc'].append(correct / max(count, 1))
        history['mq_mean'].append(float(all_mqs.mean()))
        history['mq_std'].append(float(all_mqs.std()))
        history['mq_mean_class0'].append(cls_align['mean_mq_class0'])
        history['mq_mean_class1'].append(cls_align['mean_mq_class1'])
        history['mq_gap'].append(cls_align['gap'])
        history['mq_class_auc'].append(cls_align['auc'])
        history['mq_class_acc'].append(cls_align['accuracy_bestthr'])
        history['tl_ratio'].append(float((all_mqs < 0).mean()))
        history['sl_ratio'].append(float((all_mqs > 0).mean()))
        history.setdefault('centroid_mqs', []).append(centroid_mqs)

        cmq_str = '  '.join(f'c{i}={m:+.2f}' for i, m in enumerate(centroid_mqs))
        print(f'  [{name:8s}] ep {ep+1:2d}  '
              f'loss={history["train_loss"][-1]:.4f}  '
              f'acc={history["test_acc"][-1]*100:5.2f}%  '
              f'mq[c0]={cls_align["mean_mq_class0"]:+6.2f}  '
              f'mq[c1]={cls_align["mean_mq_class1"]:+6.2f}  '
              f'gap={cls_align["gap"]:+6.2f}  '
              f'AUC={cls_align["auc"]:.3f}  '
              f'centroid:[{cmq_str}]')

    if all_mqs is not None and not nan_seen:
        history['final_mq_dist']   = all_mqs.tolist()
        history['final_labels']    = all_labels.tolist()
        history['final_mq_class0'] = all_mqs[all_labels == 0].tolist()
        history['final_mq_class1'] = all_mqs[all_labels == 1].tolist()
    return history


# ══════════════════════════════════════════════════════════════════
# Analysis & decision
# ══════════════════════════════════════════════════════════════════
def analyse(hist_L, hist_E, baseline_acc=None):
    print('\n' + '=' * 64)
    print('STEP 2 ANALYSIS — Class-Aligned mq on Serial-Dependence Sequences')
    print('=' * 64)

    final_L_acc = hist_L['test_acc'][-1] * 100 if hist_L['test_acc'] else 0
    final_E_acc = hist_E['test_acc'][-1] * 100 if hist_E['test_acc'] else 0

    print(f'\nQ1 — Stability:')
    stable = bool(
        len(hist_L['train_loss']) > 0
        and not any(hist_L.get('nan_seen', []))
        and np.isfinite(hist_L['train_loss'][-1])
    )
    if stable:
        print(f'  ✓ Lorentzian stable, final acc = {final_L_acc:.2f}%')
    else:
        print(f'  ✗ Lorentzian failed')

    print(f'\nQ2 — Task feasibility:')
    print(f'  Lorentzian: {final_L_acc:.2f}%')
    print(f'  Euclidean:  {final_E_acc:.2f}%')
    if baseline_acc is not None:
        print(f'  Lag-1 autocorr baseline: {baseline_acc*100:.2f}%')
        if final_L_acc > baseline_acc * 100 + 2:
            print(f'  ✓ Lorentzian beats trivial statistical baseline')
        else:
            print(f'  ⚠ Lorentzian barely above lag-1 baseline — it may just be'
                  f' learning autocorrelation, not anything geometrically novel')
    if final_L_acc < 60 and final_E_acc < 60:
        print(f'  ⚠ Both models near chance — task may be too hard or AR_RHO too low')
    elif final_L_acc > 90 or final_E_acc > 90:
        print(f'  ✓ Task is learnable')

    print(f'\nQ3 — Class-aligned mq (the key Step-2 question):')
    auc      = hist_L['mq_class_auc'][-1]
    mq_acc   = hist_L['mq_class_acc'][-1] * 100
    mq_c0    = hist_L['mq_mean_class0'][-1]
    mq_c1    = hist_L['mq_mean_class1'][-1]
    gap      = hist_L['mq_gap'][-1]

    # Centroid mq — what the model "thinks" each class should look like geometrically
    final_centroids = hist_L.get('centroid_mqs', [[]])[-1] if hist_L.get('centroid_mqs') else []
    if final_centroids and len(final_centroids) >= 2:
        print(f'  Learned class CENTROIDS in mq-space:')
        print(f'    centroid_0 (DEPENDENT target):  mq = {final_centroids[0]:+.3f}')
        print(f'    centroid_1 (IID target): mq = {final_centroids[1]:+.3f}')
        if final_centroids[0] < 0 < final_centroids[1]:
            print(f'    → CLEAN SPLIT: dependent centroid timelike, iid centroid spacelike '
                  f'(matches hypothesis)')
        elif final_centroids[1] < 0 < final_centroids[0]:
            print(f'    → INVERTED SPLIT: iid centroid timelike, dependent centroid spacelike '
                  f'(still class-aligned, but sign convention reversed)')
        elif (final_centroids[0] < 0) == (final_centroids[1] < 0):
            print(f'    ⚠ SAME SIGN: both centroids on same side of light cone '
                  f'(no genuine signature use)')

    print(f'\n  Class 0 (DEPENDENT) mean mq: {mq_c0:+.3f}')
    print(f'  Class 1 (IID)       mean mq: {mq_c1:+.3f}')
    print(f'  Gap: {gap:+.3f}')
    print(f'  ROC AUC of mq predicting class: {auc:.3f}  '
          f'(0.5=no info, 1.0=perfect, 0.0=perfectly inverted)')
    print(f'  Best classification accuracy from mq alone: {mq_acc:.1f}%')

    # Per-class bimodality
    if hist_L.get('final_mq_class0') and hist_L.get('final_mq_class1'):
        bi_full = bimodality_indicators(np.array(hist_L['final_mq_dist']))
        bi_c0   = bimodality_indicators(np.array(hist_L['final_mq_class0']))
        bi_c1   = bimodality_indicators(np.array(hist_L['final_mq_class1']))
        print(f'\n  Aggregate mq bimodality: BC={bi_full["bc"]:.3f}  '
              f'KDE-min={bi_full["kde_minima"]}')
        print(f'  Within-class-0 BC={bi_c0["bc"]:.3f} (should be ~unimodal)')
        print(f'  Within-class-1 BC={bi_c1["bc"]:.3f} (should be ~unimodal)')
        full_bimodal_split_by_class = (
            bi_full['kde_minima'] > 0
            and bi_c0['kde_minima'] == 0
            and bi_c1['kde_minima'] == 0
        )
        if full_bimodal_split_by_class:
            print(f'  ✓ Aggregate is bimodal AND each class is unimodal '
                  f'→ bimodality is class-driven')
        else:
            print(f'  ⚠ Bimodality is NOT cleanly split by class')

    # Euclidean baseline for context
    print(f'\n  (Euclidean for reference): '
          f'AUC={hist_E["mq_class_auc"][-1]:.3f}  '
          f'gap={hist_E["mq_gap"][-1]:+.3f}')

    print('\n' + '=' * 64)
    print('DECISION')
    print('=' * 64)

    # Two distinct conditions, kept separate (issue #3 from review):
    #
    #   hypothesis_ok           — strong hypothesis: dependent class
    #                              actually timelike (mq<0), iid class
    #                              actually spacelike (mq>0). Matches the
    #                              physics motivation of Theorem 5.
    #                              Requires REAL sign split, not just
    #                              a gap. (Issue #6 from review: gap>0
    #                              alone could fire on, e.g., mq_c0=+2,
    #                              mq_c1=+10, where neither class is
    #                              actually timelike.)
    #   class_aligned_any_dir   — mq carries class info regardless of
    #                              sign convention. Still informative,
    #                              but doesn't validate the specific
    #                              physical prediction.
    auc_dist = abs(auc - 0.5)
    sign_split_ok = (mq_c0 < 0) and (mq_c1 > 0)         # NEW (issue #6)
    inverted_sign_split = (mq_c1 < 0) and (mq_c0 > 0)
    hypothesis_ok = (auc > 0.65) and (gap > 0) and sign_split_ok
    class_aligned_any_direction = (auc_dist > 0.15)
    mq_alone_above_chance = mq_acc > 65.0

    if len(hist_L['mq_class_auc']) >= 3:
        recent_aucs = hist_L['mq_class_auc'][-3:]
        recent_consistent = all(
            (a - 0.5) * (auc - 0.5) > 0 and abs(a - 0.5) > 0.10
            for a in recent_aucs
        )
    else:
        recent_consistent = False

    if stable and hypothesis_ok and mq_alone_above_chance and recent_consistent:
        print('  ✅ GREEN — mq aligned WITH the predicted physics direction:')
        print('     class 0 (DEPENDENT) is timelike (mq<0)')
        print('     class 1 (IID) is spacelike (mq>0)')
        print('     This is direct support for the Realizability Theorem 5')
        print('     interpretation: serial dependence → timelike representation.')
        print('     Proceed to Pillar 4 AND consider drafting a paper.')
    elif stable and inverted_sign_split and mq_alone_above_chance and recent_consistent:
        print('  🟡 YELLOW (inverted sign split) — both classes have')
        print('     well-defined but inverted geometry: dependent is spacelike,')
        print('     iid is timelike. Class info encoded in geometry but in')
        print('     opposite-of-predicted direction.')
        print('     Likely cause: random τ / centroid initialization sign')
        print('     symmetry breaking. Re-run multi-seed to check stability.')
    elif stable and class_aligned_any_direction and mq_alone_above_chance and recent_consistent:
        print('  🟡 YELLOW (class-aligned but no clean sign split) —')
        print('     mq carries class information (AUC away from 0.5) but the')
        print('     two classes are NOT cleanly on opposite sides of the light')
        print('     cone. Both classes may be timelike-but-different-magnitudes,')
        print('     or both spacelike-but-different-magnitudes.')
        print(f'     mq_c0={mq_c0:+.2f}  mq_c1={mq_c1:+.2f}')
        print('     The geometry is being used as a CONTINUOUS feature, not as')
        print('     a sign-based class label. Worth a paper, but the strong')
        print('     "causality → timelike" claim is not supported.')
    elif stable and (auc_dist > 0.05 or mq_acc > 55):
        print('  🟡 YELLOW (weak) — small class-mq correlation, not robust enough')
        print('     to claim the geometry is doing real work.')
        print('     Possible fixes:')
        print('       (a) increase AR_RHO to 0.95 or 0.99')
        print('       (b) longer sequences (SEQ_LEN=64)')
        print('       (c) train longer (EPOCHS=30)')
        print('       (d) reduce hidden_dim — too much capacity may bypass geometry')
    elif stable:
        print('  ❌ RED for the strong hypothesis — mq does NOT align with class.')
        print('     The model solves the task without using indefinite structure.')
        print('     Even with the Lorentzian-inner-product head forcing geometric')
        print('     logits, the centroids found a solution that does not require')
        print('     class-specific mq signs. This is a strong falsification of')
        print('     "Lorentzian = causal" as a universal inductive bias.')
    else:
        print('  ❌ RED — Training failed.')

    return {
        'stable':                       stable,
        'final_acc_L':                  final_L_acc,
        'final_acc_E':                  final_E_acc,
        'auc_mq_class':                 auc,
        'mq_alone_acc':                 mq_acc,
        'gap':                          gap,
        'mq_c0':                        mq_c0,
        'mq_c1':                        mq_c1,
        'sign_split_ok':                sign_split_ok,
        'inverted_sign_split':          inverted_sign_split,
        'hypothesis_ok':                hypothesis_ok,
        'class_aligned_any_direction':  class_aligned_any_direction,
        'mq_alone_above':               mq_alone_above_chance,
        'final_centroids':              final_centroids,
    }


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════
def make_plots(hist_L, hist_E, out_path='step2_results.png'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'  (matplotlib unavailable: {e})')
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    epochs = list(range(1, len(hist_L['train_loss']) + 1))

    # 0,0: train loss
    axes[0, 0].plot(epochs, hist_L['train_loss'], 'o-', label='Lorentzian', color='C0')
    axes[0, 0].plot(epochs, hist_E['train_loss'], 'o-', label='Euclidean', color='C1')
    axes[0, 0].set_title('Train loss'); axes[0, 0].grid(alpha=0.3); axes[0, 0].legend()

    # 0,1: test accuracy
    axes[0, 1].plot(epochs, [a*100 for a in hist_L['test_acc']], 'o-', color='C0', label='Lorentzian')
    axes[0, 1].plot(epochs, [a*100 for a in hist_E['test_acc']], 'o-', color='C1', label='Euclidean')
    axes[0, 1].axhline(50, color='k', ls=':', alpha=0.5, label='chance')
    axes[0, 1].set_title('Test accuracy (%)'); axes[0, 1].grid(alpha=0.3); axes[0, 1].legend()

    # 0,2: per-class mq mean over training (THE key plot)
    axes[0, 2].plot(epochs, hist_L['mq_mean_class0'], 'o-', color='red', label='Lorentz: class 0 (DEPENDENT)')
    axes[0, 2].plot(epochs, hist_L['mq_mean_class1'], 'o-', color='blue', label='Lorentz: class 1 (IID)')
    axes[0, 2].axhline(0, color='k', ls='--', alpha=0.5, label='light cone')
    axes[0, 2].set_title('Per-class mq mean (Lorentzian)')
    axes[0, 2].set_xlabel('epoch'); axes[0, 2].grid(alpha=0.3); axes[0, 2].legend(fontsize=8)

    # 1,0: class-aligned AUC over training
    axes[1, 0].plot(epochs, hist_L['mq_class_auc'], 'o-', color='C0', label='Lorentzian')
    axes[1, 0].plot(epochs, hist_E['mq_class_auc'], 'o-', color='C1', label='Euclidean')
    axes[1, 0].axhline(0.5, color='k', ls=':', alpha=0.5, label='chance')
    axes[1, 0].axhspan(0.35, 0.65, alpha=0.1, color='red', label='no-info zone')
    axes[1, 0].set_title('AUC of mq → class')
    axes[1, 0].set_ylim(0, 1); axes[1, 0].grid(alpha=0.3); axes[1, 0].legend(fontsize=8)

    # 1,1: final mq distribution overlaid by class (THE key plot)
    if hist_L.get('final_mq_class0') and hist_L.get('final_mq_class1'):
        axes[1, 1].hist(hist_L['final_mq_class0'], bins=60, alpha=0.6,
                        color='red', label='class 0 (DEPENDENT)')
        axes[1, 1].hist(hist_L['final_mq_class1'], bins=60, alpha=0.6,
                        color='blue', label='class 1 (IID)')
        axes[1, 1].axvline(0, color='k', ls='--', alpha=0.7, label='light cone')
        axes[1, 1].set_title('Final mq distribution by class (Lorentzian)')
        axes[1, 1].set_xlabel('mq'); axes[1, 1].grid(alpha=0.3); axes[1, 1].legend(fontsize=8)

    # 1,2: same plot for Euclidean (for comparison)
    if hist_E.get('final_mq_class0') and hist_E.get('final_mq_class1'):
        axes[1, 2].hist(hist_E['final_mq_class0'], bins=60, alpha=0.6,
                        color='red', label='class 0 (DEPENDENT)')
        axes[1, 2].hist(hist_E['final_mq_class1'], bins=60, alpha=0.6,
                        color='blue', label='class 1 (IID)')
        axes[1, 2].axvline(0, color='k', ls='--', alpha=0.7)
        axes[1, 2].set_title('Final mq distribution by class (Euclidean baseline)')
        axes[1, 2].set_xlabel('mq'); axes[1, 2].grid(alpha=0.3); axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'  Saved plot: {out_path}')


# ══════════════════════════════════════════════════════════════════
# Single-seed run (one Lorentz + one Euclidean comparison)
# ══════════════════════════════════════════════════════════════════
def run_single_seed(seed, X_train, y_train, X_test, y_test, baseline_acc,
                    out_dir, save_artifacts=True, verbose=True,
                    spacelike_residual=0.0, head_type='linear'):
    # Seed BEFORE constructing DataLoader so the shuffle is reproducible
    # per-seed (issue #2 from review). Without this, all seeds share
    # whatever shuffle order the global RNG happened to be in when
    # main() built the loader once.
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=g,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    if verbose:
        print(f'\n=== Lorentzian (Split-Norm + Causal Residual + head={head_type}, '
              f'spacelike_residual={spacelike_residual:.2f}) — seed {seed} ===')
    model_L = LorentzianMLP(in_dim=SEQ_LEN,
                            spacelike_residual=spacelike_residual,
                            head_type=head_type).to(device)
    if verbose:
        print(f'  params: {sum(p.numel() for p in model_L.parameters()):,}')
    hist_L = train_one(model_L, train_loader, test_loader, name=f'L-s{seed}')

    if verbose:
        print(f'\n=== Euclidean (LayerNorm baseline) — seed {seed} ===')
    # Reseed for Euclidean too so its init is also reproducible per seed
    torch.manual_seed(seed); np.random.seed(seed)
    g_e = torch.Generator(); g_e.manual_seed(seed)
    train_loader_e = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=g_e,
    )
    model_E = EuclideanMLP(in_dim=SEQ_LEN).to(device)
    hist_E = train_one(model_E, train_loader_e, test_loader, name=f'E-s{seed}')

    summary = analyse(hist_L, hist_E, baseline_acc=baseline_acc)
    summary['seed'] = seed
    # Add recent_consistent so cross-seed summary can use it (issue #6)
    if len(hist_L['mq_class_auc']) >= 3:
        recent = hist_L['mq_class_auc'][-3:]
        auc_final = hist_L['mq_class_auc'][-1]
        summary['recent_consistent'] = all(
            (a - 0.5) * (auc_final - 0.5) > 0 and abs(a - 0.5) > 0.10
            for a in recent
        )
    else:
        summary['recent_consistent'] = False

    if save_artifacts:
        log = {
            'config': {
                'HIDDEN_DIM': HIDDEN_DIM, 'D_T': D_T, 'D_S': D_S,
                'SEQ_LEN': SEQ_LEN, 'AR_RHO': AR_RHO, 'NOISE_STD': NOISE_STD,
                'N_TRAIN': N_TRAIN, 'N_TEST': N_TEST, 'EPOCHS': EPOCHS,
                'LR': LR, 'BATCH_SIZE': BATCH_SIZE, 'SEED': seed,
            },
            'lorentz_history': {k: v for k, v in hist_L.items()
                                if k not in ('final_mq_dist','final_mq_class0',
                                             'final_mq_class1','final_labels')},
            'euclid_history':  {k: v for k, v in hist_E.items()
                                if k not in ('final_mq_dist','final_mq_class0',
                                             'final_mq_class1','final_labels')},
            'lorentz_final_mq_class0': hist_L.get('final_mq_class0', []),
            'lorentz_final_mq_class1': hist_L.get('final_mq_class1', []),
            'euclid_final_mq_class0':  hist_E.get('final_mq_class0', []),
            'euclid_final_mq_class1':  hist_E.get('final_mq_class1', []),
            'summary': summary,
        }
        log_path = out_dir / f'step2_log_seed{seed}.json'
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2, default=_json_safe)
        if verbose:
            print(f'  Saved: {log_path}')
        make_plots(hist_L, hist_E,
                   out_path=str(out_dir / f'step2_results_seed{seed}.png'))
    return summary


def cross_seed_summary(per_seed_results):
    """Aggregate across seeds — important for sign-convention stability."""
    print('\n' + '=' * 64)
    print(f'CROSS-SEED SUMMARY ({len(per_seed_results)} seeds)')
    print('=' * 64)
    print('\nScope: dataset is fixed across seeds; seeds vary model init,')
    print('       train shuffle order, and centroid/τ initialization only.')
    print('       This measures STABILITY OF THE GEOMETRY UNDER REINIT,')
    print('       not stability under data resampling.')

    print(f'\n{"seed":>5}  {"acc_L":>6}  {"acc_E":>6}  {"AUC":>6}  '
          f'{"gap":>7}  {"mq_c0":>7}  {"mq_c1":>7}  {"GREEN":>5}  {"sign-split":>10}')
    print('-' * 80)
    for r in per_seed_results:
        green = '✅' if r['hypothesis_ok'] else (
                '🟡' if r.get('class_aligned_any_direction') else '❌')
        ss = ('clean' if r['sign_split_ok']
              else ('inv' if r['inverted_sign_split'] else 'none'))
        print(f'{r["seed"]:>5}  {r["final_acc_L"]:>6.2f}  {r["final_acc_E"]:>6.2f}  '
              f'{r["auc_mq_class"]:>6.3f}  {r["gap"]:>+7.3f}  '
              f'{r["mq_c0"]:>+7.3f}  {r["mq_c1"]:>+7.3f}  {green:>5}  {ss:>10}')

    n_green = sum(1 for r in per_seed_results
                  if r['hypothesis_ok']
                  and r.get('mq_alone_above', False)
                  and r.get('recent_consistent', False))
    n_aligned = sum(1 for r in per_seed_results
                    if r['class_aligned_any_direction']
                    and r.get('mq_alone_above', False)
                    and r.get('recent_consistent', False))
    n_clean_split = sum(1 for r in per_seed_results if r['sign_split_ok'])
    n_inv_split = sum(1 for r in per_seed_results
                      if r['inverted_sign_split'])

    aucs = np.array([r['auc_mq_class'] for r in per_seed_results])
    gaps = np.array([r['gap'] for r in per_seed_results])
    print(f'\n  AUC mean ± std: {aucs.mean():.3f} ± {aucs.std():.3f}')
    print(f'  gap mean ± std: {gaps.mean():+.3f} ± {gaps.std():.3f}')
    print(f'\n  GREEN (hypothesis-direction, mq-above-chance, stable): '
          f'{n_green}/{len(per_seed_results)}')
    print(f'  Class-aligned (any dir, mq-above-chance, stable): '
          f'{n_aligned}/{len(per_seed_results)}')
    print(f'  Clean sign split (c0<0<c1): {n_clean_split}/{len(per_seed_results)}')
    print(f'  Inverted sign split:        {n_inv_split}/{len(per_seed_results)}')

    # Note: mq_alone_above is "any threshold direction works" (the
    # accuracy metric tries both); recent_consistent is "AUC stable on
    # the same side of 0.5 across last 3 epochs". Together they filter
    # out flukes from this count.

    print('\n  Interpretation:')
    if n_green >= 0.7 * len(per_seed_results):
        print('    ✅ Hypothesis robust across seeds — true GREEN.')
    elif n_clean_split + n_inv_split >= 0.7 * len(per_seed_results):
        print('    🟡 Sign split is consistent but direction varies by seed —')
        print('       the geometry IS being used, but which sign maps to which')
        print('       class is initialization-dependent. This is interesting in')
        print('       its own right but does NOT confirm the physics-direction')
        print('       prediction.')
    elif n_aligned >= 0.5 * len(per_seed_results):
        print('    🟡 mq carries class info on most seeds but without clean')
        print('       sign split — geometry used as continuous feature.')
    else:
        print('    ❌ Cross-seed signal weak — strong hypothesis not supported.')


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='.')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of seeds to run (default 3).')
    parser.add_argument('--spacelike-residual', type=float, default=0.0,
                        help='Decay coefficient for spacelike residual '
                             '(0=no accumulation, 1=symmetric/v6 behaviour, '
                             '0.5=soft asymmetry). Default 0.')
    parser.add_argument('--ar-rho', type=float, default=AR_RHO,
                        help=f'AR(1) coefficient for serial-dependence task. '
                             f'Default {AR_RHO}. Try 0.4 to drop lag-1 baseline '
                             f'into ~70% range and create real pressure on the '
                             f'model to use geometry beyond autocorrelation.')
    parser.add_argument('--head-type', default='mq-mlp',
                        choices=['linear', 'mq-mlp', 'quad-features'],
                        help='Head architecture. linear=original LorentzianHead '
                             '(bilinear in h, cannot use sign(mq)). '
                             'mq-mlp=MLP over [h, mq] (default for v11, '
                             'enables sign-aware decisions). '
                             'quad-features=linear over [h, ‖h_t‖², ‖h_s‖²].')
    args, _ = parser.parse_known_args()
    ar_rho = args.ar_rho     # local override
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nGenerating serial-dependence dataset (AR_RHO={ar_rho:.2f})...')
    X_train, y_train = generate_serial_sequences(
        n_per_class=N_TRAIN // 2, seq_len=SEQ_LEN,
        ar_rho=ar_rho, noise_std=NOISE_STD, seed=SEED)
    X_test, y_test = generate_serial_sequences(
        n_per_class=N_TEST // 2, seq_len=SEQ_LEN,
        ar_rho=ar_rho, noise_std=NOISE_STD, seed=SEED + 1000)
    print(f'  train shape: {X_train.shape}  test shape: {X_test.shape}')
    verify_dataset(X_train, y_train)

    # Baseline: lag-1 autocorrelation logistic regression (issue #4 from review)
    print('\nLag-1 autocorrelation logistic-regression baseline...')
    baseline_acc, baseline_w, baseline_b = lag1_autocorr_baseline(
        X_train, y_train, X_test, y_test)
    print(f'  baseline test acc = {baseline_acc*100:.2f}%  '
          f'w={baseline_w:+.3f}  b={baseline_b:+.3f}')
    if baseline_w < 0:
        print(f'  ✓ w < 0 as expected (high autocorr → predict class 0 = DEPENDENT)')
    else:
        print(f'  ⚠ w > 0 — sign inversion, check label convention')
    print('  (this is the floor a trivial statistical classifier achieves on'
          ' serial-dependence vs iid — Lorentzian model should beat it AND'
          ' show class-aligned mq, not just match it.)')

    # Multi-seed run (review suggestion: sign convention may be init-dependent)
    n_seeds = max(1, args.n_seeds)
    seeds = [SEED + i for i in range(n_seeds)]
    print(f'\nRunning {n_seeds} seed(s): {seeds}')
    print(f'Causal residual: spacelike_residual = {args.spacelike_residual:.2f}'
          f'  ({"FULL ASYMMETRY (v7 default)" if args.spacelike_residual == 0.0 else ""}'
          f'{"FULL SYMMETRY (matches v6)" if args.spacelike_residual == 1.0 else ""}'
          f'{"PARTIAL ASYMMETRY" if 0 < args.spacelike_residual < 1 else ""})')

    print(f'Head type: {args.head_type}'
          f'  ({"original LorentzianHead (bilinear, cannot use sign(mq))" if args.head_type == "linear" else "MLP head with mq access" if args.head_type == "mq-mlp" else "linear head with quadratic features"})')

    per_seed_results = []
    for s in seeds:
        summary = run_single_seed(
            s, X_train, y_train, X_test, y_test, baseline_acc,
            out_dir, save_artifacts=True, verbose=True,
            spacelike_residual=args.spacelike_residual,
            head_type=args.head_type)
        per_seed_results.append(summary)

    if n_seeds > 1:
        cross_seed_summary(per_seed_results)
        # Save aggregated results
        with open(out_dir / 'step2_cross_seed.json', 'w') as f:
            json.dump({'per_seed': per_seed_results,
                       'baseline_acc': baseline_acc,
                       'baseline_w': baseline_w,
                       'baseline_b': baseline_b}, f, indent=2,
                      default=_json_safe)
        print(f'  Saved: {out_dir / "step2_cross_seed.json"}')


if __name__ == '__main__':
    import sys
    sys.argv = ['x', '--n-seeds', '3', '--head-type', 'mq-mlp',
                '--ar-rho', '0.3', '--spacelike-residual', '0.0']
    main()