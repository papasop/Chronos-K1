"""
Lorentzian Backbone — Step 2 (v14c): SCALE CALIBRATION TEST (revised x2)
========================================================================

THE QUESTION:
  v13 found Lorentzian sign-split rate (16/24) is lower than Euclidean
  (19/24) across OOD shifts. Two possible causes:

  (A) SplitNorm + causal residual STRUCTURALLY limit sign-split robustness
  (B) The bounded |m_q|≈8 alone is the cause; pushing |m_q| up to ~50
      should close the gap.

WHAT v14c CHANGES vs v14b (per second review):
  - λ-sweep selection now requires BOTH mean|m_q|≥0.85·target AND
    p10|m_q|≥0.5·target_margin. Mean-only success can be satisfied by
    outliers leaving most samples near zero, which would make the
    calibration test inconclusive.
  - Headline adds `OOD strong sign-split AND acc>90` joint metric.
    A model can have sample_sign_acc>0.9 while classification has
    collapsed to chance — that's not a real success.

(v14b changes — retained:
  λ-sweep dry run; joint mean+margin penalty; sample-level sign metrics;
  no sklearn; stable md5 OOD seeding; hedged interpretation.)

CLI:
    python lorentz_step2_v14c_calibration.py
    # No flags. ~30-40 min on Colab GPU.
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
AR_RHO       = 0.9        # AR(1) coefficient — strong signal for clean comparison
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
    Baseline backbone: LayerNorm + standard residual + (configurable head).

    v12: head_type is configurable, same as LorentzianMLP. This allows
    Experiment 5 — Euclidean backbone + MqMLPHead — to test whether
    the Lorentzian backbone (SplitNorm + causal residual) is necessary
    for sign-split, or whether the head alone is sufficient.

    Default head_type='linear' uses a structurally matched Euclidean
    inner-product head (same as v6-v11 Euclidean baseline).
    """
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, d_t=D_T,
                 n_layers=N_LAYERS, n_classes=N_CLASSES, head_type='linear'):
        super().__init__()
        self.d_t = d_t
        self.head_type = head_type
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'fc':   nn.Linear(hidden_dim, hidden_dim),
            }) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        # Build head — same factory as LorentzianMLP
        if head_type == 'linear':
            # original Euclidean head: simple inner product to learnable centroids
            log_tau_init = torch.log(torch.expm1(torch.tensor(1.0)))
            self.log_tau   = nn.Parameter(log_tau_init)
            self.centroids = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.1)
            self.head = None       # use built-in inner product
        elif head_type == 'mq-mlp':
            self.head = MqMLPHead(hidden_dim, d_t, n_classes)
        elif head_type == 'quad-features':
            self.head = HeadWithQuadFeatures(hidden_dim, d_t, n_classes)
        else:
            raise ValueError(f'unknown head_type: {head_type!r}')

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
        if self.head is None:
            # Euclidean inner-product head (default)
            logits = (h_n @ self.centroids.T) / self.tau
        else:
            logits = self.head(h_n)
        return logits, h_n, layer_mqs

    def centroid_mq(self):
        if self.head is None:
            c = self.centroids
            return ((c[:, self.d_t:] ** 2).sum(-1) -
                    (c[:, :self.d_t] ** 2).sum(-1)).detach()
        elif hasattr(self.head, 'centroid_mq'):
            return self.head.centroid_mq()
        return None


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
def train_one(model, train_loader, test_loader, name='',
              mq_target_scale=None, lambda_scale=0.0,
              target_margin=None, lambda_margin=0.0):
    """
    Train one model.

    v14b calibration mechanics (per review):
      - The mean-|mq| penalty (`lambda_scale`, `mq_target_scale`) can be
        satisfied by a few outliers, leaving most samples near zero.
        We add a SECOND optional knob: a margin penalty
            L_margin = lambda_margin · mean( ReLU(target_margin - |mq|) ) / target_margin
        which directly penalizes samples whose |mq| < target_margin.
        This pushes the WHOLE distribution away from zero, not just
        the average.
      - Track |mq| percentiles (p10, median, p90) per epoch so we can
        diagnose distribution shape rather than just mean.

    Configurations:
      lambda_scale=0, lambda_margin=0:   no calibration (default; v13 behavior)
      lambda_scale>0, target_scale=T:    push mean(|mq|) toward T
      lambda_margin>0, target_margin=M:  push p10(|mq|) above M
      both:                              both pressures (recommended)
    """
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    history = {
        'train_loss': [], 'test_acc': [],
        'mq_mean': [], 'mq_std': [],
        'mq_mean_class0': [], 'mq_mean_class1': [],
        'mq_gap': [],
        'mq_class_auc': [],
        'mq_class_acc': [],
        'tl_ratio': [], 'sl_ratio': [],
        'nan_seen': [],
        'final_mq_dist': [],
        'final_mq_class0': [],
        'final_mq_class1': [],
        'final_labels': [],
        # v14b: calibration tracking
        'scale_loss': [],
        'margin_loss': [],
        'mq_abs_mean': [],
        'mq_abs_p10':    [],
        'mq_abs_median': [],
        'mq_abs_p90':    [],
        # v14b: sample-level sign metrics
        'sample_sign_acc': [],
    }

    use_scale_penalty  = (mq_target_scale is not None and lambda_scale > 0)
    use_margin_penalty = (target_margin   is not None and lambda_margin > 0)

    all_mqs = None
    all_labels = None
    nan_seen = False

    for ep in range(EPOCHS):
        model.train()
        total_ce = 0.; total_scale = 0.; total_margin = 0.
        n_batches = 0; nan_seen = False
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, h_n, _ = model(xb)
            ce_loss = F.cross_entropy(logits, yb)

            extra_loss = torch.zeros((), device=device)

            if use_scale_penalty or use_margin_penalty:
                mq_batch = compute_mq(h_n, model.d_t)         # (B,)
                mq_abs   = mq_batch.abs()

                if use_scale_penalty:
                    scale_loss = lambda_scale * (
                        (mq_abs.mean() - mq_target_scale) / mq_target_scale) ** 2
                    extra_loss = extra_loss + scale_loss
                    total_scale += scale_loss.item()

                if use_margin_penalty:
                    # Penalize any |mq| BELOW target_margin (push p10 up).
                    margin_loss = lambda_margin * (
                        F.relu(target_margin - mq_abs).mean() / target_margin)
                    extra_loss = extra_loss + margin_loss
                    total_margin += margin_loss.item()

            loss = ce_loss + extra_loss

            if not torch.isfinite(loss):
                nan_seen = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_ce += ce_loss.item(); n_batches += 1

        history['nan_seen'].append(nan_seen)
        if nan_seen:
            for k in ('train_loss','test_acc','mq_mean','mq_std',
                      'mq_mean_class0','mq_mean_class1','mq_gap',
                      'mq_class_auc','mq_class_acc','tl_ratio','sl_ratio',
                      'scale_loss','margin_loss','mq_abs_mean',
                      'mq_abs_p10','mq_abs_median','mq_abs_p90',
                      'sample_sign_acc'):
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

        history['train_loss'].append(total_ce / max(n_batches, 1))
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
        history['scale_loss'].append(total_scale / max(n_batches, 1)
                                     if use_scale_penalty else 0.0)
        history['margin_loss'].append(total_margin / max(n_batches, 1)
                                      if use_margin_penalty else 0.0)
        # v14b: |mq| distribution shape, not just mean
        mq_abs_arr = np.abs(all_mqs)
        history['mq_abs_mean'].append(float(mq_abs_arr.mean()))
        history['mq_abs_p10'].append(float(np.percentile(mq_abs_arr, 10)))
        history['mq_abs_median'].append(float(np.percentile(mq_abs_arr, 50)))
        history['mq_abs_p90'].append(float(np.percentile(mq_abs_arr, 90)))
        # v14b: sample-level sign accuracy (review fix #3)
        m0 = all_mqs[all_labels == 0]
        m1 = all_mqs[all_labels == 1]
        sample_sign_acc = 0.5 * float((m0 < 0).mean()) + 0.5 * float((m1 > 0).mean())
        history['sample_sign_acc'].append(sample_sign_acc)
        history.setdefault('centroid_mqs', []).append(centroid_mqs)

        cmq_str = '  '.join(f'c{i}={m:+.2f}' for i, m in enumerate(centroid_mqs))
        cal_str = ''
        if use_scale_penalty or use_margin_penalty:
            tgt_str = []
            if use_scale_penalty:
                tgt_str.append(f'mean→{mq_target_scale:.0f}')
            if use_margin_penalty:
                tgt_str.append(f'p10→{target_margin:.0f}')
            cal_str = (f'  |mq|[p10={history["mq_abs_p10"][-1]:5.1f} '
                       f'med={history["mq_abs_median"][-1]:5.1f} '
                       f'p90={history["mq_abs_p90"][-1]:5.1f}]→{",".join(tgt_str)}')
        print(f'  [{name:8s}] ep {ep+1:2d}  '
              f'loss={history["train_loss"][-1]:.4f}  '
              f'acc={history["test_acc"][-1]*100:5.2f}%  '
              f'mq[c0]={cls_align["mean_mq_class0"]:+6.2f}  '
              f'mq[c1]={cls_align["mean_mq_class1"]:+6.2f}  '
              f'gap={cls_align["gap"]:+6.2f}  '
              f'AUC={cls_align["auc"]:.3f}  '
              f'sgnAcc={sample_sign_acc:.3f}'
              f'{cal_str}'
              f'  centroid:[{cmq_str}]')

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
def _build_model(backbone_type, head_type, spacelike_residual):
    """Build a Lorentzian or Euclidean backbone with the specified head."""
    if backbone_type == 'lorentzian':
        return LorentzianMLP(in_dim=SEQ_LEN,
                             spacelike_residual=spacelike_residual,
                             head_type=head_type).to(device)
    elif backbone_type == 'euclidean':
        return EuclideanMLP(in_dim=SEQ_LEN,
                            head_type=head_type).to(device)
    else:
        raise ValueError(f'unknown backbone_type: {backbone_type!r}')


def run_single_seed(seed, X_train, y_train, X_test, y_test, baseline_acc,
                    out_dir, save_artifacts=True, verbose=True,
                    spacelike_residual=0.0, head_type='linear',
                    backbone_type='lorentzian',
                    baseline_backbone_type='euclidean',
                    baseline_head_type='linear'):
    # Seed BEFORE constructing DataLoader so the shuffle is reproducible
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=g,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    if verbose:
        print(f'\n=== {backbone_type.upper()} backbone + head={head_type}, '
              f'spacelike_residual={spacelike_residual:.2f} — seed {seed} ===')
    model_L = _build_model(backbone_type, head_type, spacelike_residual)
    if verbose:
        print(f'  params: {sum(p.numel() for p in model_L.parameters()):,}')
    hist_L = train_one(model_L, train_loader, test_loader, name=f'M-s{seed}')

    if verbose:
        print(f'\n=== Baseline: {baseline_backbone_type.upper()} backbone + '
              f'head={baseline_head_type} — seed {seed} ===')
    torch.manual_seed(seed); np.random.seed(seed)
    g_e = torch.Generator(); g_e.manual_seed(seed)
    train_loader_e = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=g_e,
    )
    model_E = _build_model(baseline_backbone_type, baseline_head_type, 1.0)
    hist_E = train_one(model_E, train_loader_e, test_loader, name=f'B-s{seed}')

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


def ood_evaluation(model, ood_configs, baseline_seed=2000, verbose=True):
    """
    Run a trained model on multiple OOD test distributions.

    v14b changes (per review):
      - Removed sklearn dependency; uses mq_class_alignment() which has
        a built-in rank-sum AUC.
      - Deterministic seed offset via stable hash (md5), not Python's
        randomized hash.
      - Sample-level sign metrics: c0_timelike_frac, c1_spacelike_frac,
        sample_sign_acc — these don't get fooled by class-mean-only checks.
      - |mq| percentiles (p10, median, p90) — these reveal whether the
        encoding has tight margins or just a few outliers driving means.

    Args:
      model: trained model with .d_t and forward returning (logits, h, _)
      ood_configs: dict of config_name -> {'ar_rho', 'noise_std'}
      baseline_seed: seed offset for OOD test data generation
      verbose: print per-config results
    """
    import hashlib
    model.eval()
    results = {}

    for cfg_name, cfg in ood_configs.items():
        # Stable, deterministic seed per OOD config (review fix #5)
        seed_offset = baseline_seed + (
            int(hashlib.md5(cfg_name.encode()).hexdigest()[:8], 16) % 10000)

        X_te, y_te = generate_serial_sequences(
            n_per_class=N_TEST // 2,
            seq_len=SEQ_LEN,
            ar_rho=cfg['ar_rho'],
            noise_std=cfg['noise_std'],
            seed=seed_offset,
        )
        X_te = X_te.to(device)
        y_te_dev = y_te.to(device)

        with torch.no_grad():
            logits, h, _ = model(X_te)
            preds = logits.argmax(-1)
            acc = (preds == y_te_dev).float().mean().item() * 100
            mq = compute_mq(h, model.d_t).cpu().numpy()

        y_np = y_te.numpy()
        mq_c0 = mq[y_np == 0]
        mq_c1 = mq[y_np == 1]

        # AUC via mq_class_alignment (review fix #4 — no sklearn)
        align = mq_class_alignment(mq, y_np)
        auc = align['auc']

        # Class-mean sign-split (legacy metric, kept for continuity)
        sign_split   = bool(mq_c0.mean() < 0 < mq_c1.mean())
        inverted     = bool(mq_c1.mean() < 0 < mq_c0.mean())

        # Sample-level sign metrics (review fix #3 — class means alone hide
        # the case where many individual samples are on the wrong side)
        c0_timelike_frac  = float((mq_c0 < 0).mean())   # fraction of class 0 with mq<0
        c1_spacelike_frac = float((mq_c1 > 0).mean())   # fraction of class 1 with mq>0
        sample_sign_acc   = 0.5 * (c0_timelike_frac + c1_spacelike_frac)

        # |mq| percentiles (review fix #2 — capture distribution shape,
        # not just mean which can be skewed by outliers)
        mq_abs = np.abs(mq)
        mq_abs_p10    = float(np.percentile(mq_abs, 10))
        mq_abs_median = float(np.percentile(mq_abs, 50))
        mq_abs_p90    = float(np.percentile(mq_abs, 90))
        mq_abs_max    = float(mq_abs.max())

        results[cfg_name] = {
            'config': dict(cfg),
            'acc': acc,
            'auc_mq': float(auc),
            'mq_c0_mean': float(mq_c0.mean()),
            'mq_c1_mean': float(mq_c1.mean()),
            'mq_c0_std':  float(mq_c0.std()),
            'mq_c1_std':  float(mq_c1.std()),
            'gap': float(mq_c1.mean() - mq_c0.mean()),
            'sign_split': sign_split,
            'inverted_split': inverted,
            'c0_timelike_frac':  c0_timelike_frac,
            'c1_spacelike_frac': c1_spacelike_frac,
            'sample_sign_acc':   float(sample_sign_acc),
            'mq_max_abs':    mq_abs_max,
            'mq_abs_p10':    mq_abs_p10,
            'mq_abs_median': mq_abs_median,
            'mq_abs_p90':    mq_abs_p90,
        }

        if verbose:
            sign_marker = ('✓ clean' if sign_split else
                           '✗ inverted' if inverted else
                           'same-side')
            print(f'  {cfg_name:30s} acc={acc:5.1f}%  AUC={auc:.3f}  '
                  f'mq[c0]={mq_c0.mean():+7.2f}  mq[c1]={mq_c1.mean():+7.2f}  '
                  f'sample-sign={sample_sign_acc:.3f}  '
                  f'|mq| p10={mq_abs_p10:5.1f} med={mq_abs_median:5.1f} '
                  f'p90={mq_abs_p90:5.1f}  {sign_marker}')

    return results


def _calibration_dry_run(X_train, y_train, X_test_id, y_test_id,
                         lambda_scale, lambda_margin,
                         mq_target_scale, target_margin,
                         seed=42, n_epochs_dry=5):
    """
    Single-seed quick check: does the calibration penalty actually move
    |m_q| toward the target? Used by run_calibration_battery to filter
    out lambda settings that fail to push the encoding scale up before
    spending compute on the full 3-seed × 8-OOD evaluation.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=g,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_id, y_test_id),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # Save & temporarily reduce EPOCHS for dry run
    global EPOCHS
    saved_epochs = EPOCHS
    EPOCHS = n_epochs_dry
    try:
        model = _build_model('lorentzian', 'mq-mlp', spacelike_residual=0.0)
        # Suppress the per-epoch print
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            hist = train_one(model, train_loader, test_loader, name='dry',
                             mq_target_scale=mq_target_scale,
                             lambda_scale=lambda_scale,
                             target_margin=target_margin,
                             lambda_margin=lambda_margin)
    finally:
        EPOCHS = saved_epochs

    if not hist['mq_abs_mean']:
        return None
    return {
        'mq_abs_mean':   hist['mq_abs_mean'][-1],
        'mq_abs_p10':    hist['mq_abs_p10'][-1],
        'mq_abs_median': hist['mq_abs_median'][-1],
        'mq_abs_p90':    hist['mq_abs_p90'][-1],
        'test_acc':      hist['test_acc'][-1] * 100,
        'sample_sign_acc': hist['sample_sign_acc'][-1],
    }


def run_calibration_battery(ar_rho_train=0.9, noise_std_train=0.5,
                            n_seeds=3,
                            mq_target_scale=50.0,
                            target_margin=30.0,
                            lambda_sweep=(0.5, 2.0, 5.0, 10.0),
                            lambda_margin=2.0,
                            out_dir='.'):
    """
    v14b CALIBRATION TEST.

    Hypothesis:
      v13 found Lorentzian sign-split rate (16/24) < Euclidean (19/24).
      Two possible causes:
        (A) SplitNorm + causal residual STRUCTURALLY limits robustness
        (B) Bounded |m_q|≈8 alone is the cause; pushing |m_q| up to ~50
            (Euclidean-like) should close the gap.

    Test design (v14b improvements over v14):
      1. λ-SWEEP DRY RUN (review fix #1):
         Before spending compute on the full 3-seed battery, single-seed
         dry runs at multiple `lambda_scale` values find the smallest
         λ that actually pushes mean|m_q| past 0.85·target. This avoids
         the "calibration silently failed" outcome.

      2. JOINT MEAN + MARGIN PENALTY (review fix #2):
         Mean |m_q| can be inflated by outliers while most samples stay
         near zero. We add a margin penalty pushing the p10 of |m_q|
         past `target_margin`. This forces the WHOLE distribution away
         from zero, so OOD sign-flip robustness is actually tested.

      3. SAMPLE-LEVEL SIGN METRICS (review fix #3):
         Sign-split as currently defined (class means on opposite sides)
         can be satisfied while many individual samples are on the wrong
         side. We additionally report sample_sign_acc and class sign
         fractions, and use sample_sign_acc > 0.9 as the headline.

      4. NO sklearn (review fix #4): uses built-in mq_class_alignment.
      5. STABLE OOD seeds (review fix #5): md5 hash, deterministic.
      6. INTERPRETATION HEDGED (review fix #6): we don't claim "scale
         is the only cause"; we report what closing the |m_q| gap does
         to OOD sign-split, acknowledging the penalty also alters the
         optimization trajectory.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [SEED + i for i in range(n_seeds)]

    print('\n' + '#' * 72)
    print(f'# v14c CALIBRATION BATTERY')
    print(f'# Train: AR_RHO={ar_rho_train}, NOISE_STD={noise_std_train}')
    print(f'# Calibration target: mean|m_q|≈{mq_target_scale}, '
          f'p10|m_q|≥{target_margin}')
    print(f'# λ_scale sweep: {lambda_sweep}, λ_margin: {lambda_margin}')
    print('#' * 72)

    print(f'\nGenerating training data...')
    X_train, y_train = generate_serial_sequences(
        n_per_class=N_TRAIN // 2, seq_len=SEQ_LEN,
        ar_rho=ar_rho_train, noise_std=noise_std_train, seed=SEED)
    X_test_id, y_test_id = generate_serial_sequences(
        n_per_class=N_TEST // 2, seq_len=SEQ_LEN,
        ar_rho=ar_rho_train, noise_std=noise_std_train, seed=SEED + 1000)
    verify_dataset(X_train, y_train)

    # ─────────────────────────────────────────────────────────────────
    # Step 1: λ-sweep dry run to find the smallest effective lambda_scale
    # ─────────────────────────────────────────────────────────────────
    print('\n' + '─' * 72)
    print('Step 1: λ-sweep dry run to find effective lambda_scale')
    print('─' * 72)
    print(f'{"lambda_scale":>14s} | {"mean|m_q|":>10s} | {"p10|m_q|":>9s} | '
          f'{"p90|m_q|":>9s} | {"acc":>6s} | {"sgnAcc":>7s} | status')
    print('-' * 76)

    best_lambda = None
    sweep_results = {}
    target_reached = 0.85 * mq_target_scale
    p10_target = 0.5 * target_margin

    for ls in lambda_sweep:
        res = _calibration_dry_run(
            X_train, y_train, X_test_id, y_test_id,
            lambda_scale=ls, lambda_margin=lambda_margin,
            mq_target_scale=mq_target_scale, target_margin=target_margin,
            seed=SEED, n_epochs_dry=8)
        sweep_results[ls] = res
        if res is None:
            print(f'{ls:>14.2f} | {"NaN":>10s} | --- | --- | --- | --- | training failed')
            continue

        # v14c: require BOTH mean AND p10 to reach target before selecting λ.
        # Mean-only success can be satisfied by outliers while p10 stays
        # near zero — meaning most samples are still close to the light
        # cone and OOD sign-flip robustness is unchanged.
        mean_ok = res['mq_abs_mean'] >= target_reached
        p10_ok  = res['mq_abs_p10']  >= p10_target

        if mean_ok and p10_ok:
            status = '✓ mean+p10 ok'
        elif mean_ok:
            status = '⚠ mean ok, p10 low'
        else:
            status = '⚠ below target'

        print(f'{ls:>14.2f} | {res["mq_abs_mean"]:>10.1f} | '
              f'{res["mq_abs_p10"]:>9.1f} | {res["mq_abs_p90"]:>9.1f} | '
              f'{res["test_acc"]:>5.1f}% | {res["sample_sign_acc"]:>7.3f} | {status}')

        if best_lambda is None and mean_ok and p10_ok:
            best_lambda = ls

    if best_lambda is None:
        # Fall back to the largest lambda we tried; warn.
        valid = [(ls, r) for ls, r in sweep_results.items() if r is not None]
        if not valid:
            print('\n  ✗ ALL dry runs failed. Aborting.')
            return None
        best_lambda = max(valid, key=lambda x: x[1]['mq_abs_mean'])[0]
        print(f'\n  ⚠ No λ reached target. Using λ={best_lambda} (best of sweep).')
        print(f'    Calibration may not move |m_q| to the intended scale.')
    else:
        print(f'\n  ✓ Selected λ_scale = {best_lambda} for full battery.')

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Full battery — 3 configs × n_seeds × 8 OOD configs
    # ─────────────────────────────────────────────────────────────────
    ood_configs = {}
    for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:
        ood_configs[f'rho={rho:.1f}_noise={noise_std_train}'] = {
            'ar_rho': rho, 'noise_std': noise_std_train}
    for noise in [0.25, 1.0, 2.0]:
        if abs(noise - noise_std_train) < 1e-6:
            continue
        ood_configs[f'rho={ar_rho_train}_noise={noise}'] = {
            'ar_rho': ar_rho_train, 'noise_std': noise}

    configurations = [
        ('lorentzian_uncal',
         'Lorentzian + MqMLPHead (uncalibrated, |m_q| natural)',
         {'backbone_type': 'lorentzian', 'head_type': 'mq-mlp',
          'spacelike_residual': 0.0,
          'mq_target_scale': None, 'lambda_scale': 0.0,
          'target_margin': None, 'lambda_margin': 0.0}),
        ('lorentzian_cal',
         f'Lorentzian + MqMLPHead (calibrated: mean→{mq_target_scale}, '
         f'p10≥{target_margin}, λs={best_lambda}, λm={lambda_margin})',
         {'backbone_type': 'lorentzian', 'head_type': 'mq-mlp',
          'spacelike_residual': 0.0,
          'mq_target_scale': mq_target_scale, 'lambda_scale': best_lambda,
          'target_margin': target_margin, 'lambda_margin': lambda_margin}),
        ('euclidean_uncal',
         'Euclidean + MqMLPHead (uncalibrated, naturally |m_q|≈90)',
         {'backbone_type': 'euclidean', 'head_type': 'mq-mlp',
          'spacelike_residual': 1.0,
          'mq_target_scale': None, 'lambda_scale': 0.0,
          'target_margin': None, 'lambda_margin': 0.0}),
    ]

    all_results = {}
    for cfg_name, label, cfg in configurations:
        all_results[cfg_name] = {}
        print('\n\n' + '═' * 72)
        print(f'# {label}')
        print('═' * 72)

        for s in seeds:
            print(f'\n--- {cfg_name} | Seed {s} ---')
            torch.manual_seed(s); np.random.seed(s)
            g = torch.Generator(); g.manual_seed(s)
            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=BATCH_SIZE, shuffle=True, generator=g,
            )
            test_loader = DataLoader(
                TensorDataset(X_test_id, y_test_id),
                batch_size=BATCH_SIZE, shuffle=False,
            )
            model = _build_model(cfg['backbone_type'], cfg['head_type'],
                                 cfg['spacelike_residual'])
            print(f'  params: {sum(p.numel() for p in model.parameters()):,}')
            hist = train_one(model, train_loader, test_loader,
                             name=f'{cfg_name[:6]}-s{s}',
                             mq_target_scale=cfg['mq_target_scale'],
                             lambda_scale=cfg['lambda_scale'],
                             target_margin=cfg['target_margin'],
                             lambda_margin=cfg['lambda_margin'])

            print(f'\n  OOD evaluation:')
            ood_res = ood_evaluation(model, ood_configs,
                                     baseline_seed=10000 + s)
            ood_res['_train_mq_abs_final']    = (
                hist['mq_abs_mean'][-1] if hist['mq_abs_mean'] else float('nan'))
            ood_res['_train_p10_final']       = (
                hist['mq_abs_p10'][-1] if hist['mq_abs_p10'] else float('nan'))
            ood_res['_train_acc_final']       = (
                hist['test_acc'][-1] * 100 if hist['test_acc'] else float('nan'))
            ood_res['_train_sample_sign_acc'] = (
                hist['sample_sign_acc'][-1] if hist['sample_sign_acc']
                else float('nan'))
            all_results[cfg_name][s] = ood_res

    # ─────────────────────────────────────────────────────────────────
    # Step 3: 3-way comparison report
    # ─────────────────────────────────────────────────────────────────
    print('\n\n' + '#' * 72)
    print('# CALIBRATION TEST — 3-WAY COMPARISON (v14c)')
    print('#' * 72)
    print(f'\nTraining distribution: AR_RHO={ar_rho_train}, '
          f'NOISE_STD={noise_std_train}')
    print(f'Calibration: mean|m_q|→{mq_target_scale}, p10|m_q|≥{target_margin}, '
          f'λ_scale={best_lambda}, λ_margin={lambda_margin}\n')

    cfg_names = list(ood_configs.keys())
    config_keys = ['lorentzian_uncal', 'lorentzian_cal', 'euclidean_uncal']
    config_short = ['Lorentz uncal', 'Lorentz CAL', 'Euclid uncal']

    print(f'{"OOD Config":28s} | ' +
          ' | '.join(f'{s:30s}' for s in config_short))
    print(f'{"":28s} | ' +
          ' | '.join(f'{"split  AUC   sgnAcc  p10":30s}' for _ in config_short))
    print('-' * (30 + 33 * 3))
    for cfg_name in cfg_names:
        row = f'{cfg_name:28s} | '
        cells = []
        for k in config_keys:
            seed_res = [all_results[k][s][cfg_name] for s in seeds]
            n_split = sum(1 for r in seed_res if r['sign_split'])
            mean_auc = np.mean([r['auc_mq'] for r in seed_res])
            mean_sgnacc = np.mean([r['sample_sign_acc'] for r in seed_res])
            mean_p10 = np.mean([r['mq_abs_p10'] for r in seed_res])
            cells.append(f'{n_split}/{n_seeds}    {mean_auc:.3f}  '
                         f'{mean_sgnacc:.3f}  {mean_p10:5.1f}')
        row += ' | '.join(f'{c:30s}' for c in cells)
        print(row)

    # ─────────────────────────────────────────────────────────────────
    # Headline metrics with sample-level sign accuracy
    # ─────────────────────────────────────────────────────────────────
    print('\n─── HEADLINE METRICS ───')
    headline = {}
    for k, label in zip(config_keys,
                         ['Lorentzian uncalibrated',
                          f'Lorentzian CALIBRATED (mean→{mq_target_scale}, p10≥{target_margin})',
                          'Euclidean uncalibrated']):
        n_total = len(cfg_names) * n_seeds
        n_split_total = sum(
            sum(1 for s in seeds if all_results[k][s][cfg]['sign_split'])
            for cfg in cfg_names)
        n_strong_split = sum(
            sum(1 for s in seeds
                if all_results[k][s][cfg]['sample_sign_acc'] > 0.9)
            for cfg in cfg_names)
        # v14c: strong sign-split AND task hasn't collapsed.
        # A model can be at chance on classification (acc ~50%) and
        # still produce sample_sign_acc > 0.9 if the geometry happens
        # to align — but that's not a real success.
        n_strong_and_correct = sum(
            sum(1 for s in seeds
                if all_results[k][s][cfg]['sample_sign_acc'] > 0.9
                and all_results[k][s][cfg]['acc'] > 90)
            for cfg in cfg_names)
        mean_auc_all = np.mean([
            all_results[k][s][cfg]['auc_mq']
            for s in seeds for cfg in cfg_names])
        mean_sgnacc_all = np.mean([
            all_results[k][s][cfg]['sample_sign_acc']
            for s in seeds for cfg in cfg_names])
        mean_p10_all = np.mean([
            all_results[k][s][cfg]['mq_abs_p10']
            for s in seeds for cfg in cfg_names])
        train_mq_abs = np.mean([
            all_results[k][s]['_train_mq_abs_final'] for s in seeds])
        train_p10 = np.mean([
            all_results[k][s]['_train_p10_final'] for s in seeds])
        train_acc = np.mean([
            all_results[k][s]['_train_acc_final'] for s in seeds])

        headline[k] = {
            'n_split': n_split_total,
            'n_strong_split': n_strong_split,
            'n_strong_and_correct': n_strong_and_correct,
            'mean_sgnacc': mean_sgnacc_all,
            'mean_auc': mean_auc_all,
            'train_mq_abs': train_mq_abs,
            'train_p10': train_p10,
        }

        print(f'\n  {label}:')
        print(f'    In-dist train acc:                    {train_acc:6.2f}%')
        print(f'    In-dist mean|m_q|:                    {train_mq_abs:6.1f}')
        print(f'    In-dist p10|m_q|:                     {train_p10:6.1f}')
        print(f'    OOD sign-split rate (mean ⩖ 0):      {n_split_total:>3d}/{n_total}')
        print(f'    OOD strong sign-split (sgnAcc>0.9):   {n_strong_split:>3d}/{n_total}')
        print(f'    OOD strong sign-split AND acc>90:     {n_strong_and_correct:>3d}/{n_total}')
        print(f'    OOD mean sample_sign_acc:             {mean_sgnacc_all:.3f}')
        print(f'    OOD mean mq AUC:                      {mean_auc_all:.3f}')
        print(f'    OOD mean p10|m_q|:                    {mean_p10_all:.1f}')

    # ─────────────────────────────────────────────────────────────────
    # Hedged interpretation (review fix #6)
    # ─────────────────────────────────────────────────────────────────
    print('\n─── INTERPRETATION (hedged) ───')

    h_uncal = headline['lorentzian_uncal']
    h_cal   = headline['lorentzian_cal']
    h_euc   = headline['euclidean_uncal']
    n_total = len(cfg_names) * n_seeds

    print(f'\n  Lorentzian uncal:  in-dist mean|m_q|={h_uncal["train_mq_abs"]:5.1f}, '
          f'OOD sign-split {h_uncal["n_split"]}/{n_total} (strong: {h_uncal["n_strong_split"]}/{n_total})')
    print(f'  Lorentzian CAL:    in-dist mean|m_q|={h_cal["train_mq_abs"]:5.1f}, '
          f'OOD sign-split {h_cal["n_split"]}/{n_total} (strong: {h_cal["n_strong_split"]}/{n_total})')
    print(f'  Euclidean uncal:   in-dist mean|m_q|={h_euc["train_mq_abs"]:5.1f}, '
          f'OOD sign-split {h_euc["n_split"]}/{n_total} (strong: {h_euc["n_strong_split"]}/{n_total})')
    print()

    if abs(h_cal['train_mq_abs'] - h_uncal['train_mq_abs']) < 5:
        print('  ⚠ CALIBRATION DID NOT TAKE: mean|m_q| of the calibrated run is')
        print('    indistinguishable from uncalibrated. Conclusion: cannot tell')
        print(f'    whether scale matters. Try larger lambda_sweep or longer training.')
    elif h_cal['train_p10'] < target_margin * 0.5:
        print('  ⚠ MEAN PUSHED, MARGIN DID NOT: mean|m_q| reached target but p10|m_q|')
        print(f'    is still {h_cal["train_p10"]:.1f} vs target ≥{target_margin}. The')
        print('    encoding has outliers carrying the mean while most samples are')
        print('    near zero. Increase lambda_margin or rerun with stronger pressure.')
    else:
        # Calibration succeeded — interpret the OOD outcome
        delta_split  = h_cal['n_split']  - h_uncal['n_split']
        delta_strong = h_cal['n_strong_split'] - h_uncal['n_strong_split']
        gap_to_euc   = h_euc['n_strong_split'] - h_cal['n_strong_split']

        print(f'  Calibration succeeded: mean|m_q| moved {h_uncal["train_mq_abs"]:.0f}→{h_cal["train_mq_abs"]:.0f}, '
              f'p10|m_q| moved {h_uncal["train_p10"]:.0f}→{h_cal["train_p10"]:.0f}.')
        print(f'  OOD strong sign-split (sgnAcc > 0.9):')
        print(f'    Lorentzian uncal: {h_uncal["n_strong_split"]}/{n_total}')
        print(f'    Lorentzian CAL:   {h_cal["n_strong_split"]}/{n_total}  (Δ vs uncal: {delta_strong:+d})')
        print(f'    Euclidean uncal:  {h_euc["n_strong_split"]}/{n_total}  (gap to CAL: {gap_to_euc:+d})')
        print()

        if delta_strong >= max(2, n_total // 8) and gap_to_euc <= 2:
            print('  → Pushing |m_q| up by penalty CLOSES most of the Lorentzian–Euclidean')
            print('    OOD robustness gap. Consistent with "scale was the dominant cause".')
            print('    Caveat: the penalty also alters optimization trajectory, so this')
            print('    is not proof that scale is the ONLY mechanism.')
        elif delta_strong < max(2, n_total // 8):
            print('  → Pushing |m_q| up by penalty does NOT improve OOD sign-robustness.')
            print('    SplitNorm + causal residual appear to have a structural property')
            print('    that limits sign-split robustness independent of |m_q| scale.')
            print('    Genuine backbone-level finding worth a paper subsection.')
        else:
            print('  → Pushing |m_q| up by penalty PARTIALLY closes the gap. Scale')
            print('    contributes but is not the entire story; some structural property')
            print('    of the Lorentzian backbone also matters.')

    save_path = out_dir / 'v14c_calibration_results.json'
    with open(save_path, 'w') as f:
        json.dump({
            'training_config': {
                'ar_rho_train': ar_rho_train,
                'noise_std_train': noise_std_train,
                'mq_target_scale': mq_target_scale,
                'target_margin': target_margin,
                'lambda_sweep': list(lambda_sweep),
                'lambda_scale_used': best_lambda,
                'lambda_margin': lambda_margin,
                'n_seeds': n_seeds,
                'seeds': seeds,
            },
            'sweep_results': sweep_results,
            'ood_configs': ood_configs,
            'headline': headline,
            'results': all_results,
        }, f, indent=2, default=_json_safe)
    print(f'\nSaved raw results: {save_path}')

    return all_results


if __name__ == '__main__':
    # v14c default: λ-sweep dry run (mean+p10 selection), then 3-way calibration battery.
    run_calibration_battery(
        ar_rho_train=0.9,
        noise_std_train=0.5,
        n_seeds=3,
        mq_target_scale=50.0,
        target_margin=30.0,
        lambda_sweep=(0.5, 2.0, 5.0, 10.0),
        lambda_margin=2.0,
        out_dir='.',
    )