"""
Lorentzian Backbone — Step 2 (v15c): TASK DIVERSITY BATTERY (revised x2)
========================================================================

THE QUESTION:
  Sections 4-5 of the paper establish cross-backbone direction-matching
  on AR(1) vs IID. Section 7.4 acknowledges this is a single causal
  structure. v15 tests whether the dep→timelike, indep→spacelike
  emergence generalizes across causal structures.

TASKS (categorised by what they actually compare):

  INDEP-VS-DEP tasks (literal Realizability prediction applies):
    1. ar1            AR(1) vs i.i.d.
    3. granger        x→y (causal) vs (x,y) independent. v15c: y_0
                      initialised at matched marginal scale so the
                      first y token does not leak class identity
                      through its variance.
    4. intervention   AR(1) x with y attached vs do(x)=ε with same y.
                      v15c docstring honestly notes y inherits some
                      autocorrelation from x in the observational
                      class, so y is NOT entirely uninformative.

  DEPTH-OF-DEPENDENCE task (related but distinct claim):
    2. markov2        AR(2) vs AR(1) with matched lag-1.
                      Both classes causally structured. Direction-match
                      = "deeper-dep → timelike than shallower-dep".

PROCEDURE: 4 tasks × 2 backbones (Lorentzian+MqMLPHead, Euclidean+MqMLPHead)
× 3 seeds = 24 trainings.

HEADLINE METRICS: split by category (indep_vs_dep, depth, pooled).

CHANGES vs v15b (per third review):
  - Per-task explicit single-feature baseline. Previously the helper
    auto-detected "univariate or bivariate" by checking even shape,
    which (with SEQ_LEN=32) wrongly added a fake cross-corr feature
    to ar1 and markov2. Now baselines are: lag-1(full) for ar1,
    lag-1(full) [+lag-2 reported] for markov2, cross-corr(x,y) for
    granger, lag-1(x) for intervention. All threshold sweeps and
    feature choices on TRAIN ONLY.
  - granger generator: y_0 sampled at marginal-matched scale instead
    of residual-only scale, so the first y token can no longer leak
    class identity through its variance.
  - intervention docstring sharpened: y inherits autocorrelation from
    observational x, so y is partially informative (not "side channel
    that does NOT distinguish the classes").
  - Suite-level comment block synced: explicit task-category labels.

CHANGES vs v15 (retained from v15b):
  - Fixed nested-quote f-string (Python 3.11 SyntaxError).
  - run_single_seed() propagates in_dim from data shape.
  - markov2 reported separately from indep-vs-dep tasks in headline.

CLI:
    python lorentz_step2_v15c_tasks.py
    # No flags. ~25-35 min on Colab GPU (24 trainings).
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
# v15 TASK SUITE — two interpretive categories of causal pair
# ══════════════════════════════════════════════════════════════════
# All tasks share the convention: class 0 = "more causally structured",
# class 1 = "less causally structured" (or independent), so the
# direction tag mq[c0] < 0 < mq[c1] is meaningful across tasks. But
# the EXACT statement that direction-tag corresponds to depends on
# task category:
#
#   indep_vs_dep tasks   (ar1, granger, intervention):
#     class 0 has causal structure; class 1 is causally trivial
#     (i.i.d. or do-intervened to break dependence). Direction-match
#     here = literal Realizability prediction "causally connected →
#     timelike, causally trivial → spacelike".
#
#   depth-of-dependence task  (markov2):
#     BOTH classes are causally structured time series; only the
#     DEPTH of dependence differs (AR(2) vs AR(1) with matched lag-1).
#     Direction-match here = related-but-distinct claim "deeper
#     dependence → timelike than shallower dependence".
#
# Each generator returns (X, y) with X shape (N, feat_dim) and y ∈ {0,1}.
# Univariate tasks (ar1, markov2): feat_dim = seq_len.
# Bivariate tasks (granger, intervention): feat_dim = 2 * seq_len
#   (flattened [x; y]).
# ══════════════════════════════════════════════════════════════════

def generate_markov2(n_per_class, seq_len, rho1=0.6, rho2=0.3,
                     noise_std=0.5, seed=0):
    """
    DEPTH-OF-DEPENDENCE TASK (not independent vs dependent).

    Class 0 (DEEPER, second-order Markov):
        x_t = rho1 * x_{t-1} + rho2 * x_{t-2} + ε_t
        Has BOTH lag-1 and lag-2 structure.

    Class 1 (SHALLOWER, first-order Markov, lag-1 ONLY):
        x_t = rho_match * x_{t-1} + ε'_t
        rho_match calibrated so that the FINITE-SAMPLE empirical lag-1
        autocorrelation of class 1 matches that of class 0 (correcting
        for AR-1's small-sample downward bias). Marginal variance also
        matched.

    Note on framing (v15b — important):
      Both classes are causally structured time series. This task does
      NOT pit causally connected vs causally disconnected; it pits a
      DEEPER causal chain against a SHALLOWER one. The Realizability
      "dep→timelike, indep→spacelike" prediction does not directly
      apply here. What we test is the related but distinct claim:
      under the same convention (class 0 = "more causally structured"),
      does the model still place class 0 in the timelike region?
      Direction-match for this task should be reported separately
      from the indep-vs-dep tasks.

    A single-feature lag-1 baseline should be ~chance after the
    finite-sample calibration. A model that uses higher-order
    (lag-2+) structure can solve it.

    Returns:
        X: (N, seq_len) float32
        y: (N,) int64
    """
    rng = np.random.default_rng(seed)
    burn = 200

    # --- Class 0: AR(2) ---
    eps0 = rng.normal(0, noise_std, size=(n_per_class, seq_len + burn))
    buf = np.zeros_like(eps0)
    buf[:, 0] = eps0[:, 0]
    buf[:, 1] = rho1 * buf[:, 0] + eps0[:, 1]
    for t in range(2, seq_len + burn):
        buf[:, t] = rho1 * buf[:, t-1] + rho2 * buf[:, t-2] + eps0[:, t]
    x_dep = buf[:, burn:].astype(np.float32)

    # Compute target lag-1 autocorr from class 0's finite-sample data
    ac1_target = []
    for s in x_dep:
        if s.std() > 1e-6:
            ac1_target.append(np.corrcoef(s[:-1], s[1:])[0, 1])
    ac1_target = float(np.mean(ac1_target))
    var_target = float(x_dep.var())

    # Calibrate ρ_match so that class 1's *finite-sample* lag-1 matches
    # class 0's by tuning. AR(1) with population ρ has E[r_hat] ≈ ρ - (1+3ρ)/(seq_len-1)
    # (Marriott-Pope bias). We search numerically because the bias also
    # depends on burn-in handling and is approximate.
    def _simulate_class1(rho_pop, n_calib=400):
        rng_local = np.random.default_rng(seed + 7)
        sigma = np.sqrt(var_target * (1 - rho_pop ** 2)) if abs(rho_pop) < 1 else noise_std
        eps_local = rng_local.normal(0, sigma, size=(n_calib, seq_len + burn))
        b = np.zeros_like(eps_local)
        b[:, 0] = rng_local.normal(0, np.sqrt(var_target), size=n_calib)
        for t in range(1, seq_len + burn):
            b[:, t] = rho_pop * b[:, t-1] + eps_local[:, t]
        seqs = b[:, burn:]
        acs = [np.corrcoef(s[:-1], s[1:])[0, 1] for s in seqs if s.std() > 1e-6]
        return float(np.mean(acs))

    # Bisection on ρ_pop ∈ [0, 0.99] to match ac1_target
    lo, hi = 0.0, 0.99
    rho_pop = ac1_target  # initial guess
    for _ in range(15):
        observed = _simulate_class1(rho_pop)
        if abs(observed - ac1_target) < 0.005:
            break
        if observed < ac1_target:
            lo = rho_pop
            rho_pop = (rho_pop + hi) / 2
        else:
            hi = rho_pop
            rho_pop = (rho_pop + lo) / 2

    # --- Class 1: AR(1) with calibrated ρ ---
    sigma_eps1 = np.sqrt(var_target * (1 - rho_pop ** 2))
    eps1 = rng.normal(0, sigma_eps1, size=(n_per_class, seq_len + burn))
    buf1 = np.zeros_like(eps1)
    buf1[:, 0] = rng.normal(0, np.sqrt(var_target), size=n_per_class)
    for t in range(1, seq_len + burn):
        buf1[:, t] = rho_pop * buf1[:, t-1] + eps1[:, t]
    x_iid = buf1[:, burn:].astype(np.float32)

    X = np.concatenate([x_dep, x_iid], axis=0)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    perm = rng.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


def generate_granger(n_per_class, seq_len, beta=0.6, noise_std=0.5, seed=0):
    """
    Bivariate causality test (Granger-style, structural).

    Class 0 (CAUSAL, X→Y):
        x_t = ε^x_t   (i.i.d.)
        y_t = β * x_{t-1} + ε^y_t   (caused by lagged x)
        At t=0 there is no x_{-1}, so y_0 is initialised from a draw
        with variance MATCHING class 1's marginal (see below).

    Class 1 (INDEPENDENT):
        x_t = ε^x_t   (i.i.d.)
        y_t = ε^y_t   (i.i.d., independent of x)

    Marginal-matching:
      class-0 var(y_t) for t≥1 = β² σ² + σ² = (1+β²) σ²
      class-1 var(y_t)         = (1+β²) σ²
      class-0 var(y_0) was previously σ² (mismatch); v15c uses (1+β²) σ²
      so the y series have matched marginal variance at every time step,
      not just on average. The first y token can no longer leak class
      identity through its variance.

    The remaining difference is purely the cross-series lag-1 dependence
    x → y in class 0, absent in class 1.

    Returns:
        X: (N, 2 * seq_len) float32, flattened [x_seq, y_seq]
        y: (N,) int64
    """
    rng = np.random.default_rng(seed)
    sigma_y_marginal = noise_std * np.sqrt(1 + beta ** 2)

    # --- Class 0: x i.i.d., y = β x_{t-1} + ε ---
    eps_x0 = rng.normal(0, noise_std, size=(n_per_class, seq_len))
    eps_y0 = rng.normal(0, noise_std, size=(n_per_class, seq_len))
    x0 = eps_x0.astype(np.float32)
    y0 = np.zeros((n_per_class, seq_len), dtype=np.float32)
    # v15c: initialise y_0 with matched-marginal scale, not residual scale
    y0[:, 0] = rng.normal(0, sigma_y_marginal, size=n_per_class)
    for t in range(1, seq_len):
        y0[:, t] = beta * x0[:, t-1] + eps_y0[:, t]

    # --- Class 1: x and y both i.i.d., MATCHED marginal variances ---
    eps_x1 = rng.normal(0, noise_std, size=(n_per_class, seq_len))
    eps_y1 = rng.normal(0, sigma_y_marginal, size=(n_per_class, seq_len))
    x1 = eps_x1.astype(np.float32)
    y1 = eps_y1.astype(np.float32)

    # Flatten into (N, 2*seq_len) — concat along time, [x | y]
    X0 = np.concatenate([x0, y0], axis=1)
    X1 = np.concatenate([x1, y1], axis=1)

    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    perm = rng.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


def generate_intervention(n_per_class, seq_len, beta=0.6, noise_std=0.5,
                          seed=0):
    """
    do() vs observational on a 2-variable causal chain.

    Underlying SCM: x → y, with
        x_t = α x_{t-1} + ε^x_t
        y_t = β x_{t-1} + ε^y_t
    where α=0.5 (so x has its own AR-1 structure on top of x→y).

    Class 0 (DEPENDENT, observational): both x and y evolve naturally
        per the SCM; cross-series structure x→y is present, AND x has
        its own dynamics α.

    Class 1 (TRIVIAL FOR x, do(x)=ε): we INTERVENE on x by overwriting
        it with i.i.d. noise (do(x_t = ε^x_t)), so x loses its AR-1
        structure but the SCM equation for y still uses the (now i.i.d.)
        x. y's value still depends on x_{t-1} in both classes — the
        only thing that changes between classes is the temporal
        autocorrelation of x.

    HONESTY ABOUT WHAT THIS TESTS (v15c):
      In the flattened [x|y] representation, the dominant signal is
      lag-1(x): observational class has AR(1) x with α=0.5; do(x) class
      has i.i.d. x. The y half is NOT entirely uninformative either —
      because y_t = β x_{t-1} + ε^y_t, the observational class's y
      INHERITS some of x's autocorrelation, while the do(x) class's y
      is essentially i.i.d.-plus-lagged-i.i.d. So y can also distinguish
      the classes, just less directly than x.

      The task therefore measures whether direction-matching survives
      when the causal structure is propagated through a 2-variable SCM
      and presented to the model in a flattened bivariate format. It
      is closer to "AR(1)-vs-IID through an SCM" than to a clean
      do-calculus generalisation test. A genuine do-vs-observational
      test where the intervention also distinguishes y's distribution
      directly (e.g., interventions on y, or where do() changes β)
      is left to follow-up work.

    Returns:
        X: (N, 2*seq_len) float32
        y: (N,) int64
    """
    rng = np.random.default_rng(seed)
    alpha = 0.5

    # --- Class 0: observational, x has AR-1 dynamics ---
    eps_x0 = rng.normal(0, noise_std, size=(n_per_class, seq_len))
    eps_y0 = rng.normal(0, noise_std, size=(n_per_class, seq_len))
    x0 = np.zeros((n_per_class, seq_len), dtype=np.float32)
    x0[:, 0] = eps_x0[:, 0] / np.sqrt(1 - alpha ** 2)  # stationary init
    for t in range(1, seq_len):
        x0[:, t] = alpha * x0[:, t-1] + eps_x0[:, t]
    y0 = np.zeros((n_per_class, seq_len), dtype=np.float32)
    y0[:, 0] = eps_y0[:, 0]
    for t in range(1, seq_len):
        y0[:, t] = beta * x0[:, t-1] + eps_y0[:, t]

    # --- Class 1: do(x) intervention, x is i.i.d. (matched marginal var) ---
    # observational var of x = noise_std² / (1 - α²)
    sigma_x1 = noise_std / np.sqrt(1 - alpha ** 2)
    eps_x1 = rng.normal(0, sigma_x1, size=(n_per_class, seq_len))
    eps_y1 = rng.normal(0, noise_std, size=(n_per_class, seq_len))
    x1 = eps_x1.astype(np.float32)  # do(x) makes x i.i.d.
    y1 = np.zeros((n_per_class, seq_len), dtype=np.float32)
    y1[:, 0] = eps_y1[:, 0]
    for t in range(1, seq_len):
        y1[:, t] = beta * x1[:, t-1] + eps_y1[:, t]

    X0 = np.concatenate([x0, y0], axis=1)
    X1 = np.concatenate([x1, y1], axis=1)

    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    perm = rng.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


# Task dispatcher: maps task name → (generator_fn, feat_dim, label)
def get_task_data(task_name, n_per_class, seq_len, seed):
    """Generate data for one of the v15 task suite. Returns (X, y, feat_dim)."""
    if task_name == 'ar1':
        X, y = generate_serial_sequences(
            n_per_class=n_per_class, seq_len=seq_len,
            ar_rho=0.9, noise_std=0.5, seed=seed)
        return X, y, seq_len
    elif task_name == 'markov2':
        X, y = generate_markov2(
            n_per_class=n_per_class, seq_len=seq_len,
            rho1=0.6, rho2=0.3, noise_std=0.5, seed=seed)
        return X, y, seq_len
    elif task_name == 'granger':
        X, y = generate_granger(
            n_per_class=n_per_class, seq_len=seq_len,
            beta=0.6, noise_std=0.5, seed=seed)
        return X, y, 2 * seq_len
    elif task_name == 'intervention':
        X, y = generate_intervention(
            n_per_class=n_per_class, seq_len=seq_len,
            beta=0.6, noise_std=0.5, seed=seed)
        return X, y, 2 * seq_len
    else:
        raise ValueError(f'Unknown task: {task_name!r}')


def verify_task_data(task_name, X, y):
    """Lightweight per-task sanity check: print marginals + class-0 vs class-1
    distinguishing statistic to confirm the data generator is doing what we think."""
    print(f'  Task: {task_name}')
    print(f'    X shape: {tuple(X.shape)}, y shape: {tuple(y.shape)}, '
          f'classes: {np.bincount(y.numpy())}')
    Xn = X.numpy()
    print(f'    Overall: mean={Xn.mean():+.3f}  std={Xn.std():.3f}')

    if task_name in ('ar1', 'markov2'):
        # Univariate: lag-1 and lag-2 autocorr per class
        for cls in [0, 1]:
            mask = (y == cls).numpy()
            seqs = Xn[mask]
            ac1, ac2 = [], []
            for s in seqs:
                if s.std() > 1e-6:
                    ac1.append(np.corrcoef(s[:-1], s[1:])[0, 1])
                    if len(s) > 2:
                        ac2.append(np.corrcoef(s[:-2], s[2:])[0, 1])
            cls_name = 'DEP' if cls == 0 else 'IID/M1'
            print(f'    {cls_name:6s} (cls {cls}): lag-1={np.mean(ac1):+.3f}  '
                  f'lag-2={np.mean(ac2):+.3f}  marginal_var={seqs.var():.3f}')
    elif task_name in ('granger', 'intervention'):
        # Bivariate flat: split back into x and y halves, report cross-corr
        T = Xn.shape[1] // 2
        for cls in [0, 1]:
            mask = (y == cls).numpy()
            seqs = Xn[mask]
            x_part = seqs[:, :T]
            y_part = seqs[:, T:]
            # cross-corr lag-1: corr(x_{t-1}, y_t)
            cc = []
            for i in range(len(seqs)):
                if x_part[i].std() > 1e-6 and y_part[i].std() > 1e-6:
                    cc.append(np.corrcoef(x_part[i, :-1], y_part[i, 1:])[0, 1])
            x_ac1 = []
            for i in range(len(seqs)):
                if x_part[i].std() > 1e-6:
                    x_ac1.append(np.corrcoef(x_part[i, :-1], x_part[i, 1:])[0, 1])
            cls_name = 'CAUSAL/OBS' if cls == 0 else 'INDEP/DO'
            print(f'    {cls_name:11s} (cls {cls}): '
                  f'corr(x_{{t-1}}, y_t)={np.mean(cc):+.3f}  '
                  f'lag-1(x)={np.mean(x_ac1):+.3f}  '
                  f'var(x)={x_part.var():.3f}  var(y)={y_part.var():.3f}')
    print('  → check that class-0 has the expected structure and class-1 does not.')


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
def _build_model(backbone_type, head_type, spacelike_residual, in_dim=None):
    """Build a Lorentzian or Euclidean backbone with the specified head.

    v15: in_dim defaults to SEQ_LEN (1-D univariate AR/Markov tasks),
    but can be overridden for tasks with different feature dimensions
    (e.g., Granger bivariate sequences flattened to 2*SEQ_LEN).
    """
    if in_dim is None:
        in_dim = SEQ_LEN
    if backbone_type == 'lorentzian':
        return LorentzianMLP(in_dim=in_dim,
                             spacelike_residual=spacelike_residual,
                             head_type=head_type).to(device)
    elif backbone_type == 'euclidean':
        return EuclideanMLP(in_dim=in_dim,
                            head_type=head_type).to(device)
    else:
        raise ValueError(f'unknown backbone_type: {backbone_type!r}')


def run_single_seed(seed, X_train, y_train, X_test, y_test, baseline_acc,
                    out_dir, save_artifacts=True, verbose=True,
                    spacelike_residual=0.0, head_type='linear',
                    backbone_type='lorentzian',
                    baseline_backbone_type='euclidean',
                    baseline_head_type='linear',
                    in_dim=None):
    # Seed BEFORE constructing DataLoader so the shuffle is reproducible
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)

    # v15b: derive in_dim from data if not provided, so this helper is
    # safe for bivariate tasks (Granger / intervention, X.shape[-1] = 64).
    if in_dim is None:
        in_dim = X_train.shape[-1]

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
    model_L = _build_model(backbone_type, head_type, spacelike_residual,
                           in_dim=in_dim)
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
    model_E = _build_model(baseline_backbone_type, baseline_head_type, 1.0,
                           in_dim=in_dim)
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



# ══════════════════════════════════════════════════════════════════
# v15 TASK DIVERSITY BATTERY
# ══════════════════════════════════════════════════════════════════

def _feat_lag1_full(X):
    """Lag-1 sample autocorr of full flat sequence."""
    Xn = X.numpy() if hasattr(X, 'numpy') else X
    out = np.zeros(len(Xn), dtype=np.float64)
    for i, s in enumerate(Xn):
        if s.std() > 1e-6:
            out[i] = np.corrcoef(s[:-1], s[1:])[0, 1]
    return out


def _feat_lag2_full(X):
    """Lag-2 sample autocorr of full flat sequence (used for markov2 only)."""
    Xn = X.numpy() if hasattr(X, 'numpy') else X
    out = np.zeros(len(Xn), dtype=np.float64)
    for i, s in enumerate(Xn):
        if len(s) > 2 and s.std() > 1e-6:
            out[i] = np.corrcoef(s[:-2], s[2:])[0, 1]
    return out


def _feat_lag1_x(X):
    """Lag-1 of x half (first SEQ_LEN entries) for bivariate flat input."""
    Xn = X.numpy() if hasattr(X, 'numpy') else X
    T = Xn.shape[1] // 2
    out = np.zeros(len(Xn), dtype=np.float64)
    for i, s in enumerate(Xn):
        xs = s[:T]
        if xs.std() > 1e-6:
            out[i] = np.corrcoef(xs[:-1], xs[1:])[0, 1]
    return out


def _feat_crosscorr_xy(X):
    """corr(x_{t-1}, y_t) for bivariate flat input."""
    Xn = X.numpy() if hasattr(X, 'numpy') else X
    T = Xn.shape[1] // 2
    out = np.zeros(len(Xn), dtype=np.float64)
    for i, s in enumerate(Xn):
        xs, ys = s[:T], s[T:]
        if xs.std() > 1e-6 and ys.std() > 1e-6:
            out[i] = np.corrcoef(xs[:-1], ys[1:])[0, 1]
    return out


def _fit_threshold_on_train(f_train, y_train):
    """Sweep thresholds on train; return (best_thr, best_dir, train_acc)."""
    cand = np.unique(np.concatenate([[f_train.min() - 1], np.sort(f_train),
                                     [f_train.max() + 1]]))
    best_acc, best_thr, best_dir = 0.0, 0.0, 1
    for thr in cand[::max(1, len(cand) // 200)]:
        for direction in [1, -1]:
            pred = ((direction * f_train) > (direction * thr)).astype(int)
            acc = (pred == y_train).mean()
            if acc > best_acc:
                best_acc = acc; best_thr = thr; best_dir = direction
    return best_thr, best_dir, best_acc


def lag1_baseline_acc(task_name, X_train, y_train, X_test, y_test):
    """Single-feature baseline with PER-TASK feature choice.

    v15c (per second review):
      - Picks the feature explicitly by task (no fake bivariate feature
        for univariate tasks; SEQ_LEN=32 being even no longer triggers
        the cross-corr branch on ar1/markov2 inputs).
      - For markov2 we report BOTH lag-1 and lag-2 single-feature
        baselines, with lag-1 being the headline (the task is designed
        so lag-1 is matched between classes); lag-2 reveals the easier
        single-feature alternative.
      - All threshold sweeps and feature selection happen on TRAIN ONLY.

    Returns (test_acc, feature_name, extras_dict).
    extras_dict optionally contains a {'lag2_acc': ...} for markov2.
    """
    yn_train = y_train.numpy() if hasattr(y_train, 'numpy') else y_train
    yn_test  = y_test.numpy()  if hasattr(y_test, 'numpy')  else y_test

    extras = {}

    if task_name == 'ar1':
        feat_fn, feat_name = _feat_lag1_full, 'lag-1 of full seq'
    elif task_name == 'markov2':
        feat_fn, feat_name = _feat_lag1_full, 'lag-1 of full seq'
        # Also report lag-2 for transparency
        f_tr2 = _feat_lag2_full(X_train); f_te2 = _feat_lag2_full(X_test)
        thr2, dir2, _ = _fit_threshold_on_train(f_tr2, yn_train)
        pred2 = ((dir2 * f_te2) > (dir2 * thr2)).astype(int)
        extras['lag2_acc'] = float((pred2 == yn_test).mean())
    elif task_name == 'granger':
        feat_fn, feat_name = _feat_crosscorr_xy, 'cross-corr x→y'
    elif task_name == 'intervention':
        feat_fn, feat_name = _feat_lag1_x, 'lag-1 of x half'
    else:
        raise ValueError(f'Unknown task: {task_name!r}')

    f_tr = feat_fn(X_train); f_te = feat_fn(X_test)
    thr, direction, _ = _fit_threshold_on_train(f_tr, yn_train)
    test_pred = ((direction * f_te) > (direction * thr)).astype(int)
    test_acc = float((test_pred == yn_test).mean())

    return test_acc, feat_name, extras


def run_task_diversity_battery(n_seeds=3, out_dir='.'):
    """
    v15 TASK DIVERSITY BATTERY.

    Question: does the cross-backbone direction-matching (dep→timelike,
    indep→spacelike) generalize from AR(1) to other causally structured
    vs trivial pairs?

    Procedure: for each task in {ar1, markov2, granger, intervention}:
      For each backbone in {lorentzian + MqMLPHead, euclidean + MqMLPHead}:
        For each seed in [SEED, SEED+1, SEED+2]:
          - generate train+test data
          - train 15 epochs
          - record sign-split direction, acc, mq-AUC, sample sign accuracy
      Print per-task summary.

    Total: 4 tasks × 2 backbones × n_seeds trainings.
    Headline: direction-match rate across (task × backbone × seed) where
              the model successfully sign-splits.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [SEED + i for i in range(n_seeds)]

    print('\n' + '#' * 72)
    print(f'# v15 TASK DIVERSITY BATTERY')
    print(f'# Seeds: {seeds}')
    print(f'# Tasks: ar1, markov2, granger, intervention')
    print(f'# Backbones: lorentzian+MqMLPHead, euclidean+MqMLPHead')
    print(f'# Total trainings: 4 × 2 × {n_seeds} = {4 * 2 * n_seeds}')
    print('#' * 72)

    tasks = ['ar1', 'markov2', 'granger', 'intervention']
    backbones = [
        ('lorentzian', 'Lorentzian + MqMLPHead', 0.0),
        ('euclidean',  'Euclidean + MqMLPHead',  1.0),
    ]

    # Results: all_results[task][backbone_name][seed] = metrics dict
    all_results = {t: {b[1]: {} for b in backbones} for t in tasks}
    baselines   = {}

    for task in tasks:
        print('\n\n' + '═' * 72)
        print(f'# TASK: {task}')
        print('═' * 72)

        # Generate data once per task (different seed for train/test)
        X_train, y_train, feat_dim = get_task_data(
            task, n_per_class=N_TRAIN // 2, seq_len=SEQ_LEN, seed=SEED)
        X_test, y_test, _ = get_task_data(
            task, n_per_class=N_TEST // 2, seq_len=SEQ_LEN, seed=SEED + 1000)

        verify_task_data(task, X_train, y_train)

        # Per-task explicit single-feature baseline (fit on train, eval on test)
        bl_acc, bl_name, bl_extras = lag1_baseline_acc(
            task, X_train, y_train, X_test, y_test)
        baselines[task] = {'acc': bl_acc, 'feature': bl_name, **bl_extras}
        print(f'  Single-feature baseline ({bl_name}): test acc = {bl_acc * 100:.1f}%')
        if 'lag2_acc' in bl_extras:
            print(f'    (also: lag-2 single-feature baseline = '
                  f'{bl_extras["lag2_acc"] * 100:.1f}%)')
        print(f'  Feature dim = {feat_dim}')

        for backbone_type, backbone_label, spacelike_residual in backbones:
            print(f'\n  ── Backbone: {backbone_label} ──')
            for s in seeds:
                print(f'\n  --- {task} | {backbone_type} | Seed {s} ---')
                torch.manual_seed(s); np.random.seed(s)
                g = torch.Generator(); g.manual_seed(s)
                train_loader = DataLoader(
                    TensorDataset(X_train, y_train),
                    batch_size=BATCH_SIZE, shuffle=True, generator=g,
                )
                test_loader = DataLoader(
                    TensorDataset(X_test, y_test),
                    batch_size=BATCH_SIZE, shuffle=False,
                )
                model = _build_model(backbone_type, 'mq-mlp',
                                     spacelike_residual=spacelike_residual,
                                     in_dim=feat_dim)
                print(f'    params: {sum(p.numel() for p in model.parameters()):,}')
                hist = train_one(
                    model, train_loader, test_loader,
                    name=f'{task[:4]}-{backbone_type[:4]}-s{s}')

                if hist['nan_seen'] and any(hist['nan_seen']):
                    print(f'    [seed {s} hit NaN — skipping metrics]')
                    all_results[task][backbone_label][s] = {
                        'nan_seen': True,
                    }
                    continue

                # Final-epoch metrics
                final = {
                    'test_acc':         hist['test_acc'][-1] * 100,
                    'mq_auc':           hist['mq_class_auc'][-1],
                    'mq_c0_mean':       hist['mq_mean_class0'][-1],
                    'mq_c1_mean':       hist['mq_mean_class1'][-1],
                    'mq_gap':           hist['mq_gap'][-1],
                    'mq_abs_mean':      hist['mq_abs_mean'][-1],
                    'mq_abs_p10':       hist['mq_abs_p10'][-1],
                    'mq_abs_median':    hist['mq_abs_median'][-1],
                    'mq_abs_p90':       hist['mq_abs_p90'][-1],
                    'sample_sign_acc':  hist['sample_sign_acc'][-1],
                }
                # Direction tag: matches Theorem 5 if dep→timelike (mq[c0]<0)
                #                                  AND indep→spacelike (mq[c1]>0)
                final['direction_match'] = bool(
                    final['mq_c0_mean'] < 0 < final['mq_c1_mean'])
                final['direction_inverted'] = bool(
                    final['mq_c1_mean'] < 0 < final['mq_c0_mean'])
                final['strong_sign_split'] = bool(
                    final['sample_sign_acc'] > 0.9
                    and final['direction_match'])

                all_results[task][backbone_label][s] = final

                tag = ('✓ MATCH' if final['direction_match']
                       else '✗ INVERTED' if final['direction_inverted']
                       else 'same-side')
                print(f'    final: acc={final["test_acc"]:.2f}%  '
                      f'mq_AUC={final["mq_auc"]:.3f}  '
                      f'mq[c0]={final["mq_c0_mean"]:+6.2f}  '
                      f'mq[c1]={final["mq_c1_mean"]:+6.2f}  '
                      f'sgnAcc={final["sample_sign_acc"]:.3f}  '
                      f'|mq|≈{final["mq_abs_mean"]:.1f}  {tag}')

    # ─────────────────────────────────────────────────────────────────
    # CROSS-TASK SUMMARY
    # ─────────────────────────────────────────────────────────────────
    print('\n\n' + '#' * 72)
    print('# CROSS-TASK SUMMARY')
    print('#' * 72)

    print(f'\n{"Task":<14s} {"Baseline":<22s} {"Backbone":<25s} '
          f'{"acc":>6s} {"mq_AUC":>7s} {"sgnAcc":>7s} '
          f'{"|mq|":>6s} {"match":>7s}')
    print('-' * 100)
    # ─────────────────────────────────────────────────────────────────
    # Per-task summary table — pre-compute strings for portability.
    # ─────────────────────────────────────────────────────────────────
    # Tasks split into two interpretive categories:
    #   - 'indep_vs_dep'  : class 1 is causally trivial (i.i.d. or do(x)).
    #                       direction-match = "dep→timelike, indep→spacelike"
    #                       This is the literal Realizability prediction.
    #   - 'depth'         : both classes have temporal dependence; only
    #                       the DEPTH of dependence differs. The match
    #                       rate measures "deeper-dep → timelike than
    #                       shallower-dep" — a *related* but distinct
    #                       claim from independent → spacelike.
    task_category = {
        'ar1':          'indep_vs_dep',
        'markov2':      'depth',
        'granger':      'indep_vs_dep',
        'intervention': 'indep_vs_dep',
    }

    n_total_idv = n_match_idv = n_invert_idv = n_strong_idv = 0
    n_total_dep = n_match_dep = n_invert_dep = n_strong_dep = 0
    n_total = n_match = n_invert = n_strong = 0

    for task in tasks:
        bl = baselines[task]
        bl_text = f'{bl["feature"][:18]} {bl["acc"] * 100:.0f}%'
        for backbone_type, backbone_label, _ in backbones:
            seed_res = [all_results[task][backbone_label][s] for s in seeds]
            valid = [r for r in seed_res if not r.get('nan_seen', False)]
            if not valid:
                continue
            mean_acc    = np.mean([r['test_acc']        for r in valid])
            mean_auc    = np.mean([r['mq_auc']          for r in valid])
            mean_sgnacc = np.mean([r['sample_sign_acc'] for r in valid])
            mean_mq     = np.mean([r['mq_abs_mean']     for r in valid])
            n_t = len(valid)
            n_m = sum(1 for r in valid if r['direction_match'])
            n_i = sum(1 for r in valid if r['direction_inverted'])
            n_s = sum(1 for r in valid if r['strong_sign_split'])
            n_total  += n_t; n_match  += n_m
            n_invert += n_i; n_strong += n_s
            if task_category[task] == 'indep_vs_dep':
                n_total_idv += n_t; n_match_idv  += n_m
                n_invert_idv += n_i; n_strong_idv += n_s
            else:
                n_total_dep += n_t; n_match_dep  += n_m
                n_invert_dep += n_i; n_strong_dep += n_s

            match_str = f'{n_m}/{n_t}'
            print(f'{task:<14s} '
                  f'{bl_text:<22s} '
                  f'{backbone_label:<25s} '
                  f'{mean_acc:>5.1f}% '
                  f'{mean_auc:>7.3f} '
                  f'{mean_sgnacc:>7.3f} '
                  f'{mean_mq:>6.1f} '
                  f'{match_str:>7s}')

    print('\n─── HEADLINE ───')
    print(f'  Total runs (task × backbone × seed): {n_total}')
    print()
    print(f'  Independent-vs-dependent tasks (ar1, granger, intervention)')
    print(f'    Direction-match (dep→timelike, indep→spacelike): '
          f'{n_match_idv}/{n_total_idv}')
    print(f'    Direction-inverted: {n_invert_idv}/{n_total_idv}')
    print(f'    Strong sign-split (sgnAcc>0.9 AND match):  '
          f'{n_strong_idv}/{n_total_idv}')
    print()
    print(f'  Depth-of-dependence task (markov2)')
    print(f'    Direction-match (deeper-dep→timelike, shallower-dep→spacelike): '
          f'{n_match_dep}/{n_total_dep}')
    print(f'    Direction-inverted: {n_invert_dep}/{n_total_dep}')
    print(f'    Strong sign-split (sgnAcc>0.9 AND match):  '
          f'{n_strong_dep}/{n_total_dep}')
    print()
    print(f'  POOLED (all tasks)')
    print(f'    Direction-match: {n_match}/{n_total}')
    print(f'    Direction-inverted: {n_invert}/{n_total}')
    print(f'    Strong sign-split: {n_strong}/{n_total}')
    print()
    if n_invert == 0 and n_match == n_total:
        print('  ✓ UNIVERSAL (no inversions): every successful run matches')
        print('    the predicted direction.')
        print('    For indep_vs_dep tasks this directly supports the Realizability')
        print('    "causally connected → timelike" prediction. For markov2 the')
        print('    same direction holds when "more dependent" plays the role of')
        print('    "more causally connected".')
    elif n_invert == 0 and n_match >= int(0.8 * n_total):
        print('  ✓ STRONG: no inversions; most runs match the predicted')
        print('    direction. Same-side runs reflect tasks where the model')
        print('    failed to sign-split at all, not contradictions.')
    elif n_invert > 0:
        print(f'  ⚠ MIXED: {n_invert} inverted run(s). Inspect which (task,')
        print('    backbone, seed) combinations inverted before claiming universality.')

    # Save
    save_path = out_dir / 'v15_task_diversity_results.json'
    with open(save_path, 'w') as f:
        json.dump({
            'config': {
                'n_seeds': n_seeds,
                'seeds': seeds,
                'tasks': tasks,
                'task_categories': task_category,
                'backbones': [b[1] for b in backbones],
                'SEQ_LEN': SEQ_LEN,
                'EPOCHS': EPOCHS,
            },
            'baselines': baselines,
            'results': all_results,
            'headline': {
                'pooled': {
                    'n_total': n_total, 'n_match': n_match,
                    'n_invert': n_invert, 'n_strong': n_strong,
                },
                'indep_vs_dep': {
                    'n_total': n_total_idv, 'n_match': n_match_idv,
                    'n_invert': n_invert_idv, 'n_strong': n_strong_idv,
                },
                'depth': {
                    'n_total': n_total_dep, 'n_match': n_match_dep,
                    'n_invert': n_invert_dep, 'n_strong': n_strong_dep,
                },
            },
        }, f, indent=2, default=_json_safe)
    print(f'\nSaved raw results: {save_path}')

    return all_results


if __name__ == '__main__':
    # v15 default: 4-task diversity battery
    run_task_diversity_battery(n_seeds=3, out_dir='.')