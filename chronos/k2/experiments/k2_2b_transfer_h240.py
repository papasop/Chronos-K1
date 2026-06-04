"""
EXPERIMENT K2.2-B: SYMPLECTIC TRANSFER at H=240 — TWO-PHASE (regime-gated) — COLAB
K2.2-A returned FULL_TRANSFER_CONFIRMED at H=200: on the graceful-baseline subset (22/30) the
symplectic prior beat baseline + fair energy + fair L2, and the full ‖JᵀΩJ−Ω‖ mechanism still
reduced 70.7% (vs 71.9% at H=160 — essentially no decay). This is STRONGER than K1, whose
spectral mechanism decayed to 11.9% by H=240 (TRANSFER_PERFORMANCE_ONLY). symplectic is the
first VPSL structure that is validated AND transfers as mechanism, plausibly because it is a
geometric STRUCTURE penalty (constrains the map's Jacobian every step) rather than a
single-step spectral supervision that error-accumulation dilutes.

This experiment probes the UPPER BOUND of that transfer: does the mechanism still hold at H=240?
But H=240 was NOT regime-validated by K2.0-B (which only scanned to H=200), so we must FIRST
check the baseline regime is still graceful — otherwise we risk the K1 @500 trap (baseline
diverges so much that the graceful subset is too small and any "transfer" is pure rescue).

TWO-PHASE (regime gate before prior):
  PHASE 1 (baseline-only): train N_SEEDS baselines at S=1, eval at H=240. Compute hard-div frac
    and graceful-subset size. GATE: proceed iff graceful >= GRACEFUL_MIN (15) AND hard_div <= 0.5.
    If the gate fails → STOP, report "K2 validated transfer ceiling = H=200", train NO priors.
  PHASE 2 (only if gate passes): train symplectic_l0.1 + fair energy_l0.01 + fair l2_l0.001,
    run the full K2.2-A three-stratum stratified transfer analysis at H=240.

This avoids wasting compute on priors in an invalid regime AND avoids the dishonest move of
stratifying on a graceful subset too small to support paired tests.

VERDICT:
  Phase-1 fail → REGIME_INVALID_H240_CEILING_H200 (K2's validated transfer ceiling is H=200).
  Phase-2 → same classes as K2.2-A: FULL_TRANSFER_CONFIRMED / TRANSFER_CONFIRMED /
            TRANSFER_PERFORMANCE_ONLY / CONTROL_MATCHES_SYMP / NO_TRANSFER (at H=240).

FIXES vs the prior version (audit before paper):
  (1) energy_penalty: replaced in-place zeroing of the kinetic term (ke[:, :M]=0) — which is an
      in-place op on a non-leaf graph tensor and corrupts/raises in autograd — with a mask MULTIPLY
      (identical forward value, autograd-safe). This is the same masking style already used in
      _interior_mask_vec. Affects the energy CONTROL's training gradient only.
  (2) removed a stale "Exp K2.0 ..." intro print left over from copy (mislabeled the log).
  (3) safe_w: warn when the two arrays are exactly equal (returns p=1.0) — distinguishes a true tie
      from a possible data wiring bug instead of silently returning non-significant.
  (4) moved _interior_mask_vec above its first use (style; no behavior change).
"""
import os, json, platform, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# ============================================================================
# RUN MODE
# ============================================================================
RUN_MODE      = "FULL"   # "FULL", "SCOUT" (5-seed sanity), or "SMOKE"
DETERMINISTIC = True
OVERRIDE_SEEDS  = None
OVERRIDE_EPOCHS = None
assert RUN_MODE in ("FULL","SCOUT","SMOKE"), f"RUN_MODE must be FULL/SCOUT/SMOKE, got {RUN_MODE!r}."

RESULTS_DIR = Path("/content/exp_k2_2b_fpu"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EPS = 1e-8

SMOKE = (RUN_MODE.upper()=="SMOKE")
SCOUT = (RUN_MODE.upper()=="SCOUT")
MODE_LABEL = "SMOKE" if SMOKE else ("SCOUT" if SCOUT else "FULL")

# ============================================================================
# FPU-β PHYSICS
# ============================================================================
N_MASS   = 32
BETA     = 4.0          # nonlinearity (sandbox: β=4, amp≈3 → mode-1 energy fraction 1.0→0.41,
                        # genuine mixing, at MODEST energy E0≈0.76 and bounded |q| — the
                        # non-near-floor, non-divergent sweet spot; broadband/high-E overloaded)
DT_PHYS  = 0.02         # fine physics timestep (Verlet truth conserves energy ~1e-4 here)
IC_AMP   = 3.0          # mode-1-dominant amplitude → strongly-nonlinear FPU regime
IC_PERTURB = 0.1        # small multi-mode perturbation (seed variety without changing the regime)

S_FIXED  = 1            # validated regime
HORIZON  = 240          # UPPER-BOUND probe horizon (NOT regime-validated by K2.0-B → phase-1 gate)
REF_HORIZON = 40
GRAD_CLIP = 1.0
# PHASE-1 regime gate (baseline-only @ H=240, before any prior)
GRACEFUL_MIN=15         # need >=15 graceful-baseline seeds for stable stratified paired tests
DIV_FRAC_CEIL=0.5       # baseline hard-div frac must be <=0.5 (else regime invalid, like K1 @500)

# prior config — only primary symplectic + the FAIR controls selected by K2.1-B
PRIMARY_LAMBDA=0.1
SYMP_LAMBDAS=[0.1]
ENERGY_LAMBDAS=[0.01]                # fair energy from K2.1-B (re-checked by the gate at this horizon)
L2_LAMBDAS=[0.001]                   # fair L2 from K2.1-B (re-checked by the gate at this horizon)
N_SYMP_PROBES=4
MECH_REL_THRESHOLD=0.20
# NON-DEGENERATE GATE thresholds (re-applied at this horizon; controls fair at H=160 may degrade)
INCREMENT_FAC=0.3
SYMP_FLOOR_FAC=0.3
MAXQ_LO_FAC=0.3; MAXQ_HI_FAC=5.0
RATIO_CEIL=5.0
WORSE_FAC=3.0
N_JAC_STATES=6                      # # eval states per rollout where FULL Jacobian is computed
INTERIOR_MARGIN=2                   # interior-only mask: drop this many sites at each Dirichlet end

# model / training
HIDDEN=64; N_LAYERS=3; RADIUS=2; LR=3e-4; EPOCHS=150; N_SEEDS=30
N_TRAIN_TRAJ=40; TRAIN_STEPS=80

# classifier / gate thresholds (same family as K1)
DIV_CEIL_FAC=50.0; DIV_ABS=100.0; STABLE_FAC=2.0; DIV_FRAC=0.5; STABLE_FRAC=0.1
FUNC_DIV_THR=10.0

if SMOKE:
    N_SEEDS, EPOCHS, N_TRAIN_TRAJ = 3, 12, 6
    TRAIN_STEPS=30; HORIZON=60; N_SYMP_PROBES=2; N_JAC_STATES=2
    print("⚠️  SMOKE MODE — tiny config, results NOT valid.\n")
elif SCOUT:
    N_SEEDS = 5
    print("⚠️  SCOUT MODE — 5 seeds; directional sanity only, NOT the final N=30.\n")
if OVERRIDE_SEEDS  is not None: N_SEEDS=OVERRIDE_SEEDS
if OVERRIDE_EPOCHS is not None: EPOCHS=OVERRIDE_EPOCHS

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)
DETERMINISTIC_ALGOS_OK=None
if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True); DETERMINISTIC_ALGOS_OK=True
    except Exception as e:
        DETERMINISTIC_ALGOS_OK=False; print(f"  (use_deterministic_algorithms unavailable: {e})")
    print(f"✓ cudnn deterministic ENABLED (use_deterministic_algorithms={DETERMINISTIC_ALGOS_OK})")
if DEVICE=="cpu": print("⚠️  DEVICE=cpu — FULL slow; enable GPU.\n")

print("Exp K2.2-B: Symplectic Transfer at H=240 — TWO-PHASE (regime-gated)")
print(f"  mode={MODE_LABEL}, N_SEEDS={N_SEEDS}, EPOCHS={EPOCHS}, DEVICE={DEVICE}")
print(f"  N={N_MASS}, beta={BETA}, dt_phys={DT_PHYS}, ic_amp={IC_AMP}")
print(f"  regime: FPU-β S={S_FIXED} H={HORIZON} (probe upper bound; NOT pre-validated)")
print(f"  PHASE 1 baseline-only gate: proceed iff graceful>={GRACEFUL_MIN} AND hard_div<={DIV_FRAC_CEIL}")
print(f"  PHASE 2 (if gate passes): symplectic vs FAIR energy_l0.01 vs FAIR l2_l0.001, stratified\n")

_idx = np.arange(1, N_MASS+1)
_modes = np.sin(np.outer(np.arange(1,N_MASS+1), _idx)*np.pi/(N_MASS+1))  # [k,i]
_omega = 2*np.sin(np.arange(1,N_MASS+1)*np.pi/(2*(N_MASS+1)))            # linear mode freqs

def fpu_accel(q):
    """acceleration a_i = -dH/dq_i for FPU-β, fixed ends. q shape [...,N]."""
    z = np.zeros_like(q[...,:1])
    qext = np.concatenate([z, q, z], axis=-1)          # [...,N+2]
    d = np.diff(qext, axis=-1)                          # bonds [...,N+1]
    f = d + BETA*d**3
    return f[...,1:] - f[...,:-1]                        # [...,N]

def fpu_energy(q, p):
    z = np.zeros_like(q[...,:1]); qext=np.concatenate([z,q,z],axis=-1)
    d = np.diff(qext, axis=-1)
    V = 0.5*d**2 + 0.25*BETA*d**4
    return 0.5*np.sum(p**2,axis=-1) + np.sum(V,axis=-1)

def verlet_step(q, p, dt):
    a = fpu_accel(q)
    p = p + 0.5*dt*a
    q = q + dt*p
    a = fpu_accel(q)
    p = p + 0.5*dt*a
    return q, p

def fpu_simulate(q0, p0, n_model_steps, S):
    """Roll out the TRUTH for n_model_steps model steps; each model step = S Verlet substeps
    at DT_PHYS. Returns state trajectory [n_model_steps+1, 2, N] (channels q,p)."""
    q=q0.copy(); p=p0.copy(); traj=[np.stack([q,p])]
    for _ in range(n_model_steps):
        for _ in range(S):
            q,p = verlet_step(q,p,DT_PHYS)
        traj.append(np.stack([q,p]))
    return np.array(traj)

def random_fpu_ic(rng):
    """Mode-1-dominant, modest-energy IC → strongly nonlinear (mode-mixing) FPU regime, with a
    small multi-mode perturbation for seed variety. Sandbox: energy spreads out of mode 1
    (frac 1.0→~0.4) at bounded |q| — non-near-floor and non-divergent."""
    x=np.arange(1,N_MASS+1)
    q = IC_AMP*np.sin(np.pi*x/(N_MASS+1)) + rng.standard_normal(N_MASS)*IC_AMP*IC_PERTURB
    p = rng.standard_normal(N_MASS)*IC_AMP*IC_PERTURB
    return q, p

def mode_energy(q, p):
    """linear normal-mode energies, shape [...,N]."""
    Q = np.sqrt(2/(N_MASS+1))*(q @ _modes.T)
    P = np.sqrt(2/(N_MASS+1))*(p @ _modes.T)
    return 0.5*(P**2 + (_omega**2)*Q**2)

# ============================================================================
# NORMALIZATION (fit on a sample of fine-dt trajectories)
# ============================================================================
def _fit_norm():
    rng=np.random.default_rng(999); qs=[]; ps=[]
    for _ in range(20):
        q0,p0=random_fpu_ic(rng); tr=fpu_simulate(q0,p0,TRAIN_STEPS,1)
        qs.append(tr[:,0,:]); ps.append(tr[:,1,:])
    qs=np.concatenate(qs); ps=np.concatenate(ps)
    mean=np.array([qs.mean(), ps.mean()], dtype=np.float32)
    std =np.array([qs.std()+EPS, ps.std()+EPS], dtype=np.float32)
    return mean.reshape(1,2,1), std.reshape(1,2,1)
DATA_MEAN, DATA_STD = _fit_norm()
def normalize(a):   return (a-DATA_MEAN)/DATA_STD
def denormalize(a): return a*DATA_STD+DATA_MEAN
print(f"  norm: mean={DATA_MEAN.ravel()}, std={DATA_STD.ravel()}\n")

# ============================================================================
# MODEL — 1D CNN rollout (Dirichlet → zero padding)
# ============================================================================
class FPUConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch=[2]+[HIDDEN]*(N_LAYERS-1)+[2]
        self.convs=nn.ModuleList([nn.Conv1d(ch[i],ch[i+1],2*RADIUS+1,padding=RADIUS)
                                  for i in range(N_LAYERS)])
        self.act=nn.GELU()
    def forward(self,u):                       # u: [B,2,N]
        x=u
        for i,c in enumerate(self.convs):
            x=c(x)
            if i<len(self.convs)-1: x=self.act(x)
        return u+x                              # residual: predict the increment
    def rollout(self,u0,n):
        outs=[u0]; u=u0
        for _ in range(n): u=self.forward(u); outs.append(u)
        return torch.stack(outs,dim=1)          # [B, n+1, 2, N]

def K_for_epoch(ep): return 1 if ep<EPOCHS*0.2 else (3 if ep<EPOCHS*0.55 else 5)

# ============================================================================
# SYMPLECTIC / ENERGY MACHINERY
# ============================================================================
# Phase space dim 2N; Ω = [[0,I],[-I,0]]. State tensor is [B,2,N] (ch0=q, ch1=p).
# Flatten to z=[q;p] ∈ R^{2N} for Jacobian / Ω operations.
_OMEGA=None
def _omega_mat():
    global _OMEGA
    if _OMEGA is None:
        I=torch.eye(N_MASS,device=DEVICE); Z=torch.zeros(N_MASS,N_MASS,device=DEVICE)
        _OMEGA=torch.cat([torch.cat([Z,I],1),torch.cat([-I,Z],1)],0)  # [2N,2N]
    return _OMEGA

def _state_to_z(u):        # u:[B,2,N] -> z:[B,2N]  (denormalized physical coords)
    ud=u*torch.tensor(DATA_STD,device=u.device)+torch.tensor(DATA_MEAN,device=u.device)
    return ud.reshape(u.shape[0],2*N_MASS)
def _z_to_state(z):        # inverse
    ud=z.reshape(z.shape[0],2,N_MASS)
    return (ud-torch.tensor(DATA_MEAN,device=z.device))/torch.tensor(DATA_STD,device=z.device)

def _interior_mask_vec(r):
    """zero out the boundary sites (first/last INTERIOR_MARGIN) in both q and p blocks of z.
    Implemented as a MASK MULTIPLY (autograd-safe; no in-place on a graph tensor)."""
    m=torch.ones(2*N_MASS,device=r.device)
    for blk in (0, N_MASS):
        m[blk:blk+INTERIOR_MARGIN]=0.0
        m[blk+N_MASS-INTERIOR_MARGIN:blk+N_MASS]=0.0
    return r*m

def symplectic_penalty(model, u, n_probes=N_SYMP_PROBES):
    """CHEAP random-projection surrogate for ‖JᵀΩJ−Ω‖²: E_v[‖(JᵀΩJ−Ω)v‖²], v~N(0,I).
    J is the model single-step Jacobian in PHYSICAL (q,p) coords. We need both J·v (jvp) and
    Jᵀ·w (vjp). vjp is one autograd.grad. jvp is built via the double-vjp identity:
        let g(w) = (∂out/∂z)ᵀ w = Jᵀ w   (a function of w);  then  ∂_w [g(w)·s] applied... →
        jvp = ∂_w (Jᵀ w) acting on direction, i.e. grad of (Jᵀw · const) wrt w gives J·(const).
    Concretely: Jv = grad_w( vjp(w) )·v  with w a dummy requiring grad.
    Differentiable wrt model params (create_graph=True throughout)."""
    B=u.shape[0]; Om=_omega_mat()
    def f_phys(z): return _state_to_z(model(_z_to_state(z)))   # one model step in physical coords
    z0=_state_to_z(u).detach().requires_grad_(True)
    out=f_phys(z0)                                             # [B,2N]
    pen=0.0
    for _ in range(n_probes):
        v=torch.randn(B,2*N_MASS,device=u.device)
        # jvp: J·v via double-vjp (forward-over-reverse)
        dummy=torch.zeros_like(out,requires_grad=True)
        vjp_dummy=torch.autograd.grad(out, z0, grad_outputs=dummy, create_graph=True, retain_graph=True)[0]  # Jᵀ·dummy
        Jv=torch.autograd.grad(vjp_dummy, dummy, grad_outputs=v, create_graph=True, retain_graph=True)[0]     # J·v
        OJv=Jv@Om.T                                           # Ω (J v)
        JT_OJv=torch.autograd.grad(out, z0, grad_outputs=OJv, create_graph=True, retain_graph=True)[0]        # Jᵀ Ω J v
        r=JT_OJv-(v@Om.T)                                     # (JᵀΩJ−Ω) v
        r=_interior_mask_vec(r)                               # interior-only (drop Dirichlet boundary sites)
        pen=pen+(r**2).sum(dim=1).mean()
    return pen/n_probes

def energy_penalty(model, u):
    """one-step energy drift |H(next)−H(cur)| on interior (physical coords). Differentiable.
    FIX: interior masking uses a MASK MULTIPLY rather than in-place zeroing of the kinetic term;
    in-place ops on a non-leaf graph tensor corrupt/raise in autograd. Forward value identical."""
    z=_state_to_z(u); q=z[:,:N_MASS]; p=z[:,N_MASS:]
    nxt=_state_to_z(model(u)); qn=nxt[:,:N_MASS]; pn=nxt[:,N_MASS:]
    # interior site-mask (kinetic) — autograd-safe multiply
    site_mask=torch.ones(N_MASS,device=u.device)
    site_mask[:INTERIOR_MARGIN]=0.0; site_mask[N_MASS-INTERIOR_MARGIN:]=0.0
    def H(q,p):
        z0=torch.zeros(q.shape[0],1,device=q.device)
        qe=torch.cat([z0,q,z0],1); d=qe[:,1:]-qe[:,:-1]
        V=0.5*d**2+0.25*BETA*d**4
        ke=0.5*p**2*site_mask                       # mask multiply (NOT in-place)
        return ke.sum(1)+V.sum(1)
    return (H(qn,pn)-H(q,p)).abs().mean()

# ============================================================================
# TRAIN (baseline + symplectic/energy/l2 priors)
# ============================================================================
def build_training_trajectories(seed, S):
    rng=np.random.default_rng(10000+seed)
    return [normalize(fpu_simulate(*random_fpu_ic(rng), TRAIN_STEPS, S)) for _ in range(N_TRAIN_TRAJ)]

def train_model(trajs, seed, prior=None, lam=0.0):
    """prior ∈ {None,'symplectic','energy','l2'}; loss prior on the SAME architecture."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model=FPUConvNet().to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR); mse=nn.MSELoss()
    T=trajs[0].shape[0]; data=torch.tensor(np.stack(trajs),dtype=torch.float32,device=DEVICE)
    n_traj=data.shape[0]; rng_local=np.random.default_rng(50000+seed)
    last_data=[]; last_prior=[]
    for ep in range(EPOCHS):
        K=K_for_epoch(ep); max_start=T-1-K
        if max_start<0: K=T-1; max_start=0
        model.train(); perm=torch.randperm(n_traj); record=(ep==EPOCHS-1)
        for bi in range(n_traj):
            ti=perm[bi].item(); start=int(rng_local.integers(0,max_start+1))
            cur=data[ti,start].unsqueeze(0); opt.zero_grad(); loss=0.0
            for kk in range(1,K+1):
                cur=model(cur); loss=loss+mse(cur,data[ti,start+kk].unsqueeze(0))
            loss=loss/K; data_term=float(loss.detach().cpu()); prior_term=0.0
            if prior and lam>0.0:
                u_anchor=data[ti,start].unsqueeze(0)
                if prior=="symplectic": pen=symplectic_penalty(model,u_anchor)
                elif prior=="energy":   pen=energy_penalty(model,u_anchor)
                elif prior=="l2":       pen=sum((w**2).sum() for w in model.parameters())
                else: pen=torch.tensor(0.0,device=DEVICE)
                loss=loss+lam*pen; prior_term=float((lam*pen).detach().cpu())
            loss.backward()
            if GRAD_CLIP is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if record: last_data.append(data_term); last_prior.append(prior_term)
    diag={'data_loss':float(np.mean(last_data)) if last_data else np.nan,
          'prior_loss':float(np.mean(last_prior)) if last_prior else 0.0}
    diag['prior_data_ratio']=diag['prior_loss']/(diag['data_loss']+EPS)
    return model, diag

def full_symplectic_error(model, u_batch):
    """EVAL diagnostic: full ‖JᵀΩJ−Ω‖_F and |det(J)|−1, averaged over a small state batch.
    Computes the full 2N×2N Jacobian per state via autograd (expensive → few states)."""
    Om=_omega_mat().detach().cpu().numpy(); errs=[]; vols=[]
    for i in range(u_batch.shape[0]):
        z0=_state_to_z(u_batch[i:i+1]).detach().requires_grad_(True)
        out=_state_to_z(model(_z_to_state(z0)))            # [1,2N]
        J=torch.zeros(2*N_MASS,2*N_MASS,device=z0.device)
        for k in range(2*N_MASS):
            e=torch.zeros(out.shape,device=out.device); e[0,k]=1.0
            row=torch.autograd.grad(out, z0, grad_outputs=e, retain_graph=True)[0]  # ∂out_k/∂z = row k of J
            J[k,:]=row.squeeze(0)
        Jn=J.detach().cpu().numpy()
        errs.append(np.linalg.norm(Jn.T@Om@Jn-Om,'fro'))
        vols.append(abs(abs(np.linalg.det(Jn))-1.0))
    return float(np.mean(errs)), float(np.mean(vols))

# ============================================================================
# EVALUATE one (seed, S) at HORIZON model steps
# ============================================================================
def evaluate(model, seed, S, horizon):
    model.eval()
    rng=np.random.default_rng(30000+seed+777777)   # eval IC stream, H-independent
    q0,p0=random_fpu_ic(rng)
    true=fpu_simulate(q0,p0,horizon,S)              # [H+1,2,N]
    E0=fpu_energy(true[0,0],true[0,1])
    init_max_q=float(np.max(np.abs(true[0,0])))     # initial |q| scale (for dimensionless bound)
    u0=torch.tensor(normalize(true[0:1]),dtype=torch.float32,device=DEVICE)
    # increment_norm: ‖f(u)-u‖ averaged over the first few true states (identity-collapse detector,
    # in PHYSICAL coords). A near-identity (collapsed) model has increment≈0.
    with torch.no_grad():
        n_inc=min(N_JAC_STATES, len(true))
        u_inc=torch.tensor(normalize(true[:n_inc]),dtype=torch.float32,device=DEVICE)
        f_inc=model(u_inc)
        du=denormalize(f_inc.cpu().numpy())-true[:n_inc]
        increment_norm=float(np.mean(np.sqrt(np.sum(du**2,axis=(1,2)))))
    with torch.no_grad():
        pred=model.rollout(u0,horizon).squeeze(0).cpu().numpy()  # [H+1,2,N]
    pred=denormalize(pred)
    if not (np.all(np.isfinite(pred)) and np.max(np.abs(pred[:,0,:]))<1e4):
        return {'roll_mse':1e4,'ham_drift':np.nan,'max_q':np.inf,'init_max_q':init_max_q,
                'mode_spread':np.nan,'symp_err':np.nan,'vol_err':np.nan,'increment_norm':increment_norm,
                'hard_diverged':True,'functional_diverged':True}
    T=min(len(pred),len(true))
    roll_mse=float(np.mean((pred[:T,0,:]-true[:T,0,:])**2))
    # Hamiltonian drift of the MODEL trajectory (relative to its own initial H)
    Hpred=fpu_energy(pred[:T,0,:],pred[:T,1,:])
    ham_drift=float(np.median(np.abs(Hpred-E0)/(abs(E0)+EPS)))
    max_q=float(np.max(np.abs(pred[:T,0,:])))
    # mode-energy spread: L1 distance between predicted & true normalized mode-energy dist (final)
    emp=mode_energy(pred[T-1,0,:][None],pred[T-1,1,:][None])[0]
    emt=mode_energy(true[T-1,0,:][None],true[T-1,1,:][None])[0]
    emp=emp/(emp.sum()+EPS); emt=emt/(emt.sum()+EPS)
    mode_spread=float(np.sum(np.abs(emp-emt)))
    # functional divergence (sustained tail), like K1
    per_step=np.mean((pred[:T,0,:]-true[:T,0,:])**2,axis=1)
    over=per_step>FUNC_DIV_THR
    sustained=bool(over[-1] and over[max(0,T-10):].mean()>0.8)
    # MECHANISM: full symplectic Jacobian error on a few states along the rollout (eval only)
    symp_err, vol_err = np.nan, np.nan
    if np.isfinite(pred).all():
        idxs=np.linspace(0, T-1, N_JAC_STATES).astype(int)
        states=torch.tensor(normalize(pred[idxs]),dtype=torch.float32,device=DEVICE)
        try:
            symp_err, vol_err = full_symplectic_error(model, states)
        except Exception:
            symp_err, vol_err = np.nan, np.nan
    return {'roll_mse':roll_mse,'ham_drift':ham_drift,'max_q':max_q,'init_max_q':init_max_q,
            'mode_spread':mode_spread,'symp_err':symp_err,'vol_err':vol_err,'increment_norm':increment_norm,
            'hard_diverged':bool(sustained),'functional_diverged':bool(over.any())}

# ============================================================================
# PHASE 1 — baseline-only regime gate at H=240 (before any prior)
# ============================================================================
def run_phase1():
    import time
    print(f"  PHASE 1: {N_SEEDS} baseline-only seeds @ H={HORIZON} (regime gate)\n")
    rows=[]
    for seed in range(N_SEEDS):
        trajs=build_training_trajectories(seed, S_FIXED)
        t0=time.time(); model,diag=train_model(trajs,seed,prior=None,lam=0.0); tt=time.time()-t0
        m=evaluate(model,seed,S_FIXED,HORIZON)
        m.update({'seed':seed,'model':'baseline','prior':'none','lam':0.0,
                  'data_loss':diag['data_loss'],'prior_loss':diag['prior_loss'],
                  'prior_data_ratio':diag['prior_data_ratio']})
        rows.append(m)
        if seed<3 or seed==N_SEEDS-1:
            print(f"  seed {seed+1}/{N_SEEDS} ({tt:.0f}s): roll={m['roll_mse']:.3f} hard_div={m['hard_diverged']}")
    return pd.DataFrame(rows)

def phase1_gate(bdf):
    hard_div=bdf['hard_diverged'].mean(); n_graceful=int((~bdf['hard_diverged'].astype(bool)).sum())
    func_div=bdf['functional_diverged'].mean()
    regime_invalid = hard_div>DIV_FRAC_CEIL
    proceed = (n_graceful>=GRACEFUL_MIN) and (not regime_invalid)
    print("\n"+"="*80); print(f"PHASE-1 REGIME GATE @ H={HORIZON}"); print("="*80+"\n")
    print(f"  baseline hard_div frac       = {hard_div:.2f} (ceiling {DIV_FRAC_CEIL})")
    print(f"  baseline functional_div frac = {func_div:.2f} (REPORTED only — not a gate; at longer")
    print(f"                                  horizons functional failure can precede hard-div tail)")
    print(f"  graceful-baseline seeds      = {n_graceful}/{len(bdf)} (need >= {GRACEFUL_MIN})")
    print(f"  baseline roll_MSE median     = {bdf['roll_mse'].median():.4f}")
    if func_div>0.7:
        print(f"  ⚠️ functional_div {func_div:.2f} is high (>0.7): even if hard_div passes, the graceful")
        print(f"     subset may be functionally degraded — interpret Phase-2 mechanism cautiously.")
    print(f"\n  → {'PROCEED to PHASE 2 (prior transfer)' if proceed else 'STOP — regime not testable at H='+str(HORIZON)}")
    return proceed, hard_div, n_graceful, func_div

# ============================================================================
# RUN — paired: per seed, train baseline + each prior variant, eval at H=HORIZON
# ============================================================================
def run():
    import time
    tagf=lambda l:str(l).replace('.','p')
    variants={'baseline':(None,0.0)}
    for l in SYMP_LAMBDAS:   variants[f'symplectic_l{tagf(l)}']=("symplectic",l)
    for l in ENERGY_LAMBDAS: variants[f'energy_l{tagf(l)}']=("energy",l)
    for l in L2_LAMBDAS:     variants[f'l2_l{tagf(l)}']=("l2",l)
    print(f"  PHASE 2: {len(variants)} variants × {N_SEEDS} seeds (paired); eval @ H={HORIZON}")
    print(f"  symplectic{SYMP_LAMBDAS} + energy{ENERGY_LAMBDAS} + l2{L2_LAMBDAS}\n")
    rows=[]
    for seed in range(N_SEEDS):
        trajs=build_training_trajectories(seed, S_FIXED)
        for name,(prior,lam) in variants.items():
            t0=time.time(); model,diag=train_model(trajs,seed,prior=prior,lam=lam); tt=time.time()-t0
            m=evaluate(model,seed,S_FIXED,HORIZON)
            m.update({'seed':seed,'model':name,'prior':prior or 'none','lam':lam,
                      'data_loss':diag['data_loss'],'prior_loss':diag['prior_loss'],
                      'prior_data_ratio':diag['prior_data_ratio']})
            rows.append(m)
        b=[r for r in rows if r['seed']==seed and r['model']=='baseline'][0]
        s=[r for r in rows if r['seed']==seed and r['model']==f'symplectic_l{tagf(SYMP_LAMBDAS[0])}'][0]
        print(f"  seed {seed+1}/{N_SEEDS}: base roll={b['roll_mse']:.3f} | symp roll={s['roll_mse']:.3f} symp_err={s['symp_err']:.2f}")
    return pd.DataFrame(rows)

# ============================================================================
# ANALYZE — three-stratum transfer test (pooled / graceful / rescued) + gate@H_HORIZON
# ============================================================================
def analyze(df):
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    tagf=lambda l:str(l).replace('.','p')
    Bd=df[df['model']=='baseline'].sort_values('seed')
    Sy=df[df['model']==f'symplectic_l{tagf(SYMP_LAMBDAS[0])}'].sort_values('seed')
    En=df[df['model']==f'energy_l{tagf(ENERGY_LAMBDAS[0])}'].sort_values('seed')
    L2=df[df['model']==f'l2_l{tagf(L2_LAMBDAS[0])}'].sort_values('seed')
    b_med=Bd['roll_mse'].median(); b_init_q=Bd['init_max_q'].median(); b_increment=Bd['increment_norm'].median()
    def e(x): return f"{x:.2e}" if np.isfinite(x) else "n/a"

    print("\n"+"="*80); print(f"ALL VARIANTS (FPU-β S=1 H={HORIZON}, POOLED)"); print("="*80+"\n")
    print(f"  {'model':20s} {'roll_med':>10s} {'ham_drift':>10s} {'symp_err':>10s} {'increment':>10s} {'hard_div':>9s}")
    for nm in df['model'].unique():
        g=df[df['model']==nm]
        print(f"  {nm:20s} {g['roll_mse'].median():>10.4f} {e(g['ham_drift'].median()):>10s} "
              f"{e(g['symp_err'].median()):>10s} {e(g['increment_norm'].median()):>10s} {g['hard_diverged'].mean():>9.2f}")

    # ── RE-CHECK non-degenerate gate at the transfer horizon (controls fair at H=160 may degrade) ──
    print("\n"+"="*80); print(f"NON-DEGENERATE GATE RE-CHECK @ H={HORIZON}"); print("="*80+"\n")
    inc_floor=INCREMENT_FAC*b_increment; maxq_lo=MAXQ_LO_FAC*b_init_q; maxq_hi=MAXQ_HI_FAC*b_init_q
    def l2_fair(g):
        inc=g['increment_norm'].median(); mq=g['max_q'].median(); rl=g['roll_mse'].median()
        return (np.isfinite(inc) and inc>inc_floor) and (maxq_lo<mq<maxq_hi) and (np.isfinite(rl) and rl<DIV_ABS)
    def en_fair(g):
        pdr=g['prior_data_ratio'].median(); rl=g['roll_mse'].median()
        return (np.isfinite(pdr) and pdr<RATIO_CEIL) and (rl<WORSE_FAC*b_med)
    en_ok=en_fair(En); l2_ok=l2_fair(L2)
    print(f"  energy_l0.01 @H{HORIZON}: roll={En['roll_mse'].median():.4f}, ratio={e(En['prior_data_ratio'].median())} → {'FAIR' if en_ok else f'DEGENERATE@H{HORIZON}'}")
    print(f"  l2_l0.001   @H{HORIZON}: roll={L2['roll_mse'].median():.4f}, increment={e(L2['increment_norm'].median())}, max|q|={L2['max_q'].median():.2f} → {'FAIR' if l2_ok else f'DEGENERATE@H{HORIZON}'}")
    if not en_ok: print(f"    ⚠️ energy control degrades at H={HORIZON} — excluded from transfer claim for energy")
    if not l2_ok: print(f"    ⚠️ L2 control degrades at H={HORIZON} — excluded from transfer claim for L2")

    # ── STRATA ──────────────────────────────────────────────────────────────────────
    div_mask=Bd['hard_diverged'].values.astype(bool)
    sg=Bd['seed'].values[~div_mask]; sr=Bd['seed'].values[div_mask]
    n_g=len(sg); n_r=len(sr)
    def sub(g,seeds): return g[g['seed'].isin(seeds)].sort_values('seed')

    def safe_w(a,b):
        a=np.asarray(a,float); b=np.asarray(b,float)
        if a.shape!=b.shape or len(a)==0: return np.nan
        if np.allclose(a-b,0):
            warnings.warn("safe_w: paired arrays are exactly equal (p=1.0). Expected for a true tie, "
                          "but if unexpected it can indicate a data-wiring bug (same model compared to itself).")
            return 1.0
        try: return float(stats.wilcoxon(a,b,alternative='less').pvalue)
        except ValueError: return 1.0
    def rel(b,p): return (b-p)/(b+EPS) if np.isfinite(b) and np.isfinite(p) else np.nan

    def stratum_report(name, seeds, do_tests):
        B=sub(Bd,seeds); S=sub(Sy,seeds); E=sub(En,seeds); L=sub(L2,seeds)
        print(f"\n  ── {name} (n={len(seeds)}) ──")
        print(f"    roll_med: baseline={B['roll_mse'].median():.4f} symplectic={S['roll_mse'].median():.4f} "
              f"energy={E['roll_mse'].median():.4f} l2={L['roll_mse'].median():.4f}")
        out={}
        if do_tests and len(seeds)>=6:
            p_b=safe_w(S['roll_mse'].values,B['roll_mse'].values)
            p_e=safe_w(S['roll_mse'].values,E['roll_mse'].values)
            p_l=safe_w(S['roll_mse'].values,L['roll_mse'].values)
            _pv=[1.0 if not np.isfinite(p) else p for p in (p_b,p_e,p_l)]
            rej,corr,_,_=multipletests(_pv,method='holm')
            print(f"    symp<baseline Holm={corr[0]:.4f}{'✅' if rej[0] else '❌'}  "
                  f"symp<energy Holm={corr[1]:.4f}{'✅' if rej[1] else '❌'}  "
                  f"symp<l2 Holm={corr[2]:.4f}{'✅' if rej[2] else '❌'}")
            se_b=B['symp_err'].median(); se_s=S['symp_err'].median(); red=rel(se_b,se_s)
            q4=bool(np.isfinite(red) and red>MECH_REL_THRESHOLD)
            print(f"    MECHANISM full symp_err: base={e(se_b)} symp={e(se_s)} reduction={red:.1%} → {'✅' if q4 else '❌'}")
            out={'symp_lt_base':bool(rej[0]),'symp_lt_energy':bool(rej[1]),'symp_lt_l2':bool(rej[2]),
                 'mech':q4,'symp_err_reduction':float(red) if np.isfinite(red) else None}
        elif do_tests:
            print(f"    ⚠️ n<6 — too few for paired tests")
        return out

    print("\n"+"="*80); print("THREE STRATA"); print("="*80)
    stratum_report("POOLED (all seeds; inflated by rescue — context only)", Bd['seed'].values, do_tests=True)
    graceful=stratum_report(f"GRACEFUL-baseline subset (HONEST transfer test)", sg, do_tests=True)
    stratum_report("RESCUED subset (baseline diverged; pure rescue — context only)", sr, do_tests=True)
    print(f"\n  strata sizes: pooled={len(Bd)}, graceful={n_g}, rescued={n_r}"
          + ("   ⚠️ graceful n<10" if n_g<10 else ""))

    return {'graceful':graceful,'n_graceful':int(n_g),'n_rescued':int(n_r),
            'energy_fair_at_horizon':bool(en_ok),'l2_fair_at_horizon':bool(l2_ok),
            'pooled_symp_med':float(Sy['roll_mse'].median()),'pooled_base_med':float(b_med)}

def verdict(info):
    print("\n"+"="*80); print(f"VERDICT — symplectic transfer at H={HORIZON}"); print("="*80+"\n")
    g=info.get('graceful') or {}
    print(f"  controls fair @H{HORIZON}: energy={info['energy_fair_at_horizon']}, L2={info['l2_fair_at_horizon']}")
    print(f"  graceful subset n={info['n_graceful']}" + ("  ⚠️ LOW-N" if info['n_graceful']<10 else ""))
    if g:
        print(f"  GRACEFUL: symp<base:{'✅' if g.get('symp_lt_base') else '❌'}  "
              f"symp<energy:{'✅' if g.get('symp_lt_energy') else '❌'}  "
              f"symp<l2:{'✅' if g.get('symp_lt_l2') else '❌'}  "
              f"mechanism:{'✅' if g.get('mech') else '❌'}")
        if g.get('symp_err_reduction') is not None:
            print(f"  graceful-subset symp_err reduction = {g['symp_err_reduction']:.1%}")
    print("\n"+"="*80)
    # Transfer confirmation on the GRACEFUL subset. A control that degenerated at the transfer
    # horizon cannot be a fair comparison, so its clause is WAIVED (not failed) — but the full
    # three-way claim is then reduced. Confirmation requires: symp<baseline + mechanism + symp
    # beats every FAIR control.
    base_ok = bool(g and g.get('symp_lt_base'))
    mech_ok = bool(g and g.get('mech'))
    fair_controls=[]; beaten=[]
    if info['energy_fair_at_horizon']: fair_controls.append('energy'); beaten.append(bool(g and g.get('symp_lt_energy')))
    if info['l2_fair_at_horizon']:     fair_controls.append('l2');     beaten.append(bool(g and g.get('symp_lt_l2')))
    beats_all_fair = all(beaten) if beaten else False
    both_fair = info['energy_fair_at_horizon'] and info['l2_fair_at_horizon']
    pooled_better = info['pooled_symp_med'] < info['pooled_base_med']

    if base_ok and mech_ok and beats_all_fair and both_fair:
        print(f"✅ FULL_TRANSFER_CONFIRMED — on the GRACEFUL-baseline subset at H={HORIZON}, the symplectic")
        print("   prior beats baseline, fair energy, AND fair L2 on roll_MSE, AND the full ‖JᵀΩJ−Ω‖")
        print("   mechanism still reduces >20%. The symplectic advantage TRANSFERS as mechanism (not")
        print("   rescue), and the FULL three-way separation holds at the stress horizon too.")
        print("   → transfer mechanism holds even at this upper-bound horizon.")
        v="FULL_TRANSFER_CONFIRMED"
    elif base_ok and mech_ok and beats_all_fair and not both_fair:
        print("✅ TRANSFER_CONFIRMED — on the graceful subset symplectic beats baseline AND every")
        print(f"   control still FAIR at H={HORIZON} ({fair_controls}), AND the mechanism transfers (>20%).")
        print("   The symplectic transfer itself is CONFIRMED. Only the FULL three-way claim is")
        print(f"   incomplete because a control degenerated at H={HORIZON} (not a failure of transfer —")
        print("   the degenerate control simply cannot serve as a fair comparison at this horizon).")
        print(f"   → transfer holds at H={HORIZON}; optionally re-establish the missing control.")
        v="TRANSFER_CONFIRMED"
    elif base_ok and not mech_ok:
        print("⚠️ TRANSFER_PERFORMANCE_ONLY — symplectic still lowers roll_MSE on the graceful subset,")
        print(f"   but the full symplectic mechanism no longer reduces >20% there. At H={HORIZON} the gain is")
        print("   largely rescue/damping, not confirmed symplectic-structure transfer. (Mirrors K1")
        print("   11.2-E: performance transfers, mechanism does not.) → H=240 unnecessary.")
        v="TRANSFER_PERFORMANCE_ONLY"
    elif base_ok and mech_ok and not beats_all_fair:
        print("⚠️ CONTROL_MATCHES_SYMP — on the graceful subset symplectic beats baseline with a")
        print(f"   transferred mechanism, but a FAIR control matches/beats it on roll_MSE at H={HORIZON}.")
        print("   The performance edge over fair controls does not transfer cleanly, even though the")
        print("   symplectic mechanism itself persists. → inspect the matching control before H=240.")
        v="CONTROL_MATCHES_SYMP"
    elif pooled_better and not base_ok:
        print("⚠️ TRANSFER_PERFORMANCE_ONLY — symplectic improves POOLED roll_MSE, but on the graceful")
        print("   subset it does NOT beat baseline → the pooled gain is rescue of diverged seeds, not")
        print("   a graceful-regime improvement. → H=240 unnecessary.")
        v="TRANSFER_PERFORMANCE_ONLY"
    else:
        print(f"⚠️ NO_TRANSFER — symplectic does not robustly beat baseline at H={HORIZON} (even pooled).")
        print("   The advantage validated at H=160 does not extend to this stress horizon.")
        v="NO_TRANSFER"
    print("="*80)
    if not (info['energy_fair_at_horizon'] and info['l2_fair_at_horizon']):
        print(f"\n  NOTE: a control degenerated at H={HORIZON}; the corresponding 'symp<control' clause was")
        print("  not required for confirmation, and the full three-way transfer claim is reduced.")
    print("\n⚠️  Confirmation requires the GRACEFUL-baseline subset (not pooled): pooled gains at a")
    print("    stress horizon can be pure rescue. This is the K1 11.2-E discipline.")
    return v

def main():
    print("\n"+"🔗"*20); print("EXPERIMENT K2.2-B: SYMPLECTIC TRANSFER @ H=240"); print("🔗"*20+"\n")
    config={"experiment":"k2_2b_transfer_h240","mode":MODE_LABEL,
        "deterministic":DETERMINISTIC,"use_deterministic_algorithms":DETERMINISTIC_ALGOS_OK,
        "N_MASS":N_MASS,"BETA":BETA,"DT_PHYS":DT_PHYS,"IC_AMP":IC_AMP,"IC_PERTURB":IC_PERTURB,
        "S_FIXED":S_FIXED,"HORIZON":HORIZON,"GRAD_CLIP":GRAD_CLIP,
        "HIDDEN":HIDDEN,"N_LAYERS":N_LAYERS,"RADIUS":RADIUS,"LR":LR,"EPOCHS":EPOCHS,"N_SEEDS":N_SEEDS,
        "SYMP_LAMBDAS":SYMP_LAMBDAS,"ENERGY_LAMBDAS":ENERGY_LAMBDAS,"L2_LAMBDAS":L2_LAMBDAS,
        "N_SYMP_PROBES":N_SYMP_PROBES,"N_JAC_STATES":N_JAC_STATES,"INTERIOR_MARGIN":INTERIOR_MARGIN,
        "MECH_REL_THRESHOLD":MECH_REL_THRESHOLD,
        "phase1_gate":{"GRACEFUL_MIN":GRACEFUL_MIN,"DIV_FRAC_CEIL":DIV_FRAC_CEIL},
        "nondegen_gate":{"INCREMENT_FAC":INCREMENT_FAC,"MAXQ_LO_FAC":MAXQ_LO_FAC,"MAXQ_HI_FAC":MAXQ_HI_FAC,
                         "RATIO_CEIL":RATIO_CEIL,"WORSE_FAC":WORSE_FAC},
        "controls":"FAIR configs from K2.1-B (energy_l0.01, l2_l0.001), gate re-checked at H=240",
        "strata":"pooled / graceful-baseline subset (honest) / rescued subset",
        "purpose":"two-phase: regime gate @H240 then (if passed) symplectic transfer upper-bound test",
        "fixes":"energy_penalty mask-multiply (autograd-safe); removed stale K2.0 print; safe_w tie warning",
        "python_version":platform.python_version(),"torch_version":torch.__version__,
        "cuda_available":torch.cuda.is_available(),
        "device_name":(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "numpy_version":np.__version__}
    with open(RESULTS_DIR/"config.json","w") as f: json.dump(config,f,indent=2,default=str)
    print(f"✓ Saved config → {RESULTS_DIR/'config.json'}")

    # ── PHASE 1: baseline-only regime gate ──────────────────────────────────────────
    bdf=run_phase1()
    proceed, hard_div, n_graceful, func_div = phase1_gate(bdf)
    bdf.to_csv(RESULTS_DIR/"exp_k2_2b_phase1_baseline.csv",index=False)
    if not proceed:
        print("\n"+"="*80); print("VERDICT — K2.2-B (phase 1 gate)"); print("="*80+"\n")
        print(f"⛔ REGIME_INVALID_H240_CEILING_H200 — at H=240 the baseline graceful subset is")
        print(f"   {n_graceful}/{N_SEEDS} (hard_div={hard_div:.2f}), below the {GRACEFUL_MIN}-seed bar"
              f"{' / regime invalid' if hard_div>DIV_FRAC_CEIL else ''}.")
        print(f"   A stratified transfer test here would rest on too small (or rescue-dominated) a")
        print(f"   subset. NO priors trained. K2's VALIDATED TRANSFER CEILING IS H=200")
        print(f"   (FULL_TRANSFER_CONFIRMED there). H=240 is beyond the testable graceful window.")
        print("="*80)
        v="REGIME_INVALID_H240_CEILING_H200"
        summary={'experiment':'k2_2b_transfer_h240','verdict':v,'phase':1,'horizon':HORIZON,
                 'hard_div':float(hard_div),'functional_div':float(func_div),'n_graceful':int(n_graceful),
                 'baseline_med':float(bdf['roll_mse'].median())}
        pd.DataFrame([summary]).to_csv(RESULTS_DIR/"exp_k2_2b_summary.csv",index=False)
        print(f"\n✓ Saved {RESULTS_DIR/'exp_k2_2b_summary.csv'}")
        print(f"\n\n{'='*80}\nFINAL RESULT: {v}\n{'='*80}")
        return bdf,None,v

    # ── PHASE 2: prior transfer (gate passed) ───────────────────────────────────────
    print("\n"+"="*80); print("PHASE 2 — prior transfer (regime gate passed)"); print("="*80+"\n")
    df=run(); df.to_csv(RESULTS_DIR/"exp_k2_2b_results.csv",index=False)
    print(f"\n✓ Saved {RESULTS_DIR/'exp_k2_2b_results.csv'}")
    info=analyze(df); v=verdict(info)
    g=info.get('graceful') or {}
    summary={'experiment':'k2_2b_transfer_h240','verdict':v,'phase':2,'horizon':HORIZON,
             'phase1_hard_div':float(hard_div),'phase1_functional_div':float(func_div),
             'n_graceful':info['n_graceful'],'n_rescued':info['n_rescued'],
             'energy_fair_at_horizon':info['energy_fair_at_horizon'],'l2_fair_at_horizon':info['l2_fair_at_horizon'],
             'graceful_symp_lt_base':g.get('symp_lt_base'),'graceful_symp_lt_energy':g.get('symp_lt_energy'),
             'graceful_symp_lt_l2':g.get('symp_lt_l2'),'graceful_mechanism':g.get('mech'),
             'graceful_symp_err_reduction':g.get('symp_err_reduction'),
             'pooled_base_med':info['pooled_base_med'],'pooled_symp_med':info['pooled_symp_med']}
    pd.DataFrame([summary]).to_csv(RESULTS_DIR/"exp_k2_2b_summary.csv",index=False)
    print(f"✓ Saved {RESULTS_DIR/'exp_k2_2b_summary.csv'}")
    print(f"\n\n{'='*80}\nFINAL RESULT: {v}\n{'='*80}")
    return df,info,v

if __name__ == "__main__":
    df, info, v = main()
