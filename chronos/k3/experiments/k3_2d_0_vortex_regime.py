"""
EXPERIMENT K3.2D.0: 2D VORTEX REGIME VALIDATION (Gross-Pitaevskii) — SMOKE FIRST
=================================================================================
Stage-1 of K3-2D (the 2D restart of the topological-structure track). K3 in 1D was SEALED: a valid
winding-density regime existed but the 1D topological prior could not separate from generic smoothing
(topology↔smoothness entanglement is geometric in 1D). A sandbox feasibility check confirmed 2D
vortices DO separate topology from smoothness (a smooth field can carry a localized vortex; removing
one needs a non-smooth core event). This experiment is the first concrete 2D step.

SCOPE: regime validation ONLY (no prior yet). The single question: does a GRACEFUL baseline regime
exist — a 2D CNN baseline that (a) actually trains (ref@REF clean), (b) keeps the vortex-antivortex
pair mostly intact, and (c) has a CONTINUOUS metric (vortex position error) that degrades gracefully
with horizon? Only if this passes does a 2D prior test (K3.2D.1) make sense. This mirrors K3.0-D /
K2.0 discipline: validate the regime before spending any compute on priors.

RUN SMOKE FIRST. SMOKE confirms the pipeline runs and the baseline can learn at all (clean ref@REF) —
NOT a scientific result. If SMOKE looks sane, switch RUN_MODE='FULL' for the N=30 regime decision.

System: 2D Gross-Pitaevskii, psi complex, split-step (Fourier kinetic + nonlinear phase). State =
[Re psi, Im psi] (U(1)-natural — absorbs the K3.0-D sin/cos lesson; no lifted-angle seam). Topologically
legal periodic configuration = vortex–ANTIvortex PAIR (net charge 0). Truth: settle 5 GP steps, then
each model step = S GP substeps.

METRICS:
  continuous (graceful candidate): vortex POSITION error — distance between predicted & true vortex
    core centroids (+ and − tracked separately), in lattice units.
  discrete (near-floor check): vortex COUNT / net-charge error (pair intact?).
  baseline-health: ref@REF rollout MSE on [Re,Im] (did it train?), field bounded, hard-div.

VERDICT (regime gate):
  VALID_VORTEX_REGIME    baseline trains (ref clean), pair mostly intact across H, position error rises
                         but stays bounded/graceful, hard_div low → proceed to K3.2D.1 (prior test).
  NO_GRACEFUL_BAND       position error saturates immediately / pair annihilates early / baseline can't
                         fit (ref bad) → 2D regime not testable as-is; retune (sep, S, g, grid) or stop.
"""
import os, json, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# ---- run mode ----
RUN_MODE="SMOKE"     # SMOKE first; then FULL for the N=30 regime decision
DETERMINISTIC=True
RESULTS_DIR=Path("/content/exp_k3_2d_0"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EPS=1e-8

# ---- 2D GP physics ----
L=32; DX=1.0; G=1.0; DT_PHYS=0.02
S_FIXED=4                 # GP substeps per model step
SETTLE=5                  # settle steps to clean vortex cores before t=0
PAIR_SEP=10               # vortex-antivortex separation
CORE=3.0                  # core size in tanh profile
HORIZON=40                # model steps for regime probe
REF_HORIZON=10

# ---- model / training ----
HIDDEN=48; N_LAYERS=3; RADIUS=2; LR=3e-4; EPOCHS=80; N_SEEDS=30
N_TRAIN_TRAJ=24; TRAIN_STEPS=40; GRAD_CLIP=1.0
FUNC_DIV_THR=10.0
# regime gate thresholds
REF_CEIL=0.05             # baseline must fit the short-horizon map
POS_ERR_CEIL=8.0          # final vortex-position error (lattice units) ceiling for "graceful"
PAIR_INTACT_MIN=0.6       # fraction of seeds keeping exactly one +/- pair at final H
HARD_DIV_MAX=0.5

if RUN_MODE=="SMOKE":
    N_SEEDS=3; EPOCHS=15; N_TRAIN_TRAJ=8; TRAIN_STEPS=20; HORIZON=20
    print("⚠️  SMOKE MODE — tiny config, results NOT valid (pipeline + can-baseline-learn check only).\n")

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)
if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass
print("K3.2D.0 — 2D vortex regime validation (Gross-Pitaevskii), baseline-only")
print(f"  mode={RUN_MODE}, DEVICE={DEVICE}, N_SEEDS={N_SEEDS}, EPOCHS={EPOCHS}, L={L}, H={HORIZON}")
print(f"  state=[Re psi, Im psi]; regime = vortex-antivortex pair (net charge 0)\n")

# spectral operators
_k=2*np.pi*np.fft.fftfreq(L,d=DX); _KX,_KY=np.meshgrid(_k,_k,indexing='ij'); _K2=_KX**2+_KY**2
_x=np.arange(L)*DX; _X,_Y=np.meshgrid(_x,_x,indexing='ij')

def pair_ic(rng):
    sep=PAIR_SEP+rng.uniform(-2,2); ang=rng.uniform(0,2*np.pi)
    cx,cy=L/2+rng.uniform(-3,3),L/2+rng.uniform(-3,3)
    dx_=0.5*sep*np.cos(ang); dy_=0.5*sep*np.sin(ang)
    cx1,cy1=cx-dx_,cy-dy_; cx2,cy2=cx+dx_,cy+dy_
    r1=np.sqrt((_X-cx1)**2+(_Y-cy1)**2)+1e-6; r2=np.sqrt((_X-cx2)**2+(_Y-cy2)**2)+1e-6
    theta=np.arctan2(_Y-cy1,_X-cx1)-np.arctan2(_Y-cy2,_X-cx2)
    return np.tanh(r1/CORE)*np.tanh(r2/CORE)*np.exp(1j*theta)
def gp_step(psi):
    psi=psi*np.exp(-1j*G*np.abs(psi)**2*DT_PHYS/2)
    psi=np.fft.ifft2(np.fft.fft2(psi)*np.exp(-1j*0.5*_K2*DT_PHYS))
    psi=psi*np.exp(-1j*G*np.abs(psi)**2*DT_PHYS/2); return psi
def simulate(psi0,n,S):
    psi=psi0.copy(); tr=[np.stack([psi.real,psi.imag])]
    for _ in range(n):
        for _ in range(S): psi=gp_step(psi)
        tr.append(np.stack([psi.real,psi.imag]))
    return np.array(tr)   # [n+1,2,L,L]
def settle(psi):
    for _ in range(SETTLE): psi=gp_step(psi)
    return psi

def _plaq(theta):
    def w(d):return (d+np.pi)%(2*np.pi)-np.pi
    d1=w(np.roll(theta,-1,0)-theta);d2=w(np.roll(np.roll(theta,-1,0),-1,1)-np.roll(theta,-1,0))
    d3=w(np.roll(theta,-1,1)-np.roll(np.roll(theta,-1,0),-1,1));d4=w(theta-np.roll(theta,-1,1))
    return (d1+d2+d3+d4)/(2*np.pi)
def _circular_mean_coords(idxs):
    """circular mean of lattice coordinates on a periodic L×L grid (avoids wrap-averaging errors)."""
    if len(idxs)==0: return None
    out=[]
    for ax in range(2):
        ang=2*np.pi*idxs[:,ax]/L
        m=np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
        out.append((m/(2*np.pi))*L % L)
    return np.array(out)
def vortex_centroids(state):
    psi=state[0]+1j*state[1]; w=_plaq(np.angle(psi))
    pos=np.argwhere(w>0.5); neg=np.argwhere(w<-0.5)
    cp=_circular_mean_coords(pos); cn=_circular_mean_coords(neg)
    # charge-based count (robust to a vortex core spanning multiple plaquettes)
    pos_charge=int(round(w[w>0.5].sum())) if len(pos) else 0
    neg_charge=int(round(w[w<-0.5].sum())) if len(neg) else 0
    return cp,cn,pos_charge,neg_charge
def position_error(pred_state,true_state):
    cpp,cpn,_,_=vortex_centroids(pred_state); ctp,ctn,_,_=vortex_centroids(true_state)
    errs=[]
    for a,b in [(cpp,ctp),(cpn,ctn)]:
        if a is not None and b is not None:
            d=a-b; d=(d+L/2)%L-L/2; errs.append(np.sqrt(np.sum(d**2)))  # periodic distance
        else: errs.append(POS_ERR_CEIL)  # missing vortex = max error
    return float(np.mean(errs))

# normalization
def _fit_norm():
    rng=np.random.default_rng(999); vals=[[],[]]
    for _ in range(10):
        tr=simulate(settle(pair_ic(rng)),TRAIN_STEPS,S_FIXED)
        for c in range(2): vals[c].append(tr[:,c].ravel())
    mean=np.array([np.concatenate(vals[c]).mean() for c in range(2)],dtype=np.float32).reshape(1,2,1,1)
    std=np.array([np.concatenate(vals[c]).std()+EPS for c in range(2)],dtype=np.float32).reshape(1,2,1,1)
    return mean,std
DATA_MEAN,DATA_STD=_fit_norm()
def normalize(a): return (a-DATA_MEAN)/DATA_STD
def denormalize(a): return a*DATA_STD+DATA_MEAN

class GP2DConvNet(nn.Module):
    def __init__(self,mean,std):
        super().__init__(); ch=[2]+[HIDDEN]*(N_LAYERS-1)+[2]
        self.convs=nn.ModuleList([nn.Conv2d(ch[i],ch[i+1],2*RADIUS+1,padding=RADIUS,
                                            padding_mode='circular') for i in range(N_LAYERS)])
        self.act=nn.GELU()
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.tensor(std,dtype=torch.float32))
    def forward(self,u):
        x=u
        for i,c in enumerate(self.convs):
            x=c(x)
            if i<len(self.convs)-1: x=self.act(x)
        return u+x
    def rollout(self,u0,n):
        outs=[u0]; u=u0
        for _ in range(n): u=self.forward(u); outs.append(u)
        return torch.stack(outs,dim=1)
def K_for_epoch(ep): return 1 if ep<EPOCHS*0.3 else (2 if ep<EPOCHS*0.6 else 3)

def build_training(seed):
    rng=np.random.default_rng(10000+seed)
    return [normalize(simulate(settle(pair_ic(rng)),TRAIN_STEPS,S_FIXED)) for _ in range(N_TRAIN_TRAJ)]
def train(trajs,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model=GP2DConvNet(DATA_MEAN,DATA_STD).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR); mse=nn.MSELoss()
    T=trajs[0].shape[0]; data=torch.tensor(np.stack(trajs),dtype=torch.float32,device=DEVICE)
    n=data.shape[0]; rng=np.random.default_rng(50000+seed)
    for ep in range(EPOCHS):
        K=K_for_epoch(ep); ms=T-1-K
        if ms<0: K=T-1; ms=0
        model.train(); perm=torch.randperm(n)
        for bi in range(n):
            ti=perm[bi].item(); st=int(rng.integers(0,ms+1)); cur=data[ti,st].unsqueeze(0); opt.zero_grad(); loss=0.0
            for kk in range(1,K+1):
                cur=model(cur); loss=loss+mse(cur,data[ti,st+kk].unsqueeze(0))
            (loss/K).backward()
            if GRAD_CLIP: torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step()
    return model
def evaluate(model,seed):
    model.eval(); rng=np.random.default_rng(30000+seed+777777)
    psi0=settle(pair_ic(rng))                    # ONE eval IC
    true=simulate(psi0,HORIZON,S_FIXED)
    ref=true[:REF_HORIZON+1]                      # ref = same-IC prefix (comparable to H diagnostics)
    u0=torch.tensor(normalize(true[0:1]),dtype=torch.float32,device=DEVICE)
    with torch.no_grad():
        ur=torch.tensor(normalize(ref[0:1]),dtype=torch.float32,device=DEVICE)
        predr=denormalize(model.rollout(ur,REF_HORIZON).squeeze(0).cpu().numpy())
        Tr=min(len(predr),len(ref)); ref_mse=float(np.mean((predr[:Tr]-ref[:Tr])**2))
        pred=denormalize(model.rollout(u0,HORIZON).squeeze(0).cpu().numpy())
    if not (np.all(np.isfinite(pred)) and np.max(np.abs(pred))<1e4):
        return dict(roll_mse=1e4,pos_err=POS_ERR_CEIL,pair_intact=0,ref_mse=ref_mse,hard_div=1.0)
    T=min(len(pred),len(true)); roll=float(np.mean((pred[:T]-true[:T])**2))
    pos_err=position_error(pred[T-1],true[T-1])
    _,_,pos_charge,neg_charge=vortex_centroids(pred[T-1])
    pair_intact=int(pos_charge==1 and neg_charge==-1)   # net +1/-1 pair (charge-based, robust)
    per=np.mean((pred[:T]-true[:T])**2,axis=(1,2,3)); over=per>FUNC_DIV_THR
    hard=bool(over[-1] and over[max(0,T-5):].mean()>0.8)
    return dict(roll_mse=roll,pos_err=pos_err,pair_intact=pair_intact,ref_mse=ref_mse,hard_div=1.0 if hard else 0.0)

def main():
    print("\n"+"🌀"*20); print("EXPERIMENT K3.2D.0: 2D VORTEX REGIME VALIDATION"); print("🌀"*20+"\n")
    config={"experiment":"k3_2d_0","mode":RUN_MODE,"L":L,"G":G,"S_FIXED":S_FIXED,"PAIR_SEP":PAIR_SEP,
            "HORIZON":HORIZON,"N_SEEDS":N_SEEDS,"EPOCHS":EPOCHS,"state":"[Re psi, Im psi]",
            "regime_gate":{"REF_CEIL":REF_CEIL,"POS_ERR_CEIL":POS_ERR_CEIL,"PAIR_INTACT_MIN":PAIR_INTACT_MIN,
                           "HARD_DIV_MAX":HARD_DIV_MAX},
            "device":DEVICE,"torch":torch.__version__,"numpy":np.__version__}
    with open(RESULTS_DIR/"config.json","w") as f: json.dump(config,f,indent=2,default=str)
    import time; rows=[]
    print(f"Training {N_SEEDS} baseline-only seeds @ H={HORIZON}...\n")
    for seed in range(N_SEEDS):
        t0=time.time(); model=train(build_training(seed),seed); m=evaluate(model,seed)
        m['seed']=seed; rows.append(m)
        print(f"  seed {seed+1}/{N_SEEDS} ({time.time()-t0:.0f}s): roll={m['roll_mse']:.4f} "
              f"pos_err={m['pos_err']:.2f} pair_intact={m['pair_intact']} ref@{REF_HORIZON}={m['ref_mse']:.4f} hard_div={m['hard_div']:.0f}")
    df=pd.DataFrame(rows); df.to_csv(RESULTS_DIR/"k3_2d_0_results.csv",index=False)
    ref_med=df['ref_mse'].median(); pos_med=df['pos_err'].median()
    pair_frac=df['pair_intact'].mean(); hard_frac=df['hard_div'].mean()
    print("\n"+"="*80); print(f"REGIME GATE @ H={HORIZON}"); print("="*80+"\n")
    print(f"  baseline ref@{REF_HORIZON} median = {ref_med:.4f} (need < {REF_CEIL}) → {'✓' if ref_med<REF_CEIL else '✗ baseline did not train'}")
    print(f"  vortex position error median = {pos_med:.2f} (need < {POS_ERR_CEIL}) → {'✓' if pos_med<POS_ERR_CEIL else '✗ saturated'}")
    print(f"  pair-intact fraction = {pair_frac:.2f} (need >= {PAIR_INTACT_MIN}) → {'✓' if pair_frac>=PAIR_INTACT_MIN else '✗ pair annihilates'}")
    print(f"  hard-div fraction = {hard_frac:.2f} (need <= {HARD_DIV_MAX}) → {'✓' if hard_frac<=HARD_DIV_MAX else '✗'}")
    trained=ref_med<REF_CEIL; graceful=pos_med<POS_ERR_CEIL; pair_ok=pair_frac>=PAIR_INTACT_MIN; div_ok=hard_frac<=HARD_DIV_MAX
    valid = trained and graceful and pair_ok and div_ok
    print("\n"+"="*80)
    if RUN_MODE=="SMOKE":
        print("⚠️  SMOKE — pipeline + can-baseline-learn check only. NOT a regime decision.")
        print(f"   {'pipeline runs, baseline learns (ref ok); promote to FULL' if trained else 'baseline did NOT learn even loosely — fix before FULL (sep/S/g/grid/epochs)'}")
        v="SMOKE_OK" if trained else "SMOKE_BASELINE_NOT_LEARNING"
    elif valid:
        print("✅ VALID_VORTEX_REGIME — baseline trains, pair mostly intact, position error graceful,")
        print("   hard-div low. A continuous (vortex-position) metric degrades gracefully. → proceed to")
        print("   K3.2D.1 (2D topological prior test with off-target / smoothness / increment controls).")
        v="VALID_VORTEX_REGIME"
    else:
        print("⚠️ NO_GRACEFUL_BAND — regime not testable as-is:")
        if not trained: print("   baseline did not train (ref bad)")
        if not graceful: print("   position error saturated (no graceful band)")
        if not pair_ok: print("   pair annihilates too often")
        if not div_ok: print("   too much hard divergence")
        print("   Retune (sep, S, g, grid, epochs) or conclude 2D regime needs a different setup.")
        v="NO_GRACEFUL_BAND"
    print("="*80)
    pd.DataFrame([{'experiment':'k3_2d_0','verdict':v,'mode':RUN_MODE,'ref_med':ref_med,
                   'pos_med':pos_med,'pair_frac':pair_frac,'hard_frac':hard_frac}]).to_csv(
        RESULTS_DIR/"k3_2d_0_summary.csv",index=False)
    print(f"\n✓ Saved to {RESULTS_DIR}")
    print(f"\n{'='*80}\nFINAL: {v}{' (SMOKE)' if RUN_MODE=='SMOKE' else ''}\n{'='*80}")
    return df,v

if __name__=="__main__":
    df,v=main()
