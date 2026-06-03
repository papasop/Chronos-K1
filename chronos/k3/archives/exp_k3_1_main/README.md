# K3.1 MAIN: Winding-Density Continuity Prior Test

Status: archived negative Stage-2 prior test

Verdict:

```text
NO_EFFECT
```

## Purpose

K3.1 tests whether a winding-density continuity prior can pass the VPSL
Stage-2 prior test in the periodic Sine-Gordon graceful regime found by
K3.0-D.

Scientific question:

Does the topological-density continuity prior outperform:

- no-prior baseline
- matched-ratio off-target continuity
- generic Laplacian smoothness
- generic increment L2 regularization

## System

- equation: periodic Sine-Gordon
- representation: `[sin(u), cos(u), u_t]`
- grid: `NX=256`, `L=50.0`
- horizon: `H=160`
- initial condition: one boosted kink with randomized position / velocity
- honest target: winding-density / local topological structure

Non-claim:

```text
This is not integer topological-charge certification.
```

## Groups

| Group | Prior | Lambda | Prior / Roll Ratio |
| --- | --- | ---: | ---: |
| baseline | none | 0 | n/a |
| density_l0.1 | winding_density_continuity | 0.1 | ~0.51 |
| density_l1.0 | winding_density_continuity | 1.0 | ~1.08 |
| offtarget_l0.01 | off_target_continuity | 0.01 | ~0.83 |
| offtarget_l0.1 | off_target_continuity | 0.1 | ~5.7 |
| smoothness_l0.1 | gradient_smoothness | 0.1 | ~0.50 |
| increment_l2_l0.1 | increment_l2 | 0.1 | ~0.36 |

The primary fairness comparison is matched by prior / rollout ratio:

```text
density_l1.0 vs offtarget_l0.01
```

The equal-lambda `offtarget_l0.1` group is an over-regularized upper-bound
reference only. It is not a pass / fail fairness gate.

## Archived Headline Results

| Group | roll_MSE median | winding-density error median | wind_err | inc_norm |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0068 | 0.0379 | 0.20 | 0.0107 |
| density_l0.1 | 0.0059 | 0.0377 | 0.10 | 0.0103 |
| density_l1.0 | 0.0087 | 0.0419 | 0.07 | 0.0106 |
| offtarget_l0.01 | 0.0067 | 0.0383 | 0.20 | 0.0103 |
| offtarget_l0.1 | 0.0039 | 0.0251 | 0.07 | 0.0077 |
| smoothness_l0.1 | 0.0059 | 0.0343 | 0.17 | 0.0104 |
| increment_l2_l0.1 | 0.0065 | 0.0364 | 0.20 | 0.0106 |

Mechanism diagnostic:

```text
wdens_err reduction = -10.7%
integer winding intact fraction = 0.93
```

The density prior did not beat baseline, did not beat the matched-ratio
off-target continuity control, and did not pass the winding-density mechanism
test.

## Files

```text
exp_k3_1_main/
├── README.md
├── config.json
├── k3_1_main.py
├── results/
│   ├── k3_1_main_results.csv
│   └── k3_1_main_summary.csv
├── analysis/
│   └── analyze_results.ipynb
└── docs/
    └── EXPERIMENT_DESIGN.md
```

## Run Archive Entry

From the repository root:

```bash
python chronos/k3/archives/exp_k3_1_main/k3_1_main.py
```

This regenerates the archived headline result CSV and summary CSV.

It does not rerun the full GPU Colab training sweep.
