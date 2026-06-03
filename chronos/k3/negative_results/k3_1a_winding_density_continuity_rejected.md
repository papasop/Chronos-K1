# K3.1-A Negative Result: Winding-Density Continuity Prior

Status:

```text
NO_EFFECT
```

System:

```text
Periodic Sine-Gordon
```

Representation:

```text
[sin(u), cos(u), u_t]
```

Target:

```text
winding-density / local topological structure
```

Non-claim:

```text
integer topological-charge certification
```

## Context

K3.0-D found a valid winding-density regime for periodic Sine-Gordon. K3.1-A
then tested whether a Form-C winding-density continuity prior could improve
rollout performance and mechanism alignment at `H=160`.

The fair primary comparison used matched prior / rollout ratio:

- `density_l1.0`, ratio approximately `1.08`
- `offtarget_l0.01`, ratio approximately `0.83`

The equal-lambda `offtarget_l0.1` arm is retained only as an over-regularized
upper-bound reference, not as a pass / fail control.

## Archived Result

| Group | roll_MSE median | winding-density error median | wind_err | inc_norm |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0068 | 0.0379 | 0.20 | 0.0107 |
| density_l0.1 | 0.0059 | 0.0377 | 0.10 | 0.0103 |
| density_l1.0 | 0.0087 | 0.0419 | 0.07 | 0.0106 |
| offtarget_l0.01 | 0.0067 | 0.0383 | 0.20 | 0.0103 |
| offtarget_l0.1 | 0.0039 | 0.0251 | 0.07 | 0.0077 |
| smoothness_l0.1 | 0.0059 | 0.0343 | 0.17 | 0.0104 |
| increment_l2_l0.1 | 0.0065 | 0.0364 | 0.20 | 0.0106 |

Mechanism:

```text
winding-density error reduction = -10.7%
integer winding intact fraction = 0.93
```

## Failure Points

- `density_l1.0` did not beat baseline.
- `density_l1.0` did not beat the matched-ratio off-target continuity control.
- `density_l0.1` did not beat generic smoothness.
- `density_l0.1` did not beat increment L2.
- Winding-density error worsened rather than improving by the required `20%`.

## Interpretation

The tested Form-C winding-density continuity prior has no VPSL-supported
structural value in this K3.1-A setting.

This does not reject all winding-density or topological priors. It rejects this
specific continuity-form prior in a one-dimensional periodic Sine-Gordon
regime.

## Archive Links

- `chronos/k3/archives/exp_k3_1_main/`
- `chronos/k3/experiments/k3_1_winding_density_prior.py`
- `chronos/k3/results/k3_1_summary.csv`
