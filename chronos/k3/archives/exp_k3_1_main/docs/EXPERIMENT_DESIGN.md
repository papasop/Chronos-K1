# K3.1 Experiment Design

## 1. Background

K3.0-D found a valid periodic Sine-Gordon graceful regime using the angle
representation `[sin(u), cos(u), u_t]`. K3.1 tests whether a winding-density
continuity prior has structural value in that regime.

The claim boundary is deliberately narrow:

```text
winding-density / local topological structure
```

This experiment does not certify integer topological charge.

## 2. Key Design Decisions

### 2.1 Matched Ratio, Not Matched Lambda

At equal lambda, the off-target continuity prior has a much larger prior /
rollout ratio than the density prior. The primary fair comparison is therefore
matched by regularization strength:

- `density_l1.0`: prior / rollout ratio approximately `1.08`
- `offtarget_l0.01`: prior / rollout ratio approximately `0.83`

### 2.2 Role Of offtarget_l0.1

`offtarget_l0.1` has prior / rollout ratio around `5.7`. It is retained as a
strong over-regularized upper-bound reference, not as a fairness gate.

### 2.3 Gates

All key groups must satisfy:

- `inc_norm >= 0.30 * true_inc_median`
- `hard_div < 0.2`
- `ref_mse < 1.0`

These gates avoid identity collapse, divergent rollouts, and short-horizon
training failure.

## 3. Statistical Method

- paired Wilcoxon
- one-sided alternative: `less`
- Holm-Bonferroni correction
- alpha: `0.05`

Primary questions:

- Q1: `density_l1.0 < baseline`
- Q2: `density_l1.0 < offtarget_l0.01`
- Q3: `density_l0.1 < smoothness_l0.1`
- Q4: `density_l0.1 < increment_l2_l0.1`

Mechanism:

- winding-density error reduction at least `20%`
- integer winding intact fraction at least `0.6`

## 4. Verdict Taxonomy

| Verdict | Condition | Meaning |
| --- | --- | --- |
| WINDING_DENSITY_CONFIRMED | Q1-Q4 all pass plus mechanism | Topological-density alignment has structural value |
| DENSITY_HELP_BUT_NOT_OFFTARGET | Q1 passes but Q2 fails | Continuity helps, but topology is not separated |
| SMOOTHNESS_ONLY | Density loses to generic controls or lacks mechanism | Generic regularization explains the gain |
| NO_EFFECT | Q1 fails | Density prior does not beat baseline |
| DEGENERATE_INVALID | A key gate fails | Experiment invalid |

## 5. Archived Outcome

K3.1 returns:

```text
NO_EFFECT
```

The density prior failed Q1, Q2, Q3, Q4, and the winding-density mechanism
diagnostic.

## 6. Implication

K3.1 is a negative Stage-2 prior result. It does not reject all topological
structures, but it does reject this winding-density continuity prior in this
one-dimensional periodic Sine-Gordon setting.
