# S0-E1 Toy Simulation Verdict

Final verdict:

```text
S0_E1_TOY_SIM_PASSED
```

Scope:

```text
toy simulation -> diagnostic extractor -> S0 recommendation
```

Claim:

S0 can choose the expected physical language from diagnostics measured from
small deterministic toy trajectories:

- `pendulum` -> `K2_SYMPLECTIC` / `continue`
- `contact` -> `K1_LORENTZ` / `continue`
- `object_persistence` -> `K3_TOPOLOGICAL` / `do_not_promote`

## What Is Real

The pendulum signal is measured from trajectory energy drift:

```text
symplectic integrator energy drift < explicit-Euler control energy drift
```

The object-persistence signal is measured from a trajectory whose object id
disappears mid-run.

## What Is Proxy

The contact case uses a toy event-ordering/contact proxy:

```text
contact event density -> causal_violation_rate
```

This is not relativistic causality and not physics certification.

## Non-Claims

S0-E1 is not robotics.

It is not model training, not a learned classifier, and not a certification
gate. It only checks the computer-side pipeline:

```text
toy simulation -> diagnostic extractor -> S0 recommendation
```
