# S0-E0: Embodied Toy Diagnostics

This is the pre-robot embodied toy layer for Chronos-S0.

It does not control a robot.
It does not train a model.
It does not run a physical simulation yet.

It tests one thing:

```text
toy physical situation -> hand-written diagnostics -> S0 recommends a physical language
```

Passing S0-E0 means the structure-language selector is wired correctly for
simple toy diagnostic packets. It does not mean Chronos has learned from real
physical data.

Current verdict:

```text
S0_E0_TOY_SUITE_PASSED
```

Detailed verdict:

- `S0_E0_TOY_SUITE_PASSED.md`

## Toy Worlds

| Toy case | Expected S0 recommendation |
| --- | --- |
| `pendulum` | `K2_SYMPLECTIC` / `continue` |
| `causal_contact` | `K1_LORENTZ` / `continue` |
| `vortex_fail` | `K3_TOPOLOGICAL` / `do_not_promote` |
| `unknown` | `UNRESOLVED` / `do_not_promote` |

## Run

```bash
python -m chronos.embodied_toy.run_toy_suite
python -m chronos.embodied_toy.run_toy_suite --json
python -m unittest chronos.embodied_toy.tests.test_embodied_toy
```

## Boundary

S0-E0 is not robotics.

It is not a learned classifier, not a simulator, and not a certification gate.
It only checks whether S0 selects the expected physical language from
hand-written toy diagnostics.

The next step is S0-E1:

```text
toy simulation -> diagnostic extractor -> S0 recommendation
```

## S0-E1: Toy Simulation Diagnostics

S0-E1 advances the toy layer from hand-written diagnostics to measured
diagnostics from small deterministic toy trajectories.

Status:

```text
S0_E1_TOY_SIM_PASSED
```

Pipeline:

```text
toy simulation -> diagnostic extractor -> S0 recommendation
```

Toy simulation cases:

| Toy simulation | Measured diagnostic | Expected S0 recommendation |
| --- | --- | --- |
| `pendulum` | symplectic energy drift beats explicit-Euler control | `K2_SYMPLECTIC` / `continue` |
| `contact` | toy contact/event-ordering proxy | `K1_LORENTZ` / `continue` |
| `object_persistence` | object id disappears mid-trajectory | `K3_TOPOLOGICAL` / `do_not_promote` |

Run:

```bash
python -m chronos.embodied_toy.run_sim_suite
python -m chronos.embodied_toy.run_sim_suite --json
python -m unittest chronos.embodied_toy.tests.test_embodied_sim
```

Boundary:

S0-E1 is not robotics, not model training, not learned classification, and not
physical certification. The contact case is explicitly a toy proxy, not
relativistic causality.

Detailed verdict:

- `S0_E1_TOY_SIM_PASSED.md`
