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

## S0-E2: Active Toy Exploration

S0-E2 advances from passive toy simulation to deterministic active exploration.

Status:

```text
S0_E2_ACTIVE_TOY_PASSED
```

Pipeline:

```text
agent chooses action by novelty -> toy world responds -> diagnostic probe -> S0 recommendation
```

The active explorer uses a lightweight novelty heuristic:

```text
candidate next-state distance from the set of visited states
```

This is RND-lite only. No neural network is trained.

Acceptance checks:

- active exploration covers more coarse state cells than a random-action control
- the K2 diagnostic probe launches from an active reached state
- symplectic energy drift is measured against an explicit-Euler control
- S0 recommends `K2_SYMPLECTIC` / `continue`

Run:

```bash
python -m chronos.embodied_toy.run_active_suite
python -m chronos.embodied_toy.run_active_suite --json
python -m unittest chronos.embodied_toy.tests.test_embodied_active
```

Boundary:

S0-E2 is not robotics, not RL training, not a neural network, not online
learning, and not physical certification.

Detailed verdict:

- `S0_E2_ACTIVE_TOY_PASSED.md`

## S0-E2b: Active Diagnostic Value

S0-E2b checks whether active exploration has real diagnostic value, not merely
whether the active-to-diagnostic loop is wired.

Status:

```text
S0_E2B_ACTIVE_VALUE_PASSED
```

Pipeline:

```text
partitioned toy world -> active reaches structure zone -> diagnostic probe -> S0 recommendation
```

The partitioned rail world only exposes a K2 structure signal in the far zone.
The active explorer reliably reaches that zone; the random-action control does
not under the fixed control seeds.

Observed recommendation split:

```text
active -> K2_SYMPLECTIC / continue
random -> UNRESOLVED / do_not_promote
```

Run:

```bash
python -m chronos.embodied_toy.run_active_value_suite
python -m chronos.embodied_toy.run_active_value_suite --json
python -m unittest chronos.embodied_toy.tests.test_embodied_active_value
```

Boundary:

S0-E2b is not robotics, not RL training, not a neural network, not online
learning, and not physical certification.

Detailed verdict:

- `S0_E2B_ACTIVE_VALUE_PASSED.md`
