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
