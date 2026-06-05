# K3-E2b Active Topology Regime Search

Status:

```text
ACTIVE_DIAGNOSTIC_VALUE_PASSED
```

K3-E2b is a deterministic toy active-search layer for K3 topology regime
discovery.

It asks one narrow question:

```text
Can guided active regime search find a topology-trackable toy regime better
than blind random action choice with the same budget?
```

## Result

On the transparent toy topology-survival landscape:

- active guided search reaches a `transport_ok` regime
- random action choice with the same budget does not
- active routes through S0 as `K3_TOPOLOGICAL / continue`
- random routes through S0 as `K3_TOPOLOGICAL / do_not_promote`
- random reaches transport OK in `0/20` seeds at the checked settings

The headline result is:

```text
guided regime search beats blind random search
```

## Boundary

This is not K3 prior validation.

It is not GP truth, not proof that topological priors work, not robotics, not
RL training, not CNN training, and not a full Gross-Pitaevskii run.

The active search is guided: it evaluates neighboring toy regimes and moves
toward the best observed topology score. Novelty only breaks ties. The tests
also record that pure novelty alone does not solve this product-shaped toy
landscape.

## Run

```bash
python -m chronos.k3.run_active_topology_search
python -m chronos.k3.run_active_topology_search --json
python -m unittest chronos.k3.tests.test_active_topology_search
```

## Future Path

```text
K3-E2b toy landscape -> K3-E2c cheap GP truth-only active search -> K3.2D regime validation -> K3 prior test
```
