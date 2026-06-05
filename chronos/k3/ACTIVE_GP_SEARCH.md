# K3-E2d Discriminating GP Truth-Only Active Search

Status:

```text
GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED
```

K3-E2d is a cheap real-GP active regime-search step for K3 topology
discovery. It changes the K3-E2c measurement paradigm from discrete charge
survival to continuous vortex-position transport.

## What Changed

K3-E2c used a discrete phase-winding lifetime metric. At cheap resolution, that
landscape was too trivial: random search reached stable charge-survival regimes
too often.

K3-E2d tracks phase-singularity positions and scores how close the +1/-1 pair
stays to the intended static positions. A `push` dimension creates a broad
failure region by moving the pair away from the intended transport target.

## Admission Criteria

The pass verdict requires all of:

- random success count at most `5/20`
- active search reaches `transport_ok`
- active best score exceeds random median best score by `SCORE_MARGIN`

## Boundary

This is not K3 prior validation.

It is not proof that topological priors work, not CNN training, not robotics,
and not a full-resolution Gross-Pitaevskii experiment. It is guided active
regime search on a cheap CPU GP evaluator.

`mean_pos_err` is a measured position error in physical coordinate units, not
grid-index units and not a lifetime proxy.

S0 still never certifies.

## Run

```bash
python -m chronos.k3.run_active_gp_search
python -m chronos.k3.run_active_gp_search --json
python -m chronos.k3.run_active_gp_search --memory chronos_memory/events.jsonl
python -m unittest chronos.k3.tests.test_active_gp_search
```
