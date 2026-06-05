# K3-E2d Discriminating GP Truth-Only Active Search Verdict

Final verdict:

```text
GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED
```

Scope:

```text
cheap real-GP active regime search with continuous vortex-position transport
```

K3-E2d changes the K3-E2c measurement paradigm. Instead of scoring only
discrete phase-winding charge survival, it tracks phase-singularity positions
and uses a continuous transport score measuring how closely the +1/-1 pair
stays near its intended positions.

## Result

Admission criteria for active diagnostic value all pass:

- random success count is at most `5/20`
- active reaches `transport_ok`
- active best score beats the random median by more than `SCORE_MARGIN`

The observed route is:

```text
active -> K3_TOPOLOGICAL / continue
random -> K3_TOPOLOGICAL / do_not_promote
```

## Boundary

This is not K3 prior validation.

It is not proof that topological priors work, not CNN training, not robotics,
and not a full-resolution Gross-Pitaevskii run. It is guided active regime
search on a cheap CPU GP evaluator.

The metric is position-based vortex transport. `mean_pos_err` is reported in
physical coordinate units, not grid-index units.

S0 still never certifies.

## Run

```bash
python -m chronos.k3.run_active_gp_search
python -m chronos.k3.run_active_gp_search --json
python -m chronos.k3.run_active_gp_search --memory chronos_memory/events.jsonl
python -m unittest chronos.k3.tests.test_active_gp_search
```
