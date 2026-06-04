# S0-E2b Active Diagnostic Value Verdict

Final verdict:

```text
S0_E2B_ACTIVE_VALUE_PASSED
```

Scope:

```text
partitioned toy world -> active exploration reaches structure zone -> diagnostic probe -> S0 recommendation
```

Claim:

Active exploration has measured diagnostic value in a partitioned toy world:

- active exploration reliably reaches the far structure zone
- random action choice rarely reaches that zone under the fixed control seeds
- a probe launched from the active reached state sees the K2 symplectic signal
- a probe launched from the random reached state remains unresolved

Observed recommendation split:

```text
active -> K2_SYMPLECTIC / continue
random -> UNRESOLVED / do_not_promote
```

## What Is Real

The diagnostic answer depends on the reached state:

- near zone: dissipative probe, no symplectic gap, no K2 recommendation
- far zone: conservative probe, symplectic energy drift beats control, K2
  recommendation

This closes the S0-E2 gap: active exploration is not only wired into the
diagnostic loop, it is necessary for the correct diagnosis in this toy.

## Non-Claims

S0-E2b is not robotics.

It is not RL training, not a neural network, not online learning, and not
physics certification. Novelty remains a deterministic distance proxy.
