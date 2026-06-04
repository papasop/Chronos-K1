# S0-E2 Active Toy Exploration Verdict

Final verdict:

```text
S0_E2_ACTIVE_TOY_PASSED
```

Scope:

```text
active toy exploration -> diagnostic probe -> S0 recommendation
```

Claim:

A deterministic novelty-driven toy explorer can choose actions that cover more
of the pendulum toy state space than a random-action control, then launch a K2
diagnostic probe from an actively reached state.

The S0 recommendation from that measured diagnostic is:

```text
K2_SYMPLECTIC / continue
```

## What Is Real

The active-to-diagnostic loop is real in this toy:

- active exploration selects actions by state novelty
- active exploration covers more coarse state cells than random action choice
- the K2 diagnostic probe starts from an active reached state
- the symplectic energy drift is measured against an explicit-Euler control

## What Is Proxy

Novelty is RND-lite:

```text
candidate next-state distance from visited states
```

No neural network is trained or distilled.

## Non-Claims

S0-E2 is not robotics.

It is not RL training, not a neural network, not online learning, and not
physics certification. It only checks the computer-side active loop:

```text
agent chooses action by novelty -> toy world responds -> diagnostic probe -> S0 recommendation
```
