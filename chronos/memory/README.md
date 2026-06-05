# S0-M0: Memory Logging Layer

S0-M0 is an append-only audit trail for Chronos runs.

Current verdict:

```text
S0_M0_MEMORY_LOGGING_PASSED
```

It records:

- which module or experiment ran
- the verdict and S0 recommendation
- an optional score
- payload metadata
- the mandatory claim boundary

It does not learn, rank, select, update S0, or feed back into any
recommendation. A layer that uses memory to change behavior would be a separate
measured layer, not S0-M0.

## Boundary

Every `MemoryEvent` must include a non-empty `claim_boundary`. Recording a
success without stating what the result does and does not establish is rejected.

`allowed_action` mirrors S0's never-certify boundary:

- `continue`
- `archive`
- `do_not_promote`

No `certified` or `promote` action can be recorded.

## Run

```bash
python -m chronos.memory.run_memory_demo
python -m chronos.memory.run_memory_demo --json
python -m unittest chronos.memory.tests.test_memory_logging
```

The storage format is plain JSONL on disk. There is no ClickHouse, cloud
service, or server.

Detailed verdict:

- `S0_M0_MEMORY_LOGGING_PASSED.md`
