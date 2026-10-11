# NNML2 F16 rebuild transition v1

NNIS changes an already-materialized F16 execution plan by rebuilding a second
model. It does not mutate resident weights in place and does not claim a
zero-downtime hot switch.

## Transaction

`F16RebuildTransitionV1` executes one admitted transition in this order:

1. validate the source/target plans and the owned-allocation memory envelope;
2. close admission and quiesce all sessions;
3. require the original logical/source weights to remain available;
4. build the target candidate while the source model remains authoritative;
5. verify the candidate through the runtime-owned qualification path;
6. invalidate the old sessions and KV state;
7. commit the verified candidate; or roll back to the source on any failure.

The transaction is one-shot. A failed or completed transaction cannot be
replayed; callers must create a new transaction and repeat memory admission.

## Memory admission

`F16RebuildMemoryBudgetV1` checks:

```text
source_resident_owned_bytes + target_peak_owned_bytes
    <= transition_budget_bytes
```

Both inputs must come from NNIS-owned allocation evidence or a conservative
upper bound. This is an owned-allocation gate, not evidence of physical GPU page
residency, allocator overhead, process-wide VRAM, or system capacity.

## Runtime adapter contract

`F16RebuildTransitionRuntime` is implemented by the serving owner. It must:

- reject new sessions after quiescence begins;
- report zero active sessions before rebuild;
- retain source weights and the source model until commit succeeds;
- validate the rebuilt candidate before exposing it;
- invalidate sessions and KV before commit;
- make rollback idempotent and restore source admission.

The coordinator deliberately does not invent session or KV migration. Schema
v1 resets both, matching `F16ExecutionTransitionRequirementsV1`.

## Evidence and limits

A successful `F16RebuildTransitionReceiptV1` records the exact plans, required
double-resident owned bytes, budget headroom, candidate verification and
session/KV reset. Its `live_transition_claimed` field is always false and
validation rejects any contrary record.

Unit tests cover ordering, budget rejection before quiescence, residual active
sessions, missing source weights, rebuild/verification/reset/commit failures,
rollback and strict wire decoding. These software tests do not qualify latency,
availability, model quality, or physical-memory behavior on production
hardware.
