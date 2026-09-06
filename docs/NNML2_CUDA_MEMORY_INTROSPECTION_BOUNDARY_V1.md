# NNML2 CUDA memory introspection boundary v1

## Audit result

NNIS currently exposes `cuMemGetInfo` through the CUDA Driver API for memory introspection. That call reports free and total device memory for the current context/device environment. It is useful as a global telemetry snapshot, but it does not attribute memory to one NNIS allocation graph and it does not prove physical page residency of individual `cuMemAlloc` allocations.

The CUDA Driver API also defines pointer/allocation metadata surfaces such as address-range and pointer attributes. Those surfaces can describe allocation boundaries or pointer properties. They do not, for the ordinary `cuMemAlloc` weight allocations used by NNIS, establish a general per-allocation physical-residency measurement.

Managed-memory range attributes have different semantics and do not justify relabeling NNIS ordinary device allocations as physically resident evidence.

## NNIS rule

Until NNIS has a backend primitive and qualification procedure whose semantics directly support physical residency or attributable process-wide GPU memory, contracts must use narrower names:

- exact owned allocation bytes for NNIS-owned `DeviceBuffer` graphs;
- scoped allocation-lifetime peaks during a controlled operation; or
- explicitly global free/total memory snapshots when such telemetry is added.

A free/total delta must not be presented as exact ownership attribution without controlling and recording all other allocation sources in the process/device environment.

## Current non-claims

PRs #119, #120 and #121 therefore remain allocation-ownership/lifetime evidence, not physical-residency evidence. The failed-materialization evidence built on top of those contracts likewise records the NNIS-owned allocation state observed at failure detection and makes no physical-residency or process-wide-VRAM claim.
