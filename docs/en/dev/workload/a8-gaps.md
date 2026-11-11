# pypto4A8 integration questions

Design notes · 2026-11-11

The [workload plan](a8-plan.md) is organized around target adaptation, CPU
interpretation and timing replay. The following questions guide implementation.

## 1. Define the target meaning of early Tile IR

The matmul IR already contains `Mat`/`Acc` storage and execution-role annotations.
We need a minimal PE configuration specifying where inputs, accumulators and
outputs live, which storage classes each operation accepts, and which transfers
or layout conversions connect them.

The first adaptation should produce a traceable mapping for load, store, matmul
and matmul_acc, including supported dtypes, shapes and valid extents. This also
fixes the expected operation count for the first workload.

## 2. Make interpretation preserve dataflow

The interpreter needs scalar arithmetic, loops, loop-carried values, branches
and tile functional evaluation. When a branch reads a tile result, that value
must be available during recording.

The main bookkeeping issue is object identity. A view shares storage with its
source; accumulation reads and writes related state; a reused address can hold
a new object generation. The trace must retain these relationships and the
actual accessed ranges. The first tests should cover accumulation, overlapping
views, strided/tail accesses and address reuse.

## 3. Turn tile operations into timing work

The PE profile must specify scheduling-window and queue sizes, issue rules,
Cube/Vector/memory resources, tile storage capacity and release policy.
Operation costs need shape- and dtype-dependent latency, initiation interval,
transfer bytes and bandwidth consumption.

Replay combines producer dependencies with hazards from overlapping storage.
A small set of examples should demonstrate independent operation overlap,
accumulator stalls, memory contention and capacity backpressure. The numerical
reference and timing checks provide complementary validation.

## 4. Extend from one PE to several

Multiple PEs require per-thread streams, placement, explicit synchronization
and a shared-resource budget. Replay derives cross-thread ordering from data
and synchronization dependencies. Initial workloads should have deterministic,
synchronized execution paths.

WSE's compute and bandwidth figures are whole-core budgets. The configuration
must state how those budgets are distributed across PEs and which resources
remain shared.

## 5. Give WSE communication an explicit contract

A Calendar transfer needs a communication description alongside the tile trace:

| Part | Information to settle |
| ---- | --------------------- |
| Participants and placement | Collective kind, group, source/root, reduction, node mapping and one communication owner per node |
| Routing and service | Per-source `routeBits`, logical key, `keyId`, `opcode`, and the matching `CalReg` image and version |
| Storage geometry | Symmetric arena, source `selfOff`, rows, row bytes, strides, destination ranges and capacity |
| Transfer identity | Launch, epoch, source/segment identity, duplicate detection and drain rules for reuse |
| Completion | Unique committed remote payload bytes counted toward `expVal`, local self-copy coverage, write visibility and destination publication |

The current whitepaper's sequence is: prepare source data, register reception
and reserve resources, establish bounded send service, align participants,
send, then collect drain/write-completion evidence and publish the destination.
Registration and data arrival have separate completion events, allowing send
progress while reception is pending. Source and destination generations remain
valid through their respective transfer obligations.

The adapter must define these events and their dependencies, as well as plan
installation and reconfiguration at a drained boundary. A small collective
example should verify routing, byte accounting, repeated epochs and consumer
readiness.

## 6. Connect Calendar to a timed WSE model

The existing `wse-model` NoC implementation provides routing and byte accounting.
Performance replay needs Calendar release times, link/port service, buffering,
backpressure, receive-write bandwidth and clock-domain conversion.

Several target choices also need to be fixed in the configuration: the 40/48-node
topology and route width, local SRAM addressing/banks/ports, and its relationship
to the core's memory model. Whitepaper decision 0007 establishes the
superscalarNPU direction; migrating the executable model is an integration task.

## Working references

Use PyPTO's `ConvertTensorToTileOps` output and operator definitions for workload
semantics; SuperScalarModel's `modelSpec/superscalar_model_top_framework.md` for
PE/resource organization; and wse-model's
`docs/design/wse-system-architecture-whitepaper.md`, decision 0007 and
`src/wse_model/noc/network.py` for the communication contract and current model.
Record the revisions used by each experiment in its manifest.
