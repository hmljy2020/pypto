# pypto4A8 integration gaps

This is the implementation backlog for the [workload plan](a8-plan.md), recorded
on 2026-11-11. The gaps concern trace semantics, target mapping, and performance
modeling. Reproducing a BlockISA decoder or scalar execution pipeline is outside
the agreed initial scope.

## Trace production and normalization

| Gap | Existing starting point | Required work |
| --- | ----------------------- | ------------- |
| Executable producer | PyPTO exposes early Tile IR; `pto-isa` provides an optional CPU reference | Implement target adaptation followed by a supported Tile IR interpreter; evaluate CPU instrumentation separately |
| Scalar/control evaluation | Loops, branches, offsets, and shape expressions remain around tile operations | Evaluate their values during recording; ignore their timing, not their effects |
| Pass boundary | `ConvertTensorToTileOps` already produces `Mat`/`Acc` and execution-role annotations; later passes add more specialization | Capture IR objects immediately after this pass, validate prerequisites, and freeze the target adaptation recipe |
| Operand completeness | CPU records include opcode, worker IDs, sequence, tile address/shape/layout/dtype, and scalar operands | Capture missing strides, backing identity, memory space, allocation extent versus valid extent, and template attributes |
| Dependencies | Input/output roles and IR effects provide a starting point | Preserve read-modify-write, aliasing, memory hazards, views, object generations, and actual synchronization |
| Trace identity | CPU addresses and per-worker sequence exist | Normalize to launch/thread/op/object identities; handle repeated launches and address reuse |
| Reproducibility | PyPTO and its runtime pin an external `pto-isa` revision | Record producer, pass recipe, revisions, input identity, profile and schema versions; explicitly select the CPU library when using instrumentation |

The local `pto-isa` checkout has `PTO_CPU_SIM_TRACE_MODE`, JSONL export, and
`ResetInstructionTrace()` / `DumpInstructionTraceJson()` APIs. Direct kernel
calls require explicit collection/export on the executing thread; they do not
automatically invoke the launch wrapper. Instrumentation is a useful producer,
not yet the complete workload contract. In particular, template arguments and
all synchronization effects cannot be assumed to appear in existing records.

PyPTO does not vendor `pto-isa` in its own source tree. Its runtime resolver uses
`runtime/pto_isa.pin`, requires a clean matching managed checkout, and exports
but does not read `PTO_ISA_ROOT` as an override. At review time the pin was
`f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7`; the inspected sibling checkout was
`49adb4a6`. Trace support in that sibling is not proof of support in the pinned
version. See `python/pypto/runtime/pto_isa.py` in the PyPTO source.

## Target adaptation before interpretation

These gaps belong to the main producer path; they must not be deferred to
ad-hoc trace cleanup or silently inferred by the timing model.

| Gap | Required closure |
| --- | ---------------- |
| Legal operation subset | Check dtype, shape, valid extents and semantic attributes; define supported decomposition and explicit rejection before recording |
| Early architecture bindings | Map `Mat`/`Acc`, matrix-bridge metadata and execution roles to the target profile; preserve user-visible semantics |
| Required data movement | Decide target storage transitions/layout conversions explicitly; distinguish metadata-only views from copies; do not blindly retain or delete Ascend staging |
| Functional completeness | Evaluate loop carries, branches and scalar expressions; execute tile values needed for data-dependent control; reject unsupported behavior |
| Dynamic dependency identity | Build per-instance producer/read/write relationships, logical address ranges, aliases and generations while interpreting; metadata-only operations still affect this state |
| Physical resource ownership | State storage classes and lifetime requirements in the workload; leave finite allocation timing, capacity stalls and issue scheduling to replay |
| Completeness claim | Validate a logical operation stream for one supported execution, not a fully encoded/allocated target program; require separate resource and communication profiles |

For the matmul, preserve 13 logical operations when the selected target profile
requires no extra transfers. If adaptation adds staging, derive the expected
count from that profile. The 21-operation Ascend trace is not the target's
mandatory sequence. Any CPU-instrumented cross-check must reconcile these
semantic differences before comparing operation counts or performance.

## superscalarNPU adaptation

One executable PTO tile operation is one scheduling unit. The remaining gaps
are the semantics and cost of those units, not missing block encodings.

| Gap | Decision or required behavior |
| --- | ----------------------------- |
| Operation coverage | Define supported load/store, matrix, vector, reduction, cast and move operations with dtype/layout/shape constraints; fail unsupported combinations |
| Tile storage mapping | Map PyPTO/PTO storage descriptions to the chosen PE storage profile; old Ascend L0/L1/UB names do not establish superscalarNPU storage equivalence |
| Dependencies and reuse | Track RAW dependencies and storage hazards; define renaming/reuse policy, preserve required WAR/WAW ordering, and account for allocation lifetime |
| Admission and scheduling | Specify finite window/queue sizes, admission width, issue arbitration, resource occupancy, completion and release; no detailed instruction decoder is required |
| Whole-operation timing | Define latency, initiation interval, throughput and port occupancy as functions of shape, dtype and layout; per-micro-op latency tables cannot be copied as whole-tile latency |
| Memory timing | Model transfer bytes, strides, partial tiles, bandwidth/latency, outstanding requests and contention at a declared abstraction level |
| Metadata versus work | Views and assignments may only change object metadata; distinguish these from actual copies, layout conversion and data movement |
| Architectural profile | Start with one single-thread PE; explicitly allocate compute/storage budgets rather than inheriting all four-PE assumptions |
| Multi-PE extension | Add per-thread streams, PE mapping, shared resources and event/barrier semantics; recording order across threads must not impose a global serialization |

The four-PE reference organization includes per-PE Cube/Vector execution
resources and shared structures. It does not imply multiplying WSE's stated
whole-core throughput by four. Scaling from one PE to several requires a
declared distribution of the aggregate compute and memory budgets.

Ignoring scalar timing makes the model suitable for studying tile scheduling
under that assumption; it does not establish full-core IPC or scalar-limited
kernel performance. A fixed trace also cannot predict a different execution
path caused by timing-dependent concurrency. Initially support deterministic,
synchronized workloads; reject unsupported polling/race-dependent behavior.

## WSE Calendar communication ABI

A normal PTO load/store supplies a data transfer but does not determine the
collective participants, physical node placement, route, Calendar service, or
receive completion protocol. These must come from explicit communication
semantics and a companion plan, not inferred from the load/store opcode alone.

The following names describe model events, not new machine instruction encodings.

| Gap | Required information or behavior |
| --- | -------------------------------- |
| Communication intent | Collective type, group, participants, root/source, reduction operation, logical transfer identity and per-node communication owner |
| Placement and routing | Logical thread/PE → WSE node mapping, topology, per-source `routeBits`, forwarding/landing semantics and route validation |
| Calendar plan binding | Bind logical key, `keyId`, `opcode`, route table and `CalReg` image to one versioned plan; an opcode alone cannot reconstruct the image |
| Geometry and addressing | Symmetric arena/backing, source `selfOff`, row count/bytes/strides, destination ranges, capacity, alignment and segmentation |
| Round identity | Match launch, opcode domain, epoch and segment/source identities; define reuse/drain rules and reject stale arrivals |
| Receive accounting | `expVal` counts unique, successfully committed remote payload bytes; exclude headers, padding, duplicates and local self-copy |
| Admission and alignment | Reserve source/destination storage and receive/completion capacity before group alignment; aligned nodes must have bounded send/receive service |
| Send and receive progress | Separate receive registration completion from data completion; final receive wait must not block send issue/progress; distinguish source-read completion, send drain and receive completion |
| Visibility and lifetime | Retain source and destination generations; publish destination readiness only after coverage, write visibility, local-copy obligations and receive accounting are satisfied |
| Local self-copy and reduction | State fabric self-delivery policy, local-copy work, reduction semantics and resource costs; local bytes must not accidentally satisfy remote `expVal` |
| Deployment and recovery | Validate compatible topology/ABI/plan versions; install `CalReg` in a quiescent window; drain in-flight responsibilities before epoch/backing reuse or reconfiguration |
| NoC timing | Implement Calendar release, link/port contention, forwarding/landing service, buffering, backpressure and clock-domain conversion; route volume accounting is insufficient |

The new WSE whitepaper requires **receive registration and resource admission
before alignment**. The old Calendar document describes alignment before MTE4
receive setup. Follow the new integration contract while preserving the route,
epoch and byte-accounting concepts; do not directly copy the old MTE/PIPE binding.

A conceptual event sequence is:

```text
producer ready + source retained
  -> post_receive + destination/resources reserved
  -> bounded send service ready
  -> group align
  -> send + optional local self-copy
  -> send drain and receive writes/accounting complete
  -> publish destination generation -> dependent compute
```

Actual overlap is expressed by dependencies, not a forced global sequential
timeline. Route/epoch scalar preparation may be resolved offline; resource
admission, alignment, data movement and visibility remain modeled events.

## Existing WSE model versus target design

- `src/wse_model/noc/network.py` explicitly models routing and byte volume,
  not network timing. `CalReg` content and service timing still need a model.
- Decision 0007 adopts superscalarNPU in the whitepaper without migrating the
  Python/ACIR implementation. Existing core and memory behavior is not proof
  that the new target is implemented.
- The whitepaper specifies local SRAM; the old model's `LocalDram` and 2 KiB
  page rules must not silently become new target constraints. Resolve local
  addressing/cache behavior versus the SSM unified-GM/coherence assumptions.
- The 40-node Calendar baseline uses 80-bit route maps; the 40/48-node topology
  question remains open. Keep topology and route width versioned together.
- WSE throughput and bandwidth numbers are aggregate target budgets, not
  calibrated SSM parameters. Keep unresolved target choices visible in profiles.

## Reviewed source baseline

These are local review baselines, not a promise that every checkout was clean
or that every specification was implemented. Freeze exact revisions and relevant
working-tree changes when generating reproducible workloads.

| Repository | Reviewed HEAD | Relevant files |
| ---------- | ------------- | -------------- |
| PyPTO | `705cb86f` | `python/pypto/runtime/pto_isa.py`, `src/ir/op/tile_ops/matmul.cpp`, `src/ir/op/tile_ops/transform.cpp` |
| PTO Tile Library (`pto-isa`) | `49adb4a6` | `include/pto/cpu/trace.hpp`, `include/pto/common/pto_instr.hpp`, `docs/coding/cpu_sim.md` |
| SuperScalarModel | `5341e55dd` | `modelSpec/superscalar_model_top_framework.md`, `TimingSim/pe/vec/VecValuInfo.h` |
| wse-model | `e9ecb1a` | `docs/design/wse-system-architecture-whitepaper.md`, `docs/design/wse-calendar-scheme.md`, decision 0007, `src/wse_model/noc/network.py` |

Here `pto-isa` means the CPU-capable Tile Library. The separate `pto-spec`
repository and SuperScalarModel's block instruction definitions are different
artifacts; sharing the PTO name does not make their interfaces interchangeable.
