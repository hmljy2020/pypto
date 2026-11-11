# pypto4A8 workload plan

Status: agreed modeling direction, implementation pending. Recorded on 2026-11-11.
The concrete file layout and trace fields below are proposals, not existing APIs.
See [integration gaps](a8-gaps.md) for remaining work and source references.

## Objective and agreed scope

Use PyPTO to generate workloads for a WSE system with superscalarNPU compute
cores. Develop in the PyPTO fork on branch `pypto4A8`; upstream PRs are not a
short-term objective. PyPTO owns workload generation, while the performance
model owns scheduling and timing.

The initial model accepts a saved stream of concrete PTO tile operations.
One executable tile operation is one scheduling/execution unit. Start with one
single-thread PE and model tile-level out-of-order execution. The input does
not require BlockISA decoding, a scalar pipeline, or internal micro-op splitting.
Multiple PEs and WSE Calendar communication are subsequent milestones.

Scalar **timing** is excluded. Scalar values still have to be computed when
they determine loop counts, branches, addresses, shapes, masks, or operation
attributes. Ignoring scalar timing does not permit ignoring these semantics.

## Generation and replay

```text
PyPTO operator + concrete inputs / shapes / scalar arguments
  -> PyPTO Tile IR immediately after ConvertTensorToTileOps
  -> target adaptation passes (legality, storage/movement, execution/sync)
  -> CPU Tile IR interpretation + recording
  -> normalize and validate PTO operation trace
  -> workload package
  -> tile-level superscalar replay
  -> WSE integration with a separate communication plan
```

The primary producer is an interpreter over target-adapted early Tile IR.
CPU instrumentation remains an optional producer and validation path:

| Producer | Approach | Trade-off and current decision |
| -------- | -------- | ------------------------------ |
| Tile IR interpreter | Adapt early Tile IR to our target, then evaluate control flow, scalar expressions and tile operations | Primary development path; record concrete operands, object identities and dependencies |
| PTO CPU instrumentation | Generate PTO, use PTOAS to generate C++, then execute against the `pto-isa` CPU implementation with tracing | Optional validation or alternative producer; requires normalization of target-specific staging, layout and storage reuse |

CPU instrumentation is an implementation alternative within the agreed
execute-record-replay approach. It does not change the model interface. The
PTO-to-C++ CPU path has not yet been demonstrated end to end for our generated
matmul; generating `.pto` with `skip_ptoas=True` does not validate that path.
It is not a prerequisite for the interpreter milestone. Both producers must
preserve the same declared target semantics before comparing their traces.

“Static trace” means a saved, parameter-resolved **dynamic execution path**,
not all possible executions of the program. Input-dependent branches require
the relevant data or a functional reference evaluator. Record the input identity
and execution assumptions. Timing-dependent races, polling, and adaptive behavior
cannot be generalized from one recorded run and are outside the initial scope.

During recording, the CPU completes the functional execution. During replay,
the simulator admits operations into a bounded window, waits for dependencies
and resources, issues ready operations, and computes completion times. CPU
timestamps and CPU completion order are not the simulated timing or a total
dependency order. Sequential recording must still expose independent operations.

## Selected IR boundary and target passes

Start immediately after `ConvertTensorToTileOps`, before subsequent target
storage/layout specialization. Traverse PyPTO IR objects directly: `passes_dump`
Python files are readable IR renderings, not MLIR converted to Python or the
interpreter's input format. No PTO MLIR generation is required on the main path.

This early boundary is not fully architecture-neutral. The inspected matmul
already has `Mat`/`Acc`, a compiler matrix-bridge attribute, and
`InCore`/`CHIP_DIE`/`SubWorker` annotations. Map or validate those explicitly;
do not assume they identify WSE nodes, PE placement, or target storage capacity.

Implement three logical adaptation stages; they need not be three separate passes:

| Stage | Responsibility | Initial scope |
| ----- | -------------- | ------------- |
| Operation legalization and support checks | Validate operations, dtype, shape and attributes; decompose operations only when required by the target | Load, store, matmul and accumulating matmul; reject unsupported combinations |
| Storage and movement adaptation | Map source storage requirements to the target profile; preserve aliases/views and insert required movement/layout conversion | Resolve `Mat`/`Acc`; do not inherit the Ascend `Mat -> Left/Right` staging path automatically |
| Execution identity and synchronization adaptation | Map execution scopes and retain explicit ordering/synchronization semantics | One logical thread on one PE; extend placement and Calendar semantics later |

The handoff between stages is explicit:

| Component | Decides | Does not precompute |
| --------- | ------- | ------------------- |
| Adaptation passes | Legal operations, storage classes, required transfers, execution scope and synchronization meaning | Issue times or an unlimited physical tile allocation |
| Interpreter | Concrete path, scalar values, logical addresses/views, dynamic object generations, operation instances and dependencies | Target performance or a global cross-thread timing order |
| Replay model | Finite resource allocation, admission, issue/completion, contention, storage release and backpressure | Missing operation semantics or unrecorded branch outcomes |

For example, whether a staging copy is required belongs to target adaptation;
when it can execute and overlap with matrix computation belongs to replay.
An initial profile without additional staging yields 13 matmul-example operations;
a profile requiring transfers must explicitly add them and update the expected count.

## Trace completeness boundary

The output is a complete **logical PTO operation stream for the supported subset,
given inputs and resolved execution path**, with object/memory metadata. It is
not a fully allocated or encoded machine instruction stream. `tile.matmul_acc`
can become a normalized `TMATMUL_ACC` record without passing through PTO MLIR.

Input bindings provide concrete tensor descriptors and logical address ranges;
physical placement is handled by the chosen target policy. Views that emit no
execution operation must still update alias metadata. Data-dependent branches
require actual functional tile results or a reference evaluator; recording only
operation names is insufficient. Unknown semantics must produce an error rather
than an apparently complete trace. Calendar plans and timing/resource profiles
remain additional inputs to the model.

## Proposed workload contract

```text
workload/
  manifest.json       # schema, toolchain revisions, inputs, target profile
  objects.json        # allocations, generations, views, aliases, memory spaces
  trace.jsonl         # operation instances, operands, attributes, dependencies
  communication.json  # optional logical collectives and node placement
  calendar.json       # optional routes, CalReg plan, ABI/version identity
```

The filenames and serialization are provisional. Trace identity must include
launch and logical thread, even when the initial workload has only one of each.
Required information includes:

- Unique operation instance and per-thread sequence; source operation identity.
- Opcode and all semantic attributes, including transpose, dtype conversion,
  rounding, accumulation, valid shape, and masks where applicable.
- Logical object, generation, view offset, shape, strides, layout, and dtype.
  Host pointer values alone are not persistent object identities.
- Read/write roles, producer dependencies, overlapping memory accesses,
  lifetime information, and explicit synchronization when present.
- Concrete scalar operands, with no scalar execution latency in the first model.
- Target and toolchain versions, pass recipe, input/seed identity, and declared
  unsupported features. Unsupported semantics must fail explicitly.

For example, a proposed normalized record could look like this. It illustrates
the contract; it is not the current `pto-isa` JSON schema:

```json
{
  "id": "launch0.thread0.op8",
  "thread": 0,
  "sequence": 8,
  "opcode": "TMATMUL_ACC",
  "reads": ["a1.v0", "b1.v0", "acc.v0"],
  "writes": ["acc.v1"],
  "depends_on": ["launch0.thread0.op6", "launch0.thread0.op7", "launch0.thread0.op2"],
  "attrs": {"m": 128, "n": 128, "k": 128, "acc_dtype": "fp32"}
}
```

The object table binds these names to storage and views. Logical versions
express dataflow; they do not imply unlimited physical tile storage. Replay
must account for allocation, aliasing, reuse, and capacity backpressure.

## First matmul workload

Use A `[128, 512]` FP16, B `[512, 128]` FP16, and C `[128, 128]` FP32,
with K split into four chunks of 128:

```text
load A0, B0 -> matmul     -> acc0
load A1, B1 -> matmul_acc(acc0) -> acc1
load A2, B2 -> matmul_acc(acc1) -> acc2
load A3, B3 -> matmul_acc(acc2) -> acc3
store acc3 -> C
```

At the early Tile IR boundary this is 8 loads, 1 matmul, 3 accumulating
matmuls, and 1 store: 13 executable tile operations. The inspected memory-space
lowering adds 8 staging moves, producing 21 operations at that later boundary.
The count depends on the pass recipe; these are not two interchangeable traces.
Select and record the boundary explicitly instead of parsing numbered dump names.
The observed 21-operation trace comes from Ascend lowering, not a required
superscalarNPU execution sequence.

The first replay must preserve the accumulator chain while permitting independent
loads to overlap within finite resource limits. Check numerical results with a
documented precision policy; an FP16-product CPU reference need not match a
full-FP32 reference bit for bit.

## Repository ownership and proposed layout

Keep PyPTO and `pto-isa` as separate repositories, with coordinated revisions.
The fork is [hmljy2020/pypto4A8](https://github.com/hmljy2020/pypto4A8), using
`pypto4A8` locally and remotely. Keep `origin` for the source repository and
`fork` for the personal fork. A `pto-isa` fork/branch is needed if its hooks change;
that is a follow-up task, not something this document claims is already done.

```text
python/pypto/workload/
  export.py                 # generation entry point
  pipeline.py               # early Tile IR boundary and target pass recipe
  transforms/               # target legality, storage/movement, execution/sync
  interpreter.py            # primary functional execution and trace producer
  cpu_trace.py              # optional instrumentation adapter / validation
  objects.py                # object identity, aliases, lifetimes
  trace.py                  # normalized trace contract
  validate.py               # supported subset and consistency checks
  targets/a8.py             # target mapping and workload profile
tests/ut/workload/           # semantic and schema checks
tests/st/workload/           # CPU generation/replay integration
docs/en/dev/workload/        # authoritative design and interface documents
docs/zh/dev/workload/        # synchronized Chinese documentation
```

Add a workload-specific toolchain configuration recording repository URLs,
commits, and explicit development paths. Do not edit the default resolver's
managed cache or assume `PTO_ISA_ROOT` selects a modified checkout. Preserve
the default runtime dependency contract and use an explicit CPU workload entry.
Include actual revisions and dirty-state identification in generated manifests.
Do not introduce a new compiler backend merely to export traces.

The `pto-isa` repository owns instruction instrumentation and CPU semantics.
PyPTO owns the normalized workload schema and generation documentation.
The model repository owns resource profiles, scheduling, and calibrated timing.

Keep temporary example code, compilation products, and review artifacts under
the enclosing workspace's `workspace/matmul-tile-ir/`, outside the PyPTO checkout.
This temporary directory may be cleaned and is not a reproducible fixture or
a tracked documentation dependency. Permanent implementation, tests and design
documents belong in the repository paths above.

## Delivery sequence and acceptance

1. Define the supported operation subset, trace schema, object model, and
   single-thread PE profile. Make unresolved parameters explicit.
2. Add the target adaptation stages at the early Tile IR boundary, then implement
   CPU interpretation for the supported scalar/control and tile subset. Validate
   required transfers and storage mappings before producing the trace.
3. Export and validate the matmul workload; check operation counts at the chosen
   pass boundary and target profile, addresses, accumulator dependencies, and
   numerical output. CPU instrumentation may later provide an independent check.
4. Connect a bounded tile-level replay; test dependency stalls, independent
   overlap, resource contention, tile capacity, and memory backpressure.
5. Extend to views, partial tiles, vector operations, and reuse; reject unsupported
   operations rather than silently dropping them.
6. Add multiple logical threads/PEs with explicit synchronization and shared
   resource accounting, then add Calendar communication and NoC timing.

Validation should cover in-place accumulation, overlapping views, reuse of an
address by a new generation, strided/tail transfers, and producer equivalence
if both producers are implemented. Calendar tests must additionally cover
duplicate/late segments, visibility before consumer issue, admission failure,
and receive/send progress. Numerical correctness alone does not validate timing.
