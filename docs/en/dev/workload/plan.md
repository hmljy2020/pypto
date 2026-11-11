# PyPTO workload plan

Design notes · 2026-11-11 · CPU interpreter implemented; timing replay planned

The goal is to turn PyPTO operators into workloads for a WSE system with
superscalarNPU cores. The first milestone is a single-thread, single-PE model
that schedules and executes whole tile operations out of order.

## The chosen path

```text
PyPTO operator
  -> early Tile IR after ConvertTensorToTileOps
  -> target adaptation passes
  -> CPU interpretation with input descriptions (optional payloads)
  -> PTO operation trace + object descriptions
  -> superscalarNPU timing replay
```

We start from the internal PyPTO IR objects immediately after
`ConvertTensorToTileOps`. At this point, tile computation, loads/stores and
control flow are explicit. The Python pass dumps provide a readable view of
these objects.

This IR already carries storage annotations such as `Mat` and `Acc`.
Our adaptation passes give them a precise meaning for the target PE.

## What each stage does

Target adaptation covers three tasks:

1. **Operation support:** define legal operations, dtypes, shapes and attributes,
   with decomposition rules where needed.
2. **Storage and movement:** map storage classes, retain views and aliases,
   and insert the transfers and layout conversions required by the target.
3. **Execution and synchronization:** map execution scopes to threads/PEs and
   preserve ordering and synchronization. Initially, all work uses one thread/PE.

The interpreter executes loops, branches and scalar expressions and describes tile
operations to record concrete operation instances. It tracks their inputs, outputs,
accessed ranges and dependencies without requiring tile payloads. Optional functional
execution computes numerical values. A payload-dependent branch requires functional
execution; trace-only execution reports the originating operation when a value is
unknown. Simulated time comes from tile operations and modeled synchronization.

The replay model reads the trace into a finite window. It chooses ready
operations according to dependencies and available resources, then models
execution, completion and storage release. Whether a transfer is required is
settled during adaptation; when it runs and overlaps with computation is settled
by replay.

## What the workload contains

The saved trace describes one execution for given inputs. Each operation carries
its opcode, concrete operands and attributes, launch/thread identity, and
producer dependencies. Object descriptions carry shape, dtype, stride, layout,
logical address range, aliases and generation identity.

The proposed package has three parts: `trace.jsonl`, `objects.json` and
`manifest.json`. The manifest records input identity, compiler revision,
adaptation recipe, target configuration and format version. Together they allow
the same workload to be replayed under different timing configurations.

## Start with matrix multiplication

Use FP16 A `[128, 512]` and B `[512, 128]`, producing FP32 C `[128, 128]`.
Split K into four chunks of 128:

```text
load A0, B0 -> matmul                  -> acc0
load A1, B1 -> matmul_acc(acc0, A1, B1) -> acc1
load A2, B2 -> matmul_acc(acc1, A2, B2) -> acc2
load A3, B3 -> matmul_acc(acc2, A3, B3) -> acc3
store acc3 -> C
```

The generated early Tile IR gives 13 operations after loop expansion:
8 loads, 4 matrix operations and 1 store. Target adaptation adds any required
transfers. This makes the operation count an explicit consequence of the chosen
storage configuration.

The first working example should produce correct numerical results and a trace
that preserves the accumulator chain while exposing independent loads for
overlap. Replay should show the effects of dependencies, bandwidth and finite
tile storage.

## Development order and locations

First define the trace/object format and a minimal PE storage configuration.
Then implement the adaptation and interpreter for load, store, matmul,
matmul_acc and their surrounding scalar/control expressions. Connect the matmul
trace to timing replay, then extend to views, partial tiles and vector operations.
Multi-PE execution and WSE Calendar communication follow this first working path.

Development lives on the fork's `dev_workload` branch. The CPU interpreter lives in
`python/pypto/workload/`, with tests in `tests/ut/workload/` and
`tests/st/workload/` (planned system coverage). The model owns scheduling and timing. Temporary examples
and generated artifacts live in the enclosing workspace's
`workspace/matmul-tile-ir/` directory.

The concrete questions to resolve are collected in [integration gaps](gaps.md).

## Available CPU interpreter

`pypto.workload.TileInterpreter` directly executes internal `ir.Program` objects.
It currently depends on CPU PyTorch but never launches a device. Trace capture is
independent of numerical execution:

```python
from pypto import DataType
from pypto.workload import TensorSpec, TileInterpreter

inputs = {
    "a": TensorSpec(shape=(128, 512), dtype=DataType.FP16),
    "b": TensorSpec(shape=(512, 128), dtype=DataType.FP16),
    "c": TensorSpec(shape=(128, 128), dtype=DataType.FP32),
}
interpreter = TileInterpreter(max_steps=1_000_000)
result = interpreter.run(
    program, entry="matmul_k_blocked", inputs=inputs,
    mode="trace_only", capture_trace=True,
)
outputs = result.outputs  # tuple of detached object descriptions / scalar values
result.write_trace("workspace/matmul-tile-ir/trace.jsonl")
```

`trace_only` is the default. It executes scalar/control flow and tile semantics,
updates object versions and dependencies, and never invokes numerical handlers or
allocates tile payloads. It accepts `TensorSpec` or CPU Tensor inputs, including a
mixture. Tensors contribute only metadata and remain unchanged, even after stores.
`TensorSpec` accepts a two-dimensional nonnegative shape, PyPTO FP16/FP32/BF16, BOOL or 8/16/32/64-bit signed/unsigned integer dtype and
optional nonnegative element strides; omitted strides describe contiguous storage.
Repeated use of the same input instance preserves shared identity; equal but distinct
specs represent separate inputs. Distinct Tensor views sharing storage are rejected.

`mode="functional"` requires actual CPU Tensor inputs. It uses the same semantic
handlers and effect processing, then calls numerical handlers. Its object outputs
remain Tensors and stores update destination inputs in place. `result.mode` records
the chosen mode. `capture_trace=False` disables recording in either mode without
changing execution, operation IDs, versions or producers. Each run resets objects,
steps and trace while preserving the instance's registry. Input keys match entry
parameter names or unambiguous names before the compiler's `__ssa_vN` suffix.

The interpreter supports scalar constants, arithmetic, comparisons, boolean short
circuiting, casts, tuples, assignments, function calls with independent variable
frames, ordinary/in-core/hierarchy scopes, `if`, sequential `for`/`while`, `yield`
and `return`. Frames use `Var.unique_id`. The step budget counts expressions,
statements and empty loop iterations.

The built-ins are `tile.load/store/move/matmul/matmul_acc`, using two-dimensional
descriptors. Functional payload support remains FP16/FP32. Their semantics validate parameters and describe accesses,
output shapes, valid rectangles, dtype and logical storage. Functional loads honor
actual PyTorch strides, copy only the valid rectangle and zero-fill tile padding.
Moves create independent copies. Stores update only the source's valid rectangle.
Matmul converts inputs to FP32 and accumulates in FP32; this is a functional reference,
not hardware bit-exact/TF32 emulation. Lhs valid K determines the product; rhs valid K
must cover it. `matmul_acc` mutates and versions its accumulator; `init_cond=True`
overwrites only the computed rectangle and omits the accumulator from its reads.
Overwrite dependencies still retain the previous writer. Integer test inputs compare
exactly; random inputs allow FP32 reduction-order differences.

The trace-only registry additionally describes `tile.create`, `add/sub/mul/maximum`,
`muls`, `exp`, `cast`, `row_max/row_sum` and `row_expand_sub/mul/div`. A declarative
operation table supplies arity, attribute support, operation family and PTO mapping;
a common handler reads output metadata from `call.type` and records effects. Reduction
scratch is a write, so reuse retains the previous writer dependency. `tile.create`
records a logical allocation with no PTO instruction mapping. These operations do
not yet provide numerical compute handlers; functional mode reports that limitation.

Additional descriptions cover `div/adds/abs/sqrt/log`, `row_min/col_sum`,
`col_expand_mul`, `full/cmps/sel/mgather` and `slice/reshape/transpose`. Binary
operations accept singleton-dimension broadcasting. Comparison results retain the
IR's packed UINT8 geometry, rather than pretending they are dense BOOL tiles.
`sel` records mask/value reads and UINT32 scratch writes. Vec `mgather` reads the
source tensor and INT32 index tile; absent index payloads leave its access region
explicitly `data_dependent`, with whole-object producer dependencies. This does not
supply concrete gathered addresses for timing replay. Mat gather remains unsupported.

Slice and logical reshape create descriptor views, not new storage.
Snapshots include `storage_id`, `storage_offset` (logical unpacked element units),
`view_of` and `view_producer`. Views keep distinct object IDs and shapes/strides but
share storage version and writer. Descriptor creation emits no PTO instruction or
storage write (`memory_access=False`); consumers depend on both view creation and
the latest storage writer. Writing multiple aliases in one operation advances the
shared storage version once. Current view writes conservatively invalidate the whole
allocation; byte-range hazard tracking is not implemented. Non-contiguous reshape
uses `stride=null` and an explicit `index_map` containing its source shape, stride,
offset and any preceding mapping. This composes logical row-major reshapes without
inventing an affine stride or allocating/copying tile payloads. A slice of such a
view composes another mapping and uses `storage_offset=null` when the offset is
not affine. Rank-dropping slices and separately supplied overlapping Tensor inputs
remain rejected.
Transpose is a real `pto.ttrans` operation with independent output storage and optional
scratch writes. The 14-probe `workspace/tile-trace-audit-more/audit_more.py` covers
these paths plus RMSNorm at both Tile checkpoints. These added handlers are trace-only;
functional mode still reports missing numerical implementations.

The next description set adds `relu/minimums/rsqrt/row_prod/col_max/row_argmax`,
`assemble/concat/cmp/transpose_view` and pure `row_expand/col_expand`. Column sum
accepts optional scratch; cast accepts saturation mode and optional scratch.
Index reductions produce INT32 and write their scratch. Binary operations validate
physical broadcasting but take output validity directly from the IR, preserving
differing input tail rectangles instead of treating them as broadcast dimensions.
`assemble` records a read/write insertion into target storage, updating existing
aliases; `concat` allocates independent output. `transpose_view` shares storage,
whereas `transpose` remains a real copy. Pure expansion marks its shape-template
operand with `shape_only_inputs` so it need not be interpreted as a payload read.

Tensor-to-Tile conversion materializes both vector expansions for `[1,N] op [M,1]`
before the binary operation. It keeps operand order and no longer misclassifies
cross broadcasting as row-expand arithmetic. The third workspace audit compiles
20 probes and captures both checkpoints, in addition to the preceding 21 probes.
Its C++ regression must use a rebuilt extension. On this workspace the incremental
build helper under `workspace/tile-trace-audit-round3/incremental-build/` rebuilds
the modified object using this checkout's Ninja command and links against this
checkout's unchanged objects, placing all new products and a Python package overlay
under that workspace directory. Set `PYTHONPATH` to its `python/` directory before
running the audit or tests; the default in-tree extension requires a normal rebuild.

Move accepts explicit `blayout/slayout` consistent with its output type, including
Cube-to-Vector layouts. Layout remains metadata rather than packed CPU storage.
Early matmul may consume Vec objects before storage inference inserts Cube staging;
the interpreter describes that IR without enforcing final device placement.
The workspace audit `workspace/tile-trace-audit/audit.py` compiles seven probes,
including vector fusion, softmax and two-block online-softmax FlashAttention, then
captures both early and storage-inferred checkpoints using pure TensorSpecs.

`RuntimeObject` stores shape, PyPTO dtype, element strides, valid shape, storage class,
layout, identity, version and producer independently of its optional `data` payload.
`describe()` returns detached JSON-compatible metadata. `Mat/Left/Right/Acc/Vec`
are logical storage classes with recorded implicit or explicit layouts; CPU payloads are unpacked.
Explicit tile strides/offsets, physical MemRefs, packed Tensor layouts, unsupported
attributes and parallel loops are rejected. The compiler's early Mat bridge and
call-direction metadata do not change these semantics; `mx_tensor_views_blocked`
needs no runtime action for plain ND inputs. Cache policies, phased/atomic stores
and FIXPIPE epilogues require future handlers.

Each event contains its zero-based ID, IR/PTO names, function/source location,
concrete args/kwargs, input/output snapshots and direct producer dependencies.
Load/store include Tensor ID and offsets. Initial inputs have version zero and no
producer; every operation write advances its version, including a new tile's first
write. These are single-thread dependencies, not a complete scheduling/alias model.
The execution engine owns IDs, versions and producers; the recorder only saves
snapshots. Identical supported IR and input descriptions produce identical traces
in both modes when both modes follow the same control-flow path. JSONL is the current
export; manifest/object sidecars and timing replay remain future work.

Extensions register two separate entry points:

```python
interpreter.register_op("tile.example", semantics=describe_example, compute=compute_example)
```

Semantics receives `(interpreter, call, evaluated_args)` and returns
`OpResult(value, reads, writes, pto_op, parameters, views=())`. It validates attributes, creates
descriptors with `interpreter.new_tile(call.type)` and reports object effects without
touching payloads. `interpreter.call_attributes(call)` supplies once-evaluated kwargs.
Compute receives `(interpreter, call, evaluated_args, result)` and fills or updates
payloads; it may replace `result.value` for a payload-dependent scalar. Use
`allocate_payload(obj)` for a new zero-filled tile and `payload(obj)` to require
existing data. Neither handler may mutate IDs, versions or producer links.
Compute is optional: semantics-only operations can be traced but fail in functional
mode. Custom names need not be built-in IR operations.

A scalar derived from absent tile data is represented explicitly by
`interpreter.unknown(call)`, an `UnknownValue` retaining source operation ID, name,
function and location. Arithmetic, comparisons and casts propagate it. Object shape,
stride and dtype remain known. Branch conditions, loop bounds, addresses, shapes,
scalar mask parameters and instruction arguments/attributes require concrete values and reject
unknowns with source and consumer context. Boolean expressions evaluate only the
required operand. Unknown scalar returns remain explicit unknown values.
Unregistered semantics and unsupported attributes fail in both modes; missing compute
or a required payload fails in functional mode with operation/object context.

A failed run raises `InterpreterError`; `interpreter.recorder.records` retains the
successfully recorded prefix for diagnosis. No `ExecutionResult` is returned for a
failed run. Export complete traces only from successful results. Earlier successful
results and their output descriptions remain detached from subsequent runs.

From the enclosing workspace, run the reproducible example:

```bash
PYTHONPATH=pypto/python .venv-ir/bin/python workspace/matmul-tile-ir/run_interpreter.py
```

It obtains `ConvertTensorToTileOps` and `InferTileMemorySpace` checkpoints from the
default pipeline, captures 13/21 events using pure TensorSpecs, verifies identical
functional traces and exact integer-input numerical results, and exports JSONL under
`workspace/matmul-tile-ir/interpreter-output/`. Unit coverage in
`tests/ut/workload/test_interpreter.py` also checks random-input accuracy, zero tile
payload allocation in trace-only mode, unchanged Tensor inputs, unknown-value errors,
short circuiting, partial/non-contiguous accesses, snapshots and registry isolation.
