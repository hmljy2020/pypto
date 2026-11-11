# pypto4A8 workload plan

Design notes · 2026-11-11 · Implementation plan

The goal is to turn PyPTO operators into workloads for a WSE system with
superscalarNPU cores. The first milestone is a single-thread, single-PE model
that schedules and executes whole tile operations out of order.

## The chosen path

```text
PyPTO operator
  -> early Tile IR after ConvertTensorToTileOps
  -> target adaptation passes
  -> CPU interpretation with concrete inputs
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

The interpreter executes loops, branches and scalar expressions, evaluates tile
operations, and records concrete operation instances. It tracks their inputs,
outputs, accessed ranges and dependencies. Data-dependent control uses the
computed values. Scalar preparation contributes values; simulated time comes
from tile operations and modeled synchronization.

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

Development lives on the fork's `pypto4A8` branch. Planned producer code goes in
`python/pypto/workload/`, with tests in `tests/ut/workload/` and
`tests/st/workload/`. The model owns scheduling and timing. Temporary examples
and generated artifacts live in the enclosing workspace's
`workspace/matmul-tile-ir/` directory.

The concrete questions to resolve are collected in [integration gaps](a8-gaps.md).
