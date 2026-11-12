# SSNPU handwritten Tile IR and capture

Status: first workload implementation · 2026-11-12

`pypto.workload.ssnpu.Builder` constructs ordinary PyPTO Function, Call, TileType,
loop, branch and yield nodes with explicit SSNPU target metadata. Run the result
through `TileInterpreter.run()` and export it with `ExecutionResult.write_capture()`.
The two examples below are executable builders; no frontend parsing, automatic
partitioning, target lowering or timing model is involved.

See [plan.md](plan.md) for the existing interpreter/import history and
[gaps.md](gaps.md) for the remaining operator and timing integration work.

## Construction contract

| API | Meaning |
| --- | --- |
| `with ss.launch(name, blocks=2, threads=4) as fn` | Build a common thread entry; `fn.param` and `fn.return_type` declare its signature |
| `ss.block_id()`, `ss.thread_id()` | Independent logical block/thread indices; existing `subblock_id` remains zero |
| `cores=[...]`, `pe_mapping=[...]` on launch | Explicit placement; defaults are one Core per block and PE0–PE3 per block |
| `ss.shared_tile(shape, dtype)` | Block-owned shared handle declaration |
| `ss.load_into(handle, tensor, offsets, pes=[0])` | Single producer loads a complete tile; every executing thread binds the same immutable version |
| `ss.load_local(tensor, offsets, shape)` | Private thread-owned tile with the author's exact region |
| `ss.op(name, *operands, pes=None)` | One tile operation per participating thread, with optional explicit result shape/dtype |
| `reuse(value)` | Input-only retention qualifier; emits no instruction |
| Plain shared input | Last-use of that concrete shared version |
| `ss.barrier()` | Full-block rendezvous and completion of every preceding phase |
| `ss.for_loop`, `loop.iter_arg`, `loop.return_var`, `ss.yield_` | Existing explicit loop builder with carried values and yield |
| `ss.if_stmt`, `ss.function`, ordinary GlobalVar calls | Existing branches and helper functions |

`pes=None` (`ssnpu.ALL`) means all logical thread/PE indices in the current block.
The optional PE mapping translates those indices to physical placement. A participant
set never splits operands or results. For a branch executed by only some threads,
set the corresponding operation's `pes` explicitly; `full`, `load_local`, `op`, and
`store` accept it. Every declared participant must execute the matching dynamic site.
Different branch paths cannot silently join a shared consumption or barrier.

Tile types keep their memory space unspecified. Namespaced call attributes identify
`LocalTileReg` or `SharedTileReg`; Ascend Mat/Vec memory enums are rejected in this
context. Shared versions and transposed views retain block ownership; local tiles
cannot escape their task/block/thread. Generic arithmetic handlers receive calls
with target metadata removed, and still validate shapes, valid extents and dtypes.

The capture-only `ssnpu.*` operations use ordinary `Op`/`Call` nodes. They are not
registered for C++ lowering. `str(program)` produces readable IR including all
attributes; the printed `pl.ssnpu.*` spelling is an inspection format, not a new
Python DSL parser API. Attribute sequences use ordinary `MakeTuple` expressions,
so no C++ type or binding changes are required.

## Shared definitions, uses and barriers

A handle is identified by launch (`task_id`), block, declaration site and dynamic
call/loop path. Each `load_into` definition creates a separate immutable logical
version. Reusing a Python/IR variable name or writing the handle again never changes
an earlier value reference. Different versions use independent storage identities.

Capture may visit a consumer before the producer thread. Such consumers hold a
symbolic reference, and finalization patches producer links after all instances
have run. `TileInterpreter(thread_order=(3, 2, 1, 0))` exercises this case. Allocation
and operation IDs may differ, while normalized definitions, uses and dependencies
remain identical. Dynamic IDs are opaque stable strings for a particular IR;
changing its structure can change those IDs.

Every shared operand records `version_id`, `use_id`, `operand`, `participants` and
`retention` (`reuse` or `last-use`). All PE records for one consumption share a
`use_id`; operand indices distinguish multiple shared inputs. A transpose view is
only a descriptor operation: it keeps the source version and adds no lifetime
consumption. QK consumes the transposed K version itself.

Validation checks unique producers, consistent definitions, private ownership,
matching participants/versions/retention, barrier sites/rounds and reads after
last-use. Lifetime order comes from per-thread sequence and full-block phases,
never from global capture order. Reads on disjoint subsets need a barrier to
establish an order before a final consumption. Reuse-only versions may remain
live at launch completion; capture does not invent a last-use. Dependency cycles
caused by incompatible producer and synchronization order are rejected.

## Executable matmul

Two blocks each compute 256 rows, with two 32-row slices per PE. Each K iteration
has one shared B load, eight private A loads and eight matmul_acc operations.
The first consumption retains B; the second is its last-use.

```python
from pypto import DataType, ir
from pypto.workload import TensorSpec, TileInterpreter
from pypto.workload.ssnpu import Builder, reuse


def build_matmul():
    ss = Builder()
    with ss.launch("matmul", blocks=2) as fn:
        a = fn.param("A", ir.TensorType([512, 512], DataType.FP16))
        b = fn.param("B", ir.TensorType([512, 128], DataType.FP16))
        c = fn.param("C", ir.TensorType([512, 128], DataType.FP32))
        fn.return_type(c.type)
        bid, tid = ss.block_id(), ss.thread_id()
        row = ir.Add(
            ir.Mul(bid, ss.expr(256), DataType.INDEX, ir.Span.unknown()),
            ir.Mul(tid, ss.expr(32), DataType.INDEX, ir.Span.unknown()),
            DataType.INDEX,
            ir.Span.unknown(),
        )
        row1 = ir.Add(row, ss.expr(128), DataType.INDEX, ir.Span.unknown())
        shared = ss.shared_tile([128, 128], DataType.FP16)
        c0 = ss.full([32, 128], 0)
        c1 = ss.full([32, 128], 0)
        k = ss.var("k", ir.ScalarType(DataType.INDEX))
        with ss.for_loop(k, 0, 4, 1) as loop:
            acc0, acc1 = loop.iter_arg("acc0", c0), loop.iter_arg("acc1", c1)
            final0, final1 = loop.return_var("final0"), loop.return_var("final1")
            offset = ir.Mul(k, ss.expr(128), DataType.INDEX, ir.Span.unknown())
            bk = ss.load_into(shared, b, [offset, 0])
            a0 = ss.load_local(a, [row, offset], [32, 128])
            a1 = ss.load_local(a, [row1, offset], [32, 128])
            ss.barrier()
            out0 = ss.op("matmul_acc", acc0, a0, reuse(bk))
            out1 = ss.op("matmul_acc", acc1, a1, bk)
            ss.barrier()
            ss.yield_(out0, out1)
        ss.store(final0, c, [row, 0])
        ss.store(final1, c, [row1, 0])
        ss.return_stmt(c)
    return ir.Program([fn.get_result()], "shared_b", ir.Span.unknown()), {
        "A": TensorSpec((512, 512), DataType.FP16),
        "B": TensorSpec((512, 128), DataType.FP16),
        "C": TensorSpec((512, 128), DataType.FP32),
    }
```

Each block produces **36 loads, 32 matmul_acc operations, 8 stores and 8 tile
initializations**. There are 8 barrier rounds per block, each with 4 arrival events.
The two-block trace has 168 operations and 8 immutable B versions.

## Executable FA forward

Dense single-head attention uses Q/K/V `[256, 64]` FP16 and O `[256, 64]` FP32.
Each thread keeps 32 Q rows and private online-softmax state. Two KV chunks are
loaded by PE0; an explicit transpose view supplies Kᵀ. QK, reductions and online
accumulation use FP32; probabilities are cast to FP16 before PV. Scratch tiles for
row reductions are explicit operands required by the existing Tile IR handlers.

Using the imports above:

```python
def build_fa():
    ss = Builder()
    with ss.launch("fa", blocks=2) as fn:
        q, k, v = [
            fn.param(name, ir.TensorType([256, 64], DataType.FP16))
            for name in ("Q", "K", "V")
        ]
        output = fn.param("O", ir.TensorType([256, 64], DataType.FP32))
        fn.return_type(output.type)
        bid, tid = ss.block_id(), ss.thread_id()
        span = ir.Span.unknown()
        row = ir.Add(
            ir.Mul(bid, ss.expr(128), DataType.INDEX, span),
            ir.Mul(tid, ss.expr(32), DataType.INDEX, span),
            DataType.INDEX,
            span,
        )
        ks = ss.shared_tile([128, 64], DataType.FP16)
        vs = ss.shared_tile([128, 64], DataType.FP16)
        qi = ss.load_local(q, [row, 0], [32, 64])
        m = ss.full([32, 1], -float("inf"))
        normalizer = ss.full([32, 1], 0)
        o = ss.full([32, 64], 0)
        scratch = ss.full([32, 128], 0)
        j = ss.var("j", ir.ScalarType(DataType.INDEX))
        with ss.for_loop(j, 0, 2, 1) as loop:
            mi, li, oi = [
                loop.iter_arg(name, value)
                for name, value in zip(("m", "l", "o"), (m, normalizer, o))
            ]
            mf, lf, of = [loop.return_var(name) for name in ("mf", "lf", "of")]
            offset = ir.Mul(j, ss.expr(128), DataType.INDEX, span)
            kj = ss.load_into(ks, k, [offset, 0])
            vj = ss.load_into(vs, v, [offset, 0])
            ss.barrier()
            kt = ss.op("transpose_view", kj)
            scores = ss.op("muls", ss.op("matmul", qi, kt), 0.125)
            mn = ss.op("maximum", mi, ss.op("row_max", scores, scratch))
            alpha = ss.op("exp", ss.op("sub", mi, mn))
            p = ss.op("exp", ss.op("row_expand_sub", scores, mn))
            ln = ss.op("add", ss.op("mul", alpha, li), ss.op("row_sum", p, scratch))
            pv = ss.op("matmul", ss.op("cast", p, dtype=DataType.FP16), vj)
            on = ss.op("add", ss.op("row_expand_mul", oi, alpha), pv)
            ss.barrier()
            ss.yield_(mn, ln, on)
        out = ss.op("row_expand_div", of, lf)
        ss.store(out, output, [row, 0])
        ss.return_stmt(output)
    return ir.Program([fn.get_result()], "shared_kv", span), {
        **{name: TensorSpec((256, 64), DataType.FP16) for name in ("Q", "K", "V")},
        "O": TensorSpec((256, 64), DataType.FP32),
    }
```

Expected: **320 operation records**, including 32 QK/PV matmuls, 16 transpose views,
8 final normalizations and 8 stores. QK outputs `[32,128]`, PV and final outputs
are `[32,64]`, and normalization denominators are `[32,1]`.

## Reproduce and export

Run from the outer `hw-native-sys` root with the current checkout built and a Python
environment containing PyPTO's dependencies. Example scripts and all generated
artifacts live in `workspace/ssnpu-workload-design/`, which is temporary; permanent
regressions are in `pypto/tests/ut/workload/test_ssnpu.py`.

```bash
export PYTHONPATH="$PWD/pypto/python"
export PYTHONPYCACHEPREFIX="$PWD/workspace/ssnpu-workload-design/pycache"
python workspace/ssnpu-workload-design/matmul_ir_design.py
python workspace/ssnpu-workload-design/fa_ir_design.py
```

Outputs are `output/matmul/` and `output/fa/` below that workspace directory. Each
contains `ir.txt`, per-thread JSONL, `objects.json`, `tasks.json`, `events.jsonl`,
`capture_config.json` and `manifest.json`. Direct use is:

```python
program, inputs = build_matmul()
result = TileInterpreter().run(program, "matmul", inputs)
result.write_capture("workspace/ssnpu-workload-design/output/matmul")
```

For regressions, source `pypto/.claude/skills/testing/load-env.sh` from the PyPTO
checkout and pass `$PYPTO_TEST_JOBS` explicitly to pytest. Use absolute workspace
paths for pytest's cache and temporary directory options.

## Capture format and replay obligations

SSNPU captures use `format_version=2` and `ssnpu_spec_version=1`. Ordinary interpreter
captures retain format 1 and their existing filenames/behavior. The completion
manifest is still written last. `capture_trace=False` suppresses exports in memory
but does not disable semantic, participant, lifetime or barrier validation.

| Field | Contract |
| --- | --- |
| `task_id / block_id / subblock_id / thread_id` | Independent logical instance identity; task is the launch identity |
| `tasks[].instances[].mapping` | Physical `core_id` and `pe_id`, separate from logical identity |
| `ssnpu.dynamic_id / site / phase` | Dynamic operation grouping and barrier phase |
| `ssnpu.shared_inputs[]` | Operand-specific immutable version and reuse/last-use contract |
| Object `shared_handle / shared_version` | Declaration and definition identity; the ordinary per-storage version is 1 for immutable shared storage |
| `dependencies` | Producer/view and overlapping memory dependencies; prerequisite completion |
| `order_dependencies` | Previous operation in this thread; instruction admission/issue order |
| `sync_dependencies` | All preceding block-phase operations; prerequisite completion before starting this operation |
| `memory_accesses[]` | Storage/object ID, read/write, element offsets, valid rectangle shape, strides and storage offset; null range means conservative overlap |
| Barrier event `parameters` | Dynamic barrier ID, site, round, all participants and completion contract |
| Barrier `completion_dependencies` | Operations that must complete before the rendezvous releases |

Memory dependency analysis recognizes disjoint rectangles on an injective row-major
2D Tensor descriptor, so independent output shards do not acquire whole-Tensor
writer dependencies. Overlapping accesses with at least one writer retain RAW,
WAR and WAW ordering. Within a launch, otherwise unordered hazards use canonical
block/phase/thread order, independent of capture visitation. Unknown layouts/views
fall back to conservative overlap. Tensor snapshot versions remain capture bookkeeping;
replay uses `memory_accesses` and explicit dependencies for addressable memory.

On a producer record, the backend allocates version state and storage, then publishes
readiness when the transfer completes. Reuse keeps the version live. Last-use
completion alone is insufficient for reclamation: all its participants and all
relevant earlier reads must also finish, including reads not yet admitted into an
out-of-order window. A temporarily zero reference count must not free storage while
old reads remain unregistered. A barrier does not release a shared tile by itself.

This capture declares `shared_storage_policy=independent_versions`. An in-place
fixed-slot backend must separately add old-read-completion → new-write constraints;
it must not interpret a repeated handle as permission to overwrite a live version.
No fixed-slot annotation or automatic capacity allocation is implemented here.

Trace-only capture is the acceptance path. Existing numerical handlers remain
available; missing numerical implementations and unknown-payload control flow
raise explicit errors. End-to-end FA numerical validation, tails, masking,
automatic mapping/lowering, cycle estimates, and WSE communication are later work.

## Alignment references

Reviewed design baselines: PTO-spec `b0165320`, SuperScalarModel `637bde0ec`.
`asl/block/operands/B.IOS.asl` describes reuse/last-use masks and
`asl/block/model/state/source-lifetime.asl` tracks generation/use completion.
The timing model's `TimingSim/group/sharetreg_rat.h`,
`TimingSim/frontend/bctrl/BIssue.cpp` and `BROB.cpp` use reclamation at redefining
producer commit; adapters must reconcile that policy with the lifetime contract above.
