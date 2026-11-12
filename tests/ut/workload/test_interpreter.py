# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Functional Tile IR, control flow, extension and trace contract tests."""

import json

import pypto.language as pl
import pytest
import torch
from pypto import DataType, backend, ir
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.op import system_ops
from pypto.workload import InterpreterError, OpResult, TensorSpec, TileInterpreter, UnknownValue

S = ir.Span("interpreter_test.py", 1, 1)
INDEX = ir.ScalarType(DataType.INDEX)


def const(value):
    return ir.ConstInt(value, DataType.INDEX, S)


def program(body, params=(), returns=(INDEX,), name="main"):
    return ir.Program([ir.Function(name, list(params), returns, body, S)], "test", S)


def ret(*values):
    return ir.ReturnStmt(list(values), S)


def seq(*stmts):
    return ir.SeqStmts(list(stmts), S)


def test_identity_tuple_and_calls():
    x, y = ir.Var("same", INDEX, S), ir.Var("same", INDEX, S)
    parameter = ir.Var("same", INDEX, S)
    helper = ir.Function(
        "helper", [parameter], [INDEX], ret(ir.Add(parameter, const(3), DataType.INDEX, S)), S
    )
    call = ir.Call(ir.GlobalVar("helper"), [y], INDEX, S)
    pair = ir.MakeTuple([x, call], S)
    body = seq(
        ir.AssignStmt(x, const(11), S),
        ir.AssignStmt(y, const(17), S),
        ret(ir.TupleGetItemExpr(pair, 0, S), ir.TupleGetItemExpr(pair, 1, S)),
    )
    prog = ir.Program([helper, ir.Function("main", [], [INDEX, INDEX], body, S)], "test", S)
    assert TileInterpreter().run(prog, "main", {}).outputs == (11, 20)


@pytest.mark.parametrize("stop,expected", [(0, 7), (3, 10)])
def test_for_carried_values(stop, expected):
    index = ir.Var("i", INDEX, S)
    carry = ir.IterArg("acc", INDEX, const(7), S)
    out = ir.Var("out", INDEX, S)
    body = ir.YieldStmt([ir.Add(carry, const(1), DataType.INDEX, S)], S)
    loop = ir.ForStmt(index, const(0), const(stop), const(1), [carry], body, [out], S)
    assert TileInterpreter().run(program(seq(loop, ret(out))), "main", {}).outputs == (expected,)


@pytest.mark.parametrize("condition,expected", [(True, 4), (False, 9)])
def test_branch_yield(condition, expected):
    out = ir.Var("out", INDEX, S)
    branch = ir.IfStmt(
        ir.ConstBool(condition, S), ir.YieldStmt([const(4)], S), ir.YieldStmt([const(9)], S), [out], S
    )
    assert TileInterpreter().run(program(seq(branch, ret(out))), "main", {}).outputs == (expected,)


def test_while_and_budget():
    carry = ir.IterArg("i", INDEX, const(0), S)
    out = ir.Var("out", INDEX, S)
    loop = ir.WhileStmt(
        ir.Lt(carry, const(4), DataType.BOOL, S),
        [carry],
        ir.YieldStmt([ir.Add(carry, const(1), DataType.INDEX, S)], S),
        [out],
        S,
    )
    prog = program(seq(loop, ret(out)))
    assert TileInterpreter().run(prog, "main", {}).outputs == (4,)
    with pytest.raises(InterpreterError, match="step limit"):
        TileInterpreter(max_steps=10).run(prog, "main", {})
    endless = ir.WhileStmt(ir.ConstBool(True, S), [], seq(), [], S)
    with pytest.raises(InterpreterError, match="step limit"):
        TileInterpreter(max_steps=10).run(program(endless, returns=()), "main", {})


@pl.program
class BlockedMatmul:
    @pl.function
    def main(
        self,
        a: pl.Tensor[[128, 512], pl.FP16],
        b: pl.Tensor[[512, 128], pl.FP16],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul"):
            acc = pl.matmul(a[:, 0:128], b[0:128, :], out_dtype=pl.FP32)
            for k in pl.range(1, 4):
                k0 = k * 128
                acc = pl.matmul_acc(acc, a[:, k0 : k0 + 128], b[k0 : k0 + 128, :])
            c[:] = acc
        return c


@pytest.fixture(scope="module")
def stages():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    prog = BlockedMatmul
    found = {}
    for compiler_pass in PassManager.get_strategy(OptimizationStrategy.Default).passes:
        prog = compiler_pass(prog)
        name = compiler_pass.get_name()
        if name in ("ConvertTensorToTileOps", "InferTileMemorySpace"):
            found[name] = prog
        if name == "InferTileMemorySpace":
            break
    assert len(found) == 2
    yield found
    backend.reset_for_testing()


@pytest.mark.parametrize("random", [False, True])
def test_matmul_checkpoints(stages, random, tmp_path):
    generator = torch.Generator().manual_seed(19)
    if random:
        a = torch.randn((128, 512), generator=generator).half()
        b = torch.randn((512, 128), generator=generator).half()
    else:
        a = torch.randint(-2, 3, (128, 512), generator=generator).half()
        b = torch.randint(-2, 3, (512, 128), generator=generator).half()
    expected = a.float() @ b.float()
    interpreter = TileInterpreter()
    outputs = []
    for name, prog in stages.items():
        c = torch.zeros_like(expected)
        result = interpreter.run(prog, "main", {"a": a, "b": b, "c": c}, mode="functional")
        assert result.outputs[0] is c
        torch.testing.assert_close(c, expected, rtol=2e-5 if random else 0, atol=1e-4 if random else 0)
        outputs.append(c.clone())
        count = 13 if name == "ConvertTensorToTileOps" else 21
        assert len(result.trace) == count
        assert [r["op_id"] for r in result.trace] == list(range(count))
        loads = [r for r in result.trace if r["ir_op"] == "tile.load"]
        assert [r["parameters"]["offsets"] for r in loads] == [
            [0, 0],
            [0, 0],
            [0, 128],
            [128, 0],
            [0, 256],
            [256, 0],
            [0, 384],
            [384, 0],
        ]
        accs = [r for r in result.trace if r["ir_op"].startswith("tile.matmul")]
        assert [r["outputs"][0]["version"] for r in accs] == [1, 2, 3, 4]
        assert len({r["outputs"][0]["object_id"] for r in accs}) == 1
        for previous, current in zip(accs, accs[1:]):
            assert previous["op_id"] in current["dependencies"]
        path = tmp_path / f"{name}.jsonl"
        result.write_trace(path)
        assert [json.loads(line) for line in path.read_text().splitlines()] == result.trace
        muted = interpreter.run(
            prog, "main", {"a": a, "b": b, "c": torch.zeros_like(c)}, capture_trace=False, mode="functional"
        )
        assert muted.trace == []
        torch.testing.assert_close(muted.outputs[0], c, rtol=0, atol=0)
        assert accs[0]["outputs"][0]["version"] == 1
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def memory_program():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src, dst = ir.Var("src", typ, S), ir.Var("dst", typ, S)
    load = ir.op.tile.load(src, [2, 1], [4, 4], [2, 3], target_memory=ir.MemorySpace.Mat)
    tile = ir.Var("tile", load.type, S)
    move = ir.op.tile.move(tile, target_memory=ir.MemorySpace.Left)
    moved = ir.Var("moved", move.type, S)
    store = ir.op.tile.store(tile, [0, 0], dst)
    body = seq(
        ir.AssignStmt(tile, load, S),
        ir.AssignStmt(moved, move, S),
        ir.EvalStmt(store, S),
        ret(tile, moved, dst),
    )
    return program(body, [src, dst], [tile.type, moved.type, typ])


def test_noncontiguous_tail_copy_and_store():
    src = torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2]
    dst = torch.full((4, 4), -1.0)
    result = TileInterpreter().run(memory_program(), "main", {"src": src, "dst": dst}, mode="functional")
    tile, moved, returned = result.outputs
    assert returned is dst
    torch.testing.assert_close(tile[:2, :3], src[2:, 1:])
    assert torch.count_nonzero(tile[2:]) == 0
    assert torch.count_nonzero(tile[:, 3:]) == 0
    assert torch.equal(dst[:2, :3], src[2:, 1:])
    assert torch.all(dst[2:] == -1)
    saved = moved.clone()
    tile.fill_(99)
    torch.testing.assert_close(moved, saved)
    assert src[2, 1] != 99
    assert result.trace[0]["inputs"][0]["stride"] == [8, 2]
    assert result.trace[1]["outputs"][0]["memory_space"] == "Left"
    assert result.trace[2]["outputs"][0]["version"] == 1


@pytest.mark.parametrize("init", [False, True])
def test_accumulator_init(init):
    typ = ir.TensorType([2, 2], DataType.FP32)
    src = ir.Var("src", typ, S)
    load = ir.op.tile.load(src, [0, 0], [2, 2], target_memory=ir.MemorySpace.Mat)
    tile = ir.Var("tile", load.type, S)
    product = ir.op.tile.matmul(tile, tile)
    acc = ir.Var("acc", product.type, S)
    accumulate = ir.op.tile.matmul_acc(acc, tile, tile, init_cond=ir.ConstBool(init, S))
    prog = program(
        seq(ir.AssignStmt(tile, load, S), ir.AssignStmt(acc, product, S), ret(accumulate)), [src], [acc.type]
    )
    result = TileInterpreter().run(prog, "main", {"src": torch.ones(2, 2)}, mode="functional")
    torch.testing.assert_close(result.outputs[0], torch.full((2, 2), 2.0 if init else 4.0))
    assert result.trace[-1]["outputs"][0]["version"] == 2
    metadata = TileInterpreter().run(prog, "main", {"src": TensorSpec((2, 2), DataType.FP32)})
    assert metadata.trace == result.trace


def test_extension_and_error_context():
    call = ir.Call(ir.Op("test.double"), [const(6)], INDEX, S)
    prog = program(ret(call))
    interpreter = TileInterpreter()
    with pytest.raises(InterpreterError, match="test.double in main at.*interpreter_test.py"):
        interpreter.run(prog, "main", {})
    interpreter.register_op(
        "test.double", semantics=lambda ctx, node, args: OpResult(args[0] * 2, parameters={"factor": 2})
    )
    result = interpreter.run(prog, "main", {})
    assert result.outputs == (12,)
    assert result.trace[0]["parameters"]["factor"] == 2
    assert result.trace[0]["op_id"] == 0
    with pytest.raises(InterpreterError, match="No registered semantics"):
        TileInterpreter().run(prog, "main", {})


def test_reject_attributes_and_dtype():
    typ = ir.TensorType([2, 2], DataType.FP32)
    src = ir.Var("src", typ, S)
    call = ir.op.tile.load(src, [0, 0], [2, 2], cache=1)
    prog = program(ret(call), [src], [call.type])
    with pytest.raises(InterpreterError, match="tile.load in main.*Unsupported attributes.*cache"):
        TileInterpreter().run(prog, "main", {"src": torch.ones(2, 2)})
    with pytest.raises(InterpreterError, match="dtype"):
        TileInterpreter().run(prog, "main", {"src": torch.ones(2, 2).double()})


def test_extension_object_effects_and_snapshot():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    loaded = ir.op.tile.load(src, [0, 0], [4, 4], target_memory=ir.MemorySpace.Mat)
    tile = ir.Var("tile", loaded.type, S)
    call = ir.Call(ir.Op("test.increment"), [tile], tile.type, S)
    prog = program(seq(ir.AssignStmt(tile, loaded, S), ret(call)), [src], [tile.type])

    def increment(ctx, node, args):
        obj = args[0]
        return OpResult(obj, (obj,), (obj,), "test.increment")

    interpreter = TileInterpreter()

    def compute_increment(ctx, node, args, result):
        ctx.payload(args[0]).add_(1)

    interpreter.register_op("test.increment", semantics=increment, compute=compute_increment)
    source = torch.ones(4, 4)
    result = interpreter.run(prog, "main", {"src": source}, mode="functional")
    assert torch.all(result.outputs[0] == 2)
    assert torch.all(source == 1)
    first, second = result.trace
    assert first["outputs"][0]["version"] == second["inputs"][0]["version"] == 1
    assert second["outputs"][0]["version"] == 2
    assert second["dependencies"] == [0]


def test_dynamic_load_bounds_and_clamp():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    offset = ir.Var("offset", INDEX, S)
    for clamp in (False, True):
        call = ir.op.tile.load(src, [offset, 0], [4, 4], clamp=clamp)
        prog = program(ret(call), [src, offset], [call.type])
        inputs = {"src": torch.ones(4, 4), "offset": 2}
        if clamp:
            result = TileInterpreter().run(prog, "main", inputs, mode="functional")
            assert result.trace[0]["outputs"][0]["valid_shape"] == [2, 4]
            assert torch.all(result.outputs[0][:2] == 1)
            assert torch.all(result.outputs[0][2:] == 0)
        else:
            with pytest.raises(InterpreterError, match="tile.load.*exceeds shape"):
                TileInterpreter().run(prog, "main", inputs)


def test_reject_layout_and_aliasing():
    typ = ir.TensorType([4, 4], DataType.FP32, tensor_view=ir.TensorView(layout=ir.TensorLayout.NZ))
    src = ir.Var("src", typ, S)
    with pytest.raises(InterpreterError, match="layout"):
        TileInterpreter().run(program(ret(src), [src], [typ]), "main", {"src": torch.ones(4, 4)})
    data = torch.ones(4, 4)
    with pytest.raises(InterpreterError, match="sharing storage"):
        TileInterpreter().run(memory_program(), "main", {"src": data, "dst": data.view(4, 4)})


def test_scalar_boolean_short_circuit_and_early_return():
    missing = ir.Var("unbound", ir.ScalarType(DataType.BOOL), S)
    condition = ir.Or(ir.ConstBool(True, S), missing, DataType.BOOL, S)
    branch = ir.IfStmt(condition, ret(const(8)), ret(const(0)), [], S)
    assert TileInterpreter().run(program(seq(branch, ret(const(99)))), "main", {}).outputs == (8,)


def test_nested_call_restores_frame():
    shared = ir.Var("shared", INDEX, S)
    helper = ir.Function("helper", [shared], [INDEX], ret(shared), S)
    call = ir.Call(ir.GlobalVar("helper"), [const(3)], INDEX, S)
    main = ir.Function("main", [shared], [INDEX, INDEX], ret(call, shared), S)
    prog = ir.Program([helper, main], "frames", S)
    assert TileInterpreter().run(prog, "main", {"shared": 9}).outputs == (3, 9)


def test_empty_while_and_descending_for():
    carry = ir.IterArg("carry", INDEX, const(10), S)
    out = ir.Var("out", INDEX, S)
    loop = ir.WhileStmt(ir.ConstBool(False, S), [carry], ir.YieldStmt([const(0)], S), [out], S)
    assert TileInterpreter().run(program(seq(loop, ret(out))), "main", {}).outputs == (10,)
    index = ir.Var("index", INDEX, S)
    body = ir.YieldStmt([ir.Add(carry, index, DataType.INDEX, S)], S)
    loop = ir.ForStmt(index, const(3), const(0), const(-1), [carry], body, [out], S)
    assert TileInterpreter().run(program(seq(loop, ret(out))), "main", {}).outputs == (16,)


@pytest.mark.parametrize("capture", [False, True])
def test_metadata_only_checkpoints(stages, monkeypatch, capture, tmp_path):
    inputs = {
        "a": torch.randn(128, 512).half(),
        "b": torch.randn(512, 128).half(),
        "c": torch.full((128, 128), -17.0),
    }
    original = {name: data.clone() for name, data in inputs.items()}
    specs = {
        name: TensorSpec(tuple(data.shape), DataType.FP32 if name == "c" else DataType.FP16)
        for name, data in inputs.items()
    }
    interpreter = TileInterpreter()
    for name, prog in stages.items():
        functional = interpreter.run(
            prog, "main", {k: v.clone() for k, v in inputs.items()}, mode="functional", capture_trace=capture
        )
        functional_snapshots = [obj.describe() for obj in interpreter.objects.objects]

        def forbidden(*args, **kwargs):
            pytest.fail("trace_only invoked numerical work or allocated a tile payload")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "zeros", forbidden)
            for op, (semantics, _) in interpreter.registry.items():
                patch.setitem(interpreter.registry, op, (semantics, forbidden))
            metadata = interpreter.run(prog, "main", specs, capture_trace=capture)
            snapshots = [obj.describe() for obj in interpreter.objects.objects]
            assert snapshots == functional_snapshots
            assert all(obj.data is None for obj in interpreter.objects.objects)
            tensors = interpreter.run(prog, "main", inputs, capture_trace=capture)
            assert all(obj.data is None for obj in interpreter.objects.objects)
            assert snapshots == [obj.describe() for obj in interpreter.objects.objects]
        assert metadata.mode == tensors.mode == "trace_only"
        assert functional.mode == "functional"
        assert metadata.trace == tensors.trace == functional.trace
        assert metadata.outputs == tensors.outputs
        assert metadata.outputs[0]["version"] == 1
        assert metadata.outputs[0]["shape"] == [128, 128]
        assert len(metadata.trace) == ((13 if name == "ConvertTensorToTileOps" else 21) if capture else 0)
        for key, data in inputs.items():
            torch.testing.assert_close(data, original[key], rtol=0, atol=0)
        path = tmp_path / f"{name}.jsonl"
        metadata.write_trace(path)
        assert [json.loads(line) for line in path.read_text().splitlines()] == metadata.trace
        # Later runs and runtime/return mutations cannot change saved trace snapshots.
        interpreter.objects.objects[-1].layout["changed"] = [1]
        tensors.outputs[0]["shape"][0] = 999
        assert metadata.outputs[0]["shape"] == [128, 128]
        assert all("changed" not in obj["layout"] for event in metadata.trace for obj in event["outputs"])


@pytest.mark.parametrize("as_spec", [False, True])
def test_input_identity_and_strides(as_spec):
    data = torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2]
    value = TensorSpec((4, 4), DataType.FP32, (8, 2)) if as_spec else data
    result = TileInterpreter().run(memory_program(), "main", {"src": value, "dst": value})
    load, move, store = result.trace
    assert load["inputs"][0]["stride"] == [8, 2]
    assert load["inputs"][0]["object_id"] == store["outputs"][0]["object_id"] == 0
    assert load["inputs"][0]["version"] == 0
    assert store["outputs"][0]["version"] == 1
    assert result.outputs[-1]["object_id"] == 0
    assert result.outputs[0]["valid_shape"] == [2, 3]
    assert result.outputs[1]["memory_space"] == "Left"
    assert move["dependencies"] == [0]
    torch.testing.assert_close(data, torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2])


def test_distinct_equal_specs_have_distinct_identity():
    result = TileInterpreter().run(
        memory_program(),
        "main",
        {
            "src": TensorSpec((4, 4), DataType.FP32),
            "dst": TensorSpec((4, 4), DataType.FP32),
        },
    )
    assert result.trace[0]["inputs"][0]["object_id"] != result.trace[-1]["outputs"][0]["object_id"]


@pytest.mark.parametrize("mode", ["trace_only", "functional"])
def test_semantics_only_extension(mode):
    typ = ir.TileType([2, 2], DataType.FP32)
    call = ir.Call(ir.Op("test.describe"), [], typ, S)
    interpreter = TileInterpreter()
    interpreter.register_op(
        "test.describe",
        semantics=lambda ctx, node, args: OpResult(
            (out := ctx.new_tile(node.type)), writes=(out,), pto_op="pto.example"
        ),
    )
    if mode == "functional":
        with pytest.raises(
            InterpreterError, match="test.describe in main at.*Missing numerical implementation"
        ):
            interpreter.run(program(ret(call), returns=[typ]), "main", {}, mode=mode)
        assert interpreter.recorder.records == []
    else:
        result = interpreter.run(program(ret(call), returns=[typ]), "main", {}, mode=mode)
        assert result.outputs[0]["shape"] == [2, 2]
        assert result.trace[0]["pto_op"] == "pto.example"
        assert result.outputs[0]["version"] == 1


def scalar_read_setup():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    load = ir.op.tile.load(src, [0, 0], [4, 4], target_memory=ir.MemorySpace.Mat)
    tile = ir.Var("tile", load.type, S)
    call = ir.Call(ir.Op("test.read_scalar"), [tile], INDEX, S)
    scalar = ir.Var("scalar", INDEX, S)
    prefix = [ir.AssignStmt(tile, load, S), ir.AssignStmt(scalar, call, S)]
    interpreter = TileInterpreter()

    def describe(ctx, node, args):
        return OpResult(ctx.unknown(node), reads=(args[0],))

    def compute(ctx, node, args, result):
        result.value = int(ctx.payload(args[0])[0, 0].item())

    interpreter.register_op("test.read_scalar", semantics=describe, compute=compute)
    return interpreter, src, tile, scalar, prefix


@pytest.mark.parametrize("use", ["if", "while", "for", "offset", "attribute", "shape", "mask", "init_cond"])
def test_unknown_requires_concrete_at_consumer(use):
    interpreter, src, tile, scalar, prefix = scalar_read_setup()
    condition = ir.Gt(ir.Add(scalar, const(1), DataType.INDEX, S), const(0), DataType.BOOL, S)
    if use == "if":
        body = ir.IfStmt(condition, ret(const(1)), ret(const(0)), [], S)
    elif use == "while":
        body = ir.WhileStmt(condition, [], seq(), [], S)
    elif use == "for":
        body = ir.ForStmt(ir.Var("i", INDEX, S), const(0), scalar, const(1), [], seq(), [], S)
    elif use == "offset":
        body = ir.EvalStmt(ir.op.tile.store(tile, [scalar, 0], src), S)
    elif use == "init_cond":
        product = ir.op.tile.matmul(tile, tile)
        acc = ir.Var("acc", product.type, S)
        prefix.append(ir.AssignStmt(acc, product, S))
        body = ir.EvalStmt(ir.op.tile.matmul_acc(acc, tile, tile, init_cond=condition), S)
    else:
        typ = ir.TileType([scalar, const(4)], DataType.FP32) if use == "shape" else INDEX
        kwargs = {use: condition} if use != "shape" else {}
        call = ir.Call(ir.Op("test.consumer"), [], kwargs, typ, S)
        interpreter.register_op(
            "test.consumer",
            semantics=lambda ctx, node, args: OpResult(ctx.new_tile(node.type) if use == "shape" else 0),
        )
        body = ir.EvalStmt(call, S)
    prog = program(seq(*prefix, body, ret(const(0))), [src])
    with pytest.raises(InterpreterError, match="unknown payload value.*test.read_scalar.*main at") as exc:
        interpreter.run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)})
    assert "interpreter_test.py" in str(exc.value)
    assert "concrete value" in str(exc.value)
    # A failed consumer never becomes a successful trace event.
    assert [e["ir_op"] for e in interpreter.recorder.records] == (
        ["tile.load", "test.read_scalar", "tile.matmul"]
        if use == "init_cond"
        else ["tile.load", "test.read_scalar"]
    )


def test_unknown_propagation_and_functional_scalar():
    interpreter, src, _, scalar, prefix = scalar_read_setup()
    value = ir.Cast(ir.Neg(ir.Add(scalar, const(2), DataType.INDEX, S), DataType.INDEX, S), DataType.FP32, S)
    prog = program(seq(*prefix, ret(value)), [src], [ir.ScalarType(DataType.FP32)])
    trace_only = interpreter.run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)})
    assert isinstance(trace_only.outputs[0], UnknownValue)
    assert "test.read_scalar" in trace_only.outputs[0].sources[0]
    functional = interpreter.run(prog, "main", {"src": torch.ones(4, 4)}, mode="functional")
    assert functional.outputs == (-3.0,)
    assert trace_only.trace == functional.trace


@pytest.mark.parametrize("op,left,expected", [(ir.And, False, False), (ir.Or, True, True)])
def test_unknown_short_circuit(op, left, expected):
    interpreter, src, _, scalar, prefix = scalar_read_setup()
    condition = op(ir.ConstBool(left, S), ir.Gt(scalar, const(0), DataType.BOOL, S), DataType.BOOL, S)
    prog = program(seq(*prefix, ret(condition)), [src], [ir.ScalarType(DataType.BOOL)])
    assert interpreter.run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)}).outputs == (expected,)
    # The skipped expression must not even invoke an unregistered operation.
    missing = ir.Call(ir.Op("test.unregistered"), [], ir.ScalarType(DataType.BOOL), S)
    short = op(ir.ConstBool(left, S), missing, DataType.BOOL, S)
    assert TileInterpreter().run(program(ret(short)), "main", {}).trace == []


def test_known_metadata_drives_control_flow():
    interpreter, src, tile, _, prefix = scalar_read_setup()
    shape = ir.Call(ir.Op("test.rows"), [tile], INDEX, S)
    interpreter.register_op("test.rows", semantics=lambda ctx, node, args: OpResult(args[0].shape[0]))
    branch = ir.IfStmt(ir.Eq(shape, const(4), DataType.BOOL, S), ret(const(8)), ret(const(0)), [], S)
    result = interpreter.run(
        program(seq(prefix[0], branch), [src]), "main", {"src": TensorSpec((4, 4), DataType.FP32)}
    )
    assert result.outputs == (8,)
    assert all(obj.data is None for obj in interpreter.objects.objects)


def test_functional_missing_input_and_intermediate_payload():
    interpreter, src, tile, _, prefix = scalar_read_setup()
    prog = program(seq(prefix[0], ret(tile)), [src], [tile.type])
    with pytest.raises(InterpreterError, match="input src consumed by main.*Missing payload"):
        interpreter.run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)}, mode="functional")
    # A custom producer that neglects its payload is rejected at its numerical consumer.
    producer = ir.Call(ir.Op("test.empty_payload"), [], tile.type, S)
    interpreter.register_op(
        "test.empty_payload",
        semantics=lambda ctx, node, args: OpResult((out := ctx.new_tile(node.type)), writes=(out,)),
        compute=lambda ctx, node, args, result: None,
    )
    move = ir.op.tile.move(tile, target_memory=ir.MemorySpace.Left)
    prog = program(seq(ir.AssignStmt(tile, producer, S), ret(move)), returns=[move.type])
    with pytest.raises(InterpreterError, match="tile.move in main.*Missing payload for tile object 0"):
        interpreter.run(prog, "main", {}, mode="functional")
    assert len(interpreter.recorder.records) == 1


@pytest.mark.parametrize("mode", ["trace_only", "functional"])
def test_unsupported_semantics_and_attributes_both_modes(mode):
    call = ir.Call(ir.Op("test.missing"), [], INDEX, S)
    with pytest.raises(InterpreterError, match="test.missing in main.*No registered semantics"):
        TileInterpreter().run(program(ret(call)), "main", {}, mode=mode)
    typ = ir.TensorType([2, 2], DataType.FP32)
    src = ir.Var("src", typ, S)
    call = ir.op.tile.load(src, [0, 0], [2, 2], cache=1)
    with pytest.raises(InterpreterError, match="tile.load in main.*Unsupported attributes.*cache"):
        TileInterpreter().run(
            program(ret(call), [src], [call.type]), "main", {"src": torch.ones(2, 2)}, mode=mode
        )


@pytest.mark.parametrize(
    "shape,dtype,stride",
    [
        ((4,), DataType.FP32, None),
        ((4, -1), DataType.FP32, None),
        ((4, 4), DataType.INT4, None),
        ((4, 4), DataType.FP32, (1,)),
        ((4, 4), DataType.FP32, (-1, 1)),
    ],
)
def test_tensor_spec_validation(shape, dtype, stride):
    with pytest.raises(ValueError):
        TensorSpec(shape, dtype, stride)


def test_mode_validation():
    with pytest.raises(ValueError, match="Expected mode"):
        TileInterpreter().run(program(ret(const(0))), "main", {}, mode="guess")


@pytest.mark.parametrize("mode", ["trace_only", "functional"])
def test_attributes_evaluate_once(mode):
    interpreter = TileInterpreter()
    attr = ir.Call(ir.Op("test.attribute"), [], ir.ScalarType(DataType.BOOL), S)
    interpreter.register_op(
        "test.attribute",
        semantics=lambda ctx, node, args: OpResult(True),
        compute=lambda ctx, node, args, result: None,
    )
    typ = ir.TensorType([2, 2], DataType.FP32)
    src = ir.Var("src", typ, S)
    load = ir.op.tile.load(src, [0, 0], [2, 2])
    load = ir.Call(load.op, load.args, {"clamp": attr}, load.type, S)
    result = interpreter.run(
        program(ret(load), [src], [load.type]), "main", {"src": torch.ones(2, 2)}, mode=mode
    )
    assert [event["ir_op"] for event in result.trace] == ["test.attribute", "tile.load"]
    assert result.trace[-1]["parameters"]["kwargs"] == {"clamp": True}


@pytest.mark.parametrize("op,left", [(ir.And, True), (ir.Or, False)])
def test_unknown_boolean_required_operand(op, left):
    interpreter, src, _, scalar, prefix = scalar_read_setup()
    condition = op(ir.ConstBool(left, S), ir.Gt(scalar, const(0), DataType.BOOL, S), DataType.BOOL, S)
    prog = program(seq(*prefix, ret(condition)), [src], [ir.ScalarType(DataType.BOOL)])
    with pytest.raises(InterpreterError, match="boolean condition in main.*test.read_scalar"):
        interpreter.run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)})
    assert len(interpreter.recorder.records) == 2
    recovered = interpreter.run(program(ret(const(7))), "main", {})
    assert recovered.outputs == (7,)
    assert recovered.trace == []


@pytest.mark.parametrize("mode", ["trace_only", "functional"])
@pytest.mark.parametrize("dtype,expected", [(DataType.BOOL, True), (DataType.INT32, 3), (DataType.FP32, 3.5)])
def test_scalar_cast(mode, dtype, expected):
    cast = ir.Cast(ir.ConstFloat(3.5, DataType.FP32, S), dtype, S)
    result = TileInterpreter().run(program(ret(cast), returns=[ir.ScalarType(dtype)]), "main", {}, mode=mode)
    assert result.outputs == (expected,)
    assert type(result.outputs[0]) is type(expected)


@pytest.mark.parametrize(
    "name",
    [
        "create",
        "add",
        "sub",
        "mul",
        "maximum",
        "muls",
        "exp",
        "cast",
        "row_max",
        "row_sum",
        "row_expand_sub",
        "row_expand_mul",
        "row_expand_div",
    ],
)
def test_declared_tile_semantics(name):
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    load = ir.op.tile.load(src, [0, 0], [4, 4], target_memory=ir.MemorySpace.Vec)
    tile = ir.Var("tile", load.type, S)
    kwargs = {}
    out_type = load.type
    prefix = [ir.AssignStmt(tile, load, S)]
    if name == "create":
        args = [ir.MakeTuple([const(4), const(4)], S)]
        kwargs = {"dtype": DataType.FP32, "target_memory": ir.MemorySpace.Vec}
    elif name in {"row_max", "row_sum"}:
        scratch_call = ir.op.tile.create([4, 4], DataType.FP32, target_memory=ir.MemorySpace.Vec)
        scratch = ir.Var("scratch", scratch_call.type, S)
        prefix.append(ir.AssignStmt(scratch, scratch_call, S))
        args = [tile, scratch]
        out_type = ir.TileType([4, 1], DataType.FP32, memory_space=ir.MemorySpace.Vec)
    elif name.startswith("row_expand"):
        small = ir.op.tile.load(src, [0, 0], [4, 1], target_memory=ir.MemorySpace.Vec)
        args = [tile, small]
    elif name == "muls":
        args = [tile, ir.ConstFloat(0.5, DataType.FP32, S)]
    elif name == "cast":
        args = [tile]
        kwargs = {"target_type": DataType.FP16, "mode": "round"}
        out_type = ir.TileType([4, 4], DataType.FP16, memory_space=ir.MemorySpace.Vec)
    else:
        args = [tile] if name == "exp" else [tile, tile]
    call = ir.Call(ir.get_op(f"tile.{name}"), args, kwargs, out_type, S)
    prog = program(seq(*prefix, ret(call)), [src], [out_type])
    interpreter = TileInterpreter()
    result = interpreter.run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)})
    event = result.trace[-1]
    assert event["ir_op"] == f"tile.{name}"
    assert all(obj.data is None for obj in interpreter.objects.objects)
    assert result.outputs[0]["shape"] == ([4, 1] if name in {"row_max", "row_sum"} else [4, 4])
    assert result.outputs[0]["version"] == 1
    if name in {"row_max", "row_sum"}:
        assert len(event["inputs"]) == 1  # scratch's previous contents are not read
        assert len(event["outputs"]) == 2
        assert event["outputs"][0]["version"] == 2  # create, then scratch use
        assert event["dependencies"] == [0, 1]
    elif name == "create":
        assert event["pto_op"] is None  # allocation is not a PTO instruction
        assert event["inputs"] == []
    with pytest.raises(InterpreterError, match="Missing numerical implementation"):
        interpreter.run(prog, "main", {"src": torch.ones(4, 4)}, mode="functional")


def test_move_explicit_layout_and_early_vector_matmul():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    load = ir.op.tile.load(src, [0, 0], [4, 4], target_memory=ir.MemorySpace.Mat)
    tile = ir.Var("tile", load.type, S)
    moved = ir.op.tile.move(
        tile,
        target_memory=ir.MemorySpace.Vec,
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.none_box,
    )
    vector = ir.Var("vector", moved.type, S)
    product = ir.op.tile.matmul(vector, vector)
    prog = program(
        seq(ir.AssignStmt(tile, load, S), ir.AssignStmt(vector, moved, S), ret(product)),
        [src],
        [product.type],
    )
    interpreter = TileInterpreter()
    metadata = interpreter.run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)})
    functional = interpreter.run(prog, "main", {"src": torch.ones(4, 4)}, mode="functional")
    assert metadata.trace == functional.trace
    assert metadata.trace[1]["outputs"][0]["layout"]["blayout"] == str(ir.TileLayout.col_major)
    assert metadata.trace[2]["dependencies"] == [1]
    torch.testing.assert_close(functional.outputs[0], torch.full((4, 4), 4.0))


def test_declared_semantics_reject_malformed_operands():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    lhs = ir.op.tile.load(src, [0, 0], [4, 4])
    rhs = ir.op.tile.load(src, [0, 0], [4, 3])
    call = ir.Call(ir.get_op("tile.add"), [lhs, rhs], lhs.type, S)
    with pytest.raises(InterpreterError, match="Expected broadcast-compatible shapes"):
        TileInterpreter().run(
            program(ret(call), [src], [call.type]), "main", {"src": TensorSpec((4, 4), DataType.FP32)}
        )


@pytest.mark.parametrize(
    "dtype,label",
    [
        (DataType.BOOL, "bool"),
        (DataType.INT8, "int8"),
        (DataType.INT16, "int16"),
        (DataType.INT32, "int32"),
        (DataType.INT64, "int64"),
        (DataType.UINT8, "uint8"),
        (DataType.UINT16, "uint16"),
        (DataType.UINT32, "uint32"),
        (DataType.UINT64, "uint64"),
        (DataType.BF16, "bfloat16"),
    ],
)
def test_metadata_dtypes(dtype, label):
    typ = ir.TensorType([4, 4], dtype)
    src = ir.Var("src", typ, S)
    loaded = ir.op.tile.load(src, [0, 0], [4, 4])
    interpreter = TileInterpreter()
    result = interpreter.run(
        program(ret(loaded), [src], [loaded.type]), "main", {"src": TensorSpec((4, 4), dtype)}
    )
    assert result.outputs[0]["dtype"] == label
    assert all(obj.data is None for obj in interpreter.objects.objects)


@pytest.mark.parametrize(
    "dtype,torch_dtype",
    [(DataType.INT32, torch.int32), (DataType.BOOL, torch.bool), (DataType.BF16, torch.bfloat16)],
)
def test_actual_metadata_tensor_and_functional_dtype_limit(dtype, torch_dtype):
    typ = ir.TensorType([4, 4], dtype)
    src = ir.Var("src", typ, S)
    prog = program(ret(src), [src], [typ])
    data = torch.ones((4, 4), dtype=torch_dtype)
    interpreter = TileInterpreter()
    metadata = interpreter.run(prog, "main", {"src": data})
    assert metadata.outputs == interpreter.run(prog, "main", {"src": TensorSpec((4, 4), dtype)}).outputs
    assert torch.all(data == 1)
    with pytest.raises(InterpreterError, match="Functional payload dtype.*not supported"):
        interpreter.run(prog, "main", {"src": data}, mode="functional")


@pytest.mark.parametrize("capture", [False, True])
def test_view_storage_versions_and_nested_offsets(capture):
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    load = ir.op.tile.load(src, [0, 0], [4, 4])
    base = ir.Var("base", load.type, S)
    slice_call = ir.op.tile.slice(base, [2, 2], [1, 1])
    view = ir.Var("view", slice_call.type, S)
    nested_call = ir.op.tile.slice(view, [1, 2], [1, 0])
    nested = ir.Var("nested", nested_call.type, S)
    mutate = ir.Call(ir.Op("test.write_aliases"), [base, nested], base.type, S)
    prog = program(
        seq(
            ir.AssignStmt(base, load, S),
            ir.AssignStmt(view, slice_call, S),
            ir.AssignStmt(nested, nested_call, S),
            ir.EvalStmt(mutate, S),
            ret(base, view, nested),
        ),
        [src],
        [base.type, view.type, nested.type],
    )
    interpreter = TileInterpreter()
    interpreter.register_op(
        "test.write_aliases",
        semantics=lambda ctx, node, args: OpResult(args[0], reads=(args[1],), writes=args),
    )
    result = interpreter.run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)}, capture_trace=capture)
    base_desc, view_desc, nested_desc = result.outputs
    assert len({o["object_id"] for o in result.outputs}) == 3
    assert len({o["storage_id"] for o in result.outputs}) == 1
    assert all(o["version"] == 2 and o["producer"] == 3 for o in result.outputs)
    assert view_desc["storage_offset"] == 5
    assert nested_desc["storage_offset"] == 9
    assert view_desc["stride"] == nested_desc["stride"] == [4, 1]
    assert nested_desc["view_of"] == view_desc["object_id"]
    if capture:
        assert result.trace[1]["outputs"][0]["version"] == 1  # view creation does not write bytes
        assert result.trace[1]["pto_op"] is None
        assert result.trace[2]["dependencies"] == [0, 1]
        assert result.trace[3]["dependencies"] == [0, 2]
    else:
        assert result.trace == []


def test_reshape_alias_and_transpose_copy_identity():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    load = ir.op.tile.load(src, [0, 0], [4, 4])
    base = ir.Var("base", load.type, S)
    reshape = ir.op.tile.reshape(base, [2, 8])
    transpose = ir.op.tile.transpose(base, 0, 1)
    prog = program(
        seq(ir.AssignStmt(base, load, S), ret(base, reshape, transpose)),
        [src],
        [base.type, reshape.type, transpose.type],
    )
    result = TileInterpreter().run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)})
    original, view, copied = result.outputs
    assert original["storage_id"] == view["storage_id"] != copied["storage_id"]
    assert view["stride"] == [8, 1] and view["view_producer"] == 1
    assert copied["view_of"] is None
    assert result.trace[-1]["pto_op"] == "pto.ttrans"


@pytest.mark.parametrize("rhs_shape", [(1, 4), (4, 1), (1, 1)])
def test_binary_broadcast_descriptor(rhs_shape):
    lhs_type = ir.TensorType([4, 4], DataType.FP32)
    rhs_type = ir.TensorType(list(rhs_shape), DataType.FP32)
    a, b = ir.Var("a", lhs_type, S), ir.Var("b", rhs_type, S)
    lhs = ir.op.tile.load(a, [0, 0], [4, 4])
    rhs = ir.op.tile.load(b, [0, 0], list(rhs_shape))
    mul = ir.Call(ir.get_op("tile.mul"), [lhs, rhs], lhs.type, S)
    result = TileInterpreter().run(
        program(ret(mul), [a, b], [mul.type]),
        "main",
        {"a": TensorSpec((4, 4), DataType.FP32), "b": TensorSpec(rhs_shape, DataType.FP32)},
    )
    assert result.outputs[0]["shape"] == [4, 4]
    assert result.trace[-1]["dependencies"] == [0, 1]


def test_gather_keeps_data_dependent_address_description():
    data_type = ir.TensorType([4, 4], DataType.FP32)
    idx_type = ir.TensorType([1, 4], DataType.INT32)
    a, idx = ir.Var("a", data_type, S), ir.Var("idx", idx_type, S)
    indices = ir.op.tile.load(idx, [0, 0], [1, 4], target_memory=ir.MemorySpace.Vec)
    out_type = ir.TileType([1, 4], DataType.FP32, memory_space=ir.MemorySpace.Vec)
    gather = ir.Call(ir.get_op("tile.mgather"), [a, indices], {"coalesce": 1}, out_type, S)
    interpreter = TileInterpreter()
    result = interpreter.run(
        program(ret(gather), [a, idx], [out_type]),
        "main",
        {"a": TensorSpec((4, 4), DataType.FP32), "idx": TensorSpec((1, 4), DataType.INT32)},
    )
    event = result.trace[-1]
    assert event["parameters"]["access_region"] == "data_dependent"
    assert event["parameters"]["index_object_id"] == result.trace[0]["outputs"][0]["object_id"]
    assert event["dependencies"] == [0]
    assert len(event["inputs"]) == 2
    assert all(obj.data is None for obj in interpreter.objects.objects)


def test_packed_mask_and_selection_effects():
    typ = ir.TensorType([4, 64], DataType.FP32)
    src = ir.Var("src", typ, S)
    load = ir.op.tile.load(src, [0, 0], [4, 64], target_memory=ir.MemorySpace.Vec)
    tile = ir.Var("tile", load.type, S)
    cmp_call = ir.op.tile.cmps(tile, 0.0, cmp_type=2)
    mask = ir.Var("mask", cmp_call.type, S)
    scratch_call = ir.op.tile.create([1, 16], DataType.UINT32, target_memory=ir.MemorySpace.Vec)
    scratch = ir.Var("scratch", scratch_call.type, S)
    selected = ir.op.tile.sel(mask, tile, tile, scratch)
    prog = program(
        seq(
            ir.AssignStmt(tile, load, S),
            ir.AssignStmt(mask, cmp_call, S),
            ir.AssignStmt(scratch, scratch_call, S),
            ret(selected),
        ),
        [src],
        [selected.type],
    )
    result = TileInterpreter().run(prog, "main", {"src": TensorSpec((4, 64), DataType.FP32)})
    predicate = result.trace[1]
    assert predicate["outputs"][0]["dtype"] == "uint8"
    assert predicate["outputs"][0]["valid_shape"] == [4, 8]
    assert predicate["parameters"]["predicate_shape"] == [4, 64]
    select = result.trace[-1]
    assert len(select["inputs"]) == 3
    assert select["outputs"][0]["dtype"] == "uint32"
    assert select["outputs"][0]["version"] == 2
    assert select["dependencies"] == [0, 1, 2]


def test_noncontiguous_reshape_keeps_exact_logical_mapping():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    load = ir.op.tile.load(src, [0, 0], [4, 4])
    tile = ir.Var("tile", load.type, S)
    sliced = ir.op.tile.slice(tile, [2, 2], [1, 1])
    view = ir.Var("view", sliced.type, S)
    reshaped = ir.op.tile.reshape(view, [1, 4])
    prog = program(
        seq(ir.AssignStmt(tile, load, S), ir.AssignStmt(view, sliced, S), ret(reshaped)),
        [src],
        [reshaped.type],
    )
    result = TileInterpreter().run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)})
    desc = result.outputs[0]
    assert desc["stride"] is None  # Flattening a pitched slice is not affine.
    assert desc["storage_id"] == result.trace[0]["outputs"][0]["storage_id"]
    assert desc["storage_offset"] == 5
    assert desc["index_map"] == {
        "kind": "reshape",
        "shape": [1, 4],
        "source_shape": [2, 2],
        "source_stride": [4, 1],
        "source_offset": 5,
        "source_map": None,
        "order": "logical_row_major",
    }
    assert desc["version"] == 1
    assert result.trace[-1]["dependencies"] == [0, 1]


@pytest.mark.parametrize("name", ["relu", "minimums", "rsqrt", "row_prod", "col_max", "row_argmax"])
def test_extended_operation_descriptions(name):
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    loaded = ir.op.tile.load(src, [0, 0], [4, 4])
    tile = ir.Var("tile", loaded.type, S)
    prefix = [ir.AssignStmt(tile, loaded, S)]
    args, out_type = [tile], loaded.type
    if name == "minimums":
        args.append(ir.ConstFloat(1.0, DataType.FP32, S))
    elif name in {"row_prod", "row_argmax"}:
        scratch = ir.op.tile.create([4, 4], DataType.FP32)
        args.append(scratch)
        out_type = ir.TileType(
            [4, 1], DataType.INT32 if name == "row_argmax" else DataType.FP32, memory_space=ir.MemorySpace.Vec
        )
    elif name == "col_max":
        out_type = ir.TileType([1, 4], DataType.FP32, memory_space=ir.MemorySpace.Vec)
    call = ir.Call(ir.get_op(f"tile.{name}"), args, out_type, S)
    result = TileInterpreter().run(
        program(seq(*prefix, ret(call)), [src], [out_type]),
        "main",
        {"src": TensorSpec((4, 4), DataType.FP32)},
    )
    assert result.trace[-1]["pto_op"] is not None
    assert result.outputs[0]["version"] == 1
    if name == "row_argmax":
        assert result.outputs[0]["dtype"] == "int32"
        assert len(result.trace[-1]["outputs"]) == 2


def test_partial_binary_validity_uses_ir_metadata():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    lhs = ir.op.tile.load(src, [0, 0], [4, 4], [2, 4])
    rhs = ir.op.tile.load(src, [0, 0], [4, 4], [4, 2])
    call = ir.op.tile.add(lhs, rhs)
    result = TileInterpreter().run(
        program(ret(call), [src], [call.type]), "main", {"src": TensorSpec((4, 4), DataType.FP32)}
    )
    assert result.trace[-1]["inputs"][0]["valid_shape"] == [2, 4]
    assert result.trace[-1]["inputs"][1]["valid_shape"] == [4, 2]
    expected = [v.value for v in call.type.get_effective_tile_view().valid_shape]
    assert result.outputs[0]["valid_shape"] == expected


def test_column_reduction_optional_scratch_and_cast_attributes():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    loaded = ir.op.tile.load(src, [0, 0], [4, 4])
    tile = ir.Var("tile", loaded.type, S)
    scratch = ir.op.tile.create([4, 4], DataType.FP32)
    reduced = ir.op.tile.col_sum(tile, scratch)
    cast = ir.op.tile.cast(tile, DataType.INT32, mode="trunc", saturation_mode="off")
    prog = program(seq(ir.AssignStmt(tile, loaded, S), ret(reduced, cast)), [src], [reduced.type, cast.type])
    result = TileInterpreter().run(prog, "main", {"src": TensorSpec((4, 4), DataType.FP32)})
    col = next(e for e in result.trace if e["ir_op"] == "tile.col_sum")
    assert len(col["inputs"]) == 1 and len(col["outputs"]) == 2
    assert col["outputs"][0]["version"] == 2
    # The direct IR API canonicalizes the spelling "off" to integer zero.
    assert result.trace[-1]["parameters"]["kwargs"]["saturation_mode"] == 0
    assert result.outputs[-1]["dtype"] == "int32"


def test_assemble_updates_target_storage_and_existing_views():
    typ = ir.TensorType([4, 4], DataType.FP32)
    src = ir.Var("src", typ, S)
    loaded = ir.op.tile.load(src, [0, 0], [4, 4])
    base = ir.Var("base", loaded.type, S)
    viewed = ir.op.tile.slice(base, [2, 2], [1, 1])
    view = ir.Var("view", viewed.type, S)
    inserted = ir.op.tile.assemble(base, view, [0, 0])
    result = TileInterpreter().run(
        program(
            seq(ir.AssignStmt(base, loaded, S), ir.AssignStmt(view, viewed, S), ret(inserted, base, view)),
            [src],
            [inserted.type, base.type, view.type],
        ),
        "main",
        {"src": TensorSpec((4, 4), DataType.FP32)},
    )
    assert len({d["storage_id"] for d in result.outputs}) == 1
    assert all(d["version"] == 2 and d["producer"] == 2 for d in result.outputs)
    assert result.trace[-1]["pto_op"] == "pto.tinsert"
    assert result.trace[-1]["dependencies"] == [0, 1]


def test_transpose_view_alias_and_concat_allocation():
    typ = ir.TensorType([4, 4], DataType.FP16)
    src = ir.Var("src", typ, S)
    loaded = ir.op.tile.load(src, [0, 0], [4, 4], target_memory=ir.MemorySpace.Mat)
    base = ir.Var("base", loaded.type, S)
    transposed = ir.op.tile.transpose_view(base)
    concat = ir.op.tile.concat(base, base)
    result = TileInterpreter().run(
        program(
            seq(ir.AssignStmt(base, loaded, S), ret(base, transposed, concat)),
            [src],
            [base.type, transposed.type, concat.type],
        ),
        "main",
        {"src": TensorSpec((4, 4), DataType.FP16)},
    )
    original, view, copied = result.outputs
    assert view["storage_id"] == original["storage_id"] != copied["storage_id"]
    assert view["stride"] == [1, 4]
    assert result.trace[1]["pto_op"] is None
    assert result.trace[-1]["pto_op"] == "pto.tconcat"
    assert copied["shape"] == [4, 8]


@pl.program
class MulticoreMatmul:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[512, 128], pl.FP16],
        b: pl.Tensor[[128, 128], pl.FP16],
        c: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        row = pl.tile.get_block_idx() * 128
        at = pl.load(a, [row, 0], [128, 128])
        bt = pl.load(b, [0, 0], [128, 128])
        ct = pl.matmul(at, bt)
        c = pl.store(ct, [row, 0], c)
        return c

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[512, 128], pl.FP16],
        b: pl.Tensor[[128, 128], pl.FP16],
        c: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.manual_scope():
            c, tid = pl.spmd_submit(self.kernel, a, b, c, core_num=4)
        return c


def test_multicore_checkpoints_and_export(tmp_path):
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    prog = MulticoreMatmul
    a = torch.arange(512 * 128).reshape(512, 128).remainder(5).half()
    b = torch.ones((128, 128), dtype=torch.float16)
    inputs = {"a": a, "b": b, "c": torch.zeros((512, 128))}
    try:
        for compiler_pass in PassManager.get_strategy(OptimizationStrategy.Default).passes:
            prog = compiler_pass(prog)
            stage = compiler_pass.get_name()
            if stage not in {"ConvertTensorToTileOps", "InferTileMemorySpace"}:
                continue
            count = 16 if stage == "ConvertTensorToTileOps" else 24
            interpreter = TileInterpreter()
            captured = interpreter.run(prog, "main", inputs)
            assert torch.count_nonzero(inputs["c"]) == 0
            actual = interpreter.run(
                prog, "main", {k: v.clone() for k, v in inputs.items()}, mode="functional"
            )
            torch.testing.assert_close(actual.outputs[0], a.float() @ b.float(), rtol=0, atol=0)
            assert captured.trace == actual.trace
            assert captured.tasks == actual.tasks
            assert captured.events == actual.events
            assert len(captured.trace) == count
            assert len(captured.tasks) == 1
            assert len(captured.tasks[0]["instances"]) == 4
            stores = [op for op in captured.trace if op["ir_op"] == "tile.store"]
            assert [op["parameters"]["offsets"] for op in stores] == [[i * 128, 0] for i in range(4)]
            assert len({op["outputs"][0]["storage_id"] for op in stores}) == 1
            tiles = [op["outputs"][0] for op in captured.trace if op["ir_op"] == "tile.load"]
            assert len({tile["storage_id"] for tile in tiles}) == 8
            assert {tile["owner_instance"]["block_id"] for tile in tiles} == set(range(4))
            folder = tmp_path / stage
            captured.write_capture(folder)
            saved = {p.name: p.read_bytes() for p in folder.iterdir()}
            repeated = interpreter.run(prog, "main", inputs)
            repeated.write_capture(folder)
            assert saved == {p.name: p.read_bytes() for p in folder.iterdir()}
            manifest = json.loads((folder / "manifest.json").read_text())
            assert manifest["complete"] and len(manifest["instances"]) == 4
            for instance in manifest["instances"]:
                records = [json.loads(s) for s in (folder / instance["trace_file"]).read_text().splitlines()]
                assert len(records) == count // 4
                assert [op["instance_op_sequence"] for op in records] == list(range(count // 4))
            if stage == "InferTileMemorySpace":
                break
    finally:
        backend.reset_for_testing()


@pl.program
class MpmdTasks:
    @pl.function(type=pl.FunctionType.InCore)
    def matrix(
        self,
        a: pl.Tensor[[128, 128], pl.FP16],
        b: pl.Tensor[[128, 128], pl.FP16],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        at = pl.load(a, [0, 0], [128, 128])
        bt = pl.load(b, [0, 0], [128, 128])
        ct = pl.matmul(at, bt)
        c = pl.store(ct, [0, 0], c)
        return c

    @pl.function(type=pl.FunctionType.InCore)
    def copy(
        self, x: pl.Tensor[[128, 128], pl.FP32], y: pl.Out[pl.Tensor[[128, 128], pl.FP32]]
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        t = pl.load(x, [0, 0], [128, 128])
        y = pl.store(t, [0, 0], y)
        return y

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP16],
        b: pl.Tensor[[128, 128], pl.FP16],
        x: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        y: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        z: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.manual_scope():
            c, matrix_id = pl.submit(self.matrix, a, b, c)
            y, copy_id = pl.submit(self.copy, x, y)
            z, consumer_id = pl.submit(self.copy, c, z, deps=[matrix_id])
        return z


def test_mpmd_dependencies_and_numerical_checkpoints():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    prog = MpmdTasks
    inputs = {
        name: torch.ones((128, 128), dtype=torch.float16 if name in {"a", "b"} else torch.float32)
        for name in ("a", "b", "x", "c", "y", "z")
    }
    try:
        for compiler_pass in PassManager.get_strategy(OptimizationStrategy.Default).passes:
            prog = compiler_pass(prog)
            stage = compiler_pass.get_name()
            if stage not in {"ConvertTensorToTileOps", "InferTileMemorySpace"}:
                continue
            result = TileInterpreter().run(prog, "main", inputs)
            assert len(result.trace) == (8 if stage == "ConvertTensorToTileOps" else 10)
            assert [task["entry"] for task in result.tasks] == ["matrix", "copy", "copy"]
            assert [task["dependencies"] for task in result.tasks] == [[], [], [0]]
            assert all(task["status"] == "completed" for task in result.tasks)
            actual = TileInterpreter().run(
                prog, "main", {k: v.clone() for k, v in inputs.items()}, mode="functional"
            )
            torch.testing.assert_close(actual.outputs[0], inputs["a"].float() @ inputs["b"].float())
            assert actual.trace == result.trace and actual.tasks == result.tasks
            assert actual.events == result.events
            if stage == "InferTileMemorySpace":
                break
    finally:
        backend.reset_for_testing()


def _identity_kernel(name="kernel", kind=ir.FunctionType.InCore):
    return ir.Function(name, [], [INDEX], ret(ir.op.tile.get_block_idx()), S, type=kind)


def _dispatch(fn, *, blocks=None, predicate=None, deps=()):
    types = [*fn.return_types, ir.ScalarType(DataType.TASK_ID)]
    typ = ir.TupleType(types) if fn.return_types else types[0]
    return ir.Submit(ir.GlobalVar(fn.name), [], deps, {}, {}, typ, S, core_num=blocks, predicate=predicate)


def _orch_program(functions, body, params=(), returns=()):
    main = ir.Function("main", list(params), returns, body, S, type=ir.FunctionType.Orchestration)
    return ir.Program([*functions, main], "capture_test", S)


@pytest.mark.parametrize("wrapper_type", [ir.FunctionType.Spmd, ir.FunctionType.Group])
def test_launch_call_wrappers_and_helper_identity(wrapper_type):
    helper = _identity_kernel("helper", ir.FunctionType.Opaque)
    kernel = ir.Function(
        "kernel",
        [],
        [INDEX],
        ret(ir.Call(ir.GlobalVar("helper"), [], INDEX, S)),
        S,
        type=ir.FunctionType.InCore,
    )
    wrapper = ir.Function(
        "wrapper",
        [],
        [INDEX],
        ret(ir.Call(ir.GlobalVar("kernel"), [], INDEX, S)),
        S,
        type=wrapper_type,
        attrs={"spmd_unwrapped": True} if wrapper_type == ir.FunctionType.Group else {},
    )
    call = ir.Call(ir.GlobalVar("wrapper"), [], {}, {"core_num": const(3), "sync_start": True}, INDEX, S)
    result = TileInterpreter().run(
        _orch_program([helper, kernel, wrapper], ret(call), returns=[INDEX]), "main", {}
    )
    assert len(result.tasks) == 1
    assert result.tasks[0]["sync_start"]
    queries = [e for e in result.events if e.get("ir_op") == "tile.get_block_idx"]
    assert [e["result"] for e in queries] == [0, 1, 2]
    assert all(
        e["task_id"] == 0 and e["call_stack"] == ["main", "wrapper", "kernel", "helper"] for e in queries
    )


def test_nested_orchestration_and_graph_parent():
    kernel = _identity_kernel()
    graph = ir.Function(
        "graph",
        [],
        [INDEX],
        ret(ir.Call(ir.GlobalVar("kernel"), [], INDEX, S)),
        S,
        type=ir.FunctionType.Graph,
    )
    nested = ir.Function(
        "nested",
        [],
        [INDEX],
        ret(ir.Call(ir.GlobalVar("graph"), [], INDEX, S)),
        S,
        type=ir.FunctionType.Orchestration,
    )
    result = TileInterpreter().run(
        _orch_program(
            [kernel, graph, nested], ret(ir.Call(ir.GlobalVar("nested"), [], INDEX, S)), returns=[INDEX]
        ),
        "main",
        {},
    )
    assert [t["entry"] for t in result.tasks] == ["graph", "kernel"]
    assert [t["parent_task_id"] for t in result.tasks] == [None, 0]


def test_dynamic_blocks_loops_and_task_array_dependencies():
    kernel = ir.Function("kernel", [], [], seq(), S, type=ir.FunctionType.InCore)
    count = ir.Var("count", INDEX, S)
    dispatch = _dispatch(kernel, blocks=count)
    tid = ir.Var("tid", dispatch.type, S)
    make_array = ir.op.array.create(2, DataType.TASK_ID)
    array = ir.Var("array", make_array.type, S)
    carry = ir.IterArg("carry", array.type, array, S)
    index = ir.Var("i", INDEX, S)
    updated = ir.Var("updated", array.type, S)
    loop = ir.ForStmt(
        index,
        const(0),
        const(2),
        const(1),
        [carry],
        seq(
            ir.AssignStmt(tid, dispatch, S), ir.YieldStmt([ir.op.array.update_element(carry, index, tid)], S)
        ),
        [updated],
        S,
    )
    consumer = _dispatch(
        kernel, deps=[updated, ir.op.array.get_element(updated, 0), ir.op.system.task_invalid()]
    )
    prog = _orch_program(
        [kernel], seq(ir.AssignStmt(array, make_array, S), loop, ir.EvalStmt(consumer, S)), [count]
    )
    result = TileInterpreter().run(prog, "main", {"count": 2})
    actual = TileInterpreter().run(prog, "main", {"count": 2}, mode="functional")
    assert result.tasks == actual.tasks and result.events == actual.events
    assert [t["core_num"] for t in result.tasks] == [2, 2, 1]
    assert result.tasks[2]["dependencies"] == [0, 1]
    assert result.trace == []


@pytest.mark.parametrize("blocks,predicate", [(0, True), (4, False)])
def test_skipped_task_preserves_output_and_dependency(blocks, predicate):
    typ = ir.TensorType([2, 2], DataType.FP32)
    dst = ir.Var("dst", typ, S)
    kernel = ir.Function(
        "kernel", [(dst, ir.ParamDirection.Out)], [typ], ret(dst), S, type=ir.FunctionType.InCore
    )
    submitted_type = ir.TupleType([typ, ir.ScalarType(DataType.TASK_ID)])
    launch = ir.Submit(
        ir.GlobalVar("kernel"),
        [dst],
        [],
        {},
        {},
        submitted_type,
        S,
        core_num=const(blocks),
        predicate=ir.ConstBool(predicate, S),
    )
    submitted = ir.Var("submitted", submitted_type, S)
    second = ir.Submit(
        ir.GlobalVar("kernel"), [dst], [ir.TupleGetItemExpr(submitted, 1, S)], submitted_type, S
    )
    prog = _orch_program(
        [kernel],
        seq(
            ir.AssignStmt(submitted, launch, S),
            ir.EvalStmt(second, S),
            ret(ir.TupleGetItemExpr(submitted, 0, S)),
        ),
        [dst],
        [typ],
    )
    result = TileInterpreter().run(prog, "main", {"dst": TensorSpec((2, 2), DataType.FP32)})
    assert result.tasks[0]["status"] == "skipped" and result.tasks[0]["instances"] == []
    assert result.tasks[1]["dependencies"] == [0]
    assert result.outputs[0]["object_id"] == 0 and result.outputs[0]["version"] == 0


@pytest.mark.parametrize("blocks", [-1, 1.5])
def test_bad_launch_count(blocks):
    kernel = _identity_kernel()
    call = ir.Call(ir.GlobalVar("kernel"), [], {}, {"core_num": blocks}, INDEX, S)
    with pytest.raises(InterpreterError, match="core_num must be a nonnegative integer"):
        TileInterpreter().run(_orch_program([kernel], ir.EvalStmt(call, S)), "main", {})


def test_skipped_ambiguous_return_and_unknown_launch():
    kernel = _identity_kernel()
    with pytest.raises(InterpreterError, match="Cannot determine skipped return bindings"):
        TileInterpreter().run(
            _orch_program([kernel], ir.EvalStmt(_dispatch(kernel, blocks=const(0)), S)), "main", {}
        )
    typ = ir.TensorType([2, 2], DataType.FP32)
    src = ir.Var("src", typ, S)
    launch = _dispatch(kernel, blocks=ir.Cast(ir.op.tensor.read(src, [0, 0]), DataType.INDEX, S))
    with pytest.raises(InterpreterError, match="unknown payload value.*tensor.read"):
        TileInterpreter().run(
            _orch_program([kernel], ir.EvalStmt(launch, S), [src]),
            "main",
            {"src": TensorSpec((2, 2), DataType.FP32)},
        )


def test_device_configuration_and_sync_events(tmp_path):
    body = seq(
        ir.EvalStmt(system_ops.barrier(pipe=ir.PipeType.ALL), S),
        ir.EvalStmt(system_ops.fence(), S),
        ir.EvalStmt(system_ops.set_flag(ir.PipeType.MTE2, ir.PipeType.V, 0), S),
        ir.EvalStmt(system_ops.wait_flag(ir.PipeType.MTE2, ir.PipeType.V, 0), S),
        ir.EvalStmt(system_ops.syncall(), S),
        ir.EvalStmt(system_ops.sync_set(2, pipe=ir.PipeType.MTE3), S),
        ir.EvalStmt(system_ops.sync_wait(2, pipe=ir.PipeType.MTE2), S),
        ret(ir.op.tile.get_block_num()),
    )
    kernel = ir.Function("kernel", [], [INDEX], body, S, type=ir.FunctionType.InCore)
    dispatch = _dispatch(kernel, blocks=system_ops.available_aiv_count())
    scope = ir.RuntimeScopeStmt(
        True, body=ir.RuntimeScopeStmt(False, body=ir.EvalStmt(dispatch, S), span=S), span=S
    )
    prog = _orch_program([kernel], scope)
    with pytest.raises(InterpreterError, match="capture_config.*available_aiv_count"):
        TileInterpreter().run(prog, "main", {})
    interpreter = TileInterpreter(capture_config={"available_aiv_count": 2})
    result = interpreter.run(prog, "main", {})
    sync = [e for e in result.events if e["kind"] == "synchronization"]
    assert len(sync) == 14 and result.trace == []
    barrier = next(e for e in sync if e["ir_op"] == "system.syncall")
    assert len(barrier["parameters"]["participants"]) == 2
    scopes = [e for e in result.events if e["kind"] == "scope_enter"]
    assert [e["mode"] for e in scopes] == ["MANUAL", "AUTO"]
    assert scopes[1]["parent_scope_id"] == scopes[0]["entered_scope_id"]
    result.write_capture(tmp_path)
    assert len(list(tmp_path.glob("task_*.jsonl"))) == 2
    assert all(p.read_text() == "" for p in tmp_path.glob("task_*.jsonl"))


def test_orchestration_tensor_views_shared_across_tasks():
    typ = ir.TensorType([2, 2], DataType.FP32)
    src, dst = ir.Var("src", typ, S), ir.Var("dst", typ, S)
    loaded = ir.op.tile.load(src, [0, 0], [2, 2])
    tile = ir.Var("tile", loaded.type, S)
    kernel = ir.Function(
        "copy",
        [src, (dst, ir.ParamDirection.Out)],
        [typ],
        seq(ir.AssignStmt(tile, loaded, S), ret(ir.op.tile.store(tile, [0, 0], dst))),
        S,
        type=ir.FunctionType.InCore,
    )
    created = ir.op.tensor.create([4, 2], DataType.FP32)
    tensor = ir.Var("tensor", created.type, S)
    sliced = ir.op.tensor.slice(tensor, [2, 2], [2, 0])
    view = ir.Var("view", sliced.type, S)
    first = ir.Call(ir.GlobalVar("copy"), [src, view], typ, S)
    second = ir.Call(ir.GlobalVar("copy"), [view, dst], typ, S)
    body = seq(
        ir.AssignStmt(tensor, created, S),
        ir.AssignStmt(view, sliced, S),
        ir.EvalStmt(first, S),
        ir.EvalStmt(second, S),
        ret(dst, ir.op.tensor.dim(view, 0)),
    )
    prog = _orch_program([kernel], body, [src, dst], [typ, ir.ScalarType(DataType.INT64)])
    inputs = {"src": torch.ones((2, 2)), "dst": torch.zeros((2, 2))}
    result = TileInterpreter().run(prog, "main", inputs, mode="functional")
    torch.testing.assert_close(result.outputs[0], inputs["src"])
    assert result.outputs[1] == 2
    assert result.trace[2]["inputs"][0]["storage_id"] == result.trace[1]["outputs"][0]["storage_id"]
    assert result.trace[1]["op_id"] in result.trace[2]["dependencies"]


def test_invalid_task_id_and_global_launch_budget():
    kernel = ir.Function("kernel", [], [], seq(), S, type=ir.FunctionType.InCore)
    tid = ir.Var("tid", ir.ScalarType(DataType.TASK_ID), S)
    prog = _orch_program([kernel], ir.EvalStmt(_dispatch(kernel, deps=[tid]), S), [tid])
    with pytest.raises(InterpreterError, match="Expected TaskId dependency"):
        TileInterpreter().run(prog, "main", {"tid": 999})
    interpreter = TileInterpreter(max_steps=20)
    prog = _orch_program([kernel], ir.EvalStmt(_dispatch(kernel, blocks=const(100)), S))
    with pytest.raises(InterpreterError, match="task/block.*step limit"):
        interpreter.run(prog, "main", {})
    assert interpreter.capture.tasks[0]["status"] == "failed"
    assert len(interpreter.capture.tasks[0]["instances"]) < 100


def test_capture_disabled_keeps_task_topology():
    kernel = _identity_kernel()
    prog = _orch_program([kernel], ir.EvalStmt(_dispatch(kernel, blocks=const(3)), S))
    interpreter = TileInterpreter()
    recorded = interpreter.run(prog, "main", {})
    muted = interpreter.run(prog, "main", {}, capture_trace=False)
    assert recorded.tasks == muted.tasks
    assert muted.trace == muted.events == []


@pytest.mark.parametrize("flag", [0.0, 1.0])
def test_payload_predicate_requires_concrete_source(flag):
    typ = ir.TensorType([2, 2], DataType.FP32)
    dst = ir.Var("dst", typ, S)
    flags = ir.Var("flags", typ, S)
    kernel = ir.Function(
        "kernel", [(dst, ir.ParamDirection.Out)], [typ], ret(dst), S, type=ir.FunctionType.InCore
    )
    predicate = ir.Gt(
        ir.op.tensor.read(flags, [0, 0]), ir.ConstFloat(0.0, DataType.FP32, S), DataType.BOOL, S
    )
    launch = ir.Submit(
        ir.GlobalVar("kernel"),
        [dst],
        [],
        {},
        {},
        ir.TupleType([typ, ir.ScalarType(DataType.TASK_ID)]),
        S,
        core_num=const(2),
        predicate=predicate,
    )
    prog = _orch_program([kernel], ir.EvalStmt(launch, S), [dst, flags])
    inputs = {"dst": torch.ones((2, 2)), "flags": torch.full((2, 2), flag)}
    interpreter = TileInterpreter()
    with pytest.raises(InterpreterError, match="task launch.*unknown payload value.*tensor.read"):
        interpreter.run(prog, "main", inputs)
    assert any(e.get("ir_op") == "tensor.read" for e in interpreter.capture.events.records)
    assert interpreter.capture.tasks == []
    actual = interpreter.run(prog, "main", inputs, mode="functional")
    assert actual.tasks[0]["status"] == ("completed" if flag else "skipped")
    assert len(actual.tasks[0]["instances"]) == (2 if flag else 0)


def test_explicit_roles_and_direct_kernel_context():
    kernel = ir.Function(
        "kernel",
        [],
        [INDEX],
        ret(ir.op.tile.get_subblock_idx()),
        S,
        level=ir.Level.CHIP_DIE,
        role=ir.Role.SubWorker,
    )
    main = ir.Function(
        "main",
        [],
        [INDEX],
        ret(ir.Call(ir.GlobalVar("kernel"), [], INDEX, S)),
        S,
        level=ir.Level.CHIP,
        role=ir.Role.Orchestrator,
    )
    prog = ir.Program([kernel, main], "roles", S)
    result = TileInterpreter().run(prog, "main", {})
    assert [task["entry"] for task in result.tasks] == ["kernel"]
    direct = TileInterpreter().run(prog, "kernel", {})
    assert direct.outputs == result.outputs == (0,)
    assert direct.tasks[0]["core_num"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
