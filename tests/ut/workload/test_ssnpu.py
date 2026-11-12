# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SSNPU construction, immutable versions, collective lifetime and export contracts."""

import json
from collections import Counter, defaultdict

import pytest
import torch
from pypto import DataType, ir
from pypto.workload import InterpreterError, OpResult, TensorSpec, TileInterpreter
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


def build_fa():
    ss = Builder()
    with ss.launch("fa", blocks=2) as fn:
        q, k, v = [fn.param(name, ir.TensorType([256, 64], DataType.FP16)) for name in ("Q", "K", "V")]
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
                loop.iter_arg(name, value) for name, value in zip(("m", "l", "o"), (m, normalizer, o))
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


def run(program_inputs, entry="matmul", order=None, capture_trace=True):
    program, inputs = program_inputs
    return TileInterpreter(thread_order=order).run(program, entry, inputs, capture_trace=capture_trace)


def normalized(result):
    """Compare relations via stable instance/site identities, never allocation IDs."""

    def key(op):
        return (op["task_id"], op["block_id"], op["thread_id"], op["ssnpu"]["dynamic_id"])

    ids = {op["op_id"]: key(op) for op in result.trace}
    return sorted(
        (
            key(op),
            op["ir_op"],
            tuple(sorted(ids[d] for d in op["dependencies"])),
            tuple(sorted(ids[d] for d in op["sync_dependencies"])),
            tuple(sorted(ids[d] for d in op["order_dependencies"])),
            json.dumps(op["ssnpu"], sort_keys=True),
        )
        for op in result.trace
    )


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1)])
def test_matmul_versions_counts_and_order(order):
    workload = build_matmul()
    result = run(workload, order=order)
    assert normalized(result) == normalized(run(workload))
    shared = defaultdict(list)
    for op in result.trace:
        for use in op["ssnpu"]["shared_inputs"]:
            shared[use["version_id"]].append((op, use))
            source = op["inputs"][use["operand"]]
            assert source["version"] == 1
            assert source["memory_space"] == "SharedTileReg"
            assert source["producer"] in op["dependencies"]
    assert len(shared) == 8  # Four loop instances in each of two isolated blocks.
    assert len({uses[0][1]["handle_id"] for uses in shared.values()}) == 2
    for entries in shared.values():
        assert len(entries) == 8
        assert Counter(use["retention"] for _, use in entries) == {"reuse": 4, "last-use": 4}
        assert len({use["use_id"] for _, use in entries}) == 2
    for block in range(2):
        ops = [op for op in result.trace if op["block_id"] == block]
        assert Counter(op["pto_op"] for op in ops) == {
            "pto.tload": 36,
            "pto.tmatmul.acc": 32,
            "pto.tstore": 8,
            "pto.tdup": 8,
        }
    stores = [op for op in result.trace if op["pto_op"] == "pto.tstore"]
    store_ids = {op["op_id"] for op in stores}
    assert all(not (set(op["dependencies"]) & store_ids) for op in stores)
    assert len([event for event in result.events if event["kind"] == "ssnpu.barrier"]) == 64
    assert all(op["sync_dependencies"] for op in stores)


def test_fa_shapes_views_and_lifetime():
    workload = build_fa()
    result = run(workload, "fa", (3, 1, 0, 2))
    assert normalized(result) == normalized(run(workload, "fa"))
    matmuls = [op for op in result.trace if op["ir_op"] == "tile.matmul"]
    assert Counter(tuple(op["outputs"][0]["shape"]) for op in matmuls) == {(32, 128): 16, (32, 64): 16}
    assert all(len(op["ssnpu"]["shared_inputs"]) == 1 for op in matmuls)
    views = [op for op in result.trace if op["ir_op"] == "tile.transpose_view"]
    assert len(views) == 16
    for view in views:
        assert not view["ssnpu"]["shared_inputs"]  # A descriptor is not a payload consumption.
        assert view["inputs"][0]["shared_version"] == view["outputs"][0]["shared_version"]
        assert view["outputs"][0]["shape"] == [64, 128]
        assert view["outputs"][0]["stride"] == [1, 64]
    for op in matmuls:
        if op["outputs"][0]["shape"] == [32, 128]:
            assert op["inputs"][1]["view_producer"] in op["dependencies"]
    norms = [op for op in result.trace if op["ir_op"] == "tile.row_expand_div"]
    assert len(norms) == 8
    assert all([v["shape"] for v in op["inputs"]] == [[32, 64], [32, 1]] for op in norms)


def test_export_thread_identity_and_no_capture(tmp_path):
    result = run(build_matmul())
    result.write_capture(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["ssnpu_spec_version"] == 1
    assert manifest["complete"]
    assert len(manifest["instances"]) == 8
    for instance in manifest["instances"]:
        assert instance["subblock_id"] == 0
        assert instance["mapping"] == {"core_id": instance["block_id"], "pe_id": instance["thread_id"]}
        ops = [json.loads(line) for line in (tmp_path / instance["trace_file"]).read_text().splitlines()]
        assert len(ops) == instance["operation_count"]
        assert all(op["thread_id"] == instance["thread_id"] for op in ops)
    quiet = run(build_matmul(), capture_trace=False)
    assert quiet.trace == quiet.events == []
    assert len(quiet.objects) == len(result.objects)


def small(body, *, threads=4):
    ss = Builder()
    with ss.launch("main", threads=threads) as fn:
        tensor = fn.param("T", ir.TensorType([8, 8], DataType.FP16))
        tid = ss.thread_id()
        body(ss, tensor, tid)
        ss.return_stmt()
    return ir.Program([fn.get_result()], "test", ir.Span.unknown()), {"T": TensorSpec((8, 8), DataType.FP16)}


def branch(ss, tid, selected):
    return ss.if_stmt(ir.Eq(tid, ss.expr(selected), DataType.BOOL, ir.Span.unknown()))


def test_missing_producer_even_without_trace():
    def body(ss, tensor, tid):
        handle = ss.shared_tile([8, 8], DataType.FP16)
        with branch(ss, tid, 1):
            ss.load_into(handle, tensor, [0, 0])

    for capture in (True, False):
        with pytest.raises(InterpreterError, match="Missing producer"):
            run(small(body), "main", capture_trace=capture)


@pytest.mark.parametrize(
    "kind", ["last_use", "missing_participant", "barrier", "offset", "reuse_local", "pes"]
)
def test_invalid_contracts(kind):
    def body(ss, tensor, tid):
        handle = ss.shared_tile([4, 4], DataType.FP16)
        offsets = [tid, 0] if kind == "offset" else [0, 0]
        value = ss.load_into(handle, tensor, offsets)
        if kind == "last_use":
            ss.op("exp", value)
            ss.op("exp", reuse(value))
        elif kind == "missing_participant":
            with branch(ss, tid, 0):
                ss.op("exp", value)
        elif kind == "barrier":
            with branch(ss, tid, 0):
                ss.barrier()
        elif kind == "reuse_local":
            local = ss.full([4, 4], 0, DataType.FP16)
            ss.op("exp", reuse(local))
        elif kind == "pes":
            ss.op("exp", value, pes=[0])

    message = {
        "last_use": "last-use",
        "missing_participant": "participants",
        "barrier": "Barrier",
        "offset": "mismatch",
        "reuse_local": "reuse requires",
        "pes": "participant",
    }[kind]
    with pytest.raises(InterpreterError, match=message):
        run(small(body), "main")


@pytest.mark.parametrize("pes", [[], [0, 0], [4], [-1], [True]])
def test_invalid_participant_sets(pes):
    ss = Builder()
    with pytest.raises(ValueError, match="PE subset|Duplicate"):
        ss._call("ssnpu.barrier", [], ir.ScalarType(DataType.INDEX), pes=pes)


def test_private_escape():
    stash = []

    def body(ss, tensor, tid):
        local = ss.load_local(tensor, [0, 0], [4, 4])
        escaped = ss._call("ssnpu.test_escape", [local], local.type)
        ss.op("exp", escaped)

    workload, inputs = small(body)
    interpreter = TileInterpreter()

    def escape(ctx, call, args):
        if not stash:
            stash.append(args[0])
        return OpResult(stash[0])

    interpreter.register_op("ssnpu.test_escape", semantics=escape)
    with pytest.raises(InterpreterError, match="Private LocalTileReg"):
        interpreter.run(workload, "main", inputs)


@pytest.mark.parametrize("overlap", [True, False])
def test_memory_overlap_and_war(overlap):
    def body(ss, tensor, tid):
        offset = ss.expr(0) if overlap else ir.Mul(tid, ss.expr(4), DataType.INDEX, ir.Span.unknown())
        value = ss.load_local(tensor, [offset, 0], [4, 4])
        ss.barrier()
        ss.store(value, tensor, [offset, 0])

    result = run(small(body, threads=2), "main", (1, 0))
    loads = [op for op in result.trace if op["pto_op"] == "pto.tload"]
    stores = [op for op in result.trace if op["pto_op"] == "pto.tstore"]
    for store in stores:
        expected = {op["op_id"] for op in loads if overlap or op["thread_id"] == store["thread_id"]}
        assert expected <= set(store["dependencies"])
    first = next(op for op in stores if op["thread_id"] == 0)
    second = next(op for op in stores if op["thread_id"] == 1)
    assert (first["op_id"] in second["dependencies"]) == overlap


def test_dynamic_helper_call_path_and_preserved_old_version():
    ss = Builder()
    with ss.function("helper", type=ir.FunctionType.InCore) as helper:
        tensor = helper.param("T", ir.TensorType([8, 8], DataType.FP16))
        handle = ss.shared_tile([4, 4], DataType.FP16)
        old = ss.load_into(handle, tensor, [0, 0])
        newer = ss.load_into(handle, tensor, [4, 0])
        ss.barrier()
        ss.op("exp", old)
        ss.op("exp", newer)
        ss.return_stmt()
    with ss.launch("main") as fn:
        tensor = fn.param("T", ir.TensorType([8, 8], DataType.FP16))
        for _ in range(2):
            ss.eval_stmt(ir.Call(ir.GlobalVar("helper"), [tensor], ir.TupleType([]), ir.Span.unknown()))
        ss.return_stmt()
    program = ir.Program([helper.get_result(), fn.get_result()], "nested", ir.Span.unknown())
    result = run((program, {"T": TensorSpec((8, 8), DataType.FP16)}), "main", (3, 2, 1, 0))
    versions = {u["version_id"] for op in result.trace for u in op["ssnpu"]["shared_inputs"]}
    assert len(versions) == 4
    assert len({obj["shared_handle"] for obj in result.objects if "shared_handle" in obj}) == 2


@pytest.mark.parametrize("with_barrier", [True, False])
def test_disjoint_participants_require_lifetime_order(with_barrier):
    def body(ss, tensor, tid):
        handle = ss.shared_tile([4, 4], DataType.FP16)
        value = ss.load_into(handle, tensor, [0, 0])
        with branch(ss, tid, 1):
            ss.op("exp", reuse(value), pes=[1])
        if with_barrier:
            ss.barrier()
        with branch(ss, tid, 0):
            ss.op("exp", value, pes=[0])

    workload = small(body)
    if with_barrier:
        result = run(workload, "main", (3, 2, 1, 0))
        assert len(result.trace) == 3
    else:
        with pytest.raises(InterpreterError, match="unordered lifetime"):
            run(workload, "main")


def test_divergent_loop_barriers_and_private_operations():
    def barrier_body(ss, tensor, tid):
        index = ss.var("i", ir.ScalarType(DataType.INDEX))
        with ss.for_loop(index, 0, tid, 1):
            ss.barrier()

    with pytest.raises(InterpreterError, match="Barrier site/round mismatch"):
        run(small(barrier_body), "main")

    def private_body(ss, tensor, tid):
        with branch(ss, tid, 0):
            ss.full([4, 4], 0)

    with pytest.raises(InterpreterError, match="Unmatched operation participants"):
        run(small(private_body), "main")


def test_partial_private_operation_and_mapping():
    ss = Builder()
    with ss.launch("main", blocks=2, threads=2, cores=[2, 5], pe_mapping=[1, 3]) as fn:
        tid = ss.thread_id()
        with branch(ss, tid, 1):
            ss.full([4, 4], 0, pes=[1])
        ss.return_stmt()
    program = ir.Program([fn.get_result()], "mapping", ir.Span.unknown())
    result = run((program, {}), "main")
    assert len(result.trace) == 2
    assert [i["mapping"] for i in result.tasks[0]["instances"]] == [
        {"core_id": core, "pe_id": pe} for core in (2, 5) for pe in (1, 3)
    ]


def test_multiple_launches_isolate_handles():
    kernel, inputs = small(
        lambda ss, tensor, tid: ss.load_into(ss.shared_tile([4, 4], DataType.FP16), tensor, [0, 0])
    )
    ss = Builder()
    with ss.function("host", type=ir.FunctionType.Orchestration) as fn:
        tensor = fn.param("T", ir.TensorType([8, 8], DataType.FP16))
        for _ in range(2):
            ss.eval_stmt(ir.Call(ir.GlobalVar("main"), [tensor], ir.TupleType([]), ir.Span.unknown()))
        ss.return_stmt()
    program = ir.Program([kernel.get_function("main"), fn.get_result()], "launches", ir.Span.unknown())
    result = run((program, inputs), "host")
    assert len(result.tasks) == 2
    assert len({o["shared_handle"] for o in result.objects if "shared_handle" in o}) == 2


def test_unknown_payload_control_and_missing_numerics():
    def body(ss, tensor, tid):
        local = ss.load_local(tensor, [0, 0], [4, 4])
        unknown = ss._call("ssnpu.read_scalar", [local], ir.ScalarType(DataType.INDEX))
        with branch(ss, unknown, 0):
            ss.barrier()

    program, inputs = small(body)
    interpreter = TileInterpreter()
    interpreter.register_op(
        "ssnpu.read_scalar", semantics=lambda ctx, call, args: OpResult(ctx.unknown(call), reads=(args[0],))
    )
    with pytest.raises(InterpreterError, match="if condition.*unknown payload value"):
        interpreter.run(program, "main", inputs)
    program, _ = small(lambda ss, tensor, tid: ss.full([4, 4], 0))

    with pytest.raises(InterpreterError, match="Missing numerical implementation"):
        TileInterpreter().run(
            program, "main", {"T": torch.zeros((8, 8), dtype=torch.float16)}, mode="functional"
        )


@pytest.mark.parametrize("order", [(), (True, 1, 2, 3), (0, 0, 2, 3), (0, 1)])
def test_invalid_capture_order(order):
    with pytest.raises((ValueError, InterpreterError), match="thread_order"):
        run(build_matmul(), order=order)


def test_nonzero_producer_precedes_aliasing_consumer_store():
    # PE0 stores back into the source Tensor, but PE1 is the shared producer.
    # Sorting by PE alone would invent store0 -> load1 -> store0.
    def body(ss, tensor, tid):
        handle = ss.shared_tile([4, 4], DataType.FP16)
        value = ss.load_into(handle, tensor, [0, 0], pes=[1])
        ss.store(value, tensor, [0, 0])

    workload = small(body)
    result = run(workload, "main")
    assert normalized(result) == normalized(run(workload, "main", (3, 2, 1, 0)))
    load = next(op for op in result.trace if op["pto_op"] == "pto.tload")
    stores = [op for op in result.trace if op["pto_op"] == "pto.tstore"]
    assert not load["dependencies"]
    assert all(load["op_id"] in store["dependencies"] for store in stores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
