# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile semantics and optional FP32 numerical reference implementations."""

from dataclasses import dataclass
from functools import partial
from math import prod
from typing import Any

import torch

from pypto import DataType
from pypto.pypto_core import ir as _ir

from .runtime import OpResult, RuntimeObject


def _attrs(ctx: Any, call: _ir.Call, allowed: set[str]) -> dict[str, Any]:
    unknown = set(call.kwargs) - allowed
    metadata = set(call.attrs) - {"__compiler_tensor_to_tile_mat_bridge", "arg_directions"}
    if unknown or metadata:
        raise ValueError(f"Unsupported attributes: {sorted(unknown | metadata)}")
    kw = ctx.call_attributes(call)
    if "target_memory" in kw and kw["target_memory"] != call.type.memory_space:
        raise ValueError("target_memory disagrees with the result type")
    if "clamp" in kw and type(kw["clamp"]) is not bool:
        raise ValueError("clamp must be boolean")
    return kw


def _object(value: Any, kind: str) -> RuntimeObject:
    if not isinstance(value, RuntimeObject) or value.kind != kind:
        raise ValueError(f"Expected {kind}, got {type(value).__name__}")
    return value


def _rectangle(offset: tuple[int, ...], extent: tuple[int, ...], shape: tuple[int, ...]) -> tuple[slice, ...]:
    if (
        len(offset) != 2
        or len(extent) != 2
        or any(
            type(o) is not int or type(n) is not int or o < 0 or n < 0 or o + n > s
            for o, n, s in zip(offset, extent, shape)
        )
    ):
        raise ValueError(f"Access offset={offset}, extent={extent} exceeds shape={shape}")
    return tuple(slice(o, o + n) for o, n in zip(offset, extent))


def load(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    kw = _attrs(ctx, call, {"target_memory", "clamp"})
    if len(args) != 4:
        raise ValueError(f"Expected 4 load arguments, got {len(args)}")
    source, offset, shape, requested = args
    source = _object(source, "tensor")
    if len(shape) != 2 or len(requested) != 2 or any(v < 0 or v > s for v, s in zip(requested, shape)):
        raise ValueError(f"Invalid load shape={shape}, valid_shape={requested}")
    valid = requested
    if kw.get("clamp", False):
        valid = tuple(min(v, max(0, s - o)) for v, s, o in zip(requested, source.valid_shape, offset))
    _rectangle(offset, valid, source.valid_shape)
    out = ctx.new_tile(call.type)
    if out.shape != shape or out.valid_shape != valid:
        raise ValueError(f"Load result type disagrees with requested shape={shape}, valid_shape={valid}")
    return OpResult(
        out, (source,), (out,), "pto.tload", {"offsets": list(offset), "tensor_id": source.object_id}
    )


def store(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    if len(args) != 3:
        raise ValueError(f"Expected 3 store arguments, got {len(args)}")
    source, offset, dest = args
    source, dest = _object(source, "tile"), _object(dest, "tensor")
    _rectangle(offset, source.valid_shape, dest.valid_shape)
    return OpResult(
        dest, (source,), (dest,), "pto.tstore", {"offsets": list(offset), "tensor_id": dest.object_id}
    )


def move(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    kw = _attrs(ctx, call, {"target_memory", "blayout", "slayout"})
    view = call.type.get_effective_tile_view()
    for key in ("blayout", "slayout"):
        if key in kw and kw[key] != getattr(view, key):
            raise ValueError(f"{key} disagrees with the result type")
    if len(args) != 1:
        raise ValueError(f"Expected 1 move argument, got {len(args)}")
    source = _object(args[0], "tile")
    out = ctx.new_tile(call.type)
    if out.shape != source.shape or out.valid_shape != source.valid_shape or out.dtype != source.dtype:
        raise ValueError("Move must preserve shape, valid shape and dtype")
    return OpResult(out, (source,), (out,), "pto.tmov")


def _product_shape(lhs: RuntimeObject, rhs: RuntimeObject) -> tuple[int, int]:
    _object(lhs, "tile")
    _object(rhs, "tile")
    # Early Tile IR may feed a Vec result into matmul before the storage
    # inference pass inserts Cube staging moves. Describe that IR as written.
    m, k = lhs.valid_shape
    rk, n = rhs.valid_shape
    if lhs.shape[1] != rhs.shape[0] or k > rk or lhs.dtype != rhs.dtype:
        raise ValueError("Matmul requires matching physical K/dtype and rhs valid K >= lhs valid K")
    return m, n


def matmul(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    if len(args) != 2:
        raise ValueError(f"Expected 2 matmul arguments, got {len(args)}")
    lhs, rhs = args
    valid = _product_shape(lhs, rhs)
    out = ctx.new_tile(call.type)
    if out.dtype != DataType.FP32 or out.valid_shape != valid or out.shape != (lhs.shape[0], rhs.shape[1]):
        raise ValueError("Matmul requires FP32 output with the product valid shape")
    return OpResult(out, (lhs, rhs), (out,), "pto.tmatmul")


def matmul_acc(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    if len(args) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 matmul_acc arguments, got {len(args)}")
    acc, lhs, rhs = args[:3]
    acc = _object(acc, "tile")
    init = args[3] if len(args) == 4 else False
    if type(init) is not bool:
        raise ValueError(f"Expected boolean init_cond, got {init}")
    valid = _product_shape(lhs, rhs)
    m, n = valid
    if (
        acc.dtype != DataType.FP32
        or acc.shape != (lhs.shape[0], rhs.shape[1])
        or m > acc.valid_shape[0]
        or n > acc.valid_shape[1]
    ):
        raise ValueError("Accumulator must be FP32, match physical M/N and cover the product valid rectangle")
    return OpResult(
        acc,
        (lhs, rhs) if init else (acc, lhs, rhs),
        (acc,),
        "pto.tmatmul" if init else "pto.tmatmul.acc",
        {"init_cond": init},
    )


def compute_load(ctx: Any, call: _ir.Call, args: tuple[Any, ...], result: OpResult) -> None:
    source, offset, _, _ = args
    out = result.value
    region = _rectangle(offset, out.valid_shape, source.valid_shape)
    m, n = out.valid_shape
    ctx.allocate_payload(out)[:m, :n].copy_(ctx.payload(source)[region])


def compute_store(ctx: Any, call: _ir.Call, args: tuple[Any, ...], result: OpResult) -> None:
    source, offset, dest = args
    region = _rectangle(offset, source.valid_shape, dest.valid_shape)
    m, n = source.valid_shape
    ctx.payload(dest)[region].copy_(ctx.payload(source)[:m, :n])


def compute_move(ctx: Any, call: _ir.Call, args: tuple[Any, ...], result: OpResult) -> None:
    ctx.allocate_payload(result.value).copy_(ctx.payload(args[0]))


def _product(ctx: Any, lhs: RuntimeObject, rhs: RuntimeObject) -> torch.Tensor:
    m, k = lhs.valid_shape
    n = rhs.valid_shape[1]
    return ctx.payload(lhs)[:m, :k].float() @ ctx.payload(rhs)[:k, :n].float()


def compute_matmul(ctx: Any, call: _ir.Call, args: tuple[Any, ...], result: OpResult) -> None:
    product = _product(ctx, *args)
    ctx.allocate_payload(result.value)[: product.shape[0], : product.shape[1]].copy_(product)


def compute_matmul_acc(ctx: Any, call: _ir.Call, args: tuple[Any, ...], result: OpResult) -> None:
    acc, lhs, rhs = args[:3]
    product = _product(ctx, lhs, rhs)
    dest = ctx.payload(acc)[: product.shape[0], : product.shape[1]]
    if result.parameters["init_cond"]:
        dest.copy_(product)
    else:
        dest.add_(product)


@dataclass(frozen=True)
class OpDescription:
    """Common fresh-output operations; scratch operands are explicit writes.

    Output metadata comes from the typed IR. The family selects the small amount
    of operand compatibility checking needed before recording effects.
    """

    family: str
    arity: int | tuple[int, ...]
    pto_op: str | None
    attributes: frozenset[str] = frozenset()


_DESCRIPTIONS = {
    "create": OpDescription("create", 1, None, frozenset({"dtype", "target_memory"})),
    "add": OpDescription("binary", 2, "pto.tadd"),
    "sub": OpDescription("binary", 2, "pto.tsub"),
    "mul": OpDescription("binary", 2, "pto.tmul"),
    "maximum": OpDescription("binary", 2, "pto.tmax"),
    "muls": OpDescription("scalar", 2, "pto.tmuls"),
    "div": OpDescription("binary", 2, "pto.tdiv"),
    "adds": OpDescription("scalar", 2, "pto.tadds"),
    "abs": OpDescription("unary", 1, "pto.tabs"),
    "sqrt": OpDescription("unary", 1, "pto.tsqrt"),
    "log": OpDescription("unary", 1, "pto.tlog", frozenset({"high_precision"})),
    "row_min": OpDescription("reduction", 2, "pto.trowmin"),
    "col_sum": OpDescription("column_reduction", (1, 2), "pto.tcolsum", frozenset({"is_binary"})),
    "row_expand": OpDescription("row_expand", 2, "pto.trowexpand"),
    "col_expand": OpDescription("col_expand", 2, "pto.tcolexpand"),
    "col_expand_mul": OpDescription("column_broadcast", 2, "pto.tcolexpandmul"),
    "relu": OpDescription("unary", 1, "pto.trelu"),
    "minimums": OpDescription("scalar", 2, "pto.tmins"),
    "rsqrt": OpDescription("unary", (1, 2), "pto.trsqrt", frozenset({"high_precision"})),
    "row_prod": OpDescription("reduction", 2, "pto.trowprod"),
    "col_max": OpDescription("column_reduction", 1, "pto.tcolmax"),
    "row_argmax": OpDescription("index_reduction", 2, "pto.trowargmax"),
    "exp": OpDescription("unary", 1, "pto.texp"),
    "cast": OpDescription("cast", (1, 2), "pto.tcvt", frozenset({"target_type", "mode", "saturation_mode"})),
    "row_max": OpDescription("reduction", 2, "pto.trowmax"),
    "row_sum": OpDescription("reduction", 2, "pto.trowsum"),
    "row_expand_sub": OpDescription("broadcast", 2, "pto.trowexpandsub"),
    "row_expand_mul": OpDescription("broadcast", 2, "pto.trowexpandmul"),
    "row_expand_div": OpDescription("broadcast", 2, "pto.trowexpanddiv"),
}


def describe_op(description: OpDescription, ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    """Record a declared operation without evaluating numerical tile contents."""
    kw = _attrs(ctx, call, set(description.attributes))
    arities = (description.arity,) if isinstance(description.arity, int) else description.arity
    if len(args) not in arities:
        raise ValueError(f"Expected {description.arity} arguments, got {len(args)}")
    out = ctx.new_tile(call.type)
    if description.family in {"row_expand", "col_expand"}:
        return _describe_expand(description.family, args, out, description.pto_op)
    if description.family == "create":
        if tuple(args[0]) != out.shape or kw.get("dtype") != out.dtype:
            raise ValueError("Create shape/dtype disagree with the result type")
        return OpResult(out, writes=(out,), parameters={"allocation": True})

    source = _object(args[0], "tile")
    reads = (source,)
    scratch = ()
    expected_shape, expected_valid = source.shape, source.valid_shape
    if description.family in {"unary", "cast"} and len(args) == 2:
        scratch = (_object(args[1], "tile"),)
    if description.family == "scalar":
        if type(args[1]) not in (float, int):
            raise ValueError(f"Expected a concrete numeric scalar, got {args[1]}")
    elif description.family == "column_reduction":
        expected_shape = (1, source.shape[1])
        expected_valid = (1, source.valid_shape[1])
        if len(args) == 2:
            scratch = (_object(args[1], "tile"),)
            if scratch[0].dtype != source.dtype or scratch[0].shape != source.shape:
                raise ValueError("Column reduction scratch must match input shape/dtype")
    elif description.family in {"binary", "broadcast", "column_broadcast", "reduction", "index_reduction"}:
        rhs = _object(args[1], "tile")
        expected_shape, expected_valid, scratch = _pair_effects(description.family, source, rhs)
        if not scratch:
            reads += (rhs,)
    # Physical broadcasting is checked above. Valid rectangles are metadata
    # supplied by the IR, not tensor shapes to broadcast against each other.
    # They may differ across operands (e.g. masked tails).
    if description.family == "binary":
        expected_valid = out.valid_shape
    if out.shape != expected_shape or out.valid_shape != expected_valid:
        raise ValueError(f"Expected output shape/valid shape {expected_shape}/{expected_valid}")
    _validate_output_dtype(description.family, kw, source, out)
    return OpResult(out, reads, (*scratch, out), description.pto_op)


def _describe_expand(family: str, args: tuple[Any, ...], out: RuntimeObject, pto_op: str | None) -> OpResult:
    target, vector = (_object(arg, "tile") for arg in args)
    expected = (target.shape[0], 1) if family == "row_expand" else (1, target.shape[1])
    if vector.shape != expected or out.shape != target.shape or out.dtype != vector.dtype:
        raise ValueError(f"Expected broadcast vector {expected} matching result dtype")
    return OpResult(out, (target, vector), (out,), pto_op, {"shape_only_inputs": [target.object_id]})


def _validate_output_dtype(
    family: str, kw: dict[str, Any], source: RuntimeObject, out: RuntimeObject
) -> None:
    if family == "cast":
        _validate_cast(kw, out)
    elif family == "index_reduction":
        if out.dtype != DataType.INT32:
            raise ValueError("Index reduction must produce INT32")
    elif out.dtype != source.dtype:
        raise ValueError("Operation must preserve dtype")


def _broadcast(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    if any(a not in {b, 1} and b != 1 for a, b in zip(lhs, rhs)):
        raise ValueError(f"Expected broadcast-compatible shapes, got {lhs} and {rhs}")
    return tuple(b if a == 1 else a for a, b in zip(lhs, rhs))


def _pair_effects(family: str, source: RuntimeObject, rhs: RuntimeObject) -> tuple[Any, ...]:
    if rhs.dtype != source.dtype:
        raise ValueError("Operands must have matching dtype")
    if family in {"reduction", "index_reduction"}:
        if any(t < s for t, s in zip(rhs.shape, source.shape)):
            raise ValueError("Reduction scratch must cover the input physical shape")
        if family == "index_reduction" and rhs.shape != source.shape:
            raise ValueError("Index reduction scratch must exactly match input shape")
        return (source.shape[0], 1), (source.valid_shape[0], 1), (rhs,)
    if family != "binary":
        expected = (source.shape[0], 1) if family == "broadcast" else (1, source.shape[1])
        valid = (source.valid_shape[0], 1) if family == "broadcast" else (1, source.valid_shape[1])
        if rhs.shape != expected or rhs.valid_shape != valid:
            raise ValueError(f"Expected rhs shape/valid shape {expected}/{valid}")
    return _broadcast(source.shape, rhs.shape), source.valid_shape, ()


def describe_view(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    """Zero-copy views get descriptor IDs without creating a storage write."""
    _attrs(ctx, call, {"pad_value"} if call.op.name == _ir.get_op("tile.slice").name else set())
    source = _object(args[0], "tile")
    out = ctx.new_tile(call.type)
    if out.dtype != source.dtype or out.memory_space != source.memory_space:
        raise ValueError("View must preserve storage dtype and memory space")
    if call.op.name == _ir.get_op("tile.slice").name:
        if len(args) not in (3, 4, 5) or tuple(args[1]) != out.shape:
            raise ValueError("Expected 2D slice shape and offset matching result type")
        if len(args) == 5 and args[4]:
            raise ValueError("Rank-dropping slice views are not supported")
        offset = tuple(args[2])
        _rectangle(offset, out.shape, source.shape)
        available = tuple(min(n, max(0, v - o)) for n, v, o in zip(out.shape, source.valid_shape, offset))
        if any(v > a for v, a in zip(out.valid_shape, available)):
            raise ValueError("Slice valid rectangle exceeds source valid data")
        out.stride = source.stride
        if source.stride is None:
            out.index_map = _index_map("slice", source, out, offset=list(offset))
            out.storage_offset = None
        else:
            out.storage_offset = source.storage_offset + sum(o * t for o, t in zip(offset, source.stride))
    else:
        if len(args) != 2 or tuple(args[1]) != out.shape or prod(out.shape) != prod(source.shape):
            raise ValueError("Reshape must preserve element count and match result shape")
        if source.stride is None or (
            source.shape[0] > 1 and source.stride[0] != source.shape[1] * source.stride[1]
        ):
            out.stride = None
            out.index_map = _index_map("reshape", source, out, order="logical_row_major")
        else:
            out.stride = (out.shape[1] * source.stride[1], source.stride[1])
        out.storage_offset = source.storage_offset
    out.storage, out.view_of = source.storage, source.object_id
    return OpResult(
        out, reads=(source,), views=(out,), parameters={"aliases": source.object_id, "memory_access": False}
    )


def _index_map(kind: str, source: RuntimeObject, out: RuntimeObject, **attributes: Any) -> dict[str, Any]:
    """Compose logical coordinate transforms without enumerating elements."""
    return {
        "kind": kind,
        "shape": list(out.shape),
        "source_shape": list(source.shape),
        "source_stride": list(source.stride) if source.stride is not None else None,
        "source_offset": source.storage_offset,
        "source_map": source.index_map,
        **attributes,
    }


def describe_transpose_view(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    if len(args) != 1:
        raise ValueError("Expected one transpose_view operand")
    source = _object(args[0], "tile")
    out = ctx.new_tile(call.type)
    if out.shape != source.shape[::-1] or out.dtype != source.dtype:
        raise ValueError("Transpose view must swap physical shape and preserve dtype")
    out.storage, out.view_of, out.storage_offset = source.storage, source.object_id, source.storage_offset
    out.stride = source.stride[::-1] if source.stride is not None else None
    out.index_map = _index_map("transpose", source, out, axes=[1, 0]) if source.index_map else None
    return OpResult(out, reads=(source,), views=(out,), parameters={"memory_access": False})


def describe_assemble(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, {"pre_quant", "pre_relu"})
    if len(args) != 3:
        raise ValueError("Expected assemble target, source and offsets")
    target, source = _object(args[0], "tile"), _object(args[1], "tile")
    _rectangle(args[2], source.valid_shape, target.shape)
    # TINSERT updates its target; keep the result descriptor distinct while
    # sharing the allocation so pre-existing views see the same write generation.
    out = ctx.new_tile(call.type)
    if out.shape != target.shape or out.dtype != target.dtype:
        raise ValueError("Assemble result must preserve target shape/dtype")
    out.storage, out.view_of = target.storage, target.object_id
    out.stride, out.storage_offset, out.index_map = target.stride, target.storage_offset, target.index_map
    return OpResult(out, (target, source), (out,), "pto.tinsert", {"offsets": list(args[2])})


def describe_concat(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    if len(args) != 2:
        raise ValueError("Expected two concat operands")
    lhs, rhs = (_object(v, "tile") for v in args)
    out = ctx.new_tile(call.type)
    if lhs.dtype != rhs.dtype or out.dtype != lhs.dtype or lhs.shape[0] != rhs.shape[0]:
        raise ValueError("Concat requires matching rows/dtypes")
    if out.shape != (lhs.shape[0], lhs.shape[1] + rhs.shape[1]):
        raise ValueError("Concat output must concatenate physical columns")
    return OpResult(out, (lhs, rhs), (out,), "pto.tconcat")


def describe_cmp(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    kw = _attrs(ctx, call, {"cmp_type"})
    if len(args) != 2 or kw.get("cmp_type", 0) not in range(6):
        raise ValueError("Expected two comparison operands and cmp_type in [0, 5]")
    lhs, rhs = (_object(v, "tile") for v in args)
    _broadcast(lhs.shape, rhs.shape)
    out = ctx.new_tile(call.type)
    if lhs.dtype != rhs.dtype or out.dtype != DataType.UINT8:
        raise ValueError("Expected matching input dtypes and a packed UINT8 result")
    return OpResult(
        out,
        (lhs, rhs),
        (out,),
        "pto.tcmp",
        {
            "predicate_shape": list(_broadcast(lhs.shape, rhs.shape)),
            "encoding": "packed_bits",
        },
    )


def describe_transpose(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    if len(args) not in (3, 4) or set(args[1:3]) != {0, 1}:
        raise ValueError("Expected a 2D transpose of axes 0 and 1 with optional scratch")
    source = _object(args[0], "tile")
    out = ctx.new_tile(call.type)
    if (
        out.shape != source.shape[::-1]
        or out.valid_shape != source.valid_shape[::-1]
        or out.dtype != source.dtype
    ):
        raise ValueError("Transpose result must swap shape/valid shape and preserve dtype")
    scratch = (_object(args[3], "tile"),) if len(args) == 4 else ()
    return OpResult(out, (source,), (*scratch, out), "pto.ttrans")


def describe_full(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    kw = _attrs(ctx, call, {"dtype", "value"})
    if len(args) not in (1, 2):
        raise ValueError("Expected full shape and filling value")
    out = ctx.new_tile(call.type)
    value = args[1] if len(args) == 2 else kw.get("value")
    if tuple(args[0]) != out.shape or kw.get("dtype") != out.dtype or type(value) not in (int, float, bool):
        raise ValueError("Full shape/dtype must match result and value must be a concrete scalar")
    return OpResult(out, writes=(out,), pto_op="pto.tdup", parameters={"value": value})


def describe_cmps(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    kw = _attrs(ctx, call, {"cmp_type"})
    if len(args) != 2 or type(args[1]) not in (int, float, bool) or kw.get("cmp_type", 0) not in range(6):
        raise ValueError("Expected tile/scalar comparison and cmp_type in [0, 5]")
    source = _object(args[0], "tile")
    out = ctx.new_tile(call.type)
    # The IR owns packed mask geometry (including aligned physical columns).
    if out.dtype != DataType.UINT8:
        raise ValueError("Expected a packed UINT8 predicate result")
    return OpResult(
        out,
        (source,),
        (out,),
        "pto.tcmps",
        {"predicate_shape": list(source.valid_shape), "encoding": "packed_bits"},
    )


def describe_sel(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    if len(args) != 4:
        raise ValueError("Expected mask, lhs, rhs and scratch")
    mask, lhs, rhs, scratch = (_object(a, "tile") for a in args)
    out = ctx.new_tile(call.type)
    if mask.dtype != DataType.UINT8 or scratch.dtype != DataType.UINT32:
        raise ValueError("Expected packed UINT8 mask and UINT32 scratch")
    if lhs.shape != rhs.shape or out.shape != lhs.shape or lhs.dtype != rhs.dtype or out.dtype != lhs.dtype:
        raise ValueError("Select value operands and output must match shape/dtype")
    return OpResult(out, (mask, lhs, rhs), (scratch, out), "pto.tsel")


def describe_mgather(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    kw = _attrs(ctx, call, {"coalesce", "gather_oob", "target_memory"})
    if len(args) != 2 or call.type.memory_space not in (None, _ir.MemorySpace.Vec):
        raise ValueError("Only Vec mgather with a tensor and index tile is supported")
    source, index = _object(args[0], "tensor"), _object(args[1], "tile")
    if index.dtype != DataType.INT32 or kw.get("coalesce", 0) not in (0, 1):
        raise ValueError("Expected INT32 indices and coalesce 0 or 1")
    out = ctx.new_tile(call.type)
    if out.dtype != source.dtype:
        raise ValueError("Gather must preserve source dtype")
    # Do not invent offsets from absent index data. Depend conservatively on
    # the whole tensor, and retain the index operand for a future replay model.
    return OpResult(
        out,
        (source, index),
        (out,),
        "pto.mgather",
        {
            "tensor_id": source.object_id,
            "index_object_id": index.object_id,
            "access_region": "data_dependent",
            "addressing": "index_tile",
        },
    )


def _validate_cast(kw: dict[str, Any], out: RuntimeObject) -> None:
    if "saturation_mode" in kw and kw["saturation_mode"] not in {"on", "off", 0, 1}:
        raise ValueError(f"Unsupported saturation_mode {kw['saturation_mode']}")
    if kw.get("target_type") != out.dtype:
        raise ValueError("Cast target_type disagrees with the result type")
    if kw.get("mode", "round") not in {
        "none",
        "rint",
        "round",
        "floor",
        "ceil",
        "trunc",
        "odd",
        *range(7),
    }:
        raise ValueError(f"Unsupported cast mode {kw['mode']}")


def default_ops() -> dict[str, tuple[Any, Any]]:
    operations = {
        _ir.get_op(f"tile.{semantics.__name__}").name: (semantics, compute)
        for semantics, compute in (
            (load, compute_load),
            (store, compute_store),
            (move, compute_move),
            (matmul, compute_matmul),
            (matmul_acc, compute_matmul_acc),
        )
    }

    operations.update(
        {
            _ir.get_op(f"tile.{name}").name: (partial(describe_op, description), None)
            for name, description in _DESCRIPTIONS.items()
        }
    )
    operations.update(
        {
            _ir.get_op(f"tile.{name}").name: (handler, None)
            for name, handler in {
                "assemble": describe_assemble,
                "concat": describe_concat,
                "cmp": describe_cmp,
                "transpose_view": describe_transpose_view,
                "slice": describe_view,
                "reshape": describe_view,
                "transpose": describe_transpose,
                "full": describe_full,
                "cmps": describe_cmps,
                "sel": describe_sel,
                "mgather": describe_mgather,
            }.items()
        }
    )
    return operations
