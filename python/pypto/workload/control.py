# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Orchestration objects, identity queries and recorded synchronization."""

from typing import Any

import torch

from pypto.pypto_core import ir as _ir

from .capture import TaskId
from .ops import _attrs, _object, _rectangle
from .runtime import DTYPES, OpResult, TensorSpec, UnknownValue


def identity(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    if args:
        raise ValueError("Identity queries take no arguments")
    instance = ctx.capture.current
    if instance is None:
        raise ValueError("Block identity requires an active kernel instance")
    field = {
        _ir.get_op("tile.get_block_idx").name: "block_id",
        _ir.get_op("tile.get_block_num").name: "block_num",
        _ir.get_op("tile.get_subblock_idx").name: "subblock_id",
    }[call.op.name]
    return OpResult(getattr(instance, field), control_kind="control")


def available(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    key = call.op.name.removeprefix("system.")
    if args or key not in ctx.capture_config:
        raise ValueError(f"Device query requires capture_config[{key!r}]")
    return OpResult(ctx.capture_config[key], control_kind="control")


def array(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    if call.op.name == _ir.get_op("array.create").name:
        kw = _attrs(ctx, call, {"dtype"})
        if len(args) != 1 or type(args[0]) is not int or args[0] < 0:
            raise ValueError("Array extent must be a nonnegative integer")
        # An explicit invalid TaskId is legal in deps; an uninitialized slot is
        # not. Keep immutable tuples so SSA updates cannot mutate earlier arrays.
        value = tuple(UnknownValue((f"uninitialized {call.op.name} at {call.span}",)) for _ in range(args[0]))
        return OpResult(value, parameters={"dtype": kw["dtype"]}, control_kind="control")
    _attrs(ctx, call, set())
    values, index = args[:2]
    ctx.concrete(index, call, "array index")
    if not isinstance(values, tuple) or type(index) is not int or not 0 <= index < len(values):
        raise ValueError(f"Invalid array index {index!r}")
    if call.op.name == _ir.get_op("array.get_element").name:
        value = values[index]
    else:
        value = (*values[:index], args[2], *values[index + 1 :])
    return OpResult(value, control_kind="control")


def task_invalid(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    _attrs(ctx, call, set())
    if args:
        raise ValueError("task_invalid takes no arguments")
    return OpResult(TaskId(None), control_kind="control")


def tensor(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    if call.op.name == _ir.get_op("tensor.create").name:
        kw = _attrs(ctx, call, {"dtype", "layout", "manual_dep"})
        if kw.get("layout", _ir.TensorLayout.ND) != _ir.TensorLayout.ND:
            raise ValueError("Only ND Tensor creation is supported")
        spec = TensorSpec(tuple(args[0]), call.type.dtype)
        out = ctx.objects.create(spec, "tensor", spec.shape, "DDR", {"tensor": "ND"})
        return OpResult(out, control_kind="control")
    _attrs(ctx, call, set())
    source = _object(args[0], "tensor")
    if call.op.name == _ir.get_op("tensor.dim").name:
        axis = args[1]
        if type(axis) is not int or not -len(source.shape) <= axis < len(source.shape):
            raise ValueError(f"Invalid tensor dimension {axis!r}")
        return OpResult(source.shape[axis], control_kind="control")
    if call.op.name == _ir.get_op("tensor.read").name:
        _rectangle(tuple(args[1]), (1,) * len(source.shape), source.shape)
        return OpResult(ctx.unknown(call), control_kind="control")
    shape, offsets = tuple(args[1]), tuple(args[2])
    _rectangle(offsets, shape, source.shape)
    if len(args) > 4 and args[4]:
        raise ValueError("Rank-dropping Tensor slices are not supported")
    valid = tuple(args[3]) if len(args) > 3 else shape
    out = ctx.objects.create(
        TensorSpec(shape, source.dtype, source.stride), "tensor", valid, "DDR", source.layout
    )
    out.storage, out.view_of = source.storage, source.object_id
    out.storage_offset = source.storage_offset + sum(o * s for o, s in zip(offsets, source.stride))
    return OpResult(out, control_kind="control")


def compute_tensor(ctx: Any, call: _ir.Call, args: tuple[Any, ...], result: OpResult) -> None:
    if call.op.name == _ir.get_op("tensor.create").name:
        if result.value.dtype not in DTYPES:
            raise ValueError(f"Unsupported functional Tensor dtype {result.value.dtype}")
        result.value.data = torch.empty(result.value.shape, dtype=DTYPES[result.value.dtype])
    elif call.op.name == _ir.get_op("tensor.read").name:
        result.value = ctx.payload(args[0])[tuple(args[1])].item()
    elif call.op.name == _ir.get_op("tensor.slice").name:
        result.value.data = ctx.payload(args[0])[_rectangle(args[2], args[1], args[0].shape)]


def synchronize(ctx: Any, call: _ir.Call, args: tuple[Any, ...]) -> OpResult:
    kwargs = ctx.call_attributes(call)
    ctx.concrete(args, call, "synchronization arguments")
    instance = ctx.capture.current
    local = call.op.name in {
        _ir.get_op(f"system.{name}").name
        for name in ("bar_v", "bar_m", "bar_all", "fence", "set_flag", "wait_flag", "sync_src", "sync_dst")
    }
    participants = [instance.identity()] if local and instance else None
    if call.op.name == _ir.get_op("system.syncall").name and instance and len(args) < 2:
        participants = [
            {**instance.identity(), "block_id": block_id} for block_id in range(instance.block_num)
        ]
    return OpResult(
        None,
        parameters={
            "domain": "instance" if local else "cross_core",
            "scope": kwargs,
            "participants": participants,
            "policy": "record_and_continue",
        },
        control_kind="synchronization",
    )


def compute_control(ctx: Any, call: _ir.Call, args: tuple[Any, ...], result: OpResult) -> None:
    """Pure control semantics already produce their final value in both modes."""


def control_ops() -> dict[str, tuple[Any, Any]]:
    groups = [
        ("tile", ("get_block_idx", "get_block_num", "get_subblock_idx"), identity, compute_control),
        ("system", ("available_cluster_count", "available_aiv_count"), available, compute_control),
        ("array", ("create", "get_element", "update_element"), array, compute_control),
        ("system", ("task_invalid",), task_invalid, compute_control),
        ("tensor", ("create", "dim", "read", "slice"), tensor, compute_tensor),
        (
            "system",
            (
                "bar_v",
                "bar_m",
                "bar_all",
                "fence",
                "syncall",
                "set_flag",
                "wait_flag",
                "sync_src",
                "sync_dst",
                "sync_set",
                "sync_wait",
                "set_ffts",
            ),
            synchronize,
            compute_control,
        ),
    ]
    return {
        _ir.get_op(f"{namespace}.{name}").name: (semantics, compute)
        for namespace, names, semantics, compute in groups
        for name in names
    }
