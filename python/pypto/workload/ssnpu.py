# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit SSNPU construction on ordinary PyPTO IR nodes.

Use inherited ``function``, ``for_loop``, ``if_stmt`` and loop ``iter_arg`` /
``return_var`` builders for control flow. No PE partitioning or lowering is
performed. Tile storage is target metadata, never an Ascend memory-space enum.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from pypto import DataType, ir
from pypto.ir.builder import FunctionBuilder
from pypto.ir.utils import _get_span_or_capture

ALL = None
INDEX = ir.ScalarType(DataType.INDEX)


@dataclass(frozen=True)
class Reuse:
    """Construction-only shared-input qualifier; emits no instruction."""

    value: ir.Expr


def reuse(value: ir.Expr) -> Reuse:
    """Retain this shared version after the consuming operation."""
    return Reuse(value)


def participants(pes: Sequence[int] | None, threads: int) -> tuple[int, ...]:
    """Validate an explicit PE set without silently removing duplicates."""
    values = tuple(range(threads)) if pes is None else tuple(pes)
    if not values or any(type(p) is not int or not 0 <= p < threads for p in values):
        raise ValueError(f"Expected nonempty PE subset of range({threads}), got {values}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate participants: {values}")
    return tuple(sorted(values))


class Builder(ir.IRBuilder):
    """Build a launch function and explicit per-thread tile operations.

    Every emitted call has a stable construction site. Dynamic loop and call
    paths are added by the interpreter, so reusing a Python variable name never
    conflates definitions. Use a separate Builder for each independent program.
    """

    def __init__(self) -> None:
        super().__init__()
        self.threads = 4
        self._site = 0

    @contextmanager
    def launch(
        self,
        name: str,
        *,
        blocks: int = 1,
        threads: int = 4,
        cores: Sequence[int] | None = None,
        pe_mapping: Sequence[int] | None = None,
    ) -> Iterator[FunctionBuilder]:
        """Build an SPMD launch; placement is independent of logical identity."""
        if type(blocks) is not int or blocks <= 0 or type(threads) is not int or threads <= 0:
            raise ValueError(f"Expected positive blocks/threads, got {blocks}/{threads}")
        cores = tuple(range(blocks)) if cores is None else tuple(cores)
        pe_mapping = tuple(range(threads)) if pe_mapping is None else tuple(pe_mapping)
        for mapping, size in ((cores, blocks), (pe_mapping, threads)):
            if (
                len(mapping) != size
                or len(set(mapping)) != size
                or any(type(v) is not int or v < 0 for v in mapping)
            ):
                raise ValueError(f"Expected {size} distinct nonnegative placement IDs, got {mapping}")
        previous = self.threads
        self.threads = threads
        try:
            with self.function(
                name,
                span=_get_span_or_capture(frame_offset=2),
                type=ir.FunctionType.InCore,
                attrs={
                    "ssnpu.blocks": blocks,
                    "ssnpu.threads": threads,
                    "ssnpu.cores": self.expr(cores),
                    "ssnpu.pe_mapping": self.expr(pe_mapping),
                },
            ) as fn:
                yield fn
        finally:
            self.threads = previous

    def expr(self, value: Any) -> ir.Expr:
        """Convert a scalar or tuple to an ordinary IR expression."""
        span = ir.Span.unknown()
        if isinstance(value, ir.Expr):
            return value
        if isinstance(value, (tuple, list)):
            return ir.MakeTuple([self.expr(v) for v in value], span)
        if type(value) is bool:
            return ir.ConstBool(value, span)
        if type(value) is int:
            return ir.ConstInt(value, DataType.INDEX, span)
        if type(value) is float:
            return ir.ConstFloat(value, DataType.FP32, span)
        raise TypeError(f"Expected IR expression, scalar or tuple, got {value!r}")

    def tile_type(self, shape: Sequence[int | ir.Expr], dtype: DataType) -> ir.TileType:
        """Describe a tile without assigning an Ascend storage space."""
        return ir.TileType([self.expr(d) for d in shape], dtype)

    def _call(
        self,
        name: str,
        args: Sequence[Any],
        typ: ir.Type,
        *,
        pes: Sequence[int] | None = ALL,
        kwargs: dict[str, Any] | None = None,
        storage: str | None = None,
    ) -> ir.Var:
        span = _get_span_or_capture(frame_offset=2)
        site = self._site
        self._site += 1
        attrs: dict[str, Any] = {"ssnpu.site": site, "ssnpu.pes": self.expr(participants(pes, self.threads))}
        attrs["ssnpu.reuse"] = self.expr([i for i, arg in enumerate(args) if isinstance(arg, Reuse)])
        if storage is not None:
            attrs["ssnpu.storage"] = storage
        # Target-only operations deliberately bypass the C++ lowering registry:
        # Python cannot register C++ op schemas, and this dialect is capture-only.
        op = ir.Op(name) if name.startswith("ssnpu.") else ir.get_op(name)
        call = ir.Call(
            op,
            [self.expr(a.value if isinstance(a, Reuse) else a) for a in args],
            kwargs or {},
            attrs,
            typ,
            span,
        )
        return self.let(f"v{site}", call, span=span)

    def block_id(self) -> ir.Var:
        """Return the logical block index."""
        return self._call("ssnpu.block_id", [], INDEX)

    def thread_id(self) -> ir.Var:
        """Return the independent intra-block thread index."""
        return self._call("ssnpu.thread_id", [], INDEX)

    def shared_tile(self, shape: Sequence[int], dtype: DataType) -> ir.Var:
        """Declare a block-owned shared handle; this is not a transfer."""
        return self._call("ssnpu.shared_tile", [], self.tile_type(shape, dtype), storage="SharedTileReg")

    def load_into(
        self,
        handle: ir.Expr,
        source: ir.Expr,
        offsets: Sequence[int | ir.Expr],
        *,
        pes: Sequence[int] = (0,),
    ) -> ir.Var:
        """Bind a new immutable version on all threads; one PE emits the load."""
        if len(participants(pes, self.threads)) != 1:
            raise ValueError("load_into requires exactly one producer PE")
        return self._call(
            "ssnpu.load_into", [handle, source, offsets], handle.type, pes=pes, storage="SharedTileReg"
        )

    def load_local(
        self,
        source: ir.Expr,
        offsets: Sequence[int | ir.Expr],
        shape: Sequence[int],
        *,
        pes: Sequence[int] | None = ALL,
    ) -> ir.Var:
        """Load exactly the author's region into the current thread's tile."""
        return self._call(
            "tile.load",
            [source, offsets, shape, shape],
            self.tile_type(shape, source.type.dtype),
            pes=pes,
            storage="LocalTileReg",
        )

    def full(
        self,
        shape: Sequence[int],
        value: int | float,
        dtype: DataType = DataType.FP32,
        *,
        pes: Sequence[int] | None = ALL,
    ) -> ir.Var:
        """Initialize a local tile."""
        return self._call(
            "tile.full",
            [shape, value],
            self.tile_type(shape, dtype),
            kwargs={"dtype": dtype},
            pes=pes,
            storage="LocalTileReg",
        )

    def op(
        self,
        name: str,
        *args: ir.Expr | Reuse | int | float,
        shape: Sequence[int | ir.Expr] | None = None,
        dtype: DataType | None = None,
        pes: Sequence[int] | None = ALL,
        kwargs: dict[str, Any] | None = None,
    ) -> ir.Var:
        """Emit arithmetic or a view; generic handlers validate inferred shapes.

        Reduction scratch operands must be supplied explicitly, as in Tile IR.
        ``shape`` allows explicit result regions; it never partitions operands.
        """
        if not args:
            raise ValueError(f"Tile operation {name} requires an input tile")
        first = args[0].value if isinstance(args[0], Reuse) else args[0]
        typ = first.type
        if shape is None:
            shape = list(typ.shape)
            if name == "matmul":
                rhs = args[1].value if isinstance(args[1], Reuse) else args[1]
                shape = [typ.shape[0], rhs.type.shape[1]]
            elif name == "transpose_view":
                shape = list(reversed(typ.shape))
            elif name in {"row_max", "row_sum"}:
                shape = [typ.shape[0], 1]
        dtype = dtype or (DataType.FP32 if name in {"matmul", "matmul_acc"} else typ.dtype)
        kw = dict(kwargs or {})
        if name == "cast":
            kw["target_type"] = dtype
        return self._call(
            f"tile.{name}",
            args,
            self.tile_type(shape, dtype),
            pes=pes,
            kwargs=kw,
            storage=None if name == "transpose_view" else "LocalTileReg",
        )

    def store(
        self,
        source: ir.Expr,
        dest: ir.Expr,
        offsets: Sequence[int | ir.Expr],
        *,
        pes: Sequence[int] | None = ALL,
    ) -> ir.Var:
        """Store the author's explicit output rectangle."""
        return self._call("tile.store", [source, offsets, dest], dest.type, pes=pes)

    def barrier(self) -> None:
        """Emit a full-block phase barrier."""
        self._call("ssnpu.barrier", [], INDEX)

    def yield_(self, *values: ir.Expr) -> None:
        """Yield loop-carried values using a standard YieldStmt."""
        self.emit(ir.YieldStmt(list(values), self._capture_call_span()))
