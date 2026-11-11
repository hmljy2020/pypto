# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Payload-independent logical objects and operation effects."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from pypto import DataType
from pypto.pypto_core import ir as _ir

# Numerical payload support stays separate from metadata-only dtype support.
DTYPES = {DataType.FP16: torch.float16, DataType.FP32: torch.float32}
DTYPE_NAMES = {dtype: str(value).removeprefix("torch.") for dtype, value in DTYPES.items()}
DTYPE_NAMES.update(
    {
        getattr(DataType, name): name.lower()
        for name in (
            "BOOL",
            "INT8",
            "INT16",
            "INT32",
            "INT64",
            "UINT8",
            "UINT16",
            "UINT32",
            "UINT64",
            "BF16",
        )
    }
)

DTYPE_NAMES[DataType.BF16] = "bfloat16"


@dataclass(frozen=True, eq=False)
class TensorSpec:
    """A two-dimensional input description without numerical storage.

    Args:
        shape: Physical dimensions.
        dtype: Supported PyPTO metadata dtype (float, integer or boolean).
        stride: Element strides; defaults to a contiguous layout.
    """

    shape: tuple[int, ...]
    dtype: DataType
    stride: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if len(shape) != 2 or any(type(v) is not int or v < 0 for v in shape):
            raise ValueError(f"Expected nonnegative 2D integer shape, got {shape}")
        if self.dtype not in DTYPE_NAMES:
            raise ValueError(f"Unsupported metadata dtype, got {self.dtype}")
        stride = (max(shape[1], 1), 1) if self.stride is None else tuple(self.stride)
        if len(stride) != 2 or any(type(v) is not int or v < 0 for v in stride):
            raise ValueError(f"Expected nonnegative 2D integer stride, got {stride}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride", stride)


@dataclass(frozen=True)
class UnknownValue:
    """A payload-dependent scalar with its originating IR operation contexts."""

    sources: tuple[str, ...]

    def __bool__(self) -> bool:
        raise ValueError(f"Unknown payload value from {', '.join(self.sources)} requires a concrete value")


@dataclass
class StorageState:
    """One logical allocation generation, shared by all descriptor views."""

    storage_id: int
    version: int = 0
    producer: int | None = None


@dataclass(eq=False)
class RuntimeObject:
    """A logical descriptor; aliases share storage version and writer state."""

    object_id: int
    kind: str
    shape: tuple[int, ...]
    dtype: DataType
    stride: tuple[int, ...] | None
    valid_shape: tuple[int, ...]
    memory_space: str
    layout: dict[str, Any] = field(default_factory=dict)
    data: torch.Tensor | None = None
    storage: StorageState = field(init=False)
    storage_offset: int | None = 0
    index_map: dict[str, Any] | None = None
    view_of: int | None = None
    view_producer: int | None = None

    def __post_init__(self) -> None:
        self.storage = StorageState(self.object_id)

    @property
    def version(self) -> int:
        return self.storage.version

    @version.setter
    def version(self, value: int) -> None:
        self.storage.version = value

    @property
    def producer(self) -> int | None:
        return self.storage.producer

    @producer.setter
    def producer(self, value: int | None) -> None:
        self.storage.producer = value

    def describe(self) -> dict[str, Any]:
        """Return a detached metadata snapshot (never tensor contents)."""
        return {
            "object_id": self.object_id,
            "version": self.version,
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": DTYPE_NAMES[self.dtype],
            "stride": list(self.stride) if self.stride is not None else None,
            "valid_shape": list(self.valid_shape),
            "memory_space": self.memory_space,
            "layout": deepcopy(self.layout),
            "producer": self.producer,
            "storage_id": self.storage.storage_id,
            "storage_offset": self.storage_offset,
            "index_map": deepcopy(self.index_map),
            "view_of": self.view_of,
            "view_producer": self.view_producer,
        }


@dataclass
class OpResult:
    """Semantic value and effects; the engine owns versions and producer IDs.

    Semantics validate parameters and create descriptors without touching payloads.
    Compute receives this result and fills payloads (or replaces an unknown scalar
    value). Views share storage without writing it. Neither handler may update
    identities, versions or producer links.
    """

    value: Any
    reads: tuple[RuntimeObject, ...] = ()
    writes: tuple[RuntimeObject, ...] = ()
    pto_op: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    views: tuple[RuntimeObject, ...] = ()


class ObjectRuntime:
    """Per-run object identities, with explicit rejection of unsupported views."""

    def __init__(self) -> None:
        self.objects: list[RuntimeObject] = []
        self.inputs: list[tuple[TensorSpec | torch.Tensor, RuntimeObject]] = []

    def create(
        self,
        spec: TensorSpec,
        kind: str,
        valid_shape: tuple[int, ...],
        memory_space: str,
        layout: dict[str, Any] | None = None,
        data: torch.Tensor | None = None,
    ) -> RuntimeObject:
        if len(valid_shape) != 2 or any(
            type(v) is not int or v < 0 or v > s for v, s in zip(valid_shape, spec.shape)
        ):
            raise ValueError(f"Invalid valid shape {valid_shape} for {spec.shape}")
        obj = RuntimeObject(
            object_id=len(self.objects),
            kind=kind,
            shape=spec.shape,
            dtype=spec.dtype,
            stride=spec.stride,
            valid_shape=valid_shape,
            memory_space=memory_space,
            layout=deepcopy(layout or {}),
            data=data,
        )
        self.objects.append(obj)
        return obj

    def tensor(
        self,
        data: TensorSpec | torch.Tensor,
        typ: _ir.TensorType,
        shape: tuple[int, ...],
        mode: str,
    ) -> RuntimeObject:
        if mode == "functional" and typ.dtype not in DTYPES:
            raise ValueError(f"Functional payload dtype {typ.dtype} is not supported; expected FP16/FP32")
        if isinstance(data, torch.Tensor):
            if data.device.type != "cpu" or str(data.dtype).removeprefix("torch.") != DTYPE_NAMES.get(
                typ.dtype
            ):
                raise ValueError(f"Expected CPU tensor with dtype {typ.dtype}")
            spec = TensorSpec(tuple(data.shape), typ.dtype, tuple(data.stride()))
        elif isinstance(data, TensorSpec):
            spec = data
            if mode == "functional":
                raise ValueError(
                    "Missing payload for TensorSpec input; functional mode requires a CPU Tensor"
                )
        else:
            raise ValueError(f"Expected TensorSpec or CPU Tensor, got {type(data).__name__}")
        if spec.dtype != typ.dtype or spec.shape != shape:
            raise ValueError(f"Expected shape {shape}, dtype {typ.dtype}; got {spec.shape}, {spec.dtype}")
        view = typ.tensor_view
        if view is not None and view.layout != _ir.TensorLayout.ND:
            raise ValueError(f"Unsupported packed Tensor layout {view.layout}; expected ND")
        if typ.memref is not None or (view is not None and (view.stride or view.valid_shape)):
            raise ValueError("Explicit Tensor views/MemRefs require a view operation handler")
        if view is not None and view.pad not in (_ir.PadValue.null, _ir.PadValue.zero):
            raise ValueError(f"Unsupported Tensor padding {view.pad}")
        for original, obj in self.inputs:
            if original is data:
                return obj
            if isinstance(original, torch.Tensor) and isinstance(data, torch.Tensor):
                if torch._C._overlaps(original, data):
                    raise ValueError("Distinct input views sharing storage are not supported")
        obj = self.create(
            spec,
            "tensor",
            shape,
            "DDR",
            {"tensor": "ND"},
            data=data if mode == "functional" else None,
        )
        self.inputs.append((data, obj))
        return obj
