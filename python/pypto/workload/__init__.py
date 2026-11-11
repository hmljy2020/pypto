# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Payload-independent Tile IR tracing and optional CPU functional execution."""

from .interpreter import ComputeHandler, InterpreterError, OpHandler, TileInterpreter
from .runtime import OpResult, RuntimeObject, TensorSpec, UnknownValue
from .trace import ExecutionResult

__all__ = [
    "ComputeHandler",
    "TensorSpec",
    "UnknownValue",
    "ExecutionResult",
    "InterpreterError",
    "OpHandler",
    "OpResult",
    "RuntimeObject",
    "TileInterpreter",
]
