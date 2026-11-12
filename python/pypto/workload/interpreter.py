# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile IR semantics, optional numerical execution and operation tracing."""

import operator
import re
from collections.abc import Callable, Mapping
from typing import Any

import torch

from pypto import DataType
from pypto.pypto_core import ir as _ir

from .capture import CaptureRuntime, Launch, TaskId, json_value
from .control import control_ops
from .ops import default_ops
from .runtime import DTYPE_NAMES, DTYPES, ObjectRuntime, OpResult, RuntimeObject, TensorSpec, UnknownValue
from .trace import ExecutionResult, TraceRecorder

OpHandler = Callable[["TileInterpreter", _ir.Call, tuple[Any, ...]], OpResult]
ComputeHandler = Callable[["TileInterpreter", _ir.Call, tuple[Any, ...], OpResult], None]
_BINARY = {
    _ir.Add: operator.add,
    _ir.Sub: operator.sub,
    _ir.Mul: operator.mul,
    _ir.FloorDiv: operator.floordiv,
    _ir.FloorMod: operator.mod,
    _ir.FloatDiv: operator.truediv,
    _ir.Min: min,
    _ir.Max: max,
    _ir.Pow: operator.pow,
    _ir.Eq: operator.eq,
    _ir.Ne: operator.ne,
    _ir.Lt: operator.lt,
    _ir.Le: operator.le,
    _ir.Gt: operator.gt,
    _ir.Ge: operator.ge,
    _ir.Xor: operator.xor,
    _ir.BitAnd: operator.and_,
    _ir.BitOr: operator.or_,
    _ir.BitXor: operator.xor,
    _ir.BitShiftLeft: operator.lshift,
    _ir.BitShiftRight: operator.rshift,
}
_UNARY = {_ir.Abs: abs, _ir.Neg: operator.neg, _ir.Not: operator.not_, _ir.BitNot: operator.invert}


class InterpreterError(RuntimeError):
    """Unsupported or invalid execution, annotated with IR context."""


class _Transfer(Exception):
    def __init__(self, kind: str, values: tuple[Any, ...]) -> None:
        self.kind, self.values = kind, values


def _unwrap(value: Any, mode: str, context: str) -> Any:
    if isinstance(value, TaskId):
        return json_value(value)
    if isinstance(value, RuntimeObject):
        if mode == "trace_only":
            return value.describe()
        if value.data is None:
            raise InterpreterError(f"Missing payload for object {value.object_id} consumed by {context}")
        return value.data
    if isinstance(value, tuple):
        return tuple(_unwrap(v, mode, context) for v in value)
    if isinstance(value, list):
        return [_unwrap(v, mode, context) for v in value]
    if isinstance(value, dict):
        return {k: _unwrap(v, mode, context) for k, v in value.items()}
    return value


class TileInterpreter:
    """Execute IR semantics with optional numerical handlers and trace capture.

    Args:
        max_steps: Maximum evaluated expressions/statements per run. An empty
            loop also consumes steps; calls share the same budget.
        capture_config: Explicit positive device query values, keyed by
            available_cluster_count or available_aiv_count. No device defaults
            are inferred from the machine running the interpreter.
    """

    def __init__(
        self, max_steps: int = 1_000_000, *, capture_config: Mapping[str, int] | None = None
    ) -> None:
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError(f"max_steps must be a positive integer, got {max_steps}")
        self.max_steps = max_steps
        self.registry: dict[str, tuple[OpHandler, ComputeHandler | None]] = default_ops()
        self.registry.update(control_ops())
        self.capture_config = dict(capture_config or {})
        allowed = {"available_cluster_count", "available_aiv_count"}
        if set(self.capture_config) - allowed:
            raise ValueError(
                f"Unknown capture configuration fields: {sorted(set(self.capture_config) - allowed)}"
            )
        for key, value in self.capture_config.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"capture_config[{key!r}] must be a positive integer, got {value!r}")
        self.env: dict[int, Any] = {}
        self.function = "<entry>"
        self.steps = 0
        self.objects = ObjectRuntime()
        self.recorder = TraceRecorder(False)
        self.mode = "trace_only"
        self.next_op_id = 0
        self._active_call: tuple[_ir.Call, dict[str, Any]] | None = None
        self.call_stack: list[str] = []
        self.capture = CaptureRuntime(self, False)

    def register_op(self, name: str, *, semantics: OpHandler, compute: ComputeHandler | None = None) -> None:
        """Register semantics and an optional numerical implementation.

        Semantics receives ``(interpreter, call, args)`` and returns OpResult.
        Compute receives ``(interpreter, call, args, result)`` and fills payloads
        or replaces ``result.value`` for a payload-dependent scalar.
        """
        if not name or not callable(semantics) or (compute is not None and not callable(compute)):
            raise ValueError("Expected a nonempty op name, callable semantics and optional callable compute")
        self.registry[name] = (semantics, compute)

    def run(
        self,
        program: _ir.Program,
        entry: str,
        inputs: Mapping[str, Any],
        capture_trace: bool = True,
        *,
        mode: str = "trace_only",
    ) -> ExecutionResult:
        """Execute an entry in trace_only (default) or functional mode.

        Only functional stores update input Tensor payloads. Trace-only object
        returns are detached descriptions; scalar returns retain their values.

        Inputs match exact parameter names or unambiguous names before the
        compiler's ``__ssa_vN`` suffix. Each run resets objects, steps and trace.
        """
        if not isinstance(program, _ir.Program):
            raise TypeError(f"Expected ir.Program, got {type(program).__name__}")
        if mode not in {"trace_only", "functional"}:
            raise ValueError(f"Expected mode trace_only or functional, got {mode!r}")
        self.mode, self.next_op_id = mode, 0
        self.program = program
        self.objects, self.recorder = ObjectRuntime(), TraceRecorder(capture_trace)
        self.env, self.steps, self.function = {}, 0, entry
        self.call_stack, self._active_call = [], None
        self.capture = CaptureRuntime(self, capture_trace)
        fn = program.get_function(entry)
        if fn is None:
            raise InterpreterError(f"Entry function {entry!r} not found")
        values, used = [], set()
        for param in fn.params:
            name = param.name_hint
            key = name if name in inputs else re.sub(r"__ssa_v\d+$", "", name)
            if key not in inputs or key in used:
                raise InterpreterError(f"{entry}: missing or ambiguous input {name!r}")
            used.add(key)
            value = inputs[key]
            if isinstance(param.type, _ir.TensorType):
                try:
                    value = self.objects.tensor(
                        value,
                        param.type,
                        tuple(self.concrete(self.evaluate(v), v, "input shape") for v in param.type.shape),
                        self.mode,
                    )
                except ValueError as error:
                    raise InterpreterError(
                        f"input {name} consumed by {entry} at {param.span}: {error}"
                    ) from error
            elif not isinstance(param.type, _ir.ScalarType) or type(value) not in (int, float, bool):
                raise InterpreterError(f"{entry}: unsupported parameter {name}: {param.type}")
            values.append(value)
            self.env[param.unique_id] = value
        if set(inputs) != used:
            raise InterpreterError(f"{entry}: unexpected inputs {sorted(set(inputs) - used)}")
        with torch.no_grad():
            if self._is_orchestration(fn):
                outputs = self._invoke(fn, tuple(values))
            else:
                try:
                    outputs, _ = self.capture.launch(fn, tuple(values), Launch(), fn)
                except Exception as error:
                    raise InterpreterError(str(error)) from error
        return ExecutionResult(
            tuple(_unwrap(v, self.mode, f"return in {entry} at {fn.span}") for v in outputs),
            self.recorder.records,
            self.mode,
            self.capture.tasks,
            self.capture.events.records,
            {
                "format_version": 1,
                "complete": True,
                "entry": entry,
                "capture_trace": capture_trace,
                "max_steps": self.max_steps,
                "device": dict(self.capture_config),
                "scheduling_policy": "launch_encounter_order_then_ascending_block_id",
                "synchronization_policy": "record_and_continue",
                "storage_dependency_policy": "whole_storage_in_capture_order",
                "subblock_policy": "unsplit_zero",
            },
            [obj.describe() for obj in self.objects.objects],
        )

    def _tick(self, node: Any) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise InterpreterError(f"{self.function} at {node.span}: step limit {self.max_steps} exceeded")

    def _bind(self, variables: Any, values: tuple[Any, ...]) -> None:
        if len(variables) != len(values):
            raise InterpreterError(f"{self.function}: binding arity {len(variables)} != {len(values)}")
        self.env.update((var.unique_id, value) for var, value in zip(variables, values))

    def _invoke(self, fn: _ir.Function, args: tuple[Any, ...]) -> tuple[Any, ...]:
        previous_env, previous_function = self.env, self.function
        self.env, self.function = {}, fn.name
        self.call_stack.append(fn.name)
        try:
            unknown = set(fn.attrs) - {"mx_tensor_views_blocked", "spmd_unwrapped", "core_num", "sync_start"}
            if unknown:
                raise InterpreterError(
                    f"{fn.name} at {fn.span}: unsupported function attributes {sorted(unknown)}"
                )
            self._bind(fn.params, args)
            try:
                self._execute(fn.body)
            except _Transfer as transfer:
                if transfer.kind != "return":
                    raise InterpreterError(
                        f"{fn.name}: unexpected {transfer.kind} outside control flow"
                    ) from None
                if len(transfer.values) != len(fn.return_types):
                    raise InterpreterError(f"{fn.name}: return arity does not match signature")
                return transfer.values
            if fn.return_types:
                raise InterpreterError(f"{fn.name}: missing return")
            return ()
        finally:
            self.call_stack.pop()
            self.env, self.function = previous_env, previous_function

    def evaluate_attribute(self, value: Any) -> Any:
        """Evaluate expressions recursively in handler attributes."""
        if isinstance(value, _ir.Expr):
            return self.concrete(self.evaluate(value), value, "operation attribute")
        if isinstance(value, (list, tuple)):
            return tuple(self.evaluate_attribute(v) for v in value)
        if isinstance(value, dict):
            return {k: self.evaluate_attribute(v) for k, v in value.items()}
        return value

    def call_attributes(self, call: _ir.Call) -> dict[str, Any]:
        """Return the current call's once-evaluated keyword arguments."""
        if self._active_call is None or self._active_call[0] is not call:
            raise ValueError("Attributes are only available to the current operation handler")
        return dict(self._active_call[1])

    def new_tile(self, typ: _ir.TileType) -> RuntimeObject:
        """Create an independent tile descriptor without allocating a payload."""
        if not isinstance(typ, _ir.TileType) or typ.dtype not in DTYPE_NAMES:
            raise ValueError(f"Expected supported metadata TileType, got {typ}")
        shape = tuple(self.concrete(self.evaluate(v), v, "tile shape") for v in typ.shape)
        view = typ.get_effective_tile_view()
        valid = tuple(self.concrete(self.evaluate(v), v, "valid shape") for v in view.valid_shape)
        if view.stride or (
            view.start_offset is not None
            and self.concrete(self.evaluate(view.start_offset), view.start_offset, "tile offset") != 0
        ):
            raise ValueError("Explicit tile strides/offsets require a view operation handler")
        if typ.memref is not None:
            raise ValueError("Physical MemRef-backed Tile IR is not supported")
        space = typ.memory_space.name if typ.memory_space is not None else "Unspecified"
        if space not in {"Unspecified", "Vec", "Mat", "Left", "Right", "Acc"}:
            raise ValueError(f"Unsupported tile memory space {space}")
        # Layout is part of the descriptor even when the compiler selects a
        # non-default Cube/Vector transfer layout. Payloads remain logical ND.
        for key in ("blayout", "slayout"):
            if not isinstance(getattr(view, key), _ir.TileLayout):
                raise ValueError(f"Unsupported tile {key}: {getattr(view, key)}")
        if view.pad not in (_ir.PadValue.null, _ir.PadValue.zero):
            raise ValueError(f"Unsupported tile padding {view.pad}")
        # Layout is logical metadata, not an instruction to pack the CPU tensor.
        layout = {k: str(getattr(view, k)) for k in ("blayout", "slayout", "fractal", "pad", "compact")}
        obj = self.objects.create(TensorSpec(shape, typ.dtype), "tile", valid, space, layout)
        obj.owner_instance = self.capture.identity()
        return obj

    def payload(self, obj: RuntimeObject) -> torch.Tensor:
        """Require an existing numerical payload (errors gain consumer context)."""
        if obj.data is None:
            raise ValueError(f"Missing payload for {obj.kind} object {obj.object_id}, version {obj.version}")
        return obj.data

    def allocate_payload(self, obj: RuntimeObject) -> torch.Tensor:
        """Allocate zero-filled storage for a new contiguous tile in compute."""
        if self.mode != "functional":
            raise ValueError("Payload allocation requires functional mode")
        if obj.data is not None:
            raise ValueError(f"Object {obj.object_id} already has a payload")
        if obj.kind != "tile" or obj.stride != TensorSpec(obj.shape, obj.dtype).stride:
            raise ValueError(f"Expected contiguous tile descriptor, got object {obj.object_id}")
        obj.data = torch.zeros(obj.shape, dtype=DTYPES[obj.dtype])
        return obj.data

    def unknown(self, call: _ir.Call) -> UnknownValue:
        """Describe a payload-dependent scalar from the current operation."""
        return UnknownValue(
            (
                f"{call.op.name} (before op {self.next_op_id}, task/block {self.capture.identity()}) "
                f"in {self.function} at {call.span}",
            )
        )

    def concrete(self, value: Any, node: Any, purpose: str) -> Any:
        """Reject unknown scalars only where execution needs a concrete value."""
        if isinstance(value, UnknownValue):
            raise InterpreterError(
                f"{purpose} in {self.function} at {node.span}: unknown payload value "
                f"from {', '.join(value.sources)}; a concrete value is required"
            )
        if isinstance(value, (tuple, list)):
            for item in value:
                self.concrete(item, node, purpose)
        elif isinstance(value, dict):
            for item in value.values():
                self.concrete(item, node, purpose)
        return value

    @staticmethod
    def _unknown_from(*values: Any) -> UnknownValue | None:
        sources = tuple(dict.fromkeys(s for v in values if isinstance(v, UnknownValue) for s in v.sources))
        return UnknownValue(sources) if sources else None

    def evaluate(self, expr: _ir.Expr) -> Any:
        try:
            return self._evaluate(expr)
        except Exception as error:
            raise InterpreterError(
                f"task/block {self.capture.identity()}, {type(expr).__name__} in "
                f"{self.function} at {expr.span}: {error}"
            ) from error

    def _evaluate(self, expr: _ir.Expr) -> Any:
        self._tick(expr)
        if isinstance(expr, _ir.Var):
            return self.env[expr.unique_id]
        if isinstance(expr, (_ir.ConstInt, _ir.ConstFloat, _ir.ConstBool)):
            return expr.value
        if isinstance(expr, _ir.MakeTuple):
            return tuple(self.evaluate(v) for v in expr.elements)
        if isinstance(expr, _ir.TupleGetItemExpr):
            return self.evaluate(expr.tuple)[expr.index]
        if isinstance(expr, _ir.And):
            return bool(self.concrete(self.evaluate(expr.left), expr, "boolean condition")) and bool(
                self.concrete(self.evaluate(expr.right), expr, "boolean condition")
            )
        if isinstance(expr, _ir.Or):
            return bool(self.concrete(self.evaluate(expr.left), expr, "boolean condition")) or bool(
                self.concrete(self.evaluate(expr.right), expr, "boolean condition")
            )
        if type(expr) in _BINARY:
            lhs, rhs = self.evaluate(expr.left), self.evaluate(expr.right)
            unknown = self._unknown_from(lhs, rhs)
            return unknown if unknown is not None else _BINARY[type(expr)](lhs, rhs)
        if type(expr) in _UNARY:
            value = self.evaluate(expr.operand)
            return value if isinstance(value, UnknownValue) else _UNARY[type(expr)](value)
        if isinstance(expr, _ir.Cast):
            value = self.evaluate(expr.operand)
            if isinstance(value, UnknownValue):
                return value
            if expr.type.dtype == DataType.BOOL:
                return bool(value)
            if expr.type.dtype.is_float():
                return float(value)
            return int(value)
        if isinstance(expr, (_ir.Call, _ir.Submit)):
            return self._call(expr)

        raise InterpreterError(
            f"{self.function} at {expr.span}: unsupported expression {type(expr).__name__}"
        )

    @staticmethod
    def _is_orchestration(fn: _ir.Function) -> bool:
        return fn.func_type in {_ir.FunctionType.Orchestration, _ir.FunctionType.Graph} or (
            fn.func_type == _ir.FunctionType.Opaque and fn.role == _ir.Role.Orchestrator
        )

    def _function_call(self, expr: _ir.Call | _ir.Submit, args: tuple[Any, ...]) -> Any:
        fn = self.program.get_function(expr.op.name)
        if fn is None:
            raise ValueError(f"Unknown function {expr.op.name}")
        attrs = self.evaluate_attribute(dict(expr.attrs))
        kwargs = self.evaluate_attribute(dict(expr.kwargs))
        if kwargs:
            raise ValueError(f"Unsupported function keyword arguments: {sorted(kwargs)}")
        allowed = {"core_num", "sync_start", "predicate", "deps", "arg_directions", "allow_early_resolve"}
        if set(attrs) - allowed:
            raise ValueError(f"Unsupported call attributes: {sorted(set(attrs) - allowed)}")
        caller = self.program.get_function(self.function)
        is_submit = isinstance(expr, _ir.Submit)
        kernel_types = {
            _ir.FunctionType.InCore,
            _ir.FunctionType.AIC,
            _ir.FunctionType.AIV,
            _ir.FunctionType.Spmd,
            _ir.FunctionType.Group,
            _ir.FunctionType.Graph,
        }
        orchestration = self._is_orchestration(caller)
        launch = (
            is_submit
            or bool(set(attrs) & (allowed - {"arg_directions"}))
            or (orchestration and (fn.func_type in kernel_types or fn.role == _ir.Role.SubWorker))
        )
        if launch:
            if is_submit:
                core_num = self.evaluate(expr.core_num) if expr.core_num is not None else 1
            elif "core_num" in attrs:
                core_num = attrs["core_num"]
            else:
                core_num = self.evaluate_attribute(fn.attrs.get("core_num", 1))
            predicate = (
                self.evaluate(expr.predicate)
                if is_submit and expr.predicate is not None
                else attrs.get("predicate", True)
            )
            dependencies = (
                tuple(self.evaluate(dep) for dep in expr.deps) if is_submit else attrs.get("deps", ())
            )
            self.concrete((core_num, predicate, dependencies), expr, "task launch")
            spec = Launch(
                core_num=core_num,
                sync_start=expr.sync_start
                if is_submit
                else attrs.get("sync_start", fn.attrs.get("sync_start", False)),
                predicate=predicate,
                dependencies=self.capture.dependencies(dependencies),
                allow_early_resolve=expr.allow_early_resolve
                if is_submit
                else attrs.get("allow_early_resolve", False),
            )
            values, task_id = self.capture.launch(fn, args, spec, expr)
            if is_submit:
                return (*values, task_id) if values else task_id
        else:
            values = self._invoke(fn, args)
        return values[0] if len(values) == 1 else values

    def _call(self, expr: _ir.Call | _ir.Submit) -> Any:
        previous_call = self._active_call
        try:
            args = tuple(self.evaluate(v) for v in expr.args)
            if isinstance(expr.op, _ir.GlobalVar):
                return self._function_call(expr, args)
            if isinstance(expr, _ir.Submit):
                raise ValueError("Submit requires a GlobalVar callee")
            handlers = self.registry.get(expr.op.name)
            if handlers is None:
                raise ValueError("No registered semantics implementation")
            semantics, compute = handlers
            # Arrays can carry uninitialized slots until consumed. Indexes and
            # dependency lists are checked at their actual point of use.
            if not expr.op.name.startswith("array."):
                self.concrete(args, expr, f"{expr.op.name} instruction arguments")
            kwargs = self.evaluate_attribute(dict(expr.kwargs))
            self._active_call = (expr, kwargs)
            before = {obj.object_id: obj.describe() for obj in self.objects.objects}
            parameters = {
                "args": json_value(args),
                "kwargs": json_value(kwargs),
            }
            result = semantics(self, expr, args)
            if not isinstance(result, OpResult):
                raise TypeError("Operation semantics must return OpResult")
            self.concrete(result.parameters, expr, f"{expr.op.name} trace parameters")
            if self.mode == "functional":
                if compute is None:
                    raise ValueError("Missing numerical implementation (compute) for functional mode")
                for obj in result.reads:
                    self.payload(obj)
                compute(self, expr, args, result)
                if not expr.op.name.startswith("array."):
                    self.concrete(result.value, expr, f"{expr.op.name} functional result")
            self._finish(expr, result, before, parameters)
            return result.value
        except Exception as error:
            raise InterpreterError(f"{expr.op.name} in {self.function} at {expr.span}: {error}") from error
        finally:
            self._active_call = previous_call

    def _finish(
        self,
        call: _ir.Call,
        result: OpResult,
        before: dict[int, dict[str, Any]],
        parameters: dict[str, Any],
    ) -> None:
        """Advance effects independently of whether the recorder is enabled."""
        if result.control_kind is not None:
            if result.reads or result.writes or result.views:
                raise ValueError("Control results cannot carry tile storage effects")
            self.capture.event(
                result.control_kind,
                call,
                ir_op=call.op.name,
                parameters={**parameters, **json_value(result.parameters)},
                result=json_value(result.value),
                objects=[result.value.describe()] if isinstance(result.value, RuntimeObject) else [],
            )
            return
        op_id = self.next_op_id
        reads = [before[obj.object_id] for obj in result.reads]
        dependencies = {
            before[obj.object_id]["producer"]
            for obj in (*result.reads, *result.writes)
            if obj.object_id in before and before[obj.object_id]["producer"] is not None
        }
        # Descriptor creation and storage writes are distinct effects. Aliases
        # share storage state; a multi-view write advances that storage once.
        dependencies.update(
            before[obj.object_id]["view_producer"]
            for obj in (*result.reads, *result.writes)
            if obj.object_id in before and before[obj.object_id]["view_producer"] is not None
        )
        writes = list(dict.fromkeys(result.writes))
        updated = set()
        for obj in writes:
            if obj.storage.storage_id not in updated:
                obj.version += 1
                obj.producer = op_id
                updated.add(obj.storage.storage_id)
        for obj in result.views:
            obj.view_producer = op_id
        self.next_op_id += 1
        self.recorder.record(
            {
                "op_id": op_id,
                **self.capture.identity(),
                "instance_op_sequence": self.capture.current.op_sequence if self.capture.current else None,
                "ir_op": call.op.name,
                "pto_op": result.pto_op,
                "function": self.function,
                "location": str(call.span),
                "parameters": {**parameters, **json_value(result.parameters)},
                "inputs": reads,
                "outputs": [obj.describe() for obj in (*writes, *result.views)],
                "dependencies": sorted(dependencies),
            }
        )
        if self.capture.current is not None:
            self.capture.current.op_sequence += 1

    def _body_values(self, body: _ir.Stmt) -> tuple[Any, ...]:
        try:
            self._execute(body)
        except _Transfer as transfer:
            if transfer.kind == "yield":
                return transfer.values
            raise
        return ()

    def _execute(self, stmt: _ir.Stmt) -> None:
        try:
            self._execute_stmt(stmt)
        except _Transfer:
            raise
        except Exception as error:
            raise InterpreterError(
                f"task/block {self.capture.identity()}, {type(stmt).__name__} in "
                f"{self.function} at {stmt.span}: {error}"
            ) from error

    def _execute_stmt(self, stmt: _ir.Stmt) -> None:
        self._tick(stmt)
        if isinstance(stmt, _ir.SeqStmts):
            for child in stmt.stmts:
                self._execute(child)
        elif isinstance(stmt, _ir.AssignStmt):
            self.env[stmt.var.unique_id] = self.evaluate(stmt.value)
        elif isinstance(stmt, _ir.EvalStmt):
            self.evaluate(stmt.expr)
        elif isinstance(stmt, (_ir.ReturnStmt, _ir.YieldStmt)):
            raise _Transfer(
                "return" if isinstance(stmt, _ir.ReturnStmt) else "yield",
                tuple(self.evaluate(v) for v in stmt.value),
            )
        elif isinstance(stmt, _ir.IfStmt):
            condition = self.concrete(self.evaluate(stmt.condition), stmt, "if condition")
            body = stmt.then_body if condition else stmt.else_body
            self._bind(stmt.return_vars, self._body_values(body) if body is not None else ())
        elif isinstance(stmt, (_ir.ForStmt, _ir.WhileStmt)):
            self._execute_loop(stmt)
        elif isinstance(stmt, _ir.RuntimeScopeStmt):
            self.capture.enter_scope(stmt)
            try:
                self._execute(stmt.body)
            finally:
                self.capture.leave_scope(stmt)
        elif type(stmt) is _ir.ScopeStmt or isinstance(stmt, (_ir.InCoreScopeStmt, _ir.HierarchyScopeStmt)):
            self._execute(stmt.body)
        else:
            raise InterpreterError(
                f"{self.function} at {stmt.span}: unsupported statement {type(stmt).__name__}"
            )

    def _execute_loop(self, stmt: _ir.ForStmt | _ir.WhileStmt) -> None:
        values = tuple(self.evaluate(v.initValue) for v in stmt.iter_args)
        self._bind(stmt.iter_args, values)
        if isinstance(stmt, _ir.ForStmt):
            if stmt.kind != _ir.ForKind.Sequential:
                raise InterpreterError(f"{self.function}: only sequential for loops are supported")
            iterator = iter(
                range(
                    *(
                        self.concrete(self.evaluate(v), stmt, "for loop bound")
                        for v in (stmt.start, stmt.stop, stmt.step)
                    )
                )
            )
        while True:
            self._tick(stmt)
            if isinstance(stmt, _ir.ForStmt):
                try:
                    index = next(iterator)
                except StopIteration:
                    break
                self.env[stmt.loop_var.unique_id] = index
            elif not self.concrete(self.evaluate(stmt.condition), stmt, "while condition"):
                break
            values = self._body_values(stmt.body)
            self._bind(stmt.iter_args, values)
        self._bind(stmt.return_vars, values)
