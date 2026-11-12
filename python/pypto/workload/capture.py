# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deterministic logical launches and control events for early Tile IR."""

from dataclasses import dataclass
from typing import Any

from pypto.pypto_core import ir as _ir

from .runtime import RuntimeObject, SharedHandle, UnknownValue
from .trace import TraceRecorder


@dataclass(frozen=True)
class TaskId:
    """Typed task reference; None is the IR's explicit no-producer sentinel."""

    value: int | None


@dataclass
class Instance:
    task_id: int
    block_id: int
    block_num: int
    subblock_id: int = 0
    op_sequence: int = 0
    thread_id: int | None = None
    thread_num: int = 1

    def identity(self) -> dict[str, int]:
        return {
            **{k: getattr(self, k) for k in ("task_id", "block_id", "subblock_id")},
            **({"thread_id": self.thread_id} if self.thread_id is not None else {}),
        }


@dataclass(frozen=True)
class Launch:
    core_num: int = 1
    sync_start: bool = False
    predicate: bool = True
    dependencies: tuple[int, ...] = ()
    allow_early_resolve: bool = False


def json_value(value: Any) -> Any:
    """Snapshot runtime values without serializing payloads or Python identities."""
    if isinstance(value, SharedHandle):
        return value.describe()
    if isinstance(value, RuntimeObject):
        return {"object_id": value.object_id, "version": value.version}
    if isinstance(value, TaskId):
        return {"task_id": value.value}
    if isinstance(value, UnknownValue):
        return {"unknown_sources": list(value.sources)}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class CaptureRuntime:
    """One run's task tree, instance position, and nested runtime scopes.

    Launches run to completion in encounter order and blocks in ascending order.
    Events describe synchronization but never suspend a block. Objects and the
    interpreter's step budget remain global across all function frames.
    """

    def __init__(self, ctx: Any, enabled: bool) -> None:
        self.ctx = ctx
        self.current: Instance | None = None
        self.tasks: list[dict[str, Any]] = []
        self.events = TraceRecorder(enabled)
        self.scopes: list[int] = []
        self.next_scope_id = 0

    def identity(self) -> dict[str, Any]:
        if self.current is not None:
            return self.current.identity()
        return {"task_id": None, "block_id": None, "subblock_id": None}

    def event(self, kind: str, node: Any, **fields: Any) -> None:
        self.events.record(
            {
                "event_id": len(self.events.records),
                "kind": kind,
                **self.identity(),
                "function": self.ctx.function,
                "call_stack": list(self.ctx.call_stack),
                "location": str(node.span),
                "before_op_id": self.ctx.next_op_id,
                "instance_op_sequence": self.current.op_sequence if self.current else None,
                "scope_id": self.scopes[-1] if self.scopes else None,
                **json_value(fields),
            }
        )

    def dependencies(self, values: Any) -> tuple[int, ...]:
        result = []
        for value in values:
            if isinstance(value, (tuple, list)):
                result.extend(self.dependencies(value))
            elif isinstance(value, TaskId):
                if value.value is None:
                    continue
                if not 0 <= value.value < len(self.tasks):
                    raise ValueError(f"Invalid TaskId {value.value}")
                if self.tasks[value.value]["status"] not in {"completed", "skipped"}:
                    raise ValueError(f"TaskId {value.value} has not completed in capture order")
                result.append(value.value)
            else:
                raise ValueError(f"Expected TaskId dependency, got {value!r}")
        return tuple(dict.fromkeys(result))

    def launch(self, fn: _ir.Function, args: tuple[Any, ...], spec: Launch, node: Any) -> tuple[Any, TaskId]:
        if len(args) != len(fn.params):
            raise ValueError(f"{fn.name}: binding arity {len(fn.params)} != {len(args)}")
        if type(spec.core_num) is not int or spec.core_num < 0:
            raise ValueError(f"core_num must be a nonnegative integer, got {spec.core_num!r}")
        if type(spec.predicate) is not bool or type(spec.sync_start) is not bool:
            raise ValueError("predicate and sync_start must be boolean")
        task_id = len(self.tasks)
        parent = self.current
        task = {
            "task_id": task_id,
            "entry": fn.name,
            "function_type": fn.func_type.name,
            "arguments": json_value(args),
            "parent_task_id": parent.task_id if parent else None,
            "parent_instance": parent.identity() if parent else None,
            "core_num": spec.core_num,
            "sync_start": spec.sync_start,
            "predicate": spec.predicate,
            "allow_early_resolve": spec.allow_early_resolve,
            "dependencies": list(spec.dependencies),
            "scope_id": self.scopes[-1] if self.scopes else None,
            "location": str(node.span),
            "status": "running",
            "instances": [],
        }
        self.tasks.append(task)
        self.event("task_launch", node, launched_task_id=task_id)
        outputs = ()
        try:
            if not spec.core_num or not spec.predicate:
                # Reuse the compiler's alias analysis, which follows stores,
                # SSA returns and wrapper calls. Never guess by Out-param order.
                indices = _ir.returned_param_indices(fn, self.ctx.program)
                if len(indices) != len(fn.return_types) or any(i is None for i in indices):
                    raise ValueError(f"Cannot determine skipped return bindings for {fn.name}")
                outputs = tuple(args[i] for i in indices)
                task["status"] = "skipped"
                task["skip_reason"] = "predicate" if not spec.predicate else "zero_blocks"
            else:
                outputs = self._launch_instances(fn, args, spec, node, task)
                task["status"] = "completed"
            task["outputs"] = json_value(outputs)
            task["completion_op_id"] = self.ctx.next_op_id
        except Exception as error:
            task["status"] = "failed"
            identity = self.identity() if self.current is not parent else {"task_id": task_id}
            raise ValueError(f"task/block {identity}, launch {fn.name} at {node.span}: {error}") from error
        finally:
            self.current = parent
        self.event("task_complete", node, completed_task_id=task_id, status=task["status"])
        return outputs, TaskId(task_id)

    def _launch_instances(
        self, fn: _ir.Function, args: tuple[Any, ...], spec: Launch, node: Any, task: dict[str, Any]
    ) -> tuple[Any, ...]:
        """Instantiate blocks and optional target threads with separate placement."""
        task_id = task["task_id"]
        outputs = ()
        thread_num = fn.attrs.get("ssnpu.threads")
        if thread_num is not None:
            if type(thread_num) is not int or thread_num <= 0:
                raise ValueError(f"Invalid SSNPU thread count: {thread_num}")
            self.ctx.ssnpu.enabled = True
            order = tuple(range(thread_num)) if self.ctx.thread_order is None else self.ctx.thread_order
            if sorted(order) != list(range(thread_num)):
                raise ValueError(f"thread_order must be a permutation of range({thread_num})")
        else:
            order = (None,)
        for block_id in range(spec.core_num):
            for thread_id in order:
                self.current = Instance(
                    task_id, block_id, spec.core_num, thread_id=thread_id, thread_num=thread_num or 1
                )
                self.ctx._tick(node)
                instance = {**self.current.identity(), "status": "running", "operation_count": 0}
                if thread_id is not None:
                    cores = self.ctx.evaluate_attribute(
                        fn.attrs.get("ssnpu.cores", list(range(spec.core_num)))
                    )
                    pes = self.ctx.evaluate_attribute(
                        fn.attrs.get("ssnpu.pe_mapping", list(range(thread_num)))
                    )
                    for mapping, size in ((cores, spec.core_num), (pes, thread_num)):
                        if (
                            len(mapping) != size
                            or len(set(mapping)) != size
                            or any(type(v) is not int or v < 0 for v in mapping)
                        ):
                            raise ValueError(f"Invalid SSNPU placement: {mapping}")
                    instance["mapping"] = {"core_id": cores[block_id], "pe_id": pes[thread_id]}
                task["instances"].append(instance)
                try:
                    outputs = self.ctx._invoke(fn, args)
                except Exception:
                    instance["status"] = "failed"
                    raise
                finally:
                    instance["operation_count"] = self.current.op_sequence
                instance["status"] = "completed"
                self.event("instance_complete", node)
        return outputs

    def enter_scope(self, stmt: _ir.RuntimeScopeStmt) -> None:
        scope_id = self.next_scope_id
        self.next_scope_id += 1
        self.event(
            "scope_enter",
            stmt,
            entered_scope_id=scope_id,
            parent_scope_id=self.scopes[-1] if self.scopes else None,
            mode="MANUAL" if stmt.manual else "AUTO",
        )
        self.scopes.append(scope_id)

    def leave_scope(self, stmt: _ir.RuntimeScopeStmt) -> None:
        self.event("scope_exit", stmt)
        self.scopes.pop()
