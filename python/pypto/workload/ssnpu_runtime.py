# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SSNPU target metadata, symbolic versions and post-capture validation.

Capture order is not execution order. Stable dynamic keys link producers and
collectives after all threads have run. Lifetime validation uses thread order
and full-phase barriers; it never compares global op IDs to order readers.
"""

import heapq
import json
from collections import defaultdict
from typing import Any

from pypto.pypto_core import ir as _ir

from .ops import _rectangle
from .runtime import OpResult, RuntimeObject, SharedHandle
from .ssnpu import participants

FUNCTION_ATTRS = {"ssnpu.blocks", "ssnpu.threads", "ssnpu.cores", "ssnpu.pe_mapping"}
CALL_ATTRS = {"ssnpu.site", "ssnpu.pes", "ssnpu.reuse", "ssnpu.storage"}


class SSNPURuntime:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.handles: dict[str, SharedHandle] = {}
        self.versions: dict[str, dict[str, Any]] = {}
        self.barriers: dict[tuple[int, int, int], list[str]] = defaultdict(list)
        self.records: list[dict[str, Any]] = []
        self.enabled = False
        # Python-only op schemas are intentionally looked up by handler name;
        # there is no C++ registry registration API for this capture dialect.
        self.handlers = {
            "ssnpu.block_id": self.block_id,
            "ssnpu.thread_id": self.thread_id,
            "ssnpu.shared_tile": self.shared_tile,
            "ssnpu.load_into": self.load_into,
            "ssnpu.barrier": self.barrier,
        }

    @property
    def active(self) -> bool:
        return self.ctx.capture.current is not None and self.ctx.capture.current.thread_id is not None

    def owner(self) -> tuple[int, int]:
        instance = self.ctx.capture.current
        return instance.task_id, instance.block_id

    def thread_key(self) -> tuple[int, int, int]:
        return (*self.owner(), self.ctx.capture.current.thread_id)

    def dynamic_id(self, call: _ir.Call) -> str:
        # JSON is lossless and deterministic, unlike Python's randomized hash.
        return json.dumps(
            [*self.owner(), self.ctx.function, list(self.ctx.dynamic_path), call.attrs.get("ssnpu.site")],
            separators=(",", ":"),
        )

    def metadata(self, call: _ir.Call) -> dict[str, Any]:
        attrs = self.ctx.evaluate_attribute(dict(call.attrs))
        unknown = {k for k in attrs if k.startswith("ssnpu.")} - CALL_ATTRS
        if unknown:
            raise ValueError(f"Unknown SSNPU attributes: {sorted(unknown)}")
        if not self.active:
            if set(attrs) & CALL_ATTRS or call.op.name.startswith("ssnpu."):
                raise ValueError("SSNPU operation requires an SSNPU launch/thread context")
            return {}
        if "ssnpu.site" not in attrs:
            raise ValueError(f"SSNPU call {call.op.name} requires a stable ssnpu.site")
        if call.op.name in self.handlers:
            arity = 3 if call.op.name == "ssnpu.load_into" else 0
            if len(call.args) != arity or call.kwargs or set(attrs) - CALL_ATTRS:
                raise ValueError(f"{call.op.name} requires {arity} arguments and only SSNPU attributes")
        threads = self.ctx.capture.current.thread_num
        pes = participants(attrs.get("ssnpu.pes"), threads)
        retained = tuple(attrs.get("ssnpu.reuse", ()))
        if len(set(retained)) != len(retained) or any(
            type(i) is not int or not 0 <= i < len(call.args) for i in retained
        ):
            raise ValueError(f"Invalid reuse operand indices: {retained}")
        storage = attrs.get("ssnpu.storage")
        if storage not in {None, "LocalTileReg", "SharedTileReg"}:
            raise ValueError(f"Invalid SSNPU storage: {storage}")
        return {
            "pes": list(pes),
            "reuse": retained,
            "storage": storage,
            "site": attrs["ssnpu.site"],
            "dynamic_id": self.dynamic_id(call),
        }

    @staticmethod
    def clean_call(call: _ir.Call) -> _ir.Call:
        """Consume target attributes before handing arithmetic to generic handlers."""
        return _ir.Call(
            call.op,
            call.args,
            dict(call.kwargs),
            {k: v for k, v in call.attrs.items() if k not in CALL_ATTRS},
            call.type,
            call.span,
        )

    def block_id(self, call: _ir.Call, args: tuple[Any, ...], meta: dict[str, Any]) -> OpResult:
        return OpResult(self.ctx.capture.current.block_id, control_kind="ssnpu.identity")

    def thread_id(self, call: _ir.Call, args: tuple[Any, ...], meta: dict[str, Any]) -> OpResult:
        return OpResult(self.ctx.capture.current.thread_id, control_kind="ssnpu.identity")

    def shared_tile(self, call: _ir.Call, args: tuple[Any, ...], meta: dict[str, Any]) -> OpResult:
        shape = tuple(self.ctx.evaluate(v) for v in call.type.shape)
        if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError(f"Expected positive shared tile shape, got {shape}")
        handle = SharedHandle(meta["dynamic_id"], shape, call.type.dtype, self.owner())
        old = self.handles.setdefault(handle.handle_id, handle)
        if old != handle:
            raise ValueError(f"Shared declaration mismatch at {handle.handle_id}")
        return OpResult(
            old,
            control_kind="ssnpu.declare",
            parameters={"handle_id": handle.handle_id, "shape": list(shape), "memory_space": "SharedTileReg"},
        )

    def load_into(self, call: _ir.Call, args: tuple[Any, ...], meta: dict[str, Any]) -> OpResult:
        if len(args) != 3 or not isinstance(args[0], SharedHandle):
            raise ValueError("load_into expects shared handle, Tensor and offsets")
        handle, source, offsets = args
        if handle.owner != self.owner():
            raise ValueError("Shared handle referenced outside its launch/block")
        if not isinstance(source, RuntimeObject) or source.kind != "tensor" or source.dtype != handle.dtype:
            raise ValueError("Shared load requires a Tensor with the handle dtype")
        _rectangle(offsets, handle.shape, source.valid_shape)
        if len(meta["pes"]) != 1:
            raise ValueError("load_into requires exactly one producer PE")
        key = meta["dynamic_id"]
        signature = (handle.handle_id, source.object_id, tuple(offsets), tuple(meta["pes"]))
        version = self.versions.get(key)
        if version is None:
            obj = self.ctx.new_tile(call.type)
            if obj.shape != handle.shape or obj.dtype != handle.dtype or obj.valid_shape != handle.shape:
                raise ValueError("Shared load result type disagrees with handle")
            obj.memory_space = "SharedTileReg"
            obj.owner_instance = {"task_id": handle.owner[0], "block_id": handle.owner[1]}
            obj.shared_handle, obj.shared_version = handle.handle_id, key
            # A logical version is immutable even before its producer is captured.
            # _finish must not increment this version when the producer arrives.
            obj.version = 1
            version = {"object": obj, "signature": signature, "producer": None, "references": set()}
            self.versions[key] = version
        elif version["signature"] != signature:
            raise ValueError(f"Shared definition/version mismatch at {key}")
        tid = self.ctx.capture.current.thread_id
        if tid in version["references"]:
            raise ValueError(f"Repeated dynamic definition at {key} on thread {tid}")
        version["references"].add(tid)
        obj = version["object"]
        if tid not in meta["pes"]:
            return OpResult(obj, control_kind="ssnpu.version_ref", parameters={"shared_version": key})
        if version["producer"] is not None:
            raise ValueError(f"Multiple producers for shared version {key}")
        version["producer"] = self.ctx.next_op_id
        return OpResult(
            obj,
            (source,),
            (obj,),
            "pto.tload",
            {"offsets": list(offsets), "tensor_id": source.object_id, "shared_version": key},
        )

    def barrier(self, call: _ir.Call, args: tuple[Any, ...], meta: dict[str, Any]) -> OpResult:
        expected = list(range(self.ctx.capture.current.thread_num))
        if meta["pes"] != expected:
            raise ValueError("A full-block barrier requires all threads")
        rounds = self.barriers[self.thread_key()]
        round_id = len(rounds)
        rounds.append(meta["dynamic_id"])
        return OpResult(
            0,
            control_kind="ssnpu.barrier",
            parameters={
                "barrier_id": meta["dynamic_id"],
                "site": meta["site"],
                "round": round_id,
                "participants": expected,
                "completion": "all_prior_operations",
            },
        )

    def prepare(self, args: tuple[Any, ...], meta: dict[str, Any]) -> None:
        """Reject private escape even through tuples and helper function arguments."""
        for value in args:
            if isinstance(value, (tuple, list)):
                self.prepare(tuple(value), {**meta, "reuse": ()})
            elif isinstance(value, RuntimeObject) and value.kind == "tile":
                if value.shared_version is not None:
                    if value.owner_instance != {"task_id": self.owner()[0], "block_id": self.owner()[1]}:
                        raise ValueError("Shared tile referenced outside its launch/block")
                elif value.owner_instance != self.ctx.capture.identity():
                    raise ValueError("Private LocalTileReg referenced by another thread")
        for index in meta.get("reuse", ()):
            obj = args[index]
            if not isinstance(obj, RuntimeObject) or obj.shared_version is None:
                raise ValueError("reuse requires a shared version input")

    def effects(self, call: _ir.Call, args: tuple[Any, ...], result: OpResult, meta: dict[str, Any]) -> None:
        if result.control_kind is not None:
            return
        if self.ctx.capture.current.thread_id not in meta["pes"]:
            raise ValueError(f"Executing thread is outside operation participant set {meta['pes']}")
        if result.views:
            source = result.reads[0]
            for obj in result.views:
                obj.shared_handle, obj.shared_version = source.shared_handle, source.shared_version
                obj.owner_instance, obj.memory_space = source.owner_instance, source.memory_space
        for obj in result.writes:
            if obj.kind == "tile" and obj.shared_version is None:
                if meta["storage"] == "SharedTileReg":
                    raise ValueError("SharedTileReg writes must use load_into")
                obj.memory_space = "LocalTileReg"
            elif obj.shared_version is not None and call.op.name not in self.handlers:
                raise ValueError("Shared versions are immutable; use load_into for a new definition")
        shared = []
        if result.parameters.get("memory_access", True):
            for index, obj in enumerate(args):
                if isinstance(obj, RuntimeObject) and obj.shared_version is not None and obj in result.reads:
                    shared.append(
                        {
                            "operand": index,
                            "handle_id": obj.shared_handle,
                            "version_id": obj.shared_version,
                            "use_id": meta["dynamic_id"],
                            "participants": meta["pes"],
                            "retention": "reuse" if index in meta["reuse"] else "last-use",
                        }
                    )
        result.parameters["ssnpu"] = {
            "site": meta["site"],
            "dynamic_id": meta["dynamic_id"],
            "participants": meta["pes"],
            "shared_inputs": shared,
            "phase": len(self.barriers[self.thread_key()]),
        }

    def record(self, record: dict[str, Any], result: OpResult) -> None:
        """Keep internal records for validation even when export is disabled."""
        if not self.active:
            return
        record["ssnpu"] = record["parameters"].pop("ssnpu")
        record["memory_accesses"] = []
        offsets = result.parameters.get("offsets")
        tensor_id = result.parameters.get("tensor_id")
        for direction, objects in (("read", result.reads), ("write", result.writes)):
            for obj in objects:
                if obj.kind != "tensor":
                    continue
                extent = None
                if offsets is not None and tensor_id == obj.object_id:
                    tile = result.value if direction == "read" else result.reads[0]
                    extent = list(tile.valid_shape)
                record["memory_accesses"].append(
                    {
                        "storage_id": obj.storage.storage_id,
                        "object_id": obj.object_id,
                        "mode": direction,
                        "offsets": list(offsets) if extent is not None else None,
                        "shape": extent,
                        "stride": list(obj.stride) if obj.stride is not None else None,
                        "storage_offset": obj.storage_offset,
                    }
                )
        self.records.append(record)

    def finalize(self) -> None:
        if not self.enabled:
            return
        by_id = {r["op_id"]: r for r in self.records}
        groups: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        by_thread: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            by_thread[(record["task_id"], record["block_id"], record["thread_id"])].append(record)
            for use in record["ssnpu"]["shared_inputs"]:
                groups[(use["use_id"], use["operand"])].append((record, use))
        self._link_versions()
        self._validate_barriers(by_thread)
        self._validate_uses(groups)
        self._validate_participation()
        # Per-thread issue order and full phase completion are separate from
        # data dependencies. Backends may issue independent operations freely
        # within a phase, while respecting these explicit ordering constraints.
        for records in by_thread.values():
            previous = None
            for record in records:
                record["order_dependencies"] = [] if previous is None else [previous]
                previous = record["op_id"]
        self._memory_dependencies(self._ordered_records(by_id))

    def _validate_participation(self) -> None:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            groups[record["ssnpu"]["dynamic_id"]].append(record)
        for key, records in groups.items():
            expected = records[0]["ssnpu"]["participants"]
            if any(r["ssnpu"]["participants"] != expected for r in records):
                raise ValueError(f"Operation participant mismatch at {key}")
            if sorted(r["thread_id"] for r in records) != expected:
                raise ValueError(f"Unmatched operation participants at {key}; expected {expected}")

    def _link_versions(self) -> None:
        for key, version in self.versions.items():
            if version["producer"] is None:
                raise ValueError(f"Missing producer for shared version {key}")
        for record in self.records:
            for obj in record["inputs"] + record["outputs"]:
                key = obj.get("shared_version")
                if key is not None:
                    obj["producer"] = self.versions[key]["producer"]
                    if obj in record["inputs"]:
                        record["dependencies"].append(obj["producer"])
            record["dependencies"] = sorted(set(record["dependencies"]))

        for event in self.ctx.capture.events.records:
            for obj in event.get("objects", []):
                key = obj.get("shared_version")
                if key is not None:
                    obj["producer"] = self.versions[key]["producer"]

    def _validate_barriers(self, by_thread: dict) -> None:
        for task in self.ctx.capture.tasks:
            instances = [i for i in task["instances"] if "thread_id" in i]
            for block in sorted({i["block_id"] for i in instances}):
                threads = [i["thread_id"] for i in instances if i["block_id"] == block]
                rounds = [self.barriers[(task["task_id"], block, t)] for t in threads]
                if any(r != rounds[0] for r in rounds):
                    raise ValueError(f"Barrier site/round mismatch in task {task['task_id']} block {block}")
                phases: dict[int, list[int]] = defaultdict(list)
                for tid in threads:
                    for record in by_thread.get((task["task_id"], block, tid), []):
                        phases[record["ssnpu"]["phase"]].append(record["op_id"])
                # Include all earlier phases: an empty phase cannot erase a
                # completion requirement from an earlier nonempty phase.
                prior = []
                for phase in sorted(phases):
                    for tid in threads:
                        for record in by_thread.get((task["task_id"], block, tid), []):
                            if record["ssnpu"]["phase"] == phase:
                                record["sync_dependencies"] = list(prior)
                    prior.extend(phases[phase])
                for event in self.ctx.capture.events.records:
                    if (
                        event["kind"] == "ssnpu.barrier"
                        and event["task_id"] == task["task_id"]
                        and event["block_id"] == block
                    ):
                        phase = event["parameters"]["round"]
                        event["completion_dependencies"] = [
                            op for p, ops in phases.items() if p <= phase for op in ops
                        ]

    @staticmethod
    def _before(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return a["ssnpu"]["phase"] < b["ssnpu"]["phase"] or (
            a["thread_id"] == b["thread_id"] and a["instance_op_sequence"] < b["instance_op_sequence"]
        )

    def _validate_uses(self, groups: dict) -> None:
        by_version: dict[str, list[list[tuple[dict, dict]]]] = defaultdict(list)
        for key, entries in groups.items():
            first = entries[0][1]
            if any(use != first for _, use in entries):
                raise ValueError(f"Shared version/participant/retention mismatch at use {key}")
            tids = [r["thread_id"] for r, _ in entries]
            if sorted(tids) != first["participants"]:
                raise ValueError(
                    f"Unmatched shared-use participants at {key}: {tids}, expected {first['participants']}"
                )
            by_version[first["version_id"]].append(entries)
        for key, uses in by_version.items():
            last = [u for u in uses if u[0][1]["retention"] == "last-use"]
            if len(last) > 1:
                raise ValueError(f"Read after last-use for shared version {key}")
            if not last:
                continue  # Explicitly retained live-out; never implicitly freed.
            terminal = last[0]
            for use in uses:
                if use is terminal:
                    continue
                # Every earlier PE read must precede a final participant's
                # read, by thread order or a block barrier. Unordered disjoint
                # subsets need an explicit barrier to establish lifetime order.
                if any(not any(self._before(r, end) for end, _ in terminal) for r, _ in use):
                    raise ValueError(f"Read after last-use or unordered lifetime for shared version {key}")

    @staticmethod
    def _overlap(a: dict, b: dict) -> bool:
        if a["storage_id"] != b["storage_id"]:
            return False
        if a["shape"] is None or b["shape"] is None or a["object_id"] != b["object_id"]:
            return True
        # Only injective row-major descriptors admit rectangle disjointness.
        stride = a["stride"]
        if not stride or stride[1] != 1 or stride[0] <= 0:
            return True
        obj = a.get("tensor_shape")
        if obj is not None and stride[0] < obj[1]:
            return True
        return all(
            x < y + m and y < x + n for x, n, y, m in zip(a["offsets"], a["shape"], b["offsets"], b["shape"])
        )

    def _memory_dependencies(self, records: list[dict[str, Any]]) -> None:
        history: list[tuple[int, dict]] = []
        # Existing producer, thread and barrier edges take precedence. Only
        # otherwise unordered hazards use canonical block/phase/thread order.
        # A bare lexical sort could put PE0's store before PE1's shared load
        # feeding that store, adding an artificial cycle on an aliased Tensor.
        for record in records:
            for access in record["memory_accesses"]:
                access["tensor_shape"] = list(self.ctx.objects.objects[access["object_id"]].shape)
                for previous, old in history:
                    if (
                        previous != record["op_id"]
                        and (access["mode"] == "write" or old["mode"] == "write")
                        and self._overlap(access, old)
                    ):
                        record["dependencies"].append(previous)
                history.append((record["op_id"], access))
            record["dependencies"] = sorted(set(record["dependencies"]))

    @staticmethod
    def _ordered_records(by_id: dict[int, dict]) -> list[dict[str, Any]]:
        edges = {
            op: set(r["dependencies"] + r.get("order_dependencies", []) + r.get("sync_dependencies", []))
            & by_id.keys()
            for op, r in by_id.items()
        }
        pending = {op: len(deps) for op, deps in edges.items()}
        followers: dict[int, list[int]] = defaultdict(list)
        for op, deps in edges.items():
            for dep in deps:
                followers[dep].append(op)

        def priority(op: int) -> tuple[int, ...]:
            record = by_id[op]
            return (
                record["task_id"],
                record["block_id"],
                record["ssnpu"]["phase"],
                record["thread_id"],
                record["instance_op_sequence"],
                op,
            )

        ready = [priority(op) for op, count in pending.items() if not count]
        heapq.heapify(ready)
        ordered = []
        while ready:
            op = heapq.heappop(ready)[-1]
            ordered.append(by_id[op])
            for follower in followers[op]:
                pending[follower] -= 1
                if pending[follower] == 0:
                    heapq.heappush(ready, priority(follower))
        if len(ordered) != len(by_id):
            raise ValueError("SSNPU dependency cycle: producer/consumer order conflicts with synchronization")
        return ordered
