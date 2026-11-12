# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deterministic operation and multicore capture exports with metadata snapshots."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExecutionResult:
    """Returned values (in IR return order) and optional JSON-compatible events."""

    outputs: tuple[Any, ...]
    trace: list[dict[str, Any]]
    mode: str
    tasks: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    capture_config: dict[str, Any] = field(default_factory=dict)
    objects: list[dict[str, Any]] = field(default_factory=list)

    def write_trace(self, path: str | Path) -> None:
        """Export one operation per UTF-8 JSONL line."""
        with Path(path).open("w", encoding="utf-8") as stream:
            for record in self.trace:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_capture(self, output_dir: str | Path) -> None:
        """Export tasks, events, configuration and one JSONL per instance.

        The completion manifest is written last. An existing manifest is first
        removed so an interrupted rewrite cannot advertise a complete capture.
        Only manifest-listed instance files belong to this capture.
        """
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.json"
        manifest_path.unlink(missing_ok=True)
        instances = []
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for op in self.trace:
            key = tuple(op[k] for k in ("task_id", "block_id", "subblock_id"))
            grouped.setdefault(key, []).append(op)
        for task in self.tasks:
            for instance in task["instances"]:
                key = tuple(instance[k] for k in ("task_id", "block_id", "subblock_id"))
                filename = f"task_{key[0]}_block_{key[1]}_subblock_{key[2]}.jsonl"
                records = grouped.get(key, [])
                self._write_jsonl(folder / filename, records)
                instances.append({**instance, "trace_file": filename})
        self._write_jsonl(folder / "events.jsonl", self.events)
        self._write_json(folder / "tasks.json", self.tasks)
        self._write_json(folder / "objects.json", self.objects)
        self._write_json(folder / "capture_config.json", {**self.capture_config, "mode": self.mode})
        self._write_json(
            manifest_path,
            {
                "format_version": 1,
                "complete": self.capture_config.get("complete", False),
                "instances": instances,
                "tasks_file": "tasks.json",
                "events_file": "events.jsonl",
                "config_file": "capture_config.json",
                "objects_file": "objects.json",
            },
        )

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class TraceRecorder:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []

    def record(self, event: dict[str, Any]) -> None:
        """Save an event without owning execution identities or mutating objects."""
        if self.enabled:
            self.records.append(deepcopy(event))
