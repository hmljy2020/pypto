# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Single-thread operation trace with immutable metadata snapshots."""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecutionResult:
    """Returned values (in IR return order) and optional JSON-compatible events."""

    outputs: tuple[Any, ...]
    trace: list[dict[str, Any]]
    mode: str

    def write_trace(self, path: str | Path) -> None:
        """Export one operation per UTF-8 JSONL line."""
        with Path(path).open("w", encoding="utf-8") as stream:
            for record in self.trace:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class TraceRecorder:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []

    def record(self, event: dict[str, Any]) -> None:
        """Save an event without owning execution identities or mutating objects."""
        if self.enabled:
            self.records.append(deepcopy(event))
