"""Durable tracking of benchmark tasks skipped after an out-of-memory kill."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench_engine.core.interfaces import Example, SolverResult

OOM_SKIP_SCHEMA = 1
_OOM_TEXT = re.compile(
    r"(?:\boom\b|out of memory|killed process|memory cgroup)", re.IGNORECASE
)


def is_oom_result(result: SolverResult) -> bool:
    """Return whether a solver failure is consistent with an OOM kill.

    A negative SIGKILL return code is not perfectly distinguishable from an
    administrative kill, but in a memory-capped benchmark unit it is the signal
    reported when the kernel or systemd reclaim terminates the process.
    """
    if result.ok:
        return False
    if result.returncode == -9:
        return True
    diagnostic = " ".join(
        part or "" for part in (result.error, result.stderr_tail)
    )
    return bool(_OOM_TEXT.search(diagnostic))


def read_oom_skips(path: Path) -> dict[str, dict[str, Any]]:
    """Load valid OOM skip records keyed by task ID."""
    if not path.exists():
        return {}
    skipped: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid OOM skip JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if record.get("schema") != OOM_SKIP_SCHEMA:
                raise ValueError(
                    f"unsupported OOM skip schema in {path}:{line_number}: "
                    f"{record.get('schema')!r}"
                )
            task_id = record.get("id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(
                    f"invalid OOM skip ID in {path}:{line_number}: {task_id!r}"
                )
            skipped[task_id] = record
    return skipped


def record_oom_skip(
    path: Path,
    example: Example,
    solver_result: SolverResult,
    *,
    source: str = "solver",
) -> dict[str, Any]:
    """Durably append an OOM skip record and return it."""
    existing = read_oom_skips(path)
    if example.id in existing:
        return existing[example.id]

    record: dict[str, Any] = {
        "schema": OOM_SKIP_SCHEMA,
        "id": example.id,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source": source,
        "returncode": solver_result.returncode,
        "agent_name": solver_result.agent_name,
        "error": solver_result.error,
        "telemetry": solver_result.telemetry,
        "telemetry_events": solver_result.telemetry_events,
        "run_metrics": solver_result.run_metrics,
    }
    record.update(
        {
            "category": example.category,
            "answer_type": example.answer_type,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return record
