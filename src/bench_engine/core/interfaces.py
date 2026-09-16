"""Stable contracts shared by the runner and solver backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from bench_engine.benchmarks.base import Benchmark
    from bench_engine.core.grading import Grade


@dataclass(frozen=True)
class TaskAsset:
    """An input file or directory associated with a benchmark task."""

    name: str
    path: Path
    media_type: str = "application/octet-stream"
    role: str = "input"
    relative_path: str | None = None

    def payload(self, data_mount_path: Path | None = None) -> dict[str, str]:
        """Return the JSON-serializable metadata passed to external solvers."""
        path = (
            data_mount_path / (self.relative_path or self.name)
            if data_mount_path is not None
            else self.path
        )
        return {
            "name": self.name,
            "path": str(path),
            "media_type": self.media_type,
            "role": self.role,
        }


@dataclass(frozen=True)
class Example:
    """A normalized benchmark item."""

    id: str
    question: str
    image: str
    target: str
    answer_type: str
    category: str
    task_path: Path | None = None
    assets: tuple[TaskAsset, ...] = ()
    rubric: str = ""
    grade_threshold: float = 1.0

    def payload(self, data_mount_path: Path | None = None) -> dict[str, Any]:
        """Return the JSON-serializable item passed to external solvers."""
        payload: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "image": self.image,
            "answer_type": self.answer_type,
            "category": self.category,
        }
        if self.assets:
            payload["data"] = [asset.payload(data_mount_path) for asset in self.assets]
        return payload


@dataclass(frozen=True)
class SolverResult:
    """The normalized result returned by every solver backend."""

    response: str
    ok: bool
    returncode: int | None = None
    error: str | None = None
    session_id: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    stderr_tail: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    agent_name: str | None = None
    telemetry: dict[str, Any] | None = None
    telemetry_events: list[dict[str, Any]] = field(default_factory=list)
    run_metrics: dict[str, Any] | None = None


class Solver(Protocol):
    """A backend that can answer one benchmark prompt."""

    async def solve(
        self,
        example: Example,
        prompt: str,
        *,
        data_mount_path: Path | None = None,
    ) -> SolverResult: ...


class Grader(Protocol):
    """A backend that assigns an optional correctness verdict."""

    async def grade(
        self,
        benchmark: Benchmark,
        example: Example,
        solver_result: SolverResult,
    ) -> Grade: ...


class RawSolver(Protocol):
    """A solver capable of grading or other auxiliary prompts."""

    async def solve_raw(
        self,
        prompt: str,
        *,
        task_id: str,
        phase: str,
        answer_type: str | None,
        category: str | None,
        has_image: bool,
    ) -> SolverResult: ...
