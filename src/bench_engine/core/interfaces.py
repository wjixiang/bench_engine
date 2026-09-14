"""Stable contracts shared by the runner and solver backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from bench_engine.benchmarks.base import Benchmark
    from bench_engine.core.grading import Grade


@dataclass(frozen=True)
class Example:
    """A normalized benchmark item."""

    id: str
    question: str
    image: str
    target: str
    answer_type: str
    category: str

    def payload(self) -> dict[str, str]:
        """Return the JSON-serializable item passed to external solvers."""
        return {
            "id": self.id,
            "question": self.question,
            "image": self.image,
            "answer_type": self.answer_type,
            "category": self.category,
        }


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


class Solver(Protocol):
    """A backend that can answer one benchmark prompt."""

    async def solve(self, example: Example, prompt: str) -> SolverResult: ...


class Grader(Protocol):
    """A backend that assigns an optional correctness verdict."""

    async def grade(
        self,
        benchmark: "Benchmark",
        example: Example,
        response: str,
    ) -> "Grade": ...


class RawSolver(Protocol):
    """A solver capable of grading or other auxiliary prompts."""

    async def solve_raw(self, prompt: str, **kwargs: Any) -> SolverResult: ...
