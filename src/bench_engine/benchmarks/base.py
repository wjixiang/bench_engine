"""Benchmark adapter contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from bench_engine.core.grading import Grade

if TYPE_CHECKING:
    from bench_engine.core.interfaces import Example


class Benchmark(Protocol):
    """A benchmark schema, prompt policy, and scoring policy."""

    @property
    def name(self) -> str: ...

    def prompt(self, example: str = "Example") -> str: ...

    def grade(self, response: str, example: str = "Example") -> Grade: ...

    def extract(self, response: str, example: str = "Example") -> str | None: ...

    def grader_prompt(
        self,
        response: str,
        example: str = "Example",
    ) -> str: ...
