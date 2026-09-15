"""Benchmark orchestration, artifact I/O, and metrics."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from bench_engine.benchmarks.base import Benchmark
from bench_engine.core.grading import Grade, grade_verdict
from bench_engine.core.interfaces import (
    Example,
    Grader,
    Solver,
    SolverResult,
)

RESULT_SCHEMA = 3


class OpenAIGrader:
    """Grade a response with a native OpenAI Responses API call."""

    def __init__(
        self,
        benchmark: Benchmark,
        *,
        model: str,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if client is None and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")
        if base_url is not None and not base_url.strip():
            raise ValueError("OpenAI grader base URL must be non-blank")
        self.benchmark = benchmark
        self.model = model
        self.client = client or (
            AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
        )

    async def grade(
        self,
        benchmark: Benchmark,
        example: Example,
        response: str,
    ) -> Grade:
        del benchmark
        prompt = self.benchmark.grader_prompt(example, response)
        try:
            result = await self.client.responses.create(
                model=self.model,
                input=prompt,
            )
        except OpenAIError as exc:
            return Grade(None, None, "model", f"OpenAI grader failed: {exc}")

        text = result.output_text
        verdict = grade_verdict(text) if text else None
        if verdict is None:
            return Grade(
                None,
                None,
                "model",
                f"could not parse grader verdict: {text[-500:] if text else '<empty>'}",
            )
        extracted = self.benchmark.extract(response, example)
        return Grade(verdict, extracted, "model")


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _result_record(
    benchmark: Benchmark,
    example: Example,
    solver_result: SolverResult,
    grade: Grade,
    *,
    solver_kind: str,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "benchmark": benchmark.name,
        "id": example.id,
        "answer_type": example.answer_type,
        "category": example.category,
        "question": example.question,
        "target_answer": example.target,
        "raw_response": solver_result.response,
        "extracted_answer": grade.extracted,
        "correct": grade.correct,
        "grade_method": grade.method,
        "grade_detail": grade.detail,
        "solver_ok": solver_result.ok,
        "solver_kind": solver_kind,
        "solver_error": solver_result.error,
        "solver_returncode": solver_result.returncode,
        "solver_session_id": solver_result.session_id,
        "solver_model": solver_result.model,
        "solver_usage": solver_result.usage,
        "solver_stderr_tail": solver_result.stderr_tail,
        "solver_artifacts": solver_result.artifacts,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if record.get("schema") not in {2, RESULT_SCHEMA}:
                raise ValueError(
                    f"unsupported result schema at {path}:{line_number}: "
                    f"{record.get('schema')!r}"
                )
            records.append(record)
    return records


def completed_ids(path: Path) -> set[str]:
    """Read IDs already represented in an output artifact."""
    return {str(record["id"]) for record in _read_jsonl(path)}


def read_results(path: Path) -> list[dict[str, Any]]:
    """Load a valid result artifact for resumed summaries."""
    return _read_jsonl(path)


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute run metrics over result records."""
    scored = [record for record in records if record.get("correct") is not None]
    correct = sum(record["correct"] is True for record in scored)
    solver_errors = sum(record.get("solver_ok") is False for record in records)
    grader_errors = sum(
        record.get("grade_method") == "model" and record.get("correct") is None
        for record in records
    )

    by_answer_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        answer_type = str(record["answer_type"])
        category = str(record["category"])
        by_answer_type[answer_type]["total"] += 1
        by_category[category]["total"] += 1
        if record.get("correct") is not None:
            outcome = "correct" if record["correct"] else "incorrect"
            by_answer_type[answer_type][outcome] += 1
            by_category[category][outcome] += 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "records": len(records),
        "scored": len(scored),
        "correct": correct,
        "incorrect": len(scored) - correct,
        "unscored": len(records) - len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "solver_errors": solver_errors,
        "grader_errors": grader_errors,
        "by_answer_type": dict(by_answer_type),
        "by_category": dict(by_category),
    }


async def evaluate_examples(
    benchmark: Benchmark,
    examples: Sequence[Example],
    solver: Solver,
    *,
    solver_kind: str,
    output: Path,
    grader: Grader | None,
    jobs: int = 1,
    skip_ids: set[str] | None = None,
    dry_run: bool = False,
    data_mount_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run all examples and append one JSON object per completed item."""
    skip = skip_ids or set()
    queue = [example for example in examples if example.id not in skip]
    if dry_run:
        for example in queue:
            print(
                json.dumps(
                    {"id": example.id, "prompt": benchmark.prompt(example)},
                    ensure_ascii=False,
                )
            )
        return []

    output.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(jobs)
    records: list[dict[str, Any]] = []

    async def run_one(example: Example) -> None:
        async with semaphore:
            solver_result = await solver.solve(
                example,
                benchmark.prompt(example),
                data_mount_path=data_mount_path,
            )
            if grader is None:
                grade = benchmark.grade(solver_result.response, example)
            else:
                grade = await grader.grade(benchmark, example, solver_result.response)
            record = _result_record(
                benchmark,
                example,
                solver_result,
                grade,
                solver_kind=solver_kind,
            )
            async with write_lock:
                _write_jsonl(output, record)
                records.append(record)
                print(
                    f"[{len(records)}/{len(queue)}] id={record['id']} "
                    f"correct={record['correct']}",
                    file=sys.stderr,
                )

    await asyncio.gather(*(run_one(example) for example in queue))
    return records
