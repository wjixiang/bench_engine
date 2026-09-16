"""LLM-based rubric grader for OmicOS-BiomniBench tasks.

The grader reads ``/app/answer.txt`` (captured as ``solver_result.response``
by the solver) and ``/app/trace.md`` (recorded as a solver artifact) and asks
an OpenAI-compatible model to score the candidate against the task rubric.
The returned ``Grade.correct`` flag is set by comparing the model score
(0-100) to the example ``pass_threshold`` (0-1).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, NotFoundError, OpenAIError

from bench_engine.benchmarks.base import Benchmark
from bench_engine.core.grading import Grade
from bench_engine.core.interfaces import Example, SolverResult

logger = logging.getLogger(__name__)

_SCORE_LINE = re.compile(
    r"(?im)^\s*(?:\*\*)?\s*score\s*(?:(?:\*\*)?\s*:|:\s*(?:\*\*)?)\s*"
    r"(\d{1,3})(?:\s*/\s*100)?\s*$"
)
_MAX_TRACE_CHARS = 20_000
_MAX_ANSWER_CHARS = 8_000
_MAX_RUBRIC_CHARS = 12_000


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 200]
    tail = text[-200:]
    return f"{head}\n...[truncated {len(text) - limit} chars]...\n{tail}"


def _read_artifact(path_value: str) -> str:
    try:
        return Path(path_value).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("omicos.grader: failed to read artifact %s: %s", path_value, exc)
        return ""


def _normalize_score(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        score = int(raw)
    except ValueError:
        return None
    return max(0, min(100, score))


class OmicOSGrader:
    """Score an OmicOS task against its rubric with an OpenAI model."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
        rubric: Mapping[str, Any] | None = None,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("OmicOS grader requires a non-blank model name")
        self.model = model
        self.client = client or (
            AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
        )
        # ``rubric`` is unused today but reserved for future rubric overrides.
        del rubric

    async def grade(
        self,
        benchmark: Benchmark,
        example: Example,
        solver_result: SolverResult,
    ) -> Grade:
        del benchmark
        if not example.rubric:
            return Grade(
                None,
                solver_result.response,
                "rubric",
                "OmicOSGrader requires a non-blank rubric on the example",
            )
        answer_text = solver_result.response
        trace_path = solver_result.artifacts.get("trace")
        trace_text = _read_artifact(trace_path) if trace_path else ""
        prompt = _build_prompt(
            example.question,
            example.rubric,
            answer_text,
            trace_text,
            example.grade_threshold,
        )
        try:
            result = await self.client.responses.create(
                model=self.model, input=prompt
            )
        except OpenAIError as exc:
            if not isinstance(exc, NotFoundError):
                return Grade(
                    None,
                    answer_text,
                    "rubric",
                    f"OmicOS grader request failed: {exc}",
                )
            # Some OpenAI-compatible providers expose /models and Chat
            # Completions but not the Responses API.
            try:
                chat = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
            except OpenAIError as chat_exc:
                return Grade(
                    None,
                    answer_text,
                    "rubric",
                    f"OmicOS grader request failed: {chat_exc}",
                )
            choices = chat.choices
            text = (
                choices[0].message.content or "" if choices else ""
            )
        else:
            text = result.output_text or ""

        score = _parse_score(text)
        if score is None:
            return Grade(
                None,
                answer_text,
                "rubric",
                f"could not parse OmicOS grader score: {text[-500:] or '<empty>'}",
            )
        threshold_pct = example.grade_threshold * 100
        correct = score >= threshold_pct
        detail = (
            f"score={score}/100 threshold={threshold_pct:.0f}/100 "
            f"verdict={'PASS' if correct else 'FAIL'}"
        )
        return Grade(correct, answer_text, "rubric", detail)


def _build_prompt(
    instruction: str,
    rubric: str,
    answer: str,
    trace: str,
    threshold: float,
) -> str:
    return (
        "You are grading an OmicOS-BiomniBench data-analysis response.\n"
        "The candidate completed an open-ended task end-to-end and produced\n"
        "two artifacts: a final answer file and a trace describing the\n"
        "analysis steps. Use the rubric below to score the response on a\n"
        "scale of 0-100 (integer, where 100 means every criterion is fully\n"
        "met at level A). Apply the rubric literally; do not invent criteria.\n\n"
        f"Task instruction:\n{instruction}\n\n"
        f"Rubric (target pass score = {threshold * 100:.0f}/100):\n"
        f"{_truncate(rubric, _MAX_RUBRIC_CHARS)}\n\n"
        "Candidate answer (answer.txt):\n"
        f"{_truncate(answer or '<no answer.txt produced>', _MAX_ANSWER_CHARS)}\n\n"
        "Candidate trace (trace.md):\n"
        f"{_truncate(trace or '<no trace.md produced>', _MAX_TRACE_CHARS)}\n\n"
        "First, walk through every rubric criterion and assign a level\n"
        "(A, B, or C). Compute the integer score from the level weights and\n"
        "clamp the result to 0-100. End your reply with a single line in\n"
        "exactly this format:\n\n"
        "Score: <integer 0-100>"
    )


def _parse_score(text: str) -> int | None:
    match = _SCORE_LINE.search(text)
    if match is None:
        return None
    return _normalize_score(match.group(1))
