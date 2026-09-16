"""LLM-based rubric grader for OmicOS-BiomniBench tasks using local headless Codex.

The grader reads ``/app/answer.txt`` (captured as ``solver_result.response``
by the solver) and ``/app/trace.md`` (recorded as a solver artifact) and
shells out to ``codex exec`` to score the candidate against the task
rubric. The returned ``Grade.correct`` flag is set by comparing the model
score (0-100) to the example ``pass_threshold`` (0-1).

This is a drop-in alternative to :class:`OmicOSGrader` that runs against
the local Codex CLI instead of a remote OpenAI-compatible API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

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
        logger.warning("codex_local.grader: failed to read artifact %s: %s", path_value, exc)
        return ""


def _normalize_score(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        score = int(raw)
    except ValueError:
        return None
    return max(0, min(100, score))


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



def _build_structured_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
            "rationale": {"type": "string"},
        },
        "required": ["score"],
        "additionalProperties": False,
    }


class CodexLocalGrader:
    """Score an OmicOS task against its rubric by shelling out to ``codex exec``."""

    def __init__(
        self,
        *,
        codex_path: str = "codex",
        model: str | None = None,
        timeout: float = 900.0,
        extra_args: list[str] | None = None,
    ) -> None:
        if not shutil.which(codex_path) and not Path(codex_path).exists():
            raise FileNotFoundError(
                f"codex executable not found: {codex_path!r}"
            )
        self.codex_path = codex_path
        self.model = model
        self.timeout = timeout
        self.extra_args = list(extra_args or [])

    async def _run_codex(
        self,
        argv: list[str],
        prompt: str,
        *,
        timeout: float,
    ) -> tuple[int, bytes, bytes]:
        """Spawn ``codex exec`` and return ``(returncode, stdout, stderr)``.

        Override this in tests to inject a fake process.
        """
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        return proc.returncode, stdout, stderr

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
                "CodexLocalGrader requires a non-blank rubric on the example",
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="codex-grader-", delete=False
        ) as output_handle:
            output_path = Path(output_handle.name)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="codex-grader-",
            delete=False,
            encoding="utf-8",
        ) as schema_handle:
            json.dump(_build_structured_schema(), schema_handle)
            schema_path = Path(schema_handle.name)

        argv: list[str] = [
            self.codex_path, "exec",
            "--output-last-message", str(output_path),
            "--output-schema", str(schema_path),
            "--ephemeral",
            "--skip-git-repo-check",
            "-C", str(Path.cwd()),
        ]
        if self.model:
            argv.extend(["--model", self.model])
        argv.extend(self.extra_args)
        argv.append("-")  # read prompt from stdin

        try:
            try:
                returncode, stdout, stderr = await self._run_codex(
                    argv, prompt, timeout=self.timeout
                )
            except FileNotFoundError as exc:
                return Grade(
                    None,
                    answer_text,
                    "rubric",
                    f"failed to launch codex: {exc}",
                )
            except TimeoutError:
                return Grade(
                    None,
                    answer_text,
                    "rubric",
                    f"codex grader timed out after {self.timeout:g}s",
                )

            if returncode != 0:
                detail = (stderr or b"").decode("utf-8", errors="replace")
                return Grade(
                    None,
                    answer_text,
                    "rubric",
                    f"codex grader failed (exit {returncode}): {detail[:500]}",
                )

            output_text = output_path.read_text(encoding="utf-8", errors="replace")
            # Prefer JSON-shape parse when --output-schema is honoured.
            score: int | None = None
            try:
                parsed = json.loads(output_text)
                if isinstance(parsed, dict) and "score" in parsed:
                    score = _normalize_score(parsed["score"])
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if score is None:
                score = _parse_score(output_text)
            if score is None:
                return Grade(
                    None,
                    answer_text,
                    "rubric",
                    f"could not parse codex grader score: {output_text[-500:] or '<empty>'}",
                )

            threshold_pct = example.grade_threshold * 100
            correct = score >= threshold_pct
            detail = (
                f"score={score}/100 threshold={threshold_pct:.0f}/100 "
                f"verdict={'PASS' if correct else 'FAIL'}"
            )
            return Grade(correct, answer_text, "rubric", detail)
        finally:
            for tmp in (output_path, schema_path):
                try:
                    tmp.unlink()
                except OSError:
                    pass
