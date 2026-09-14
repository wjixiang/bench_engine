from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from openai import OpenAIError

from bench_engine.benchmarks.hle import HLE
from bench_engine.core.grading import exact_grade, grade_verdict
from bench_engine.core.interfaces import Example, Solver, SolverResult, TaskAsset
from bench_engine.core.runner import OpenAIGrader, evaluate_examples, summarize

EXAMPLE = Example(
    id="q1",
    question="What is 2 + 2?",
    image="",
    target="4",
    answer_type="exactMatch",
    category="Mathematics",
)
MC_EXAMPLE = Example(
    id="mc1",
    question="Which answer is correct?",
    image="",
    target="B",
    answer_type="multipleChoice",
    category="Test",
)


class StaticSolver(Solver):
    async def solve(self, example: Example, prompt: str) -> SolverResult:
        self.example = example
        self.prompt = prompt
        return SolverResult("work\nAnswer: 4", True)


class CoreTest(unittest.TestCase):
    def test_grading_and_verdicts(self) -> None:
        self.assertEqual(HLE.extract("Answer: **(B)**", MC_EXAMPLE), "B")
        self.assertTrue(
            exact_grade("Answer: 46.240", "46.24", answer_type="exactMatch").correct
        )
        self.assertTrue(grade_verdict("**Verdict:** CORRECT"))
        self.assertIsNone(grade_verdict("Verdict: UNKNOWN"))

    def test_runner_uses_adapter_prompt_and_grader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            solver = StaticSolver()
            records = asyncio.run(
                evaluate_examples(
                    HLE,
                    [EXAMPLE],
                    solver,
                    solver_kind="test",
                    output=output,
                    grader=None,
                )
            )
            self.assertIn("Answer: <answer>", solver.prompt)
            record = json.loads(output.read_text("utf-8"))
            self.assertEqual(record["benchmark"], "hle")
            self.assertTrue(record["correct"])
        self.assertEqual(summarize(records)["accuracy"], 1.0)

    def test_openai_grader_uses_native_responses_api(self) -> None:
        client = MagicMock()
        client.responses.create = AsyncMock(
            return_value=SimpleNamespace(output_text="Verdict: CORRECT")
        )
        grader = OpenAIGrader(HLE, model="test-grader-model", client=client)

        grade = asyncio.run(grader.grade(HLE, EXAMPLE, "work\nAnswer: 4"))

        self.assertTrue(grade.correct)
        self.assertEqual(grade.method, "model")
        client.responses.create.assert_awaited_once_with(
            model="test-grader-model",
            input=HLE.grader_prompt(EXAMPLE, "work\nAnswer: 4"),
        )

    def test_openai_grader_rejects_invalid_verdict_and_api_errors(self) -> None:
        client = MagicMock()
        client.responses.create = AsyncMock(
            return_value=SimpleNamespace(output_text="no verdict")
        )
        grader = OpenAIGrader(HLE, model="test-grader-model", client=client)
        grade = asyncio.run(grader.grade(HLE, EXAMPLE, "Answer: 4"))
        self.assertIsNone(grade.correct)
        self.assertIn("could not parse grader verdict", grade.detail)

        client.responses.create = AsyncMock(side_effect=OpenAIError("request failed"))
        grade = asyncio.run(grader.grade(HLE, EXAMPLE, "Answer: 4"))
        self.assertIsNone(grade.correct)
        self.assertIn("OpenAI grader failed", grade.detail)

    def test_openai_grader_requires_api_key_without_injected_client(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"),
        ):
            OpenAIGrader(HLE, model="test-grader-model")

    def test_hle_prompt_lists_task_image_asset(self) -> None:
        asset = TaskAsset(
            "image",
            Path("/tasks/task-1/data/image.png"),
            "image/png",
            "image",
        )
        example = Example(
            id="hle-image",
            question="Question",
            image="",
            target="A",
            answer_type="multipleChoice",
            category="Biology/Medicine",
            assets=(asset,),
        )
        prompt = replace(HLE, include_image_uri=True).prompt(example)

        self.assertIn("Image files:", prompt)
        self.assertIn("/tasks/task-1/data/image.png", prompt)


if __name__ == "__main__":
    unittest.main()
