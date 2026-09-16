"""Tests for the OmicOS-BiomniBench adapter and rubric grader."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from openai import NotFoundError, OpenAIError

from bench_engine.benchmarks.omicos import OMICOS
from bench_engine.core.interfaces import Example, SolverResult
from bench_engine.grading.omicos import OmicOSGrader, _build_prompt, _parse_score


RUBRIC_TASK = {
    "schema_version": 1,
    "id": "da-1-3",
    "instruction": "Analyze the distribution of cell subsets and identify tumor-specific cell types.",
    "category": "oncology",
    "subtask": "cell-composition",
    "grading": {
        "method": "rubric",
        "answer_type": "rubric",
        "answer": None,
        "rubric": "RUBRIC: 3 criteria each worth 33 points; full marks require A on all.",
        "pass_threshold": 0.7,
    },
    "data": [
        {
            "name": "environment",
            "path": "data",
            "media_type": "application/x-directory",
            "role": "workspace",
        }
    ],
}


def _make_example(tmp: Path) -> Example:
    task_path = tmp / "task-1"
    task_path.mkdir()
    data_real = tmp / "environment"
    data_real.mkdir()
    (data_real / "matrix.tsv").write_text("a\tb\n1\t2\n")
    (task_path / "data").symlink_to(data_real)
    (task_path / "task.json").write_text(json.dumps(RUBRIC_TASK), encoding="utf-8")
    from bench_engine.core.data import load_examples

    [example] = load_examples(tmp)
    return example


class OmicOSAdapterTest(unittest.TestCase):
    def test_prompt_includes_answer_trace_directives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            prompt = OMICOS.prompt(example)
            self.assertIn("/app/answer.txt", prompt)
            self.assertIn("/app/trace.md", prompt)
            self.assertIn(example.question, prompt)
            self.assertIn("Do not open, stage, sample, or process the raw count matrix", prompt)

    def test_extract_returns_response_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            self.assertEqual(
                OMICOS.extract("The answer is 4.", example), "The answer is 4."
            )
            self.assertIsNone(OMICOS.extract("   \n  ", example))

    def test_grade_fallback_marks_no_answer_as_incorrect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            grade = OMICOS.grade("", example)
            self.assertFalse(grade.correct)
            self.assertEqual(grade.method, "exact")
            self.assertIn("no answer", grade.detail)

    def test_grade_fallback_returns_uncorrected_for_rubric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            grade = OMICOS.grade("Some answer", example)
            self.assertIsNone(grade.correct)
            self.assertEqual(grade.method, "exact")
            self.assertIn("rubric tasks require", grade.detail)

    def test_grader_prompt_contains_rubric_and_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            prompt = OMICOS.grader_prompt(example, "candidate answer")
            self.assertIn(example.rubric, prompt)
            self.assertIn("candidate answer", prompt)


class OmicOSGraderTest(unittest.TestCase):
    def test_parse_score_accepts_bare_and_bold_lines(self) -> None:
        self.assertEqual(_parse_score("Final notes.\nScore: 73"), 73)
        self.assertEqual(_parse_score("Score: 73/100"), 73)
        self.assertEqual(_parse_score("**Score:** 0"), 0)
        self.assertEqual(_parse_score("Score: 250"), 100)  # clamped
        self.assertIsNone(_parse_score("no score here"))

    def test_build_prompt_includes_all_artifacts(self) -> None:
        prompt = _build_prompt(
            "Question text",
            "RUBRIC.",
            "answer body",
            "trace body",
            0.7,
        )
        self.assertIn("Question text", prompt)
        self.assertIn("RUBRIC.", prompt)
        self.assertIn("answer body", prompt)
        self.assertIn("trace body", prompt)
        self.assertIn("70", prompt)  # threshold shown as percent
        self.assertIn("Score: <integer 0-100>", prompt)

    def test_grader_scores_above_threshold_as_correct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            trace_file = Path(directory) / "trace.md"
            trace_file.write_text("# Steps\n- did X\n- did Y\n")
            client = MagicMock()
            client.responses.create = AsyncMock(
                return_value=SimpleNamespace(output_text="Score: 85")
            )
            grader = OmicOSGrader(model="test-grader", client=client)
            solver_result = SolverResult(
                response="my answer",
                ok=True,
                artifacts={"trace": str(trace_file)},
            )
            grade = asyncio.run(grader.grade(OMICOS, example, solver_result))
        self.assertTrue(grade.correct)
        self.assertEqual(grade.method, "rubric")
        self.assertIn("score=85/100", grade.detail)
        client.responses.create.assert_awaited_once()

    def test_grader_scores_below_threshold_as_incorrect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            client = MagicMock()
            client.responses.create = AsyncMock(
                return_value=SimpleNamespace(output_text="Score: 30")
            )
            grader = OmicOSGrader(model="test-grader", client=client)
            solver_result = SolverResult(
                response="answer",
                ok=True,
                artifacts={"trace": str(Path(directory) / "trace.md")},
            )
            grade = asyncio.run(grader.grade(OMICOS, example, solver_result))
        self.assertFalse(grade.correct)
        self.assertEqual(grade.method, "rubric")
        self.assertIn("verdict=FAIL", grade.detail)

    def test_grader_handles_missing_score_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            client = MagicMock()
            client.responses.create = AsyncMock(
                return_value=SimpleNamespace(output_text="no score returned")
            )
            grader = OmicOSGrader(model="test-grader", client=client)
            solver_result = SolverResult(
                response="answer", ok=True, artifacts={}
            )
            grade = asyncio.run(grader.grade(OMICOS, example, solver_result))
        self.assertIsNone(grade.correct)
        self.assertIn("could not parse OmicOS grader score", grade.detail)

    def test_grader_falls_back_to_chat_completions_when_responses_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            client = MagicMock()
            client.responses.create = AsyncMock(
                side_effect=NotFoundError(
                    "responses endpoint not found",
                    response=MagicMock(),
                    body=None,
                )
            )
            client.chat.completions.create = AsyncMock(
                return_value=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="Score: 88")
                        )
                    ]
                )
            )
            grader = OmicOSGrader(model="test-grader", client=client)
            solver_result = SolverResult(
                response="answer", ok=True, artifacts={}
            )

            grade = asyncio.run(grader.grade(OMICOS, example, solver_result))

        self.assertTrue(grade.correct)
        self.assertIn("score=88/100", grade.detail)
        client.chat.completions.create.assert_awaited_once()

    def test_grader_handles_openai_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            client = MagicMock()
            client.responses.create = AsyncMock(
                side_effect=OpenAIError("rate limited")
            )
            grader = OmicOSGrader(model="test-grader", client=client)
            solver_result = SolverResult(
                response="answer", ok=True, artifacts={}
            )
            grade = asyncio.run(grader.grade(OMICOS, example, solver_result))
        self.assertIsNone(grade.correct)
        self.assertIn("OmicOS grader request failed", grade.detail)
        self.assertIn("rate limited", grade.detail)

    def test_grader_rejects_blank_rubric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = _make_example(Path(directory))
            empty_example = Example(
                id=example.id,
                question=example.question,
                image=example.image,
                target=example.target,
                answer_type=example.answer_type,
                category=example.category,
                task_path=example.task_path,
                assets=example.assets,
                rubric="",
                grade_threshold=example.grade_threshold,
            )
            client = MagicMock()
            client.responses.create = AsyncMock()
            grader = OmicOSGrader(model="test-grader", client=client)
            solver_result = SolverResult(response="", ok=True, artifacts={})
            grade = asyncio.run(grader.grade(OMICOS, empty_example, solver_result))
        self.assertIsNone(grade.correct)
        self.assertIn("non-blank rubric", grade.detail)
        client.responses.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class CodexLocalGraderTest(unittest.TestCase):
    def test_build_prompt_and_parse(self) -> None:
        from bench_engine.grading.codex_local import _build_prompt, _parse_score

        prompt = _build_prompt(
            "Test instruction",
            "RUBRIC\nCriterion 1: ...",
            "My answer",
            "My trace",
            0.7,
        )
        self.assertIn("Test instruction", prompt)
        self.assertIn("RUBRIC", prompt)
        self.assertIn("My answer", prompt)
        self.assertIn("My trace", prompt)
        self.assertIn("70", prompt)  # threshold shown as percent
        self.assertIn("Score: <integer 0-100>", prompt)

        self.assertEqual(_parse_score("Final notes.\nScore: 88"), 88)
        self.assertEqual(_parse_score("Score: 88/100"), 88)
        self.assertEqual(_parse_score("**Score:** 70"), 70)
        self.assertEqual(_parse_score("score : 50"), 50)
        self.assertEqual(_parse_score("Score: 250"), 100)  # clamped
        self.assertIsNone(_parse_score("no score line"))

    def test_grader_parses_codex_json_output(self) -> None:
        import asyncio
        import os
        import shutil
        import subprocess
        from pathlib import Path

        from bench_engine.core.interfaces import Example, SolverResult
        from bench_engine.grading.codex_local import CodexLocalGrader

        if not shutil.which("codex"):
            self.skipTest("codex CLI not installed")

        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "da-test"
            task_path.mkdir()
            (task_path / "task.json").write_text("{}")
            answer = task_path / "work" / "answer.txt"
            answer.parent.mkdir(parents=True, exist_ok=True)
            answer.write_text("candidate answer", encoding="utf-8")
            trace = task_path / "work" / "trace.md"
            trace.write_text("candidate trace", encoding="utf-8")

            example = Example(
                id="da-test",
                question="Q?",
                image="",
                target="",
                answer_type="rubric",
                category="x",
                task_path=task_path,
                rubric="RUBRIC\nCriterion 1: Test.",
                grade_threshold=0.7,
            )
            solver_result = SolverResult(
                response="candidate answer",
                ok=True,
                artifacts={"trace": str(trace)},
            )

            grader = CodexLocalGrader(model="gpt-5", timeout=900.0)
            # monkey-patch _parse_score by invoking grader; we can't intercept
            # subprocess output here, so just check it constructs/launches.
            # Instead of running codex end-to-end (requires auth), validate the
            # subprocess argv shape by running with a tiny override.
            captured: dict[str, object] = {}

            class _FakeProc:
                def __init__(self):
                    self.returncode = 0
                async def communicate(self, _):
                    out_file = grader._last_output_path
                    out_file.write_text(json.dumps({"score": 85, "rationale": "ok"}))
                    return (b"", b"")
                async def wait(self):
                    pass
                def kill(self):
                    pass

            # Patch subprocess.Popen-equivalent by injecting via grader._run_subprocess
            async def fake_run(argv, stdin, timeout):
                captured["argv"] = argv
                # simulate output file write
                for i, a in enumerate(argv):
                    if a == "--output-last-message":
                        grader._last_output_path = Path(argv[i + 1])
                        grader._last_output_path.parent.mkdir(parents=True, exist_ok=True)
                grader._last_output_path.write_text(
                    json.dumps({"score": 85, "rationale": "ok"})
                )
                return 0, b"", b""

            grader._run_codex = fake_run  # type: ignore[attr-defined]
            grade = asyncio.run(grader.grade(object(), example, solver_result))
            self.assertTrue(grade.correct)
            self.assertIn("85/100", grade.detail)
            self.assertIn("codex", " ".join(captured["argv"]))  # type: ignore[arg-type]
