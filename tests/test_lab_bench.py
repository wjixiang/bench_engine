from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from bench_engine.benchmarks.lab_bench import LAB_BENCH
from bench_engine.core.interfaces import Example, TaskAsset

EXAMPLE = Example(
    id="lab-1",
    question="Question:\nWhat enzyme cuts DNA?\n\nOptions:\n(A) Ligase\n(B) EcoRI",
    image="data:image/png;base64,aGVsbG8=",
    target="B",
    answer_type="multipleChoice",
    category="SeqQA",
)


class LabBenchTest(unittest.TestCase):
    def test_prompt_extract_and_grade(self) -> None:
        prompt = LAB_BENCH.prompt(EXAMPLE)
        response = "The enzyme is EcoRI.\nAnswer: B"

        self.assertIn("biology benchmark", prompt)
        self.assertEqual(LAB_BENCH.extract(response, EXAMPLE), "B")
        self.assertTrue(LAB_BENCH.grade(response, EXAMPLE).correct)

    def test_model_grader_prompt_contains_reference_letter(self) -> None:
        prompt = LAB_BENCH.grader_prompt(EXAMPLE, "Answer: A")

        self.assertIn("Reference option letter:\nB", prompt)
        self.assertIn("Verdict: CORRECT", prompt)

    def test_prompt_lists_image_asset_path(self) -> None:
        asset = TaskAsset(
            "image",
            Path("/tasks/task-1/data/image.png"),
            "image/png",
            "image",
        )
        example = Example(
            id="lab-2",
            question="Question",
            image="",
            target="A",
            answer_type="multipleChoice",
            category="FigQA",
            assets=(asset,),
        )
        benchmark = replace(LAB_BENCH, include_image_uri=True)
        prompt = benchmark.prompt(example)

        self.assertIn("/tasks/task-1/data/image.png", prompt)


if __name__ == "__main__":
    unittest.main()
