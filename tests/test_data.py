from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from bench_engine.core.data import load_examples, validate_dataset_schema


def write_task_package(
    directory: Path,
    *,
    include_instruction_file: bool = False,
) -> Path:
    task_path = directory / "tasks" / "task-1"
    data_path = task_path / "data"
    data_path.mkdir(parents=True)
    (data_path / "input.txt").write_text("sample input", encoding="utf-8")
    task = {
        "schema_version": 1,
        "id": "task-1",
        "instruction": "Answer the question.\n\nOptions:\n(A) One\n(B) Two",
        "category": "Test",
        "grading": {
            "method": "exact",
            "answer_type": "multipleChoice",
            "answer": "B",
        },
        "data": [
            {
                "name": "input",
                "path": "data/input.txt",
                "media_type": "text/plain",
                "role": "context",
            }
        ],
    }
    (task_path / "task.json").write_text(
        json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if include_instruction_file:
        (task_path / "instruction.md").write_text(
            "Separated instruction", encoding="utf-8"
        )
    return directory / "tasks"


def valid_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["q1"],
            "question": ["What is 2 + 2?"],
            "image": ["data:image/png;base64,aGVsbG8="],
            "answer": ["4"],
            "answer_type": ["exactMatch"],
            "category": ["Mathematics"],
            "source_metadata": ["extra columns are allowed"],
        }
    )


class DataTest(unittest.TestCase):
    def test_loads_schema_v1_and_preserves_extra_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.parquet"
            valid_frame().write_parquet(path)
            examples = load_examples(path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].id, "q1")
        self.assertEqual(examples[0].image, "data:image/png;base64,aGVsbG8=")

    def test_rejects_missing_and_mistyped_columns(self) -> None:
        frame = valid_frame().drop("category").with_columns(pl.lit(1).alias("id"))
        with self.assertRaises(ValueError) as raised:
            validate_dataset_schema(frame)

        message = str(raised.exception)
        self.assertIn("missing required columns: category", message)
        self.assertIn("id must be String", message)

    def test_rejects_invalid_values(self) -> None:
        frame = valid_frame().with_columns(
            pl.lit("").alias("answer"),
            pl.lit("shortAnswer").alias("answer_type"),
            pl.lit("not-an-image").alias("image"),
        )
        frame = pl.concat([frame, frame], how="vertical")
        with self.assertRaises(ValueError) as raised:
            validate_dataset_schema(frame)

        message = str(raised.exception)
        self.assertIn("answer has 2 blank values", message)
        self.assertIn("id has 2 duplicate values", message)
        self.assertIn("answer_type has 2 invalid values", message)
        self.assertIn("image has 2 invalid values", message)

    def test_loads_task_package_with_instruction_in_task_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = write_task_package(Path(directory))
            examples = load_examples(dataset_path)
            self.assertEqual(len(examples), 1)
            example = examples[0]
            self.assertEqual(example.id, "task-1")
            self.assertIn("Options:", example.question)
            self.assertEqual(example.target, "B")
            self.assertEqual(len(example.assets), 1)
            self.assertEqual(example.assets[0].name, "input")
            self.assertTrue(example.assets[0].path.is_file())
        payload = example.payload()
        self.assertEqual(payload["data"][0]["name"], "input")
        self.assertIn("path", payload["data"][0])
        self.assertTrue(payload["data"][0]["path"].endswith("data/input.txt"))

    def test_rejects_separate_instruction_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = write_task_package(
                Path(directory), include_instruction_file=True
            )
            with self.assertRaisesRegex(ValueError, "separate instruction"):
                load_examples(dataset_path)


if __name__ == "__main__":
    unittest.main()


def write_rubric_task_package(directory: Path) -> Path:
    task_path = directory / "tasks" / "rubric-task"
    real_data = directory / "real" / "rubric-task" / "environment"
    real_data.mkdir(parents=True)
    (real_data / "matrix.tsv").write_text("cell data")
    task_path.mkdir(parents=True)
    (task_path / "data").symlink_to(real_data)
    task = {
        "schema_version": 1,
        "id": "rubric-task",
        "instruction": "Analyze the data end to end.",
        "category": "oncology",
        "subtask": "cell-composition",
        "grading": {
            "method": "rubric",
            "answer_type": "rubric",
            "answer": None,
            "rubric": "RUBRIC: 5 criteria worth 20 points each.",
            "pass_threshold": 0.6,
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
    (task_path / "task.json").write_text(
        json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return directory / "tasks"


class RubricTaskTest(unittest.TestCase):
    def test_loads_rubric_task_with_null_answer_and_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = write_rubric_task_package(Path(directory))
            examples = load_examples(dataset_path)
            self.assertEqual(len(examples), 1)
            example = examples[0]
            self.assertEqual(example.target, "")
            self.assertEqual(example.answer_type, "rubric")
            self.assertIn("RUBRIC", example.rubric)
            self.assertEqual(example.grade_threshold, 0.6)
            self.assertEqual(len(example.assets), 1)
            self.assertTrue(example.assets[0].path.is_dir())
            self.assertNotEqual(example.assets[0].path, example.task_path / "data")

    def test_accepts_threshold_alias_pass_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task_path = base / "tasks" / "rubric-task"
            (task_path / "data").mkdir(parents=True)
            (task_path / "data" / "x").write_text("x")
            task = {
                "schema_version": 1,
                "id": "rubric-task",
                "instruction": "Q.",
                "category": "x",
                "grading": {
                    "method": "rubric",
                    "answer_type": "rubric",
                    "answer": None,
                    "rubric": "R.",
                    "threshold": 0.42,
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
            (task_path / "task.json").write_text(json.dumps(task))
            [example] = load_examples(base / "tasks")
            self.assertEqual(example.grade_threshold, 0.42)

    def test_rejects_rubric_without_rubric_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task_path = base / "tasks" / "rubric-task"
            (task_path / "data").mkdir(parents=True)
            (task_path / "data" / "x").write_text("x")
            task = {
                "schema_version": 1,
                "id": "rubric-task",
                "instruction": "Q.",
                "category": "x",
                "grading": {
                    "method": "rubric",
                    "answer_type": "rubric",
                    "answer": None,
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
            (task_path / "task.json").write_text(json.dumps(task))
            with self.assertRaisesRegex(ValueError, "rubric grading requires"):
                load_examples(base / "tasks")

    def test_rejects_rubric_with_non_string_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task_path = base / "tasks" / "rubric-task"
            (task_path / "data").mkdir(parents=True)
            (task_path / "data" / "x").write_text("x")
            task = {
                "schema_version": 1,
                "id": "rubric-task",
                "instruction": "Q.",
                "category": "x",
                "grading": {
                    "method": "rubric",
                    "answer_type": "rubric",
                    "answer": 42,
                    "rubric": "R.",
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
            (task_path / "task.json").write_text(json.dumps(task))
            with self.assertRaisesRegex(ValueError, "null or a string"):
                load_examples(base / "tasks")

    def test_rejects_out_of_range_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task_path = base / "tasks" / "rubric-task"
            (task_path / "data").mkdir(parents=True)
            (task_path / "data" / "x").write_text("x")
            task = {
                "schema_version": 1,
                "id": "rubric-task",
                "instruction": "Q.",
                "category": "x",
                "grading": {
                    "method": "rubric",
                    "answer_type": "rubric",
                    "answer": None,
                    "rubric": "R.",
                    "pass_threshold": 1.5,
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
            (task_path / "task.json").write_text(json.dumps(task))
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                load_examples(base / "tasks")
