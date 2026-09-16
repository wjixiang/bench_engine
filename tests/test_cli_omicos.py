"""Tests for the OmicOS-BiomniBench CLI integration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import typer.testing

from bench_engine import cli


OMICOS_TASKS = {
    "schema_version": 1,
    "id": "rubric-task",
    "instruction": "Analyze.",
    "category": "oncology",
    "grading": {
        "method": "rubric",
        "answer_type": "rubric",
        "answer": None,
        "rubric": "RUBRIC: 3 criteria worth 33 points each.",
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


def _write_dataset(tmp: Path) -> Path:
    root = tmp
    base = root / "omicos_biomnibench" / "biomnibench_da" / "rubric-task"
    real_data = tmp / "environment"
    real_data.mkdir(parents=True, exist_ok=True)
    (real_data / "matrix.tsv").write_text("a\tb\n1\t2\n")
    base.mkdir(parents=True, exist_ok=True)
    if (base / "data").is_symlink() or (base / "data").exists():
        (base / "data").unlink()
    (base / "data").symlink_to(real_data)
    (base / "task.json").write_text(json.dumps(OMICOS_TASKS), encoding="utf-8")
    # Stub HLE datasets so the datasets command does not fail when BENCH_ENGINE_DATA_DIR
    # points at a temporary directory.
    for name in ("full", "biomedical", "biomedical_visual"):
        stub = root / "hle" / name
        stub.mkdir(parents=True, exist_ok=True)
    # Stub LAB-Bench datasets too.
    for alias in (
        "cloning_scenarios",
        "dbqa",
        "figqa",
        "litqa2",
        "protocolqa",
        "seqqa",
        "suppqa",
        "tableqa",
    ):
        (root / "lab_bench" / alias).mkdir(parents=True, exist_ok=True)
    return root


class CLIOmicOSTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = typer.testing.CliRunner()

    def test_datasets_command_includes_omicos(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BENCH_ENGINE_DATA_DIR": str(_write_dataset(Path("/tmp/omicos-cli"))),
                "OPENAI_API_KEY": "test-key",
            },
            clear=True,
        ):
            from bench_engine.core import data as data_module

            data_module.DATASET_ROOT = Path("/tmp/omicos-cli")
            result = self.runner.invoke(cli.app, ["datasets"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("omicos-biomnibench", result.output)

    def test_evaluate_dry_run_with_omicos_benchmark(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BENCH_ENGINE_DATA_DIR": str(_write_dataset(Path("/tmp/omicos-cli2"))),
                "OPENAI_API_KEY": "test-key",
            },
            clear=True,
        ):
            from bench_engine.core import data as data_module

            data_module.DATASET_ROOT = Path("/tmp/omicos-cli2")
            result = self.runner.invoke(
                cli.app,
                [
                    "evaluate",
                    "--benchmark",
                    "omicos-biomnibench",
                    "--limit",
                    "1",
                    "--dry-run",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("rubric-task", result.output)
        self.assertIn("/app/answer.txt", result.output)

    def test_evaluate_rejects_unknown_grader(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BENCH_ENGINE_DATA_DIR": str(_write_dataset(Path("/tmp/omicos-cli3"))),
                "OPENAI_API_KEY": "test-key",
            },
            clear=True,
        ):
            from bench_engine.core import data as data_module

            data_module.DATASET_ROOT = Path("/tmp/omicos-cli3")
            result = self.runner.invoke(
                cli.app,
                [
                    "evaluate",
                    "--benchmark",
                    "omicos-biomnibench",
                    "--limit",
                    "1",
                    "--dry-run",
                    "--grader",
                    "nope",
                ],
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--grader must be", result.output)

    def test_evaluate_omicos_grader_requires_model(self) -> None:
        with patch.dict(
            "os.environ",
            {"BENCH_ENGINE_DATA_DIR": str(_write_dataset(Path("/tmp/omicos-cli4")))},
            clear=True,
        ):
            from bench_engine.core import data as data_module

            data_module.DATASET_ROOT = Path("/tmp/omicos-cli4")
            result = self.runner.invoke(
                cli.app,
                [
                    "evaluate",
                    "--benchmark",
                    "omicos-biomnibench",
                    "--limit",
                    "1",
                    "--dry-run",
                    "--grader",
                    "omicos",
                ],
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("requires --grader-model", result.output)

    def test_gateway_mode_requires_mount_path_and_accepts_concurrency(self) -> None:
        common = [
            "evaluate",
            "--benchmark",
            "omicos-biomnibench",
            "--limit",
            "1",
            "--dry-run",
            "--autonomics-gateway",
        ]
        with patch.dict(
            "os.environ",
            {"BENCH_ENGINE_DATA_DIR": str(_write_dataset(Path("/tmp/omicos-cli5")))},
            clear=True,
        ):
            from bench_engine.core import data as data_module

            data_module.DATASET_ROOT = Path("/tmp/omicos-cli5")
            missing_mount = self.runner.invoke(cli.app, common)
            concurrent = self.runner.invoke(
                cli.app,
                [
                    *common,
                    "--data-mount-path",
                    "/tmp/omicos-cli5/mounts",
                    "--jobs",
                    "2",
                ],
            )

        self.assertNotEqual(missing_mount.exit_code, 0)
        self.assertIn("requires --data-mount-path", missing_mount.output)
        self.assertEqual(concurrent.exit_code, 0, concurrent.output)

    def test_data_mount_requires_gateway_vfs_mode(self) -> None:
        with patch.dict(
            "os.environ",
            {"BENCH_ENGINE_DATA_DIR": str(_write_dataset(Path("/tmp/omicos-cli6")))},
            clear=True,
        ):
            from bench_engine.core import data as data_module

            data_module.DATASET_ROOT = Path("/tmp/omicos-cli6")
            result = self.runner.invoke(
                cli.app,
                [
                    "evaluate",
                    "--benchmark",
                    "omicos-biomnibench",
                    "--limit",
                    "1",
                    "--dry-run",
                    "--data-mount-path",
                    "/tmp/omicos-cli6/mounts",
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--data-mount-path requires --autonomics-gateway", result.output)


if __name__ == "__main__":
    unittest.main()
