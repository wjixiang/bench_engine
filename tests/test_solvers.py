from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from bench_engine.core.interfaces import Example, TaskAsset
from bench_engine.solvers.autonomics_solver import AutonomicsTuiSolver
from bench_engine.solvers.custom import CustomCommandSolver
from bench_engine.solvers.mounting import mount_example_data
from tests.test_core import EXAMPLE


class SolverTest(unittest.TestCase):
    def test_custom_solver_success_and_logs(self) -> None:
        command = "python -c 'import sys; sys.stdin.read(); print(\"Answer: 4\")'"
        solver = CustomCommandSolver(command, timeout=5)
        with self.assertLogs("bench_engine.solvers.custom", level="INFO"):
            result = asyncio.run(solver.solve(EXAMPLE, "prompt"))
        self.assertTrue(result.ok)
        self.assertEqual(result.response, "Answer: 4")

    def test_custom_solver_timeout(self) -> None:
        solver = CustomCommandSolver(
            "python -c 'import time; time.sleep(2)'", timeout=0.01
        )
        result = asyncio.run(solver.solve(EXAMPLE, "prompt"))
        self.assertFalse(result.ok)
        error = result.error
        self.assertTrue(error is not None and "timed out" in error)

    def test_autonomics_missing_executable(self) -> None:
        solver = AutonomicsTuiSolver(Path("/does-not-exist/tui"), timeout=1)
        with self.assertLogs("bench_engine.solvers.autonomics_solver", level="INFO"):
            result = asyncio.run(solver.solve(EXAMPLE, "prompt"))
        self.assertFalse(result.ok)
        error = result.error
        self.assertTrue(error is not None and "does not exist" in error)

    def test_custom_solver_receives_mounted_data_and_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-data.txt"
            source.write_text("input", encoding="utf-8")
            example = Example(
                id="task-1",
                question="Question",
                image="",
                target="A",
                answer_type="multipleChoice",
                category="Test",
                assets=(
                    TaskAsset(
                        "input",
                        source,
                        "text/plain",
                        "context",
                        relative_path="data/input.txt",
                    ),
                ),
            )
            mount_root = root / "mounts"
            helper = root / "print_payload.py"
            helper.write_text(
                "import json, os, sys\n"
                "payload = json.load(sys.stdin)\n"
                "print(payload['data'][0]['path'])\n"
                "print(os.getcwd())\n",
                encoding="utf-8",
            )
            solver = CustomCommandSolver(
                f"python {helper}",
                timeout=5,
            )

            result = asyncio.run(
                solver.solve(example, "prompt", data_mount_path=mount_root)
            )

        self.assertTrue(result.ok)
        mounted_input, mounted_cwd = result.response.splitlines()
        self.assertEqual(mounted_input, str(mount_root / "task-1" / "data/input.txt"))
        self.assertEqual(mounted_cwd, str(mount_root / "task-1"))

    def test_data_mount_is_idempotent_and_rejects_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("input", encoding="utf-8")
            example = Example(
                id="task-1",
                question="Question",
                image="",
                target="A",
                answer_type="multipleChoice",
                category="Test",
                assets=(
                    TaskAsset(
                        "input",
                        source,
                        relative_path="data/input.txt",
                    ),
                ),
            )
            mount_root = root / "mounts"
            first = mount_example_data(example, mount_root)
            second = mount_example_data(example, mount_root)

            assert first is not None and second is not None
            self.assertEqual(first, second)
            self.assertEqual(
                (first / "data/input.txt").resolve(),
                source.resolve(),
            )

            unsafe = Example(
                id="../unsafe",
                question="Question",
                image="",
                target="A",
                answer_type="multipleChoice",
                category="Test",
            )
            with self.assertRaisesRegex(ValueError, "unsafe task id"):
                mount_example_data(unsafe, mount_root)


if __name__ == "__main__":
    unittest.main()
