from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench_engine.core.interfaces import Example, SolverResult, TaskAsset
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

    def test_autonomics_command_uses_a_unique_named_agent(self) -> None:
        solver = AutonomicsTuiSolver(
            Path("/usr/bin/autonomics"),
            timeout=123,
            model="provider:model",
            profile=Path("/profiles/benchmark.md"),
        )

        argv = solver._build_argv(
            Path("/tmp/last-message.txt"),
            Path("/tmp/manifest.json"),
            task_id="DA-1/3",
        )

        name = argv[argv.index("--name") + 1]
        second = solver._build_argv(
            Path("/tmp/last-message.txt"),
            Path("/tmp/manifest.json"),
            task_id="DA-1/3",
        )[argv.index("--name") + 1]
        self.assertRegex(name, r"^be_da_1_3_[0-9a-f]{8}$")
        self.assertNotEqual(name, second)

        long_name = solver._build_argv(
            Path("/tmp/last-message.txt"),
            Path("/tmp/manifest.json"),
            task_id="a" * 100,
        )[argv.index("--name") + 1]
        self.assertLessEqual(len(long_name), 32)
        self.assertTrue(long_name.startswith("be_aaaaaaaaaaaaaaaaaaaa_"))

        self.assertNotIn("--ephemeral", argv)
        self.assertNotIn("--backend", argv)
        self.assertNotIn("--data-mount", argv)
        self.assertNotIn("--workspace", argv)
        self.assertNotIn("--resume-workspace", argv)
        self.assertIn("--no-memory", argv)
        self.assertEqual(argv[-1], "-")

    def test_autonomics_gateway_uses_fresh_session_and_virtual_paths(self) -> None:
        captured: dict[str, object] = {}

        class StubAutonomicsSolver(AutonomicsTuiSolver):
            async def solve_raw(
                self,
                prompt: str,
                *,
                task_id: str = "<raw>",
                phase: str = "raw",
                answer_type: str | None = None,
                category: str | None = None,
                has_image: bool = False,
                working_directory: Path | None = None,
                data_mount: Path | None = None,
                workspace: Path | None = None,
                resume_workspace: bool = False,
                session_id: str | None = None,
            ) -> SolverResult:
                captured.update(
                    {
                        "prompt": prompt,
                        "workspace": workspace,
                    }
                )
                assert workspace is not None
                (workspace / "answer.txt").write_text(
                    "gateway answer", encoding="utf-8"
                )
                (workspace / "trace.md").write_text("gateway trace", encoding="utf-8")
                return SolverResult("final message", True)

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
            solver = StubAutonomicsSolver(
                Path("/usr/bin/autonomics"),
                timeout=5,
                use_gateway=True,
            )

            with patch(
                "bench_engine.solvers.autonomics_solver.virtual_path",
                side_effect=lambda path: f"/virtual/{path.name}",
            ):
                result = asyncio.run(
                    solver.solve(
                        example,
                        "prompt",
                        data_mount_path=root / "mounts",
                    )
                )

            prompt = str(captured["prompt"])
            self.assertIn("resident Autonomics gateway VFS", prompt)
            self.assertIn("/virtual/data", prompt)
            self.assertIn("/virtual/work/answer.txt", prompt)

        self.assertEqual(result.response, "gateway answer")
        self.assertEqual(sorted(result.artifacts), ["answer", "trace"])

    def test_autonomics_collects_answer_and_trace_artifacts(self) -> None:
        captured: dict[str, object] = {}

        class StubAutonomicsSolver(AutonomicsTuiSolver):
            async def solve_raw(
                self,
                prompt: str,
                *,
                task_id: str = "<raw>",
                phase: str = "raw",
                answer_type: str | None = None,
                category: str | None = None,
                has_image: bool = False,
                working_directory: Path | None = None,
                data_mount: Path | None = None,
                workspace: Path | None = None,
                resume_workspace: bool = False,
            ) -> SolverResult:
                captured.update(
                    {
                        "prompt": prompt,
                        "data_mount": data_mount,
                        "workspace": workspace,
                        "resume_workspace": resume_workspace,
                    }
                )
                assert workspace is not None
                (workspace / "answer.txt").write_text(
                    "artifact answer", encoding="utf-8"
                )
                (workspace / "trace.md").write_text("trace", encoding="utf-8")
                return SolverResult("final message", True)

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
            solver = StubAutonomicsSolver(Path("/usr/bin/autonomics"), timeout=5)

            result = asyncio.run(
                solver.solve(example, "prompt", data_mount_path=root / "mounts")
            )
            assert isinstance(captured["data_mount"], Path)
            assert isinstance(captured["workspace"], Path)
            self.assertTrue((captured["data_mount"] / "input.txt").is_file())
            self.assertTrue(
                (captured["data_mount"].parent / "data/input.txt").is_symlink()
            )
            self.assertTrue((captured["workspace"] / "answer.txt").is_file())

        self.assertEqual(result.response, "artifact answer")
        self.assertTrue(result.ok)
        self.assertEqual(sorted(result.artifacts), ["answer", "trace"])
        self.assertIn("/data", str(captured["prompt"]))
        self.assertIn("/app", str(captured["prompt"]))
        self.assertEqual(captured["resume_workspace"], False)

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
            self.assertTrue((first / "work").is_dir())

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
