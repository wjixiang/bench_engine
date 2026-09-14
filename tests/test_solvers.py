from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from bench_engine.solvers.custom import CustomCommandSolver
from bench_engine.solvers.tui import AutonomicsTuiSolver

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
        solver = CustomCommandSolver("python -c 'import time; time.sleep(2)'", timeout=0.01)
        result = asyncio.run(solver.solve(EXAMPLE, "prompt"))
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.error)

    def test_tui_missing_executable(self) -> None:
        solver = AutonomicsTuiSolver(Path("/does-not-exist/tui"), timeout=1)
        with self.assertLogs("bench_engine.solvers.tui", level="INFO"):
            result = asyncio.run(solver.solve(EXAMPLE, "prompt"))
        self.assertFalse(result.ok)
        self.assertIn("does not exist", result.error)


if __name__ == "__main__":
    unittest.main()
