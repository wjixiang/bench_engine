"""Custom external solver backend."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from pathlib import Path

from bench_engine.core.interfaces import Example, Solver, SolverResult
from bench_engine.solvers.mounting import mount_example_data

logger = logging.getLogger(__name__)


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]


def _json_or_text(text: str) -> str:
    try:
        value = json.loads(text)
    except json.JSONDecodeError, TypeError:
        return text
    if isinstance(value, dict):
        for key in ("answer", "response", "final_message", "output"):
            if isinstance(value.get(key), str):
                return value[key]
    return text


class CustomCommandSolver(Solver):
    """Run a stdin/stdout command for one benchmark item."""

    def __init__(self, command: str, timeout: float) -> None:
        self.argv = shlex.split(command)
        self.timeout = timeout
        if not self.argv:
            raise ValueError("solver command is empty")
        self.data_mount_path: Path | None = None

    async def solve(
        self,
        example: Example,
        prompt: str,
        *,
        data_mount_path: Path | None = None,
    ) -> SolverResult:
        del prompt
        self.data_mount_path = data_mount_path
        try:
            mounted_path = mount_example_data(example, data_mount_path)
        except (OSError, ValueError) as exc:
            return SolverResult("", False, error=f"failed to mount task data: {exc}")
        started = time.perf_counter()
        logger.info(
            "solver.start backend=custom-command id=%s command=%r "
            "question_chars=%d image_chars=%d answer_type=%s",
            example.id,
            shlex.join(self.argv),
            len(example.question),
            len(example.image),
            example.answer_type,
        )
        payload = json.dumps(example.payload(mounted_path), ensure_ascii=False)
        try:
            process = await asyncio.create_subprocess_exec(
                *self.argv,
                cwd=mounted_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError) as exc:
            result = SolverResult("", False, error=f"failed to start solver: {exc}")
            self._log_end(example.id, started, result)
            return result

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(payload.encode("utf-8")), timeout=self.timeout
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            result = SolverResult(
                "",
                False,
                process.returncode,
                error=f"solver timed out after {self.timeout:g}s",
            )
            self._log_end(example.id, started, result)
            return result

        response = _json_or_text(stdout.decode("utf-8", errors="replace")).strip()
        returncode = process.returncode
        result = SolverResult(
            response=response,
            ok=returncode == 0,
            returncode=returncode,
            error=None if returncode == 0 else f"solver exited with code {returncode}",
            stderr_tail=_tail(stderr.decode("utf-8", errors="replace")),
        )
        self._log_end(example.id, started, result)
        return result

    def _log_end(self, task_id: str, started: float, result: SolverResult) -> None:
        logger.log(
            logging.INFO if result.ok else logging.WARNING,
            "solver.end backend=custom-command id=%s elapsed=%.3fs ok=%s "
            "returncode=%s response_chars=%d error=%r",
            task_id,
            time.perf_counter() - started,
            result.ok,
            result.returncode,
            len(result.response),
            result.error,
        )
