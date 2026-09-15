"""Autonomics headless solver backend."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from bench_engine.core.interfaces import Example, Solver, SolverResult
from bench_engine.solvers.mounting import mount_example_data

DEFAULT_AUTONOMICS = Path(
    "/mnt/projects/autonomics_projects/autonomics/target/release/autonomics"
)
# Kept for callers that still import the old name.
DEFAULT_TUI = DEFAULT_AUTONOMICS
logger = logging.getLogger(__name__)


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]


class AutonomicsTuiSolver(Solver):
    """Drive Autonomics headless mode in isolated benchmark sessions."""

    def __init__(
        self,
        executable: Path,
        timeout: float,
        *,
        model: str | None = None,
        profile: Path | None = None,
    ) -> None:
        self.executable = executable
        self.timeout = timeout
        self.model = model
        self.profile = profile
        self.data_mount_path: Path | None = None

    async def solve(
        self,
        example: Example,
        prompt: str,
        *,
        data_mount_path: Path | None = None,
    ) -> SolverResult:
        self.data_mount_path = data_mount_path
        data_mount: Path | None = None
        workspace: Path | None = None
        resume_workspace = False
        try:
            mounted_path = mount_example_data(example, data_mount_path)
        except (OSError, ValueError) as exc:
            return SolverResult("", False, error=f"failed to mount task data: {exc}")
        if mounted_path is not None:
            prompt += (
                "\n\nBenchmark inputs are mounted read-only at /data.\n"
                "Write benchmark output files under /app. If the task asks for "
                "answer.txt or trace.md, use /app/answer.txt and /app/trace.md."
            )
            data_mount = (mounted_path / "data").resolve()
            workspace = (mounted_path / "work").resolve()
            resume_workspace = any(workspace.iterdir())
        result = await self.solve_raw(
            prompt,
            task_id=example.id,
            phase="answer",
            answer_type=example.answer_type,
            category=example.category,
            has_image=bool(
                example.image or any(asset.role == "image" for asset in example.assets)
            ),
            working_directory=mounted_path,
            data_mount=data_mount,
            workspace=workspace,
            resume_workspace=resume_workspace,
        )
        if workspace is not None:
            result = self._apply_workspace_artifacts(result, workspace)
        return result

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
        started = time.perf_counter()
        logger.info(
            "solver.start backend=autonomics id=%s phase=%s executable=%s "
            "model=%s timeout=%.3fs prompt_chars=%d answer_type=%s "
            "category=%s image=%s",
            task_id,
            phase,
            self.executable,
            self.model or "<active>",
            self.timeout,
            len(prompt),
            answer_type or "<unknown>",
            category or "<unknown>",
            has_image,
        )
        if not self.executable.is_file():
            result = SolverResult(
                "",
                False,
                error=f"Autonomics executable does not exist: {self.executable}",
            )
            self._log_start_failed(task_id, phase, started, result.error)
            return result

        with tempfile.TemporaryDirectory(prefix="bench-engine-tui-") as directory:
            output = Path(directory) / "last-message.txt"
            manifest = Path(directory) / "manifest.json"
            argv = self._build_argv(
                output,
                manifest,
                data_mount=data_mount,
                workspace=workspace,
                resume_workspace=resume_workspace,
            )

            logger.info("autonomics.spawn id=%s phase=%s argv=%s", task_id, phase, argv)
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=working_directory,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (FileNotFoundError, PermissionError) as exc:
                result = SolverResult(
                    "",
                    False,
                    error=f"failed to start Autonomics: {exc}",
                )
                self._log_start_failed(task_id, phase, started, result.error)
                return result

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode("utf-8")),
                    timeout=self.timeout + 30,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                result = SolverResult(
                    "",
                    False,
                    process.returncode,
                    error=f"Autonomics did not stop within {self.timeout + 30:g}s",
                )
                self._log_end(
                    task_id,
                    phase,
                    started,
                    result,
                    response_chars=0,
                )
                return result

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            session_id: str | None = None
            model: str | None = self.model
            usage: dict[str, Any] | None = None
            for line in stdout_text.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                session_id = event.get("session_id") or session_id
                model = event.get("model") or model
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]

            manifest_data: dict[str, Any] = {}
            try:
                manifest_data = json.loads(manifest.read_text("utf-8"))
            except OSError, json.JSONDecodeError:
                pass
            session_id = manifest_data.get("session_id") or session_id
            model = manifest_data.get("model") or model
            usage = manifest_data.get("usage") or usage

            response = ""
            try:
                response = output.read_text("utf-8", errors="replace").strip()
            except OSError:
                pass

            if not response and workspace is not None:
                answer_path = workspace / "answer.txt"
                if answer_path.is_file():
                    response = answer_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()

            returncode = process.returncode
            status = manifest_data.get("status")
            ok = returncode == 0 and response != "" and status == "completed"
            error = None
            if not ok:
                error = (
                    f"Autonomics produced no final message; code={returncode}, "
                    f"status={status or 'unknown'}"
                    if not response
                    else (
                        f"Autonomics exited with code {returncode}; "
                        f"status={status or 'unknown'}"
                    )
                )
            result = SolverResult(
                response=response,
                ok=ok,
                returncode=returncode,
                error=error,
                session_id=session_id,
                model=model,
                usage=usage,
                stderr_tail=_tail(stderr_text),
            )
            self._log_end(
                task_id,
                phase,
                started,
                result,
                response_chars=len(response),
            )
            return result

    def _build_argv(
        self,
        output: Path,
        manifest: Path,
        *,
        data_mount: Path | None,
        workspace: Path | None,
        resume_workspace: bool,
    ) -> list[str]:
        argv = [
            str(self.executable),
            "run",
            "--json",
            "--ephemeral",
            "--backend",
            "in-process",
            "--timeout",
            str(int(self.timeout)),
            "--output-last-message",
            str(output),
            "--manifest",
            str(manifest),
        ]
        if data_mount is not None:
            argv.extend(("--data-mount", f"{data_mount}=/data"))
        if workspace is not None:
            argv.extend(("--workspace", f"{workspace}=/app"))
        if resume_workspace:
            argv.append("--resume-workspace")
        if self.model is not None:
            argv.extend(("--model", self.model))
        if self.profile is not None:
            argv.extend(("--profile", str(self.profile)))
        argv.append("-")
        return argv

    @staticmethod
    def _collect_artifacts(workspace: Path) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        for filename in ("answer.txt", "trace.md"):
            path = workspace / filename
            if path.is_file():
                key = filename.removesuffix(".txt").removesuffix(".md")
                artifacts[key] = str(path)
        return artifacts

    @classmethod
    def _apply_workspace_artifacts(
        cls,
        result: SolverResult,
        workspace: Path,
    ) -> SolverResult:
        artifacts = cls._collect_artifacts(workspace)
        if result.ok and "answer" in artifacts:
            answer_text = (
                (workspace / "answer.txt")
                .read_text(encoding="utf-8", errors="replace")
                .strip()
            )
            if answer_text:
                result = replace(result, response=answer_text)
        return replace(result, artifacts=artifacts)

    def _log_start_failed(
        self,
        task_id: str,
        phase: str,
        started: float,
        error: str | None,
    ) -> None:
        logger.warning(
            "solver.end backend=autonomics-tui id=%s phase=%s elapsed=%.3fs "
            "ok=False returncode=none response_chars=0 error=%r",
            task_id,
            phase,
            time.perf_counter() - started,
            error,
        )

    def _log_end(
        self,
        task_id: str,
        phase: str,
        started: float,
        result: SolverResult,
        *,
        response_chars: int,
    ) -> None:
        logger.log(
            logging.INFO if result.ok else logging.WARNING,
            "solver.end backend=autonomics-tui id=%s phase=%s elapsed=%.3fs "
            "ok=%s returncode=%s response_chars=%d error=%r",
            task_id,
            phase,
            time.perf_counter() - started,
            result.ok,
            result.returncode,
            response_chars,
            result.error,
        )
