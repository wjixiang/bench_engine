"""Typer command-line interface for the benchmark driver."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from bench_engine.benchmarks.hle import HLE, load_hle_examples
from bench_engine.core.data import load_examples
from bench_engine.core.runner import (
    ModelGrader,
    completed_ids,
    evaluate_examples,
    read_results,
    summarize,
)
from bench_engine.solvers.custom import CustomCommandSolver
from bench_engine.solvers.tui import DEFAULT_TUI, AutonomicsTuiSolver

app = typer.Typer(
    help="Run and score local benchmark datasets.",
    no_args_is_help=True,
)


def _default_output() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"bench-engine-{stamp}.jsonl"


def _default_tui(override: Path | None) -> Path:
    if override is not None:
        return override
    configured = os.environ.get("BENCH_ENGINE_TUI")
    if configured:
        return Path(configured)
    return DEFAULT_TUI


@app.command()
def datasets() -> None:
    """List registered benchmark datasets and row counts."""
    for info in load_hle_datasets():
        typer.echo(f"{info.name}\t{info.rows}\t{info.path}")


def load_hle_datasets():
    from bench_engine.benchmarks.hle import hle_datasets

    return hle_datasets()


@app.command()
def evaluate(
    benchmark: Annotated[
        str,
        typer.Option(
            "--benchmark",
            "-b",
            help="Registered benchmark: hle-all, hle-biomedical, or "
            "hle-biomedical-visual.",
        ),
    ] = "hle-biomedical",
    data: Annotated[
        Path | None,
        typer.Option("--data", help="Override path to a parquet file."),
    ] = None,
    offset: Annotated[int, typer.Option(min=0)] = 0,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Maximum number of questions to run."),
    ] = None,
    question_id: Annotated[
        list[str] | None,
        typer.Option(
            "--question-id",
            "--id",
            help="Question ID to include; repeat for multiple IDs.",
        ),
    ] = None,
    answer_type: Annotated[
        str | None,
        typer.Option("--answer-type", case_sensitive=False),
    ] = None,
    category: Annotated[str | None, typer.Option("--category")] = None,
    shuffle: Annotated[bool, typer.Option("--shuffle/--no-shuffle")] = False,
    seed: Annotated[int, typer.Option(min=0)] = 0,
    solver_command: Annotated[
        str | None,
        typer.Option(
            "--solver-command",
            help="Custom argv command; receives JSON on stdin and response on stdout.",
        ),
    ] = None,
    tui: Annotated[
        Path | None,
        typer.Option("--tui", help="Autonomics TUI executable."),
    ] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    profile: Annotated[
        Path | None, typer.Option("--profile", help="Autonomics agent profile.")
    ] = None,
    include_image_uri: Annotated[
        bool,
        typer.Option(
            "--include-image-uri/--no-image-uri",
            help="Include image data URIs in TUI prompts (text transport caveat).",
        ),
    ] = False,
    timeout: Annotated[float, typer.Option(min=1)] = 1800.0,
    grader_mode: Annotated[
        str,
        typer.Option("--grader", help="Exact deterministic grading or model grading."),
    ] = "exact",
    grader_model: Annotated[
        str | None, typer.Option("--grader-model")
    ] = None,
    jobs: Annotated[int, typer.Option(min=1, max=32)] = 1,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    summary: Annotated[
        Path | None,
        typer.Option("--summary", help="Summary JSON path; default is beside output."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip IDs already in --output."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print prompts as JSONL without invoking a model."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress solver progress logs."),
    ] = False,
) -> None:
    """Run a solver over selected questions and write JSONL results."""
    if data is not None and benchmark != "hle-all":
        raise typer.BadParameter("--data requires --benchmark hle-all")
    if grader_mode not in {"exact", "model"}:
        raise typer.BadParameter("--grader must be 'exact' or 'model'")
    if grader_mode == "model" and solver_command is not None:
        raise typer.BadParameter(
            "TUI model grading is not available with --solver-command; use exact"
        )
    if not quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )

    try:
        if data is not None:
            examples = load_examples(
                data.expanduser().resolve(),
                offset=offset,
                limit=limit,
                ids=question_id,
                answer_type=answer_type,
                category=category,
                shuffle=shuffle,
                seed=seed,
            )
        else:
            names = {
                "hle-all": "all",
                "hle-biomedical": "biomedical",
                "hle-biomedical-visual": "biomedical_visual",
            }
            if benchmark not in names:
                choices = ", ".join(names)
                raise ValueError(
                    f"unknown benchmark {benchmark!r}; choose one of: {choices}"
                )
            examples = load_hle_examples(
                names[benchmark],
                offset=offset,
                limit=limit,
                ids=question_id,
                answer_type=answer_type,
                category=category,
                shuffle=shuffle,
                seed=seed,
            )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not examples:
        typer.echo("No questions matched the selection.", err=True)
        raise typer.Exit(code=1)
    if output is None:
        output = _default_output()
    if output.exists() and not resume and not dry_run:
        raise typer.BadParameter(
            f"output exists: {output}; use --resume to continue it or choose --output"
        )

    run_benchmark = (
        replace(HLE, include_image_uri=True) if include_image_uri else HLE
    )

    if solver_command is not None:
        try:
            solver = CustomCommandSolver(solver_command, timeout)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        solver_kind = f"command:{shlex.join(solver.argv)}"
        engine_grader = None
    else:
        executable = _default_tui(tui)
        solver = AutonomicsTuiSolver(
            executable,
            timeout,
            model=model,
            profile=profile,
        )
        solver_kind = f"tui:{solver.executable}"
        engine_grader = None
        if grader_mode == "model":
            engine_grader = ModelGrader(
                HLE,
                AutonomicsTuiSolver(
                    executable,
                    min(timeout, 600.0),
                    model=grader_model,
                    profile=profile,
                ),
            )

    try:
        skip = completed_ids(output) if resume and output.exists() else set()
        asyncio.run(
            evaluate_examples(
                run_benchmark,
                examples,
                solver,
                solver_kind=solver_kind,
                output=output,
                grader=engine_grader,
                jobs=jobs,
                skip_ids=skip,
                dry_run=dry_run,
            )
        )
        all_records = read_results(output)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except KeyboardInterrupt:
        typer.echo("Interrupted; completed records remain in --output.", err=True)
        raise

    if dry_run:
        return

    summary_path = output.with_suffix(".summary.json") if summary is None else summary
    metrics = summarize(all_records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2))
    typer.echo(f"Results: {output}")
    typer.echo(f"Summary: {summary_path}")


def main() -> None:
    app()
