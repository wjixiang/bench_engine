"""Typer command-line interface for the benchmark driver."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shlex
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from dotenv import load_dotenv

from bench_engine.benchmarks.hle import HLE
from bench_engine.benchmarks.lab_bench import LAB_BENCH, LAB_BENCH_DATASETS
from bench_engine.benchmarks.omicos import OMICOS
from bench_engine.core.data import (
    DatasetInfo,
    dataset_path,
    inspect_dataset,
    load_examples,
)
from bench_engine.core.interfaces import Example
from bench_engine.core.runner import (
    OpenAIGrader,
    completed_ids,
    evaluate_examples,
    read_results,
    summarize,
)
from bench_engine.grading import CodexLocalGrader, OmicOSGrader
from bench_engine.solvers.autonomics_solver import (
    DEFAULT_AUTONOMICS,
    AutonomicsTuiSolver,
)
from bench_engine.solvers.custom import CustomCommandSolver

load_dotenv(Path(".env"))

app = typer.Typer(
    help="Run and score local benchmark datasets.",
    no_args_is_help=True,
)

OMICOS_BIOMNIBENCH_DATASETS = {
    "omicos-biomnibench": "omicos_biomnibench:biomnibench_da",
}


def _default_output() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"bench-engine-{stamp}.jsonl"


def _default_autonomics(override: Path | None) -> Path:
    if override is not None:
        return override
    configured = os.environ.get("BENCH_ENGINE_AUTONOMICS") or os.environ.get(
        "BENCH_ENGINE_TUI"
    )
    if configured:
        return Path(configured)
    return DEFAULT_AUTONOMICS


@app.command()
def datasets() -> None:
    """List registered benchmark datasets and row counts."""
    infos = [
        *load_hle_datasets(),
    ]
    for alias, dataset_name in LAB_BENCH_DATASETS.items():
        path = dataset_path(dataset_name)
        infos.append(DatasetInfo(alias, path, int(inspect_dataset(path).rows)))
    for alias, dataset_name in OMICOS_BIOMNIBENCH_DATASETS.items():
        path = dataset_path(dataset_name)
        infos.append(DatasetInfo(alias, path, int(inspect_dataset(path).rows)))
    for info in infos:
        typer.echo(f"{info.name}\t{info.rows}\t{info.path}")


def load_hle_datasets():
    from bench_engine.benchmarks.hle import hle_datasets

    return hle_datasets()


def _load_lab_bench_examples(
    benchmark: str,
    *,
    offset: int,
    limit: int | None,
    ids: list[str] | None,
    answer_type: str | None,
    category: str | None,
    shuffle: bool,
    seed: int,
) -> list[Example]:
    dataset_names = (
        (LAB_BENCH_DATASETS[benchmark],)
        if benchmark in LAB_BENCH_DATASETS
        else tuple(LAB_BENCH_DATASETS.values())
    )
    examples: list[Example] = []
    for dataset_name in dataset_names:
        examples.extend(
            load_examples(
                dataset_path(dataset_name),
                offset=0,
                limit=None,
                ids=ids,
                answer_type=answer_type,
                category=category,
                shuffle=False,
                seed=seed,
            )
        )
    if shuffle:
        random.Random(seed).shuffle(examples)
    end = None if limit is None else offset + limit
    return examples[offset:end]


@app.command()
def evaluate(
    benchmark: Annotated[
        str,
        typer.Option(
            "--benchmark",
            "-b",
            help="Registered benchmark: hle-all, hle-biomedical, "
            "hle-biomedical-visual, lab-bench, or a lab-<category> dataset.",
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
    autonomics: Annotated[
        Path | None,
        typer.Option(
            "--autonomics",
            "--tui",
            help="Autonomics headless executable.",
        ),
    ] = None,
    autonomics_gateway: Annotated[
        bool,
        typer.Option(
            "--autonomics-gateway/--autonomics-ephemeral",
            help=(
                "Reuse the resident Autonomics gateway (one fresh fallback identity "
                "per task) or run isolated in-process Autonomics processes."
            ),
        ),
    ] = False,
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
    data_mount_path: Annotated[
        Path | None,
        typer.Option(
            "--data-mount-path",
            "--data-mount",
            help="Mount task data below this directory for each solver process.",
        ),
    ] = None,
    grader_mode: Annotated[
        str,
        typer.Option(
            "--grader",
            help="Grader mode: 'exact', 'model', 'omicos' for BigModel-compatible rubric scoring, or 'codex-local' for local headless codex exec.",
        ),
    ] = "exact",
    grader_model: Annotated[
        str | None,
        typer.Option(
            "--grader-model",
            help="OpenAI model used by the native LLM grader.",
        ),
    ] = None,
    grader_base_url: Annotated[
        str | None,
        typer.Option(
            "--grader-base-url",
            help="OpenAI-compatible API base URL for the native LLM grader.",
        ),
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
        typer.Option(
            "--dry-run", help="Print prompts as JSONL without invoking a model."
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress solver progress logs."),
    ] = False,
) -> None:
    """Run a solver over selected questions and write JSONL results."""
    if data is not None and benchmark != "hle-all":
        raise typer.BadParameter("--data requires --benchmark hle-all")
    if grader_mode not in {"exact", "model", "omicos", "codex-local"}:
        raise typer.BadParameter(
            "--grader must be 'exact', 'model', 'omicos', or 'codex-local'"
        )
    selected_grader_model = grader_model or os.environ.get("OPENAI_GRADER_MODEL")
    selected_grader_base_url = grader_base_url or os.environ.get("OPENAI_BASE_URL")
    if grader_mode in {"model", "omicos"} and not selected_grader_model:
        raise typer.BadParameter(
            "--grader model/omicos requires --grader-model or OPENAI_GRADER_MODEL"
        )
    if not quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
    if autonomics_gateway:
        if solver_command is not None:
            raise typer.BadParameter(
                "--autonomics-gateway cannot be combined with --solver-command"
            )
        if data_mount_path is None:
            raise typer.BadParameter(
                "--autonomics-gateway requires --data-mount-path under the gateway VFS root"
            )

    base_benchmark = HLE
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
            hle_benchmarks = {
                "hle-all": ("hle:full", HLE),
                "hle-biomedical": ("hle:biomedical", HLE),
                "hle-biomedical-visual": ("hle:biomedical_visual", HLE),
            }
            choices = [
                *hle_benchmarks,
                "lab-bench",
                *LAB_BENCH_DATASETS,
                *OMICOS_BIOMNIBENCH_DATASETS,
            ]
            if benchmark in hle_benchmarks:
                dataset_name, base_benchmark = hle_benchmarks[benchmark]
                examples = load_examples(
                    dataset_path(dataset_name),
                    offset=offset,
                    limit=limit,
                    ids=question_id,
                    answer_type=answer_type,
                    category=category,
                    shuffle=shuffle,
                    seed=seed,
                )
            elif benchmark == "lab-bench" or benchmark in LAB_BENCH_DATASETS:
                base_benchmark = LAB_BENCH
                examples = _load_lab_bench_examples(
                    benchmark,
                    offset=offset,
                    limit=limit,
                    ids=question_id,
                    answer_type=answer_type,
                    category=category,
                    shuffle=shuffle,
                    seed=seed,
                )
            elif benchmark in OMICOS_BIOMNIBENCH_DATASETS:
                base_benchmark = OMICOS
                examples = load_examples(
                    dataset_path(OMICOS_BIOMNIBENCH_DATASETS[benchmark]),
                    offset=offset,
                    limit=limit,
                    ids=question_id,
                    answer_type=answer_type,
                    category=category,
                    shuffle=shuffle,
                    seed=seed,
                )
            else:
                raise ValueError(
                    f"unknown benchmark {benchmark!r}; choose one of: {choices}"
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
        replace(base_benchmark, include_image_uri=True)
        if include_image_uri
        else base_benchmark
    )

    engine_grader = None
    if grader_mode == "model":
        try:
            engine_grader = OpenAIGrader(
                base_benchmark,
                model=selected_grader_model,
                base_url=selected_grader_base_url,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif grader_mode == "omicos":
        try:
            engine_grader = OmicOSGrader(
                model=selected_grader_model,
                base_url=selected_grader_base_url,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif grader_mode == "codex-local":
        codex_path = os.environ.get("BENCH_ENGINE_CODEX", "codex")
        try:
            engine_grader = CodexLocalGrader(
                codex_path=codex_path,
                model=selected_grader_model,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    if solver_command is not None:
        try:
            solver = CustomCommandSolver(solver_command, timeout)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        solver_kind = f"command:{shlex.join(solver.argv)}"
    else:
        executable = _default_autonomics(autonomics)
        solver = AutonomicsTuiSolver(
            executable,
            timeout,
            model=model,
            profile=profile,
            use_gateway=autonomics_gateway,
        )
        solver_kind = (
            f"autonomics-gateway:{solver.executable}"
            if autonomics_gateway
            else f"autonomics:{solver.executable}"
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
                data_mount_path=(
                    data_mount_path.expanduser().resolve()
                    if data_mount_path is not None
                    else None
                ),
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
