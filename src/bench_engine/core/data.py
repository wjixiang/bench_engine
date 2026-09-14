"""Dataset loading, generic schemas, and selection."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from bench_engine.core.interfaces import Example, TaskAsset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DATASET_ROOT = Path(
    os.environ.get("BENCH_ENGINE_DATA_DIR", DEFAULT_DATASET_ROOT)
).expanduser()

DATASET_SCHEMA_VERSION = 1
TASK_SCHEMA_VERSION = 1
DATASET_SCHEMA: Mapping[str, type[pl.DataType]] = {
    "id": pl.String,
    "question": pl.String,
    "image": pl.String,
    "answer": pl.String,
    "answer_type": pl.String,
    "category": pl.String,
}
ANSWER_TYPES = frozenset(("exactMatch", "multipleChoice"))
GRADING_METHODS = frozenset(("exact",))


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    path: Path
    rows: int


def dataset_path(name: str, *, data_root: Path | None = None) -> Path:
    """Resolve a registered dataset by adapter and dataset name."""
    if ":" not in name:
        raise ValueError(f"dataset must use 'adapter:dataset' format: {name!r}")
    adapter, dataset = name.split(":", 1)
    root = data_root if data_root is not None else DATASET_ROOT
    path = root / adapter / f"{dataset}.parquet"
    if path.is_file():
        return path
    path = root / adapter / dataset
    if path.is_dir():
        return path
    if path.is_file():
        return path
    else:
        raise FileNotFoundError(f"dataset does not exist: {path}")
    return path


def inspect_dataset(path: Path) -> DatasetInfo:
    if path.is_dir():
        rows = sum(
            1 for task_path in path.iterdir() if (task_path / "task.json").is_file()
        )
    else:
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    return DatasetInfo(path.stem, path, int(rows))


def validate_dataset_schema(frame: pl.DataFrame) -> None:
    """Validate the canonical columns required by the benchmark runner."""
    errors: list[str] = []

    actual_schema = frame.schema
    missing = [column for column in DATASET_SCHEMA if column not in actual_schema]
    if missing:
        errors.append(f"missing required columns: {', '.join(missing)}")
    type_errors = [
        f"{column} must be {dtype} (got {actual_schema[column]})"
        for column, dtype in DATASET_SCHEMA.items()
        if column in actual_schema and actual_schema[column] != dtype
    ]
    if type_errors:
        errors.append("; ".join(type_errors))
    if missing or type_errors:
        raise ValueError(
            f"dataset schema v{DATASET_SCHEMA_VERSION} is invalid: " + "; ".join(errors)
        )

    null_counts = frame.select(
        pl.col(column).null_count().alias(column) for column in DATASET_SCHEMA
    ).row(0, named=True)
    null_errors = [
        f"{column} has {count} null values"
        for column, count in null_counts.items()
        if count
    ]
    if null_errors:
        errors.append("; ".join(null_errors))

    required_text = ("id", "question", "answer", "answer_type", "category")
    blank_counts = frame.select(
        pl.col(column).fill_null("").str.strip_chars().eq("").sum().alias(column)
        for column in required_text
    ).row(0, named=True)
    blank_errors = [
        f"{column} has {count} blank values"
        for column, count in blank_counts.items()
        if count
    ]
    if blank_errors:
        errors.append("; ".join(blank_errors))

    duplicate_ids = int(frame.select(pl.col("id").is_duplicated().sum()).item())
    if duplicate_ids:
        errors.append(f"id has {duplicate_ids} duplicate values")

    invalid_answer_types = int(
        frame.select(~pl.col("answer_type").fill_null("").is_in(list(ANSWER_TYPES)))
        .sum()
        .item()
    )
    if invalid_answer_types:
        valid = ", ".join(sorted(ANSWER_TYPES))
        errors.append(
            f"answer_type has {invalid_answer_types} invalid values; expected one of: {valid}"
        )

    image = pl.col("image").fill_null("")
    invalid_images = int(
        frame.select((image != "").and_(~image.str.starts_with("data:image/")))
        .sum()
        .item()
    )
    if invalid_images:
        errors.append(
            f"image has {invalid_images} invalid values; use an empty string or an image data URI"
        )

    if errors:
        raise ValueError(
            f"dataset schema v{DATASET_SCHEMA_VERSION} is invalid: " + "; ".join(errors)
        )


def load_examples(
    path: Path,
    *,
    offset: int = 0,
    limit: int | None = None,
    ids: list[str] | None = None,
    answer_type: str | None = None,
    category: str | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[Example]:
    """Load benchmark rows while preserving source order by default."""
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    if path.is_dir():
        return _load_task_examples(
            path,
            offset=offset,
            limit=limit,
            ids=ids,
            answer_type=answer_type,
            category=category,
            shuffle=shuffle,
            seed=seed,
        )

    frame = pl.read_parquet(path)
    validate_dataset_schema(frame)

    if ids:
        frame = frame.filter(pl.col("id").is_in(list(ids)))
    if answer_type is not None:
        frame = frame.filter(pl.col("answer_type") == answer_type)
    if category is not None:
        frame = frame.filter(pl.col("category") == category)
    if shuffle:
        frame = frame.sample(n=len(frame), seed=seed, shuffle=True)

    selected = frame.slice(offset, limit)
    return [
        Example(
            id=row["id"],
            question=row["question"],
            image=row["image"],
            target=row["answer"],
            answer_type=row["answer_type"],
            category=row["category"],
        )
        for row in selected.iter_rows(named=True)
    ]


def _load_task_examples(
    path: Path,
    *,
    offset: int,
    limit: int | None,
    ids: list[str] | None,
    answer_type: str | None,
    category: str | None,
    shuffle: bool,
    seed: int,
) -> list[Example]:
    examples = [
        _load_task_package(task_path)
        for task_path in sorted(path.iterdir())
        if (task_path / "task.json").is_file()
    ]
    ids_set = set(ids) if ids else None
    examples = [
        example
        for example in examples
        if (ids_set is None or example.id in ids_set)
        and (answer_type is None or example.answer_type == answer_type)
        and (category is None or example.category == category)
    ]
    if len({example.id for example in examples}) != len(examples):
        raise ValueError(f"task dataset has duplicate IDs: {path}")
    if shuffle:
        random.Random(seed).shuffle(examples)
    end = None if limit is None else offset + limit
    return examples[offset:end]


def _load_task_package(task_path: Path) -> Example:
    metadata_path = task_path / "task.json"
    try:
        task = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid task package metadata {metadata_path}: {exc}"
        ) from exc

    instruction_path = task_path / "instruction.md"
    if instruction_path.is_file():
        raise ValueError(
            f"task package uses a separate instruction file; embed instruction in task.json: {instruction_path}"
        )

    required = ("id", "instruction", "category", "grading")
    missing = [field for field in required if field not in task]
    if missing:
        raise ValueError(
            f"task schema v{TASK_SCHEMA_VERSION} is missing required fields: {', '.join(missing)}"
        )

    task_id = task.get("id")
    instruction = task.get("instruction")
    category = task.get("category")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"task id must be a non-blank string: {metadata_path}")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(
            f"task instruction must be a non-blank string: {metadata_path}"
        )
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"task category must be a non-blank string: {metadata_path}")

    grading = task.get("grading")
    if not isinstance(grading, dict):
        raise ValueError(f"task grading must be an object: {metadata_path}")  # noqa: TRY004
    grading_required = ("method", "answer_type", "answer")
    grading_missing = [field for field in grading_required if field not in grading]
    if grading_missing:
        raise ValueError(
            f"task grading is missing required fields: {', '.join(grading_missing)}"
        )
    method = grading.get("method")
    answer_type = grading.get("answer_type")
    answer = grading.get("answer")
    if method not in GRADING_METHODS:
        valid = ", ".join(sorted(GRADING_METHODS))
        raise ValueError(f"task grading method must be one of: {valid}; got {method!r}")
    if answer_type not in ANSWER_TYPES:
        valid = ", ".join(sorted(ANSWER_TYPES))
        raise ValueError(
            f"task answer_type must be one of: {valid}; got {answer_type!r}"
        )
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            f"task grading answer must be a non-blank string: {metadata_path}"
        )
    assets: list[TaskAsset] = []
    asset_names: set[str] = set()
    data = task.get("data", [])
    if not isinstance(data, list):
        raise ValueError(f"task data must be an array: {metadata_path}")  # noqa: TRY004
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(  # noqa: TRY004
                f"task data[{index}] must be an object: {metadata_path}"
            )
        name = entry.get("name")
        relative_path = entry.get("path")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"task data[{index}].name must be a non-blank string")
        if name in asset_names:
            raise ValueError(f"task data has duplicate name {name!r}: {metadata_path}")
        asset_names.add(name)
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"task data[{index}].path must be a non-blank string")
        asset_path = (task_path / relative_path).resolve()
        try:
            asset_path.relative_to(task_path.resolve() / "data")
        except ValueError as exc:
            raise ValueError(
                f"task data path must stay under data/: {relative_path!r}"
            ) from exc
        if not asset_path.is_file():
            raise ValueError(f"task data file does not exist: {asset_path}")
        media_type = entry.get("media_type", "application/octet-stream")
        role = entry.get("role", "input")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError(
                f"task data[{index}].media_type must be a non-blank string"
            )
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"task data[{index}].role must be a non-blank string")
        assets.append(TaskAsset(name, asset_path, media_type, role))

    return Example(
        id=task_id,
        question=instruction,
        image="",
        target=answer,
        answer_type=answer_type,
        category=category,
        task_path=task_path.resolve(),
        assets=tuple(assets),
    )
