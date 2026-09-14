"""Dataset loading, generic schemas, and selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from bench_engine.core.interfaces import Example

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DATASET_ROOT = Path(
    os.environ.get("BENCH_ENGINE_DATA_DIR", DEFAULT_DATASET_ROOT)
).expanduser()


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
    if not path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {path}")
    return path


def inspect_dataset(path: Path) -> DatasetInfo:
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    return DatasetInfo(path.stem, path, int(rows))


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

    frame = pl.read_parquet(path)
    missing = [
        column
        for column in ("id", "question", "image", "answer", "answer_type", "category")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"dataset is missing required columns: {', '.join(missing)}")

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
            id=str(row["id"]),
            question=str(row["question"]),
            image=str(row["image"] or ""),
            target=str(row["answer"]),
            answer_type=str(row["answer_type"]),
            category=str(row["category"]),
        )
        for row in selected.iter_rows(named=True)
    ]
