#!/usr/bin/env python3
"""Normalize OmicOS ``.txt.gz`` inputs as plain TSV/CSV files."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Conversion:
    source: Path
    destination: Path
    table_format: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompress OmicOS .txt.gz files and replace task instruction "
            "references with their .tsv or .csv equivalents."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("datasets/omicos_biomnibench/biomnibench_da"),
        help="OmicOS task-package collection",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the conversion; without this flag, only print a dry run.",
    )
    return parser.parse_args()


def _detect_format(source: Path) -> str:
    with gzip.open(source, "rt", encoding="utf-8", errors="replace") as handle:
        line = handle.readline()
    if not line:
        raise ValueError(f"compressed table is empty: {source}")
    return "tsv" if line.count("\t") >= line.count(",") else "csv"


def _atomic_json_write(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_decompress(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, gzip.open(
            input_handle, "rb"
        ) as decompressed, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(decompressed, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _destination(source: Path, data_root: Path) -> tuple[Path, str] | None:
    if not source.name.endswith(".txt.gz"):
        return None
    table_format = _detect_format(source)
    relative = source.relative_to(data_root)
    stem = source.name[: -len(".txt.gz")]
    destination = data_root / relative.with_name(f"{stem}.{table_format}")
    return destination, table_format


def _replace_value(value: object, replacements: dict[str, str]) -> object:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_value(item, replacements)
            for key, item in value.items()
        }
    return value


def _task_paths(dataset: Path) -> list[Path]:
    if (dataset / "task.json").is_file():
        return [dataset]
    return [
        path
        for path in sorted(dataset.iterdir())
        if (path / "task.json").is_file()
    ]


def normalize(dataset: Path, *, apply: bool) -> int:
    converted = 0
    for task_path in _task_paths(dataset):
        metadata_path = task_path / "task.json"
        task = json.loads(metadata_path.read_text(encoding="utf-8"))
        data_entry = next(
            (entry for entry in task.get("data", []) if entry.get("path")),
            None,
        )
        if data_entry is None:
            continue
        data_root = (task_path / str(data_entry["path"])).resolve()
        if not data_root.is_dir():
            continue

        conversions: list[Conversion] = []
        for source in sorted(data_root.rglob("*.txt.gz")):
            result = _destination(source, data_root)
            if result is None:
                continue
            destination, table_format = result
            conversions.append(Conversion(source, destination, table_format))

        replacements: dict[str, str] = {}
        for conversion in conversions:
            replacements[conversion.source.name] = conversion.destination.name
            if apply:
                if conversion.source.exists():
                    _atomic_decompress(conversion.source, conversion.destination)
                elif not conversion.destination.exists():
                    raise FileNotFoundError(
                        "source and destination are both absent: "
                        f"{conversion.source}"
                    )
            converted += 1

        if not replacements:
            continue

        normalized = _replace_value(task, replacements)
        if normalized != task:
            if apply:
                _atomic_json_write(metadata_path, normalized)
            else:
                print(f"would update {metadata_path}")

        if apply:
            for conversion in conversions:
                if conversion.source.exists():
                    conversion.source.unlink()

        print(
            f"{'updated' if apply else 'would update'} "
            f"{task_path.name}: {len(conversions)} file(s)"
        )
    print(f"total: {converted} file(s)")
    return 0


def main() -> int:
    args = parse_args()
    return normalize(args.dataset.expanduser(), apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
