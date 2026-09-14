#!/usr/bin/env python3
"""Convert the HLE parquet table to task packages."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl

TASK_SCHEMA_VERSION = 1
CANONICAL_COLUMNS = {
    "id",
    "question",
    "image",
    "answer",
    "answer_type",
    "category",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path("datasets/hle/full.parquet"),
        help="Source HLE full parquet file",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("datasets/hle"),
        help="Output HLE dataset root",
    )
    parser.add_argument(
        "--biomedical-source",
        type=Path,
        default=Path("datasets/hle/biomedical.parquet"),
        help="Source parquet defining the biomedical subset",
    )
    parser.add_argument(
        "--biomedical-visual-source",
        type=Path,
        default=Path("datasets/hle/biomedical_visual.parquet"),
        help="Source parquet defining the biomedical visual subset",
    )
    return parser.parse_args()


def decode_image(value: str) -> tuple[bytes, str, str] | None:
    if not value:
        return None
    prefix, separator, encoded = value.partition(",")
    if not separator or not prefix.startswith("data:") or ";base64" not in prefix:
        raise ValueError(f"unsupported image value: {prefix!r}")
    media_type = prefix.removeprefix("data:").removesuffix(";base64")
    if not media_type.startswith("image/"):
        raise ValueError(f"unsupported image media type: {media_type!r}")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 image data") from exc
    extension = media_type.removeprefix("image/").replace("jpeg", "jpg")
    return image_bytes, media_type, extension


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    excluded = CANONICAL_COLUMNS | {"image_preview", "rationale_image"}
    return {key: value for key, value in row.items() if key not in excluded}


def write_task(row: dict[str, Any], collection_path: Path) -> None:
    task_id = str(row["id"])
    task_path = collection_path / task_id
    if task_path.exists():
        shutil.rmtree(task_path)
    task_path.mkdir(parents=True)

    data: list[dict[str, str]] = []
    decoded = decode_image(str(row["image"]))
    instruction = str(row["question"])
    if decoded is not None:
        image_bytes, media_type, extension = decoded
        (task_path / "data").mkdir()
        image_name = f"image.{extension}"
        (task_path / "data" / image_name).write_bytes(image_bytes)
        data.append(
            {
                "name": "image",
                "path": f"data/{image_name}",
                "media_type": media_type,
                "role": "image",
            }
        )
        instruction += f"\n\nInput data:\n- image: data/{image_name} ({media_type})"

    task = {
        "schema_version": TASK_SCHEMA_VERSION,
        "id": task_id,
        "instruction": instruction,
        "category": str(row["category"]),
        "grading": {
            "method": "exact",
            "answer_type": str(row["answer_type"]),
            "answer": str(row["answer"]),
        },
        "metadata": metadata(row),
        "data": data,
    }
    (task_path / "task.json").write_text(
        json.dumps(task, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def collection_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"subset parquet does not exist: {path}")
    return set(pl.read_parquet(path)["id"].cast(pl.String).to_list())


def write_subset(output: Path, name: str, ids: set[str]) -> None:
    collection_path = output / name
    if collection_path.exists():
        shutil.rmtree(collection_path)
    collection_path.mkdir(parents=True)
    for task_id in sorted(ids):
        (collection_path / task_id).symlink_to(f"../full/{task_id}")


def main() -> None:
    args = parse_args()
    frame = pl.read_parquet(args.source)
    for column in ("id", "question", "image", "answer", "answer_type", "category"):
        if column not in frame.columns:
            raise ValueError(f"source parquet is missing column {column!r}")

    full_path = args.output / "full"
    if full_path.exists():
        shutil.rmtree(full_path)
    full_path.mkdir(parents=True)
    for row in frame.iter_rows(named=True):
        write_task(row, full_path)

    subsets = {
        "biomedical": collection_ids(args.biomedical_source),
        "biomedical_visual": collection_ids(args.biomedical_visual_source),
    }
    for name, ids in subsets.items():
        write_subset(output=args.output, name=name, ids=ids)

    for legacy in (
        args.output / "full.parquet",
        args.output / "biomedical.parquet",
        args.output / "biomedical_visual.parquet",
    ):
        if legacy.is_file():
            legacy.unlink()
    print(f"wrote {frame.height} task packages to {full_path}")


if __name__ == "__main__":
    main()
