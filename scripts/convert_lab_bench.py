#!/usr/bin/env python3
"""Convert LAB-Bench public JSONL files to the Bench Engine dataset schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

ALPHABET = string.ascii_uppercase
REFUSE_CHOICE = "Insufficient information to answer the question"
CANONICAL_COLUMNS = ("id", "question", "image", "answer", "answer_type", "category")
TASK_SCHEMA_VERSION = 1
PUBLIC_CATEGORIES = (
    "CloningScenarios",
    "DbQA",
    "FigQA",
    "LitQA2",
    "ProtocolQA",
    "SeqQA",
    "SuppQA",
    "TableQA",
)
CATEGORY_FILES = {
    "CloningScenarios": "cloning_scenarios",
    "DbQA": "dbqa",
    "FigQA": "figqa",
    "LitQA2": "litqa2",
    "ProtocolQA": "protocolqa",
    "SeqQA": "seqqa",
    "SuppQA": "suppqa",
    "TableQA": "tableqa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path("/mnt/base/benchmark/lab_bench/LAB-Bench"),
        help="LAB-Bench repository root",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("datasets/lab_bench"),
        help="Output dataset directory",
    )
    return parser.parse_args()


def stable_permutation(identifier: str, size: int) -> list[int]:
    """Return a reproducible permutation for one LAB-Bench item."""
    seed = int.from_bytes(
        hashlib.sha256(identifier.encode("utf-8")).digest()[:16], "big"
    )
    permutation = list(range(size))
    random.Random(seed).shuffle(permutation)
    return permutation


def combine_images(images: Sequence[Image.Image]) -> Image.Image:
    if len(images) == 1:
        return images[0]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    combined = Image.new("RGB", (width, height), "white")
    top = 0
    for image in images:
        if image.mode == "RGBA":
            white = Image.new("RGBA", image.size, "white")
            white.alpha_composite(image)
            image = white.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        combined.paste(image, ((width - image.width) // 2, top))
        top += image.height
    return combined


def image_paths(source: Path, row: dict[str, Any]) -> list[Path]:
    if "figure-path" in row:
        return [source / row["figure-path"]]
    elif "table-path" in row:
        return [source / path for path in row["table-path"]]
    return []


def write_data_files(
    source: Path,
    row: dict[str, Any],
    task_path: Path,
) -> list[dict[str, str]]:
    paths = image_paths(source, row)
    if not paths:
        return []

    data_path = task_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        source_path = paths[0]
        destination = data_path / f"image{source_path.suffix.lower()}"
        shutil.copyfile(source_path, destination)
        with Image.open(source_path) as image:
            image_format = (image.format or "PNG").lower()
        media_type = "image/jpeg" if image_format == "jpg" else f"image/{image_format}"
    else:
        destination = data_path / "tables.png"
        images = [Image.open(path) for path in paths]
        try:
            combine_images(images).save(destination, format="PNG")
        finally:
            for image in images:
                image.close()
        media_type = "image/png"

    return [
        {
            "name": "image",
            "path": f"data/{destination.name}",
            "media_type": media_type,
            "role": "image",
        }
    ]


def question_text(row: dict[str, Any]) -> str:
    sections: list[str] = []
    if row.get("protocol"):
        sections.append(f"Protocol:\n{row['protocol']}")
    if row.get("paper-title"):
        sections.append(f"Paper title: {row['paper-title']}")
    if row.get("source"):
        sections.append(f"DOI: {row['source']}")
    sections.append(f"Question:\n{row['question']}")
    return "\n\n".join(sections)


def write_task_package(
    source: Path,
    output: Path,
    category: str,
    subtask: str,
    row: dict[str, Any],
) -> None:
    choices = [row["ideal"], REFUSE_CHOICE, *row["distractors"]]
    permutation = stable_permutation(str(row["id"]), len(choices))
    rendered_choices = [
        f"({letter}) {choices[index]}" for letter, index in zip(ALPHABET, permutation)
    ]
    answer = ALPHABET[permutation.index(0)]
    instruction = "\n\n".join(
        (question_text(row), "Options:\n" + "\n".join(rendered_choices))
    )
    task_id = str(row["id"])
    task_path = output / CATEGORY_FILES[category] / task_id
    if task_path.exists():
        shutil.rmtree(task_path)
    task_path.mkdir(parents=True)
    data = write_data_files(source / category, row, task_path)
    if data:
        instruction += "\n\nInput data:\n" + "\n".join(
            f"- {entry['name']}: {entry['path']} ({entry['media_type']})"
            for entry in data
        )

    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"question", "ideal", "distractors", "figure-path", "table-path"}
    }
    task = {
        "schema_version": TASK_SCHEMA_VERSION,
        "id": task_id,
        "instruction": instruction,
        "category": category,
        "subtask": subtask,
        "grading": {
            "method": "exact",
            "answer_type": "multipleChoice",
            "answer": answer,
        },
        "metadata": metadata,
        "data": data,
    }
    (task_path / "task.json").write_text(
        json.dumps(task, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"LAB-Bench source does not exist: {source}")

    total_rows = 0
    for category in PUBLIC_CATEGORIES:
        for path in sorted((source / category).glob("*-public.jsonl")):
            subtask = path.name.removesuffix("-public.jsonl")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    write_task_package(
                        source, args.output, category, subtask, json.loads(line)
                    )
                    total_rows += 1
        legacy_parquet = args.output / f"{CATEGORY_FILES[category]}.parquet"
        if legacy_parquet.is_file():
            legacy_parquet.unlink()

    print(f"wrote {total_rows} task packages to {args.output}")


if __name__ == "__main__":
    main()
