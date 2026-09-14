"""Humanity's Last Exam benchmark adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bench_engine.core.data import (
    DatasetInfo,
    dataset_path,
    inspect_dataset,
    load_examples,
)
from bench_engine.core.grading import Grade, exact_grade, extract_answer
from bench_engine.core.interfaces import Example

MULTIPLE_CHOICE = "multipleChoice"


def _has_image(example: Example) -> bool:
    return bool(example.image or any(asset.role == "image" for asset in example.assets))


@dataclass(frozen=True)
class HLEBenchmark:
    """HLE prompt and exact-scoring behavior."""

    name: str = "hle"
    include_image_uri: bool = False

    def prompt(self, example: Example) -> str:
        instruction = (
            "Answer with the option letter only."
            if example.answer_type == MULTIPLE_CHOICE
            else "Answer with the exact, concise value only."
        )
        prompt = (
            "Answer this closed-book benchmark question without searching the web "
            "or using external tools.\n\n"
            f"Question:\n{example.question}\n\n"
        )
        if example.image and self.include_image_uri:
            prompt += f"Image data URI:\n{example.image}\n\n"
        elif _has_image(example) and self.include_image_uri:
            image_paths = "\n".join(
                f"- {asset.path}" for asset in example.assets if asset.role == "image"
            )
            prompt += f"Image files:\n{image_paths}\n\n"
        return (
            prompt
            + f"{instruction}\n"
            + "The final line must use exactly this format:\n"
            + "Answer: <answer>"
        )

    def extract(self, response: str, example: Example) -> str | None:
        return extract_answer(
            response, multiple_choice=example.answer_type == MULTIPLE_CHOICE
        )

    def grade(self, response: str, example: Example) -> Grade:
        return exact_grade(
            response,
            example.target,
            answer_type=example.answer_type,
        )

    def grader_prompt(self, example: Example, response: str) -> str:
        return (
            "You are grading a Humanity's Last Exam response. Decide whether "
            "the candidate answers the question with the same factual value as "
            "the reference. Ignore formatting, explanation length, and confidence. "
            "Minor notation or wording differences are acceptable only when they "
            "do not change the answer. If the candidate is empty, ambiguous, or "
            "refuses to answer, grade it as incorrect.\n\n"
            f"Question:\n{example.question}\n\n"
            f"Reference answer:\n{example.target}\n\n"
            f"Candidate response:\n{response}\n\n"
            "The final line must use exactly one of these formats:\n"
            "Verdict: CORRECT\n"
            "Verdict: INCORRECT"
        )


HLE = HLEBenchmark()


def resolve_path(name: str, override: Path | None = None) -> Path:
    if override is not None:
        path = override.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"dataset does not exist: {path}")
        return path
    if name == "all":
        name = "full"
    return dataset_path(f"hle:{name}")


def hle_datasets() -> list[DatasetInfo]:
    return [
        DatasetInfo(name, path, int(inspect_dataset(path).rows))
        for name, path in (
            ("hle-all", resolve_path("all")),
            ("hle-biomedical", resolve_path("biomedical")),
            ("hle-biomedical-visual", resolve_path("biomedical_visual")),
        )
    ]


def load_hle_examples(name: str, **kwargs: object) -> list[Example]:
    return load_examples(resolve_path(name), **kwargs)  # type: ignore[arg-type]
