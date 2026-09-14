"""LAB-Bench prompt and scoring adapter."""

from __future__ import annotations

from dataclasses import dataclass

from bench_engine.core.grading import Grade, exact_grade, extract_answer
from bench_engine.core.interfaces import Example


def _has_image(example: Example) -> bool:
    return bool(example.image or any(asset.role == "image" for asset in example.assets))


LAB_BENCH_DATASETS = {
    "lab-cloning-scenarios": "lab_bench:cloning_scenarios",
    "lab-dbqa": "lab_bench:dbqa",
    "lab-figqa": "lab_bench:figqa",
    "lab-litqa2": "lab_bench:litqa2",
    "lab-protocolqa": "lab_bench:protocolqa",
    "lab-seqqa": "lab_bench:seqqa",
    "lab-suppqa": "lab_bench:suppqa",
    "lab-tableqa": "lab_bench:tableqa",
}


@dataclass(frozen=True)
class LabBenchBenchmark:
    """LAB-Bench multiple-choice behavior."""

    name: str = "lab_bench"
    include_image_uri: bool = False

    def prompt(self, example: Example) -> str:
        has_image = _has_image(example)
        prompt = (
            "Answer this biology benchmark question using the supplied question, "
            + ("image, " if has_image else "")
            + "context, and options.\n\n"
            + f"Question:\n{example.question}\n\n"
        )
        if example.image and self.include_image_uri:
            prompt += f"Image data URI:\n{example.image}\n\n"
        elif has_image and self.include_image_uri:
            image_paths = "\n".join(
                f"- {asset.path}" for asset in example.assets if asset.role == "image"
            )
            prompt += f"Image files:\n{image_paths}\n\n"
        return (
            prompt
            + "Answer with the option letter only.\n"
            + "The final line must use exactly this format:\n"
            + "Answer: <letter>"
        )

    def extract(self, response: str, example: Example) -> str | None:
        return extract_answer(response, multiple_choice=True)

    def grade(self, response: str, example: Example) -> Grade:
        return exact_grade(
            response,
            example.target,
            answer_type=example.answer_type,
        )

    def grader_prompt(self, example: Example, response: str) -> str:
        return (
            "You are grading a LAB-Bench multiple-choice response. Decide whether "
            "the candidate selects the same option letter as the reference. Ignore "
            "formatting and explanation length. If the candidate is empty, ambiguous, "
            "or refuses to answer, grade it as incorrect.\n\n"
            f"Question:\n{example.question}\n\n"
            f"Reference option letter:\n{example.target}\n\n"
            f"Candidate response:\n{response}\n\n"
            "The final line must use exactly one of these formats:\n"
            "Verdict: CORRECT\n"
            "Verdict: INCORRECT"
        )


LAB_BENCH = LabBenchBenchmark()
