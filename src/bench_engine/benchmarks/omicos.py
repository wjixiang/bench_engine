"""OmicOS-BiomniBench benchmark adapter.

OmicOS-BiomniBench tasks are open-ended data-analysis questions. Each task
ships with a rubric that defines graded criteria; the answer text and the
analysis trace are produced by the solver into ``/app/answer.txt`` and
``/app/trace.md`` and are surfaced to the rubric grader as solver artifacts.

The adapter formats the task instruction for the solver, falls back to a
trivial exact-style check when no model grader is configured, and exposes
the rubric and trace prompt expected by the OmicOS rubric grader.
"""

from __future__ import annotations

from dataclasses import dataclass

from bench_engine.core.grading import Grade
from bench_engine.core.interfaces import Example


RUBRIC_ANSWER_TYPE = "rubric"


def _has_image(example: Example) -> bool:
    return bool(example.image or any(asset.role == "image" for asset in example.assets))


@dataclass(frozen=True)
class OmicOSBenchmark:
    """Prompt and scoring behavior for OmicOS-BiomniBench tasks."""

    name: str = "omicos_biomnibench"
    include_image_uri: bool = False

    def prompt(self, example: Example) -> str:
        has_image = _has_image(example)
        parts = [
            (
                "Solve this OmicOS-BiomniBench data-analysis task end-to-end."
                " Use the provided workspace under /app, write your final answer"
                " to /app/answer.txt, and document every step of your analysis"
                " in /app/trace.md."
            ),
            "",
            f"Task instruction:\n{example.question}",
        ]
        if example.image and self.include_image_uri:
            parts.append(f"Image data URI:\n{example.image}")
        elif has_image and self.include_image_uri:
            image_paths = "\n".join(
                f"- {asset.path}" for asset in example.assets if asset.role == "image"
            )
            parts.append(f"Image files:\n{image_paths}")
        parts.append("")
        parts.append(
            "Final answer file (/app/answer.txt) must contain the answer in plain"
            " text. Trace file (/app/trace.md) must record your analysis steps,"
            " decisions, intermediate results, and references."
        )
        parts.append(
            "Before reading a very large input file, check whether smaller"
            " metadata files already provide the required rows, columns, labels,"
            " or sample mappings. Do not scan or load a large raw matrix merely"
            " to inspect its schema; use bounded metadata inspection instead."
        )
        question = example.question.casefold()
        if "distribution of cell" in question or "cell subsets" in question:
            parts.append(
                "For this cell-distribution question, use the annotated"
                " cell-level metadata table directly when it supplies tissue,"
                " treatment, and cell-type labels. Do not open, stage, sample,"
                " or process the raw count matrix; counts are not needed for"
                " this composition analysis."
            )
        return "\n".join(parts)

    def extract(self, response: str, example: Example) -> str | None:
        """Return the candidate answer verbatim; rubric tasks do not extract."""
        del example
        cleaned = response.strip()
        return cleaned or None

    def grade(self, response: str, example: Example) -> Grade:
        """Deterministic fallback used when no model grader is configured.

        Without the rubric grader there is no reliable way to score open-ended
        analysis tasks; this fallback simply records that the answer was
        produced without invoking the LLM judge.
        """
        del example
        cleaned = response.strip()
        if not cleaned:
            return Grade(False, None, "exact", "no answer text was produced")
        return Grade(
            None,
            cleaned,
            "exact",
            "rubric tasks require --grader omicos or --grader model",
        )

    def grader_prompt(self, example: Example, response: str) -> str:
        """Compose a generic rubric-grading prompt for downstream graders.

        Concrete graders (e.g. :class:`OmicOSGrader`) prefer to build their own
        prompt that includes ``trace.md`` content; this default helper is kept
        for benchmarks that pair with :class:`OpenAIGrader`.
        """
        return (
            "You are grading an OmicOS-BiomniBench response. The candidate was"
            " given an open-ended data-analysis task and must satisfy the rubric"
            " criteria.\n\n"
            f"Task instruction:\n{example.question}\n\n"
            f"Rubric:\n{example.rubric}\n\n"
            f"Candidate answer:\n{response}\n\n"
            "The final line must use exactly one of these formats:\n"
            "Verdict: CORRECT\n"
            "Verdict: INCORRECT"
        )


OMICOS = OmicOSBenchmark()
