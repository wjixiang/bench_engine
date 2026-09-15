"""Answer extraction and deterministic grading."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

ANSWER_LINE = re.compile(
    r"(?im)^\s*(?:\*\*)?\s*answer\s*(?:(?:\*\*)?\s*:|:\s*(?:\*\*)?)\s*" r"(.*?)\s*$"
)
MC_LETTER = re.compile(r"^\(?\**([A-Z])\**\)?(?:[.)])?\s*$")


@dataclass(frozen=True)
class Grade:
    correct: bool | None
    extracted: str | None
    method: str
    detail: str = ""


def _clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    value = value.removeprefix("```text").removeprefix("```")
    value = value.removesuffix("```").strip()
    if len(value) >= 4 and value.startswith("**") and value.endswith("**"):
        value = value[2:-2].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"', "`"}:
        value = value[1:-1].strip()
    return value


def extract_answer(response: str, *, multiple_choice: bool = False) -> str | None:
    """Extract the value from a required final ``Answer:`` line."""
    response = _clean_text(response)
    if not response:
        return None

    marker_values = [value.strip() for value in ANSWER_LINE.findall(response)]
    candidate = marker_values[-1] if marker_values else response.splitlines()[-1]
    candidate = _clean_text(candidate)
    if not candidate:
        return None

    if multiple_choice:
        match = MC_LETTER.match(candidate)
        if match:
            return match.group(1)
        match = re.search(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])", candidate)
        if match:
            return match.group(1)
    return candidate


def _normalize(value: str) -> str:
    value = _clean_text(value)
    value = value.removeprefix("\\(").removesuffix("\\)").strip()
    value = value.removeprefix("\\[").removesuffix("\\]").strip()
    if len(value) >= 2 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    value = value.rstrip(" .;。")
    return re.sub(r"\s+", " ", value)


def _numeric_equal(left: str, right: str) -> bool:
    try:
        return Decimal(_normalize(left).replace(",", "")) == Decimal(
            _normalize(right).replace(",", "")
        )
    except InvalidOperation, ValueError:
        return False


def exact_grade(
    response: str,
    target: str,
    *,
    answer_type: str,
) -> Grade:
    """Grade a multiple-choice or exact-match response."""
    multiple_choice = answer_type == "multipleChoice"
    extracted = extract_answer(response, multiple_choice=multiple_choice)
    if extracted is None:
        return Grade(False, None, "exact", "no answer could be extracted")

    if multiple_choice:
        normalized_target = _normalize(target).strip()
        target_match = MC_LETTER.match(normalized_target)
        expected = target_match.group(1) if target_match else normalized_target[:1]
        return Grade(extracted.upper() == expected.upper(), extracted, "exact")

    normalized_extracted = _normalize(extracted)
    normalized_target = _normalize(target)
    correct = (
        normalized_extracted.casefold() == normalized_target.casefold()
        or _numeric_equal(normalized_extracted, normalized_target)
    )
    return Grade(correct, extracted, "exact")


def grade_verdict(response: str) -> bool | None:
    """Parse a model grader's required final ``Verdict:`` line."""
    match = re.search(
        r"(?im)^\s*(?:\*\*)?\s*verdict\s*(?:(?:\*\*)?\s*:|:\s*(?:\*\*)?)\s*"
        r"(correct|incorrect)(?:\*\*)?",
        response,
    )
    if match is None:
        return None
    return match.group(1).casefold() == "correct"
