"""Output contract for the Reviewer Agent, plus a dependency-free validator."""

from __future__ import annotations

from typing import List, TypedDict

VALID_SCORE_KEYS = {"clarity", "flow", "tone", "structure", "limitations_honesty"}


class Hallucination(TypedDict):
    location: str  # e.g. "Results > H2"
    claim: str  # the specific claim/sentence in the paper
    issue: str  # concretely what's wrong / not grounded in upstream data


class CitationIssue(TypedDict):
    location: str
    issue: str


class ResultsAccuracyIssue(TypedDict):
    location: str
    claimed: str
    actual: str


class HypothesisCoverageIssue(TypedDict):
    hypothesis_id: str
    location: str
    issue: str


class QualityScores(TypedDict):
    clarity: int
    flow: int
    tone: int
    structure: int
    limitations_honesty: int


class ReviewOutput(TypedDict):
    iteration: int
    hallucinations: List[Hallucination]
    citation_issues: List[CitationIssue]
    results_accuracy_issues: List[ResultsAccuracyIssue]
    hypothesis_coverage_issues: List[HypothesisCoverageIssue]
    quality_scores: QualityScores
    overall_pass: bool
    feedback_for_writer: str
    generated_at: str
    model: str


class SchemaValidationError(ValueError):
    """Raised when an assembled result doesn't match ReviewOutput."""


def _check_fields(obj: dict, fields: list[tuple[str, type]], path: str, errors: list[str]) -> None:
    for key, expected_type in fields:
        if key not in obj:
            errors.append(f"{path}.{key} is missing")
        elif not isinstance(obj[key], expected_type):
            errors.append(f"{path}.{key} should be {expected_type.__name__}, got {type(obj[key]).__name__}")


def validate_output(data: dict) -> None:
    """Raises SchemaValidationError (with every problem found, not just the
    first) if `data` doesn't match ReviewOutput."""
    errors: List[str] = []

    if not isinstance(data, dict):
        raise SchemaValidationError(f"top-level output should be an object, got {type(data).__name__}")

    _check_fields(
        data,
        [
            ("iteration", int),
            ("hallucinations", list),
            ("citation_issues", list),
            ("results_accuracy_issues", list),
            ("hypothesis_coverage_issues", list),
            ("quality_scores", dict),
            ("overall_pass", bool),
            ("feedback_for_writer", str),
            ("generated_at", str),
            ("model", str),
        ],
        "output",
        errors,
    )

    for i, item in enumerate(data.get("hallucinations", []) or []):
        if not isinstance(item, dict):
            errors.append(f"output.hallucinations[{i}] should be an object")
            continue
        _check_fields(item, [("location", str), ("claim", str), ("issue", str)], f"output.hallucinations[{i}]", errors)

    for i, item in enumerate(data.get("citation_issues", []) or []):
        if not isinstance(item, dict):
            errors.append(f"output.citation_issues[{i}] should be an object")
            continue
        _check_fields(item, [("location", str), ("issue", str)], f"output.citation_issues[{i}]", errors)

    for i, item in enumerate(data.get("results_accuracy_issues", []) or []):
        if not isinstance(item, dict):
            errors.append(f"output.results_accuracy_issues[{i}] should be an object")
            continue
        _check_fields(
            item, [("location", str), ("claimed", str), ("actual", str)], f"output.results_accuracy_issues[{i}]", errors
        )

    for i, item in enumerate(data.get("hypothesis_coverage_issues", []) or []):
        if not isinstance(item, dict):
            errors.append(f"output.hypothesis_coverage_issues[{i}] should be an object")
            continue
        _check_fields(
            item, [("hypothesis_id", str), ("location", str), ("issue", str)], f"output.hypothesis_coverage_issues[{i}]", errors
        )

    scores = data.get("quality_scores")
    if not isinstance(scores, dict):
        errors.append("output.quality_scores is missing or not an object")
    else:
        missing_keys = VALID_SCORE_KEYS - scores.keys()
        if missing_keys:
            errors.append(f"output.quality_scores is missing key(s): {sorted(missing_keys)}")
        for key, value in scores.items():
            if key not in VALID_SCORE_KEYS:
                errors.append(f"output.quality_scores has unrecognized key {key!r}")
            elif not isinstance(value, int) or isinstance(value, bool) or not (1 <= value <= 5):
                errors.append(f"output.quality_scores.{key} should be an integer 1-5, got {value!r}")

    if errors:
        raise SchemaValidationError("; ".join(errors))
