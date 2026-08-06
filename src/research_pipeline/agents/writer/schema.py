"""Output contract for the Writer Agent, plus a dependency-free validator.

Unlike the upstream agents, the Writer Agent's primary deliverable is the PDF
itself — this schema covers the JSON summary written alongside it (see
writer_agent.py's module docstring).
"""

from __future__ import annotations

from typing import List, TypedDict


class WriterAgentOutput(TypedDict):
    paper_path: str
    sections_generated: List[str]
    hypotheses_supported: List[str]
    hypotheses_refuted: List[str]
    hypotheses_inconclusive: List[str]
    citations_used: List[str]  # paper ids, from the Literature Agent's output
    notes_for_review: List[str]
    generated_at: str
    model: str


class SchemaValidationError(ValueError):
    """Raised when an assembled result doesn't match WriterAgentOutput."""


def _check_fields(obj: dict, fields: list[tuple[str, type]], path: str, errors: list[str]) -> None:
    for key, expected_type in fields:
        if key not in obj:
            errors.append(f"{path}.{key} is missing")
        elif not isinstance(obj[key], expected_type):
            errors.append(f"{path}.{key} should be {expected_type.__name__}, got {type(obj[key]).__name__}")


def validate_output(data: dict) -> None:
    """Raises SchemaValidationError (with every problem found, not just the
    first) if `data` doesn't match WriterAgentOutput."""
    errors: List[str] = []

    if not isinstance(data, dict):
        raise SchemaValidationError(f"top-level output should be an object, got {type(data).__name__}")

    _check_fields(
        data,
        [
            ("paper_path", str),
            ("sections_generated", list),
            ("hypotheses_supported", list),
            ("hypotheses_refuted", list),
            ("hypotheses_inconclusive", list),
            ("citations_used", list),
            ("notes_for_review", list),
            ("generated_at", str),
            ("model", str),
        ],
        "output",
        errors,
    )

    for key in ("sections_generated", "hypotheses_supported", "hypotheses_refuted", "hypotheses_inconclusive", "citations_used", "notes_for_review"):
        value = data.get(key)
        if isinstance(value, list) and not all(isinstance(item, str) for item in value):
            errors.append(f"output.{key} should be a list of strings")

    supported = set(data.get("hypotheses_supported", []) or [])
    refuted = set(data.get("hypotheses_refuted", []) or [])
    inconclusive = set(data.get("hypotheses_inconclusive", []) or [])
    overlap = (supported & refuted) | (supported & inconclusive) | (refuted & inconclusive)
    if overlap:
        errors.append(f"output hypotheses_supported/refuted/inconclusive overlap on id(s): {sorted(overlap)} — each hypothesis belongs to exactly one bucket")

    if errors:
        raise SchemaValidationError("; ".join(errors))
