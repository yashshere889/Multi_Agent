"""Output contract for the Hypothesis Agent, plus a dependency-free validator.

This is the schema an Experiment Planner Agent (or anything else downstream)
can rely on. Field shapes below double as documentation; `validate_output`
enforces them at runtime before an agent run is allowed to return/persist.
"""

from __future__ import annotations

from typing import List, TypedDict


class MethodEntry(TypedDict):
    method: str
    papers_using_it: List[str]  # paper ids
    notes: str


class GapEntry(TypedDict):
    gap: str
    supporting_evidence: List[str]  # paper ids
    notes: str


class SuggestedVariables(TypedDict):
    independent: List[str]
    dependent: List[str]


class Hypothesis(TypedDict):
    id: str
    statement: str
    rationale: str
    related_gaps: List[str]
    related_methods: List[str]
    suggested_variables: SuggestedVariables


class HypothesisAgentOutput(TypedDict):
    literature_summary: str
    methods_overview: List[MethodEntry]
    gaps: List[GapEntry]
    hypotheses: List[Hypothesis]  # always exactly 3
    source_paper_ids: List[str]
    generated_at: str
    model: str


class SchemaValidationError(ValueError):
    """Raised when an assembled result doesn't match HypothesisAgentOutput."""


def _check_fields(obj: dict, fields: list[tuple[str, type]], path: str, errors: list[str]) -> None:
    for key, expected_type in fields:
        if key not in obj:
            errors.append(f"{path}.{key} is missing")
        elif not isinstance(obj[key], expected_type):
            errors.append(f"{path}.{key} should be {expected_type.__name__}, got {type(obj[key]).__name__}")


def validate_output(data: dict) -> None:
    """Raises SchemaValidationError (with every problem found, not just the first) if
    `data` doesn't match HypothesisAgentOutput."""
    errors: List[str] = []

    if not isinstance(data, dict):
        raise SchemaValidationError(f"top-level output should be an object, got {type(data).__name__}")

    _check_fields(
        data,
        [
            ("literature_summary", str),
            ("source_paper_ids", list),
            ("generated_at", str),
            ("model", str),
            ("methods_overview", list),
            ("gaps", list),
            ("hypotheses", list),
        ],
        "output",
        errors,
    )

    for i, method in enumerate(data.get("methods_overview", []) or []):
        if not isinstance(method, dict):
            errors.append(f"output.methods_overview[{i}] should be an object")
            continue
        _check_fields(method, [("method", str), ("papers_using_it", list), ("notes", str)], f"output.methods_overview[{i}]", errors)

    for i, gap in enumerate(data.get("gaps", []) or []):
        if not isinstance(gap, dict):
            errors.append(f"output.gaps[{i}] should be an object")
            continue
        _check_fields(gap, [("gap", str), ("supporting_evidence", list), ("notes", str)], f"output.gaps[{i}]", errors)

    hypotheses = data.get("hypotheses", [])
    if isinstance(hypotheses, list) and len(hypotheses) != 3:
        errors.append(f"output.hypotheses should contain exactly 3 items, got {len(hypotheses)}")

    for i, hyp in enumerate(hypotheses or []):
        if not isinstance(hyp, dict):
            errors.append(f"output.hypotheses[{i}] should be an object")
            continue
        path = f"output.hypotheses[{i}]"
        _check_fields(
            hyp,
            [
                ("id", str),
                ("statement", str),
                ("rationale", str),
                ("related_gaps", list),
                ("related_methods", list),
            ],
            path,
            errors,
        )
        variables = hyp.get("suggested_variables")
        if not isinstance(variables, dict):
            errors.append(f"{path}.suggested_variables is missing or not an object")
        else:
            _check_fields(variables, [("independent", list), ("dependent", list)], f"{path}.suggested_variables", errors)

    if errors:
        raise SchemaValidationError("; ".join(errors))
