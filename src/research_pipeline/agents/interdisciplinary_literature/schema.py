"""Output contract for the Interdisciplinary Literature Agent, plus a
dependency-free validator.

This is the schema the Hypothesis Agent (or anything else downstream) can rely
on. Field shapes below double as documentation; `validate_output` enforces them
at runtime before an agent run is allowed to return/persist.

Note that `papers` is deliberately the *merged* pool (the in-domain papers this
agent was handed, plus the cross-field papers it found), in the same shape the
Hypothesis Agent already consumes — so the Hypothesis Agent needs no change to
how it ingests papers. `cross_field_papers` is the subset of `papers` this
agent added, each carrying the `discipline` it came from.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class FieldEntry(TypedDict):
    field: str
    rationale: str
    queries: List[str]


class BridgeInsight(TypedDict):
    insight: str
    source_field: str
    connection_to_core_problem: str
    supporting_paper_ids: List[str]


class InterdisciplinaryLiteratureOutput(TypedDict):
    papers: List[dict]  # in-domain papers + cross_field_papers, deduped
    core_paper_ids: List[str]
    cross_field_papers: List[dict]  # subset of `papers`, each with a "discipline"
    fields_explored: List[FieldEntry]
    bridge_insights: List[BridgeInsight]
    research_question: Optional[str]
    generated_at: str
    model: str


class SchemaValidationError(ValueError):
    """Raised when an assembled result doesn't match InterdisciplinaryLiteratureOutput."""


def _check_fields(obj: dict, fields: list[tuple[str, type]], path: str, errors: list[str]) -> None:
    for key, expected_type in fields:
        if key not in obj:
            errors.append(f"{path}.{key} is missing")
        elif not isinstance(obj[key], expected_type):
            errors.append(f"{path}.{key} should be {expected_type.__name__}, got {type(obj[key]).__name__}")


def validate_output(data: dict) -> None:
    """Raises SchemaValidationError (with every problem found, not just the first)
    if `data` doesn't match InterdisciplinaryLiteratureOutput."""
    errors: List[str] = []

    if not isinstance(data, dict):
        raise SchemaValidationError(f"top-level output should be an object, got {type(data).__name__}")

    _check_fields(
        data,
        [
            ("papers", list),
            ("core_paper_ids", list),
            ("cross_field_papers", list),
            ("fields_explored", list),
            ("bridge_insights", list),
            ("generated_at", str),
            ("model", str),
        ],
        "output",
        errors,
    )

    if "research_question" not in data:
        errors.append("output.research_question is missing")
    elif data["research_question"] is not None and not isinstance(data["research_question"], str):
        errors.append(f"output.research_question should be str or null, got {type(data['research_question']).__name__}")

    for i, field in enumerate(data.get("fields_explored", []) or []):
        path = f"output.fields_explored[{i}]"
        if not isinstance(field, dict):
            errors.append(f"{path} should be an object")
            continue
        _check_fields(field, [("field", str), ("rationale", str), ("queries", list)], path, errors)

    for i, insight in enumerate(data.get("bridge_insights", []) or []):
        path = f"output.bridge_insights[{i}]"
        if not isinstance(insight, dict):
            errors.append(f"{path} should be an object")
            continue
        _check_fields(
            insight,
            [
                ("insight", str),
                ("source_field", str),
                ("connection_to_core_problem", str),
                ("supporting_paper_ids", list),
            ],
            path,
            errors,
        )

    # cross_field_papers is documented as a subset of papers, so a run that
    # reports a cross-field paper it didn't actually merge in would hand the
    # Hypothesis Agent an id it can never resolve. Checked deterministically
    # here rather than trusted, in the same spirit as the Writer Agent's
    # citation resolution.
    papers = data.get("papers")
    cross_field = data.get("cross_field_papers")
    if isinstance(papers, list) and isinstance(cross_field, list):
        merged_titles = {p.get("title") for p in papers if isinstance(p, dict)}
        orphans = [
            p.get("title") for p in cross_field if isinstance(p, dict) and p.get("title") not in merged_titles
        ]
        if orphans:
            errors.append(f"output.cross_field_papers contains paper(s) not present in output.papers: {orphans}")

    for i, paper in enumerate(cross_field or []):
        if not isinstance(paper, dict):
            errors.append(f"output.cross_field_papers[{i}] should be an object")
        elif not isinstance(paper.get("discipline"), str):
            errors.append(f"output.cross_field_papers[{i}].discipline is missing or not a string")

    if errors:
        raise SchemaValidationError("; ".join(errors))
