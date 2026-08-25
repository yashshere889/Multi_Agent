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


class RankingEntry(TypedDict):
    hypothesis_id: str
    rank: int  # 1 = best; the ranks are a permutation of 1..len(hypotheses)
    score: float
    justification: str


class HypothesisAgentOutput(TypedDict):
    literature_summary: str
    methods_overview: List[MethodEntry]
    gaps: List[GapEntry]
    hypotheses: List[Hypothesis]  # always exactly 3
    ranking: List[RankingEntry]  # one per hypothesis, ranks a permutation of 1..3
    selected_hypothesis_id: str  # the id ranked 1 — derived, never the model's own pick
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


# The fields every hypothesis must carry, and their types. A single list
# because two things read it: this module's validator, and
# hypothesis_agent's repair_keys call, which renames a mangled key
# (`rationationale`) back onto the name meant. If those two ever disagreed, the
# repair would fix keys the validator does not want and miss ones it does.
HYPOTHESIS_FIELDS: list[tuple[str, type]] = [
    ("id", str),
    ("statement", str),
    ("rationale", str),
    ("related_gaps", list),
    ("related_methods", list),
]
# suggested_variables is checked separately (it is an object with its own two
# fields), but the repair still needs to know the name is expected.
HYPOTHESIS_FIELD_NAMES: list[str] = [name for name, _ in HYPOTHESIS_FIELDS] + [
    "suggested_variables"
]


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
            ("ranking", list),
            ("selected_hypothesis_id", str),
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
        _check_fields(hyp, HYPOTHESIS_FIELDS, path, errors)
        variables = hyp.get("suggested_variables")
        if not isinstance(variables, dict):
            errors.append(f"{path}.suggested_variables is missing or not an object")
        else:
            _check_fields(variables, [("independent", list), ("dependent", list)], f"{path}.suggested_variables", errors)

    _validate_ranking(data, hypotheses, errors)

    if errors:
        raise SchemaValidationError("; ".join(errors))


def _validate_ranking(data: dict, hypotheses: list, errors: List[str]) -> None:
    """The ranking has to be a total order over exactly the hypotheses that were
    generated, and `selected_hypothesis_id` has to be the one it puts first —
    all checkable exactly, so none of it is taken on trust. Same style as
    experiment_planner/schema.py's priority_order check."""
    ranking = data.get("ranking", [])
    if not isinstance(ranking, list):
        return  # already reported as a type error above

    hypothesis_ids = [h.get("id") for h in (hypotheses or []) if isinstance(h, dict)]
    ranked_ids: List[str] = []
    ranks: List[int] = []
    for i, entry in enumerate(ranking):
        path = f"output.ranking[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{path} should be an object")
            continue
        _check_fields(entry, [("hypothesis_id", str), ("rank", int), ("justification", str)], path, errors)
        # bool is an int subclass, so it would sail through the check above.
        if isinstance(entry.get("rank"), int) and not isinstance(entry.get("rank"), bool):
            ranks.append(entry["rank"])
        score = entry.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            errors.append(f"{path}.score should be a number, got {type(score).__name__}")
        ranked_ids.append(entry.get("hypothesis_id"))

    if hypothesis_ids and set(ranked_ids) != set(hypothesis_ids):
        errors.append(
            f"output.ranking hypothesis_ids {sorted(set(map(str, ranked_ids)))} don't match "
            f"output.hypotheses ids {sorted(set(map(str, hypothesis_ids)))}"
        )
    if sorted(ranks) != list(range(1, len(ranking) + 1)):
        errors.append(f"output.ranking ranks should be exactly 1..{len(ranking)}, got {sorted(ranks)}")

    selected = data.get("selected_hypothesis_id")
    top = [entry.get("hypothesis_id") for entry in ranking if isinstance(entry, dict) and entry.get("rank") == 1]
    if isinstance(selected, str) and top and selected != top[0]:
        errors.append(
            f"output.selected_hypothesis_id is {selected!r}, but the hypothesis ranked 1 is {top[0]!r}"
        )
    if isinstance(selected, str) and hypothesis_ids and selected not in hypothesis_ids:
        errors.append(f"output.selected_hypothesis_id {selected!r} isn't one of the generated hypothesis ids")
