"""Output contract for the Experiment Planner Agent, plus a dependency-free validator.

This is the schema a Coder Agent (or anything else downstream) can rely on.
Field shapes below double as documentation; `validate_output` enforces them at
runtime before an agent run is allowed to return/persist.
"""

from __future__ import annotations

import re
from typing import List, Optional, TypedDict

VALID_COMPLEXITIES = {"low", "medium", "high"}

_HYPOTHESIS_ID_RE = re.compile(r"\bH\d+\b")


class Variables(TypedDict):
    independent: List[str]
    dependent: List[str]


class DataRequirements(TypedDict):
    source: str
    description: str
    preprocessing_steps: List[str]


class Method(TypedDict):
    name: str
    description: str
    reused_from_literature: bool


class Evaluation(TypedDict):
    metrics: List[str]
    baseline: str
    success_criteria: str


class ImplementationStep(TypedDict):
    step: int
    description: str


class ExperimentPlan(TypedDict):
    hypothesis_id: str
    feasible: bool
    feasibility_notes: str
    objective: str
    variables: Variables
    design: str
    data_requirements: DataRequirements
    methods: List[Method]
    evaluation: Evaluation
    implementation_steps: List[ImplementationStep]
    estimated_complexity: str  # "low" | "medium" | "high"
    risks: List[str]


class PriorityEntry(TypedDict):
    hypothesis_id: str
    rank: int
    justification: str


class ExperimentPlannerOutput(TypedDict):
    experiment_plans: List[ExperimentPlan]  # one per input hypothesis, always
    shared_infrastructure: List[str]
    priority_order: List[PriorityEntry]
    source_hypothesis_ids: List[str]
    generated_at: str
    model: str


class SchemaValidationError(ValueError):
    """Raised when an assembled result doesn't match ExperimentPlannerOutput."""


def _check_fields(obj: dict, fields: list[tuple[str, type]], path: str, errors: list[str]) -> None:
    for key, expected_type in fields:
        if key not in obj:
            errors.append(f"{path}.{key} is missing")
        elif not isinstance(obj[key], expected_type):
            errors.append(f"{path}.{key} should be {expected_type.__name__}, got {type(obj[key]).__name__}")


def priority_order_errors(priority_entries: list, expected_ids: list[str], path_prefix: str = "") -> List[str]:
    """Checks a raw priority_order list's per-entry shape, that its
    hypothesis_ids are exactly `expected_ids` (each once), and that ranks are
    a permutation of 1..len(priority_entries). Returns human-readable
    problems; empty means clean.

    Factored out of validate_output so the Experiment Planner Agent's own
    pre-assembly repair/coercion step (experiment_planner_agent.py's
    _plan_cross_cutting) checks the model's raw cross-cutting response
    against the exact same rule the final schema validation enforces —
    "valid" can't drift into two different definitions between the two call
    sites."""
    errors: List[str] = []
    ids: List[str] = []
    ranks: List[int] = []
    for i, entry in enumerate(priority_entries):
        path = f"{path_prefix}priority_order[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{path} should be an object")
            continue
        _check_fields(entry, [("hypothesis_id", str), ("rank", int), ("justification", str)], path, errors)
        ids.append(entry.get("hypothesis_id"))
        if isinstance(entry.get("rank"), int):
            ranks.append(entry["rank"])

    if expected_ids and set(ids) != set(expected_ids):
        errors.append(
            f"{path_prefix}priority_order hypothesis_ids {sorted(set(ids))} don't match "
            f"expected hypothesis_ids {sorted(set(expected_ids))}"
        )
    if ranks and sorted(ranks) != list(range(1, len(priority_entries) + 1)):
        errors.append(f"{path_prefix}priority_order ranks should be exactly 1..{len(priority_entries)}, got {ranks}")
    return errors


def clean_shared_infrastructure(entries: list, expected_ids: list[str]) -> tuple[List[str], List[str]]:
    """Drops any shared_infrastructure entry that names a hypothesis id
    outside this run's actual plan set — the same "strip it, don't print it"
    rule the Writer applies to unresolved citations. CROSS_CUTTING_PROMPT's
    own worked example names a hypothesis id ("H1"/"H3") to illustrate the
    shape; a small model asked for cross-cutting notes has been observed
    paraphrasing that example back nearly verbatim (2026-08-14, job
    10229968: an "H1 and H2" entry produced for a run that only ever planned
    H2) instead of generating real content or returning the empty list the
    prompt asks for when nothing is genuinely shared. Any entry mentioning a
    hypothesis id that isn't one of expected_ids is deterministically not
    this run's own content, whether it's the copied example or something
    else invented. Returns (kept, dropped) — the caller logs `dropped`
    rather than silently losing it."""
    expected_set = set(expected_ids)
    kept: List[str] = []
    dropped: List[str] = []
    for entry in entries or []:
        if not isinstance(entry, str):
            dropped.append(repr(entry))
            continue
        mentioned = set(_HYPOTHESIS_ID_RE.findall(entry))
        if mentioned and not mentioned.issubset(expected_set):
            dropped.append(entry)
            continue
        kept.append(entry)
    return kept, dropped


def validate_output(data: dict, expected_hypothesis_ids: Optional[list[str]] = None) -> None:
    """Raises SchemaValidationError (with every problem found, not just the
    first) if `data` doesn't match ExperimentPlannerOutput. If
    `expected_hypothesis_ids` is given, also checks every one of them has a
    corresponding experiment plan (even if flagged infeasible)."""
    errors: List[str] = []

    if not isinstance(data, dict):
        raise SchemaValidationError(f"top-level output should be an object, got {type(data).__name__}")

    _check_fields(
        data,
        [
            ("experiment_plans", list),
            ("shared_infrastructure", list),
            ("priority_order", list),
            ("source_hypothesis_ids", list),
            ("generated_at", str),
            ("model", str),
        ],
        "output",
        errors,
    )

    plans = data.get("experiment_plans", []) or []
    plan_ids: List[str] = []
    for i, plan in enumerate(plans):
        path = f"output.experiment_plans[{i}]"
        if not isinstance(plan, dict):
            errors.append(f"{path} should be an object")
            continue
        _check_fields(
            plan,
            [
                ("hypothesis_id", str),
                ("feasible", bool),
                ("feasibility_notes", str),
                ("objective", str),
                ("design", str),
                ("methods", list),
                ("implementation_steps", list),
                ("estimated_complexity", str),
                ("risks", list),
            ],
            path,
            errors,
        )
        plan_ids.append(plan.get("hypothesis_id"))

        variables = plan.get("variables")
        if not isinstance(variables, dict):
            errors.append(f"{path}.variables is missing or not an object")
        else:
            _check_fields(variables, [("independent", list), ("dependent", list)], f"{path}.variables", errors)

        data_requirements = plan.get("data_requirements")
        if not isinstance(data_requirements, dict):
            errors.append(f"{path}.data_requirements is missing or not an object")
        else:
            _check_fields(
                data_requirements,
                [("source", str), ("description", str), ("preprocessing_steps", list)],
                f"{path}.data_requirements",
                errors,
            )

        for j, method in enumerate(plan.get("methods", []) or []):
            if not isinstance(method, dict):
                errors.append(f"{path}.methods[{j}] should be an object")
                continue
            _check_fields(
                method, [("name", str), ("description", str), ("reused_from_literature", bool)], f"{path}.methods[{j}]", errors
            )

        evaluation = plan.get("evaluation")
        if not isinstance(evaluation, dict):
            errors.append(f"{path}.evaluation is missing or not an object")
        else:
            _check_fields(
                evaluation, [("metrics", list), ("baseline", str), ("success_criteria", str)], f"{path}.evaluation", errors
            )

        for j, step in enumerate(plan.get("implementation_steps", []) or []):
            if not isinstance(step, dict):
                errors.append(f"{path}.implementation_steps[{j}] should be an object")
                continue
            _check_fields(step, [("step", int), ("description", str)], f"{path}.implementation_steps[{j}]", errors)

        complexity = plan.get("estimated_complexity")
        if complexity not in VALID_COMPLEXITIES:
            errors.append(f"{path}.estimated_complexity should be one of {sorted(VALID_COMPLEXITIES)}, got {complexity!r}")

    priority_entries = data.get("priority_order", []) or []
    errors.extend(priority_order_errors(priority_entries, plan_ids, path_prefix="output."))

    if expected_hypothesis_ids is not None:
        missing = set(expected_hypothesis_ids) - set(plan_ids)
        extra = set(plan_ids) - set(expected_hypothesis_ids)
        if missing:
            errors.append(f"output.experiment_plans is missing plan(s) for hypothesis id(s): {sorted(missing)}")
        if extra:
            errors.append(f"output.experiment_plans has plan(s) for unrecognized hypothesis id(s): {sorted(extra)}")

    if errors:
        raise SchemaValidationError("; ".join(errors))


def narrow_to_hypotheses(output: dict, hypothesis_ids: list[str]) -> dict:
    """A planner output covering only the named hypotheses, still valid.

    Dropping plans is not a filter over one list: `priority_order` must stay a
    permutation of 1..n over exactly the surviving plan ids (see
    priority_order_errors), and `source_hypothesis_ids` must agree with them.
    Filtering `experiment_plans` alone produces a document that
    `validate_output` rejects and `run_coder_agent` therefore refuses — so the
    re-rank lives here, beside the rule it exists to preserve, rather than in
    whichever caller happened to need it first.

    Order is taken from the *original* priority_order, not from the caller's
    list, so re-running two of three experiments keeps the ranking the planner
    justified. A surviving plan the priority_order never mentioned (a document
    written before that field was required) is appended after the ranked ones.

    Raises SchemaValidationError if none of the ids has a plan — the caller
    asked for something this output cannot supply, and a valid-but-empty
    document would fail later, further from the cause.
    """
    wanted = set(hypothesis_ids)
    plans = [p for p in output.get("experiment_plans") or [] if isinstance(p, dict) and p.get("hypothesis_id") in wanted]
    if not plans:
        raise SchemaValidationError(
            f"no experiment plan for hypothesis id(s) {sorted(wanted)} in this planner output"
        )

    kept_ids = [p.get("hypothesis_id") for p in plans]
    ranked = [e for e in output.get("priority_order") or [] if isinstance(e, dict) and e.get("hypothesis_id") in kept_ids]
    unranked = [pid for pid in kept_ids if pid not in {e.get("hypothesis_id") for e in ranked}]

    priority_order = [
        {**entry, "rank": rank}
        for rank, entry in enumerate(sorted(ranked, key=lambda e: e.get("rank", 0)), start=1)
    ]
    priority_order += [
        {"hypothesis_id": pid, "rank": len(priority_order) + i, "justification": ""}
        for i, pid in enumerate(unranked, start=1)
    ]

    return {
        **output,
        "experiment_plans": plans,
        "priority_order": priority_order,
        "source_hypothesis_ids": kept_ids,
    }
