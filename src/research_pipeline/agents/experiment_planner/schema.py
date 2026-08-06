"""Output contract for the Experiment Planner Agent, plus a dependency-free validator.

This is the schema a Coder Agent (or anything else downstream) can rely on.
Field shapes below double as documentation; `validate_output` enforces them at
runtime before an agent run is allowed to return/persist.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

VALID_COMPLEXITIES = {"low", "medium", "high"}


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
    priority_ids: List[str] = []
    ranks: List[int] = []
    for i, entry in enumerate(priority_entries):
        path = f"output.priority_order[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{path} should be an object")
            continue
        _check_fields(entry, [("hypothesis_id", str), ("rank", int), ("justification", str)], path, errors)
        priority_ids.append(entry.get("hypothesis_id"))
        if isinstance(entry.get("rank"), int):
            ranks.append(entry["rank"])

    if plan_ids and set(priority_ids) != set(plan_ids):
        errors.append(
            f"output.priority_order hypothesis_ids {sorted(set(priority_ids))} don't match "
            f"output.experiment_plans hypothesis_ids {sorted(set(plan_ids))}"
        )
    if ranks and sorted(ranks) != list(range(1, len(priority_entries) + 1)):
        errors.append(f"output.priority_order ranks should be exactly 1..{len(priority_entries)}, got {ranks}")

    if expected_hypothesis_ids is not None:
        missing = set(expected_hypothesis_ids) - set(plan_ids)
        extra = set(plan_ids) - set(expected_hypothesis_ids)
        if missing:
            errors.append(f"output.experiment_plans is missing plan(s) for hypothesis id(s): {sorted(missing)}")
        if extra:
            errors.append(f"output.experiment_plans has plan(s) for unrecognized hypothesis id(s): {sorted(extra)}")

    if errors:
        raise SchemaValidationError("; ".join(errors))
