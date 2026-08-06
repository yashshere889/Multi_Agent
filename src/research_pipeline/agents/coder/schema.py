"""Output contract for the Coder Agent, plus a dependency-free validator."""

from __future__ import annotations

from typing import List, Optional, TypedDict

VALID_STATUSES = {"completed", "code_generated_not_run", "skipped"}


class Results(TypedDict):
    metrics: dict
    meets_success_criteria: object  # bool, or the literal string "unknown"
    notes: str


class ExperimentResult(TypedDict):
    hypothesis_id: str
    status: str  # "completed" | "code_generated_not_run" | "skipped"
    reason: str  # required (non-empty) for "skipped" / "code_generated_not_run"
    code_path: Optional[str]  # None only when status == "skipped"
    assumptions_made: List[str]
    results: Optional[Results]  # only present when status == "completed"


class CoderAgentOutput(TypedDict):
    experiments: List[ExperimentResult]
    shared_infrastructure_path: str
    source_hypothesis_ids: List[str]
    generated_at: str
    model: str


class SchemaValidationError(ValueError):
    """Raised when an assembled result doesn't match CoderAgentOutput."""


def _check_fields(obj: dict, fields: list[tuple[str, type]], path: str, errors: list[str]) -> None:
    for key, expected_type in fields:
        if key not in obj:
            errors.append(f"{path}.{key} is missing")
        elif not isinstance(obj[key], expected_type):
            errors.append(f"{path}.{key} should be {expected_type.__name__}, got {type(obj[key]).__name__}")


def validate_output(data: dict, expected_hypothesis_ids: Optional[list[str]] = None) -> None:
    """Raises SchemaValidationError (with every problem found, not just the
    first) if `data` doesn't match CoderAgentOutput. If
    `expected_hypothesis_ids` is given, also checks every one of them has a
    corresponding entry (even if skipped)."""
    errors: List[str] = []

    if not isinstance(data, dict):
        raise SchemaValidationError(f"top-level output should be an object, got {type(data).__name__}")

    _check_fields(
        data,
        [
            ("experiments", list),
            ("shared_infrastructure_path", str),
            ("source_hypothesis_ids", list),
            ("generated_at", str),
            ("model", str),
        ],
        "output",
        errors,
    )

    experiments = data.get("experiments", []) or []
    experiment_ids: List[str] = []
    for i, exp in enumerate(experiments):
        path = f"output.experiments[{i}]"
        if not isinstance(exp, dict):
            errors.append(f"{path} should be an object")
            continue
        _check_fields(
            exp, [("hypothesis_id", str), ("status", str), ("reason", str), ("assumptions_made", list)], path, errors
        )
        experiment_ids.append(exp.get("hypothesis_id"))

        status = exp.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"{path}.status should be one of {sorted(VALID_STATUSES)}, got {status!r}")
            continue

        if status in ("skipped", "code_generated_not_run") and not (exp.get("reason") or "").strip():
            errors.append(f"{path}.reason is required (non-empty) when status is {status!r}")

        code_path = exp.get("code_path")
        if status == "skipped":
            if code_path not in (None, ""):
                errors.append(f"{path}.code_path should be null when status is 'skipped', got {code_path!r}")
        elif not code_path or not isinstance(code_path, str):
            errors.append(f"{path}.code_path is required (non-empty string) when status is {status!r}")

        results = exp.get("results")
        if status == "completed":
            if not isinstance(results, dict):
                errors.append(f"{path}.results is required (an object) when status is 'completed'")
            else:
                _check_fields(results, [("metrics", dict), ("notes", str)], f"{path}.results", errors)
                meets = results.get("meets_success_criteria")
                if not (isinstance(meets, bool) or meets == "unknown"):
                    errors.append(f"{path}.results.meets_success_criteria should be true, false, or \"unknown\", got {meets!r}")
        elif results is not None:
            errors.append(f"{path}.results should be null when status is {status!r}")

    if expected_hypothesis_ids is not None:
        missing = set(expected_hypothesis_ids) - set(experiment_ids)
        extra = set(experiment_ids) - set(expected_hypothesis_ids)
        if missing:
            errors.append(f"output.experiments is missing entries for hypothesis id(s): {sorted(missing)}")
        if extra:
            errors.append(f"output.experiments has entries for unrecognized hypothesis id(s): {sorted(extra)}")

    if errors:
        raise SchemaValidationError("; ".join(errors))
