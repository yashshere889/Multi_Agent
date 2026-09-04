"""Output contract for the Coder Agent, plus a dependency-free validator."""

from __future__ import annotations

from typing import TypedDict

VALID_STATUSES = {
    "completed",
    "code_generated_not_run",
    "skipped",
    "submitted_to_slurm",
    # A job that was submitted, ran, and ended badly — set only by reconcile.py,
    # never by a run itself, since a run never sees its own jobs finish. Distinct
    # from "code_generated_not_run" because that one means nothing ever executed;
    # this code did execute, on the cluster, and failed there. Both read as
    # inconclusive to the Writer, but only one of them has a log worth reading.
    "slurm_job_failed",
}

VALID_ERROR_SOURCES = {
    # Renamed from "invalid_json" when generated code moved off the JSON
    # transport onto llm_sections.py's delimited format — the failure it names
    # is now "the response wasn't in the required section format at all".
    "invalid_format",
    "missing_sections",
    # A code section defines no function with the name run.py's fixed
    # orchestration calls — see sandbox.check_required_function_names.
    "missing_required_function",
    # A required function's body is empty, `pass`, `...`, or a bare
    # NotImplementedError raise — see sandbox.check_nontrivial_function_bodies.
    "empty_body",
    "compile_check",
    # The rendered run.py uses a name it never binds — a helper that was
    # referenced but never written, or one whose import a regeneration dropped.
    # Parses fine, so compile_check clears it; only a NameError at execution
    # would otherwise reveal it. See sandbox.check_undefined_names.
    "undefined_name",
    "static_lint",
    # load_data reads a local file with nothing to fall back on if it isn't there
    # — see sandbox.check_data_fallback.
    "missing_data_fallback",
    # A real dataset was found and offered but load_data() shows no sign of
    # using it, and it wasn't declined in assumptions_made either — see
    # sandbox.check_hf_dataset_usage.
    "ignored_available_dataset",
    "run_experiment",
    # The four below split cases off `run_experiment`, which used to absorb
    # every runtime failure regardless of kind. Each needs a different repair
    # and `run_experiment`'s — regenerate the code — is wrong for all four.
    # See diagnose.classify_execution_failure.
    #
    # A package the code correctly imports but nothing installed. Repaired by
    # installing it and re-running the same code; regenerating cannot install.
    "missing_dependency",
    # Something the code refers to no longer exists. Two halves, same repair
    # and same route: an import no installable package provides (pymc3, pystan,
    # theano...), and a call a major release deleted (pandas' fillna(method=),
    # DataFrame.append, numpy's scalar aliases — see diagnose.REMOVED_APIS).
    # The opposite of missing_dependency: only new code can fix it, and an
    # install that "succeeds" changes nothing. One narrow shape inside the
    # second half is repaired deterministically instead — see
    # repair.patch_removed_pandas_fillna.
    "obsolete_dependency",
    # A native library. pip cannot supply it, so neither repair applies and the
    # experiment stops with something a human can act on.
    "missing_system_library",
    # Out of time or memory. Repaired by shrinking the run's own cost knobs
    # deterministically — no model call — before asking for a rethink.
    "resource_limit",
    "results_json",
    # results.json parsed fine but its metrics look hollow (empty, NaN/Infinity,
    # a placeholder string, or every numeric metric exactly 0) — see
    # sandbox.check_results_plausibility.
    "implausible_results",
    "self_review",
}

ERROR_SUMMARY_MAX_CHARS = 2000


class Results(TypedDict):
    metrics: dict
    meets_success_criteria: object  # bool, or the literal string "unknown"
    notes: str


class FixAttempt(TypedDict):
    attempt: int  # 1-indexed
    error_source: str  # one of VALID_ERROR_SOURCES
    error_summary: str  # truncated to ERROR_SUMMARY_MAX_CHARS
    code_path: str  # snapshot of the code that produced this error
    resolved: bool  # whether the regeneration cleared the check that failed
    # How many trailing entries (this one included) share this error_source —
    # see coder_agent._consecutive_error_streak. Not in _check_fix_history's
    # required fields below: summaries written before this field existed still
    # have to load, same "checked when present but not required" rule as the
    # rest of this function.
    same_error_streak: int
    # Which code sections the regeneration that followed this failure was asked
    # for — [] meaning all of them. See coder_agent._target_sections. Optional,
    # same rule as same_error_streak.
    regenerated_sections: list[str]
    # What this attempt's code assumed, carried per-attempt because the code
    # this run ends up reporting is not always the last one generated: when the
    # fix budget runs out, coder_agent restores whichever attempt got furthest
    # (see _best_candidate), and the final generation's assumptions would then
    # describe a program that is no longer on disk. Optional, same rule as
    # same_error_streak.
    assumptions_made: list[str]


class ExperimentResult(TypedDict):
    hypothesis_id: str
    status: str  # one of VALID_STATUSES
    reason: str  # required (non-empty) for every status except "completed"
    code_path: str | None  # None only when status == "skipped"
    assumptions_made: list[str]
    results: Results | None  # only present when status == "completed"
    fix_attempts: int
    fix_history: list[FixAttempt]
    slurm_job_id: str | None  # set only when status == "submitted_to_slurm"
    starter_used: (
        str  # starters.STARTERS id this plan's prompts were grounded in, or "" for "general"
    )
    # Per-input record of what data this experiment actually used: real, or a
    # labelled surrogate and why. Optional so summaries written before this
    # existed still load, same rule as same_error_streak. See provenance.py —
    # when any input is a surrogate, results.meets_success_criteria is the
    # string "unknown" rather than a bool, which the Writer reads as
    # "inconclusive" instead of publishing a verdict off synthesized data.
    data_provenance: dict
    # What this run cost to finish: which cost knobs repair.downscale had to
    # shrink to fit the timeout, and whether shrinking them changed what the
    # experiment measures. Optional, same rule as data_provenance. When it did
    # — halved epochs, a smaller ensemble — results.meets_success_criteria is
    # the string "unknown" for the same reason a surrogate input makes it one:
    # an undertrained model's metrics lose to a baseline on their own merits,
    # and the Writer would publish that False as a refutation. See
    # compute_provenance.py.
    compute_provenance: dict


class CoderAgentOutput(TypedDict):
    experiments: list[ExperimentResult]
    shared_infrastructure_path: str
    source_hypothesis_ids: list[str]
    generated_at: str
    model: str
    # When reconcile.py last asked SLURM what became of this run's submitted
    # jobs. Absent until one has, which is most summaries — same
    # "checked when present but not required" rule as the fields above.
    reconciled_at: str


class SchemaValidationError(ValueError):
    """Raised when an assembled result doesn't match CoderAgentOutput."""


def _check_fields(obj: dict, fields: list[tuple[str, type]], path: str, errors: list[str]) -> None:
    for key, expected_type in fields:
        if key not in obj:
            errors.append(f"{path}.{key} is missing")
        elif not isinstance(obj[key], expected_type):
            errors.append(
                f"{path}.{key} should be {expected_type.__name__}, got {type(obj[key]).__name__}"
            )


def _check_fix_history(exp: dict, path: str, errors: list[str]) -> None:
    """Fix-loop bookkeeping is checked when present but not required: summaries
    written before the fix loop existed still have to load, since reading a
    previous run's JSON off disk is a supported way to chain agents."""
    if "fix_attempts" not in exp and "fix_history" not in exp:
        return

    attempts = exp.get("fix_attempts")
    history = exp.get("fix_history")
    if not isinstance(attempts, int):
        errors.append(f"{path}.fix_attempts should be int, got {type(attempts).__name__}")
        return
    if not isinstance(history, list):
        errors.append(f"{path}.fix_history should be list, got {type(history).__name__}")
        return

    if attempts < 0:
        errors.append(f"{path}.fix_attempts should be >= 0, got {attempts}")
    if len(history) != attempts:
        errors.append(
            f"{path}.fix_history has {len(history)} entries but fix_attempts is {attempts}"
        )

    for j, entry in enumerate(history):
        entry_path = f"{path}.fix_history[{j}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_path} should be an object")
            continue
        _check_fields(
            entry,
            [
                ("attempt", int),
                ("error_source", str),
                ("error_summary", str),
                ("code_path", str),
                ("resolved", bool),
            ],
            entry_path,
            errors,
        )
        source = entry.get("error_source")
        if source is not None and source not in VALID_ERROR_SOURCES:
            errors.append(
                f"{entry_path}.error_source should be one of {sorted(VALID_ERROR_SOURCES)}, got {source!r}"
            )
        summary = entry.get("error_summary")
        if isinstance(summary, str) and len(summary) > ERROR_SUMMARY_MAX_CHARS:
            errors.append(
                f"{entry_path}.error_summary exceeds {ERROR_SUMMARY_MAX_CHARS} characters"
            )
        # Optional (see FixAttempt) — checked only when present, same reason
        # error_summary's length check above isn't in _check_fields either.
        if "same_error_streak" in entry and not isinstance(entry["same_error_streak"], int):
            errors.append(
                f"{entry_path}.same_error_streak should be int, "
                f"got {type(entry['same_error_streak']).__name__}"
            )


def validate_output(data: dict, expected_hypothesis_ids: list[str] | None = None) -> None:
    """Raises SchemaValidationError (with every problem found, not just the
    first) if `data` doesn't match CoderAgentOutput. If
    `expected_hypothesis_ids` is given, also checks every one of them has a
    corresponding entry (even if skipped)."""
    errors: list[str] = []

    if not isinstance(data, dict):
        raise SchemaValidationError(
            f"top-level output should be an object, got {type(data).__name__}"
        )

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
    # `str | None`, not `list[str]`: a malformed entry missing hypothesis_id
    # still gets an (empty) slot here, so the expected-ids diff below reports it
    # as missing rather than silently lining up with the wrong plan.
    experiment_ids: list[str | None] = []
    for i, exp in enumerate(experiments):
        path = f"output.experiments[{i}]"
        if not isinstance(exp, dict):
            errors.append(f"{path} should be an object")
            continue
        _check_fields(
            exp,
            [("hypothesis_id", str), ("status", str), ("reason", str), ("assumptions_made", list)],
            path,
            errors,
        )
        experiment_ids.append(exp.get("hypothesis_id"))
        _check_fix_history(exp, path, errors)

        status = exp.get("status")
        if status not in VALID_STATUSES:
            errors.append(
                f"{path}.status should be one of {sorted(VALID_STATUSES)}, got {status!r}"
            )
            continue

        if status != "completed" and not (exp.get("reason") or "").strip():
            errors.append(f"{path}.reason is required (non-empty) when status is {status!r}")

        job_id = exp.get("slurm_job_id")
        if status == "submitted_to_slurm":
            if not job_id or not isinstance(job_id, str):
                errors.append(
                    f"{path}.slurm_job_id is required (non-empty string) when status is 'submitted_to_slurm'"
                )
        # "completed" and "slurm_job_failed" may carry one and need not: an
        # experiment run locally has no job, and one reconcile.py resolved from
        # the cluster keeps the id of the job that produced its numbers — which
        # is the only link from a result back to its sbatch log. Requiring null
        # here (as this did when submission was the end of the story) would make
        # every reconciled experiment fail validation.
        elif status in ("skipped", "code_generated_not_run") and job_id is not None:
            errors.append(f"{path}.slurm_job_id should be null when status is {status!r}")

        code_path = exp.get("code_path")
        if status == "skipped":
            if code_path not in (None, ""):
                errors.append(
                    f"{path}.code_path should be null when status is 'skipped', got {code_path!r}"
                )
        elif not code_path or not isinstance(code_path, str):
            errors.append(
                f"{path}.code_path is required (non-empty string) when status is {status!r}"
            )

        results = exp.get("results")
        if status == "completed":
            if not isinstance(results, dict):
                errors.append(f"{path}.results is required (an object) when status is 'completed'")
            else:
                _check_fields(
                    results, [("metrics", dict), ("notes", str)], f"{path}.results", errors
                )
                meets = results.get("meets_success_criteria")
                if not (isinstance(meets, bool) or meets == "unknown"):
                    errors.append(
                        f'{path}.results.meets_success_criteria should be true, false, or "unknown", got {meets!r}'
                    )
        elif results is not None:
            errors.append(f"{path}.results should be null when status is {status!r}")

    if expected_hypothesis_ids is not None:
        missing = set(expected_hypothesis_ids) - set(experiment_ids)
        extra = set(experiment_ids) - set(expected_hypothesis_ids)
        if missing:
            errors.append(
                f"output.experiments is missing entries for hypothesis id(s): {sorted(missing)}"
            )
        if extra:
            errors.append(
                "output.experiments has entries for unrecognized hypothesis id(s): "
                f"{sorted(extra, key=str)}"
            )

    if errors:
        raise SchemaValidationError("; ".join(errors))
