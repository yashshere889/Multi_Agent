"""One generated experiment, in the detail the progress panel has no room for.

The progress panel gives each experiment a line: hypothesis id, status, a fix
count, a success-criteria chip. That is the right density for watching six
stages go by, and it is not enough to answer the questions someone actually
asks afterwards — *why* did this take three fix attempts, what did it fail with
each time, what data did it really use, and why is the verdict "unknown" when
there are metrics sitting right there.

All of that is already recorded. `ExperimentResult` carries `fix_history` (a
snapshot path and the concrete error per attempt), `data_provenance` (per input:
real, or a labelled surrogate and why), and the fields `provenance.apply_to_results`
stamps on when it withholds a verdict. This module reduces the Coder Agent's own
summary file into display shapes; it computes nothing about the experiment that
the agent did not already decide.

Reading the summary file rather than the event stream is deliberate. The stage
event carries `stages.py`'s reduction — five fields per experiment — because
that is what the progress panel needs; the file on disk is the whole
`CoderAgentOutput`. Same rule as everywhere else in this package: the server's
only view of a run is the files under its run directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Written by CoderAgent._write_summary as coder_agent_summary_<UTC timestamp>.json.
# A run can hold more than one — the agent writes an interim summary as well as
# a final one — so the newest wins, which is also the most complete.
SUMMARY_GLOB = "coder_agent_summary_*.json"

# The files a generated experiment leaves in its own directory, in the order
# they are worth opening. Absent ones are simply not offered: an experiment that
# was never run has no results.json, and one that didn't need a cluster has no
# run.sbatch.
EXPERIMENT_FILES = (
    ("run.py", "the generated program"),
    ("results.json", "what it produced"),
    ("data_provenance.json", "what data it used"),
    ("requirements.txt", "its dependencies"),
    ("README.md", "the agent's own notes"),
    ("run.sbatch", "the cluster job template"),
)

# Statuses whose experiment produced no metrics, so the results panel would be
# an empty box rather than a finding.
NO_RESULTS_STATUSES = frozenset({"skipped", "code_generated_not_run", "submitted_to_slurm"})


@dataclass(frozen=True)
class FixAttemptView:
    attempt: int
    error_source: str
    error_summary: str
    resolved: bool
    same_error_streak: int
    code_rel: Optional[str]


@dataclass(frozen=True)
class ExperimentView:
    hypothesis_id: str
    status: str
    reason: str
    starter_used: str
    slurm_job_id: Optional[str]
    metrics: dict
    notes: str
    verdict: Any
    model_reported_verdict: Any
    verdict_withheld_because: str
    methodological_validity: str
    assumptions: list[str]
    inputs: list[dict]
    surrogate_count: int
    fix_attempts: list[FixAttemptView]
    files: list[tuple[str, str, str]]  # (name, why it's worth opening, rel path)
    code_rel: Optional[str]

    @property
    def verdict_withheld(self) -> bool:
        """Whether the Coder Agent refused to turn these metrics into a verdict.

        The string "unknown" is the signal — `writer_agent.compute_hypothesis_verdict`
        maps it to "inconclusive", where `False` would have mapped to "refuted".
        Worth showing loudly: an experiment with real-looking metrics and no
        verdict looks like a bug until you know it was a decision."""
        return self.verdict == "unknown"

    @property
    def ok(self) -> bool:
        return self.status == "completed"


def find_summary(run_dir: str | Path) -> Optional[Path]:
    """The newest Coder Agent summary in this run's outputs, or None."""
    outputs = Path(run_dir) / "outputs"
    if not outputs.is_dir():
        return None
    found = sorted(outputs.glob(SUMMARY_GLOB))
    return found[-1] if found else None


def _relative_to_run(run_dir: Path, path: Any) -> Optional[str]:
    """A run-relative path the artifact browser can link to, or None.

    `code_path` is absolute — the Coder Agent records where it actually wrote,
    and under the web UI that is `CODER_EXPERIMENTS_DIR` inside this run. A run
    whose experiments went somewhere else entirely (a CLI run with the default
    `experiments/`, later copied here) has code the browser cannot serve, and
    None is how the template knows not to offer a dead link.
    """
    if not path:
        return None
    try:
        candidate = Path(str(path)).resolve()
        return candidate.relative_to(run_dir.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def _fix_attempt_views(run_dir: Path, history: list) -> list[FixAttemptView]:
    views = []
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        views.append(
            FixAttemptView(
                attempt=entry.get("attempt", 0),
                error_source=entry.get("error_source", "") or "unknown",
                error_summary=entry.get("error_summary", "") or "",
                resolved=bool(entry.get("resolved")),
                # Absent in summaries written before the streak counter existed;
                # 1 is the truthful default — this attempt, and no repeat.
                same_error_streak=entry.get("same_error_streak") or 1,
                code_rel=_relative_to_run(run_dir, entry.get("code_path")),
            )
        )
    return views


def _files_for(run_dir: Path, code_path: Any) -> list[tuple[str, str, str]]:
    code_rel = _relative_to_run(run_dir, code_path)
    if not code_rel:
        return []
    directory = run_dir / code_rel
    found = []
    for name, why in EXPERIMENT_FILES:
        if (directory / name).is_file():
            found.append((name, why, f"{code_rel}/{name}"))
    return found


def _view(run_dir: Path, experiment: dict) -> ExperimentView:
    results = experiment.get("results") or {}
    provenance = experiment.get("data_provenance") or {}
    return ExperimentView(
        hypothesis_id=experiment.get("hypothesis_id", "") or "?",
        status=experiment.get("status", "") or "unknown",
        reason=experiment.get("reason", "") or "",
        starter_used=experiment.get("starter_used", "") or "",
        slurm_job_id=experiment.get("slurm_job_id"),
        metrics=results.get("metrics") or {},
        notes=results.get("notes", "") or "",
        verdict=results.get("meets_success_criteria"),
        model_reported_verdict=results.get("model_reported_meets_success_criteria"),
        verdict_withheld_because=results.get("verdict_withheld_because", "") or "",
        methodological_validity=results.get("methodological_validity", "")
        or provenance.get("methodological_validity", "")
        or "",
        assumptions=[a for a in experiment.get("assumptions_made") or [] if a],
        inputs=[i for i in provenance.get("inputs") or [] if isinstance(i, dict)],
        surrogate_count=provenance.get("surrogate_count") or 0,
        fix_attempts=_fix_attempt_views(run_dir, experiment.get("fix_history") or []),
        files=_files_for(run_dir, experiment.get("code_path")),
        code_rel=_relative_to_run(run_dir, experiment.get("code_path")),
    )


def load_experiments(run_dir: str | Path) -> tuple[list[ExperimentView], Optional[str]]:
    """Every experiment this run generated, plus the run-relative path of the
    summary they came from (so the page can link to the raw file).

    `([], None)` covers "the Coder stage hasn't run", "it ran but wrote nothing
    readable", and "this run stopped before the Coder stage" alike — all three
    are the same thing to the page: nothing to inspect yet.
    """
    root = Path(run_dir)
    summary_path = find_summary(root)
    if not summary_path:
        return [], None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, UnicodeDecodeError):
        # A summary being written right now, or a torn file. The rest of the
        # page is still worth rendering, so this is not an error to raise.
        return [], None

    experiments = summary.get("experiments")
    if not isinstance(experiments, list):
        return [], None

    views = [_view(root, e) for e in experiments if isinstance(e, dict)]
    return views, _relative_to_run(root, summary_path)


def find_experiment(views: list[ExperimentView], hypothesis_id: str) -> Optional[ExperimentView]:
    for view in views:
        if view.hypothesis_id == hypothesis_id:
            return view
    return None
