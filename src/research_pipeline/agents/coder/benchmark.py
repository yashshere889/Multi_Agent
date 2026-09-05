"""Score a set of Coder Agent runs, and say whether a change made them better.

Every improvement to this agent so far was justified by a post-mortem: a Barkla
sweep fails, someone reads `fix_history` by hand, a repair gets written. That
finds real defects — most of `diagnose.py` and `repair.py` came from it — but it
cannot answer the question that actually matters before merging anything, which
is whether the change helped *overall*. A repair that fixes one failure mode and
costs two elsewhere reads identically in a post-mortem.

So: a fixed corpus of experiment plans (`benchmark_plans/`), run under a fixed
configuration, scored into numbers that can be compared against the same corpus
run before the change. `scripts/analyze_coder_fix_history.py` already aggregates
error_source frequency across whatever runs happen to exist; this is the other
half — a *held-out* set that does not change under you, and a comparison between
two runs of it.

The one metric to look at first is **interpretable**, not completed. An
experiment that runs to completion and has its verdict withheld — synthetic
inputs (provenance.py) or a run truncated to fit its budget
(compute_provenance.py) — produced no finding a paper can state. Optimising
`completed` alone is how you get a pipeline that always finishes and never
concludes anything.

Scoring is a pure function of `coder_agent_summary_*.json` files, deliberately:
it reads the same artefacts a production run writes, so it grades an ordinary
sweep's output directory as readily as a benchmark run's. Nothing here imports
the rest of the agent or calls a model.

A caution the report itself repeats: the corpus is small. With a dozen cases one
case is eight percentage points, so `format_comparison` prints counts beside
every rate and flags a delta that rests on a single case. A benchmark that
launders one flaky run into a confident percentage is worse than no benchmark.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PLANS_DIR = Path(__file__).resolve().parent / "benchmark_plans"
SUMMARY_GLOB = "**/coder_agent_summary_*.json"

# Statuses grouped by what they mean for a paper, which is not the same as what
# they mean for the agent. "deferred" covers both "we never ran it" and "the
# cluster has it" — from here neither has produced a number.
DEFERRED_STATUSES = frozenset({"code_generated_not_run", "submitted_to_slurm"})


@dataclass
class CaseOutcome:
    """One experiment's result, reduced to the fields worth comparing."""

    hypothesis_id: str
    status: str
    fix_attempts: int
    # Ran AND is allowed to carry a verdict. The headline number: an experiment
    # whose verdict was withheld contributed nothing a paper can state.
    interpretable: bool
    real_data: bool
    ran_at_full_size: bool
    # The failure still standing when the loop stopped, or "" — the same
    # "unresolved_at_end" question scripts/analyze_coder_fix_history.py asks,
    # which is the one that identifies what actually cost an experiment.
    unresolved_error: str
    starter_used: str
    source: str

    @property
    def completed(self) -> bool:
        return self.status == "completed"


@dataclass
class Score:
    cases: list[CaseOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def completed(self) -> int:
        return sum(1 for case in self.cases if case.completed)

    @property
    def interpretable(self) -> int:
        return sum(1 for case in self.cases if case.interpretable)

    @property
    def deferred(self) -> int:
        return sum(1 for case in self.cases if case.status in DEFERRED_STATUSES)

    @property
    def skipped(self) -> int:
        return sum(1 for case in self.cases if case.status == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for case in self.cases if case.status == "slurm_job_failed")

    @property
    def real_data(self) -> int:
        return sum(1 for case in self.cases if case.real_data)

    @property
    def truncated(self) -> int:
        """Completed experiments whose run was shrunk to fit its budget."""
        return sum(1 for case in self.cases if case.completed and not case.ran_at_full_size)

    @property
    def total_fix_attempts(self) -> int:
        return sum(case.fix_attempts for case in self.cases)

    @property
    def attempts_to_green(self) -> float | None:
        """Mean fix attempts spent on experiments that did complete.

        Restricted to the ones that made it on purpose: averaging in the
        experiments that exhausted the budget measures how often the budget is
        exhausted, which `completed` already says, and drags this number toward
        `max_fix_attempts` regardless of whether the successes got cheaper.
        """
        finished = [case.fix_attempts for case in self.cases if case.completed]
        return sum(finished) / len(finished) if finished else None

    @property
    def unresolved_errors(self) -> Counter[str]:
        return Counter(case.unresolved_error for case in self.cases if case.unresolved_error)

    def by_id(self) -> dict[str, CaseOutcome]:
        return {case.hypothesis_id: case for case in self.cases}


def _unresolved_error(experiment: dict) -> str:
    """The error_source still standing when the fix loop stopped, or "".

    The *last* history entry, and only when it was never resolved. An earlier
    failure that a regeneration cleared is not what cost this experiment, and
    counting it would rank the noisiest category above the one actually ending
    runs.
    """
    history = experiment.get("fix_history") or []
    if not history or experiment.get("status") == "completed":
        return ""
    last = history[-1]
    if not isinstance(last, dict) or last.get("resolved"):
        return ""
    return str(last.get("error_source") or "")


def outcome_for(experiment: dict, source: str = "") -> CaseOutcome:
    results = experiment.get("results") or {}
    meets = results.get("meets_success_criteria")
    status = str(experiment.get("status") or "")
    provenance = experiment.get("data_provenance") or {}
    compute = experiment.get("compute_provenance") or {}
    return CaseOutcome(
        hypothesis_id=str(experiment.get("hypothesis_id") or ""),
        status=status,
        fix_attempts=int(experiment.get("fix_attempts") or 0),
        # `is True/is False` rather than a truthiness test: the withheld verdict
        # is the *string* "unknown", which is truthy, and the whole point of
        # this field is telling those two apart.
        interpretable=status == "completed" and (meets is True or meets is False),
        real_data=bool(provenance.get("all_inputs_real")),
        # Absent for a summary written before compute provenance existed, and
        # for one that never executed anything. Treated as full size: this
        # metric exists to catch runs that were *shrunk*, and calling every old
        # summary truncated would make a comparison against one meaningless.
        ran_at_full_size=bool(compute.get("ran_at_full_size", True)),
        unresolved_error=_unresolved_error(experiment),
        starter_used=str(experiment.get("starter_used") or ""),
        source=source,
    )


def load_summaries(root: Path) -> list[Path]:
    """Every coder summary under `root` (a file is returned as-is).

    Sorted so a report's case order is stable between runs of the same
    directory — a diff of two reports should show what changed, not what got
    globbed in a different order.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.glob(SUMMARY_GLOB))


def score(root: Path) -> Score:
    """Score every experiment in every summary under `root`.

    A file that does not parse, or has no `experiments` list, is skipped with a
    warning rather than aborting: one interrupted run's half-written summary
    should not hide every other run's data — the same rule
    scripts/analyze_coder_fix_history.py follows, and for the same reason.
    """
    result = Score()
    for path in load_summaries(root):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        experiments = payload.get("experiments") if isinstance(payload, dict) else None
        if not isinstance(experiments, list):
            logger.warning("Skipping %s: no experiments list", path)
            continue
        for experiment in experiments:
            if isinstance(experiment, dict):
                result.cases.append(outcome_for(experiment, source=path.name))
    return result


@dataclass
class Change:
    """One case whose outcome differs between the two runs."""

    hypothesis_id: str
    before: str
    after: str
    direction: str  # "better" | "worse" | "changed"


# What counts as progress, worst to best. A case moving up this ladder improved
# even when both ends are failures — "the cluster job failed" is further than
# "we never generated anything runnable".
_OUTCOME_LADDER = [
    "skipped",
    "slurm_job_failed",
    "code_generated_not_run",
    "submitted_to_slurm",
    "completed",
    "interpretable",
]


def _rung(case: CaseOutcome) -> int:
    label = "interpretable" if case.interpretable else case.status
    return _OUTCOME_LADDER.index(label) if label in _OUTCOME_LADDER else 0


def compare(baseline: Score, candidate: Score) -> list[Change]:
    """Per-case outcome changes between two runs of the same corpus.

    The aggregate rates say whether things got better; this says *what* moved,
    which is the half that tells you where to look. Matched on hypothesis_id,
    since that is what a corpus case is identified by. A case present in only
    one run is reported rather than dropped — usually it means the corpus itself
    changed, which invalidates the comparison and should be visible, not silent.
    """
    before, after = baseline.by_id(), candidate.by_id()
    changes: list[Change] = []
    for hid in sorted(set(before) | set(after)):
        was, now = before.get(hid), after.get(hid)
        if was is None or now is None:
            changes.append(
                Change(
                    hid,
                    before="(absent)" if was is None else _label(was),
                    after="(absent)" if now is None else _label(now),
                    direction="changed",
                )
            )
            continue
        if _label(was) == _label(now):
            continue
        direction = "better" if _rung(now) > _rung(was) else "worse"
        changes.append(Change(hid, _label(was), _label(now), direction))
    return changes


def _label(case: CaseOutcome) -> str:
    return "interpretable" if case.interpretable else case.status


def _rate(count: int, total: int) -> str:
    return f"{count}/{total}" + (f" ({count / total:.0%})" if total else "")


def format_score(result: Score, title: str = "Coder benchmark") -> str:
    lines = [f"{title} — {result.total} experiment(s)", ""]
    if not result.total:
        return lines[0] + "\n(no coder summaries found)"

    lines += [
        f"  interpretable      {_rate(result.interpretable, result.total)}"
        "   <- the one that matters: ran AND carries a verdict",
        f"  completed          {_rate(result.completed, result.total)}",
        f"  deferred           {_rate(result.deferred, result.total)}",
        f"  failed on cluster  {_rate(result.failed, result.total)}",
        f"  skipped            {_rate(result.skipped, result.total)}",
        "",
        f"  real data          {_rate(result.real_data, result.total)}",
        f"  truncated to fit   {_rate(result.truncated, result.total)}",
        f"  fix attempts       {result.total_fix_attempts} total",
    ]
    mean = result.attempts_to_green
    if mean is not None:
        lines.append(
            f"  attempts to green  {mean:.2f} mean, over the {result.completed} that completed"
        )

    if result.unresolved_errors:
        lines += ["", "  still failing when the loop stopped:"]
        for source, count in result.unresolved_errors.most_common():
            lines.append(f"    {count:>3}  {source}")
    return "\n".join(lines)


def format_comparison(baseline: Score, candidate: Score) -> str:
    """A before/after report, with the honesty caveats printed rather than left
    to the reader to remember."""
    metrics = [
        ("interpretable", baseline.interpretable, candidate.interpretable),
        ("completed", baseline.completed, candidate.completed),
        ("real data", baseline.real_data, candidate.real_data),
        ("truncated to fit", baseline.truncated, candidate.truncated),
        ("fix attempts (total)", baseline.total_fix_attempts, candidate.total_fix_attempts),
    ]
    lines = [
        f"baseline: {baseline.total} experiment(s)   candidate: {candidate.total} experiment(s)",
        "",
    ]
    if baseline.total != candidate.total:
        lines.append(
            "  !! different numbers of experiments — these runs are not comparable as rates. "
            "Check the corpus is the same on both sides."
        )
        lines.append("")

    for name, before, after in metrics:
        delta = after - before
        arrow = "  " if delta == 0 else ("+" if delta > 0 else "")
        lines.append(f"  {name:<22} {before:>4} -> {after:>4}   {arrow}{delta if delta else ''}")

    if baseline.total:
        lines += [
            "",
            f"  One case is {1 / baseline.total:.0%} of this corpus. A delta of ±1 is one "
            "experiment, not a trend —",
            "  read the per-case changes below before concluding anything from the numbers above.",
        ]

    changes = compare(baseline, candidate)
    lines += ["", f"  per-case changes ({len(changes)}):"]
    if not changes:
        lines.append("    (none — every case landed the same way)")
    for change in changes:
        mark = {"better": "+", "worse": "-", "changed": "?"}[change.direction]
        lines.append(f"    {mark} {change.hypothesis_id}: {change.before} -> {change.after}")
    return "\n".join(lines)


def corpus_cases(plans_dir: Path | None = None) -> list[Path]:
    """The frozen plan files, in a stable order."""
    return sorted(Path(plans_dir or PLANS_DIR).glob("*.json"))
