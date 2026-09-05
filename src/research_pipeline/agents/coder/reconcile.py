"""Find out what became of the SLURM jobs a previous run submitted, and fold
their results back into that run's summary.

Submission is asynchronous and the pipeline is not. A plan that needs a GPU
this process doesn't have, or that is `high` complexity, is written out as
`run.sbatch` and — under `CODER_AUTO_SUBMIT_SLURM` — submitted; the run then
records `submitted_to_slurm` and moves on, because the job will still be
queued long after the process exits. Nothing ever went back to look. The
Writer's own status table says it plainly: "no results are available". So
every experiment big enough to need the cluster was, by construction, one the
paper could never report.

This is the missing half. It reads a `coder_agent_summary_*.json`, asks
`sacct` what happened to each job it recorded, and for the ones that finished
cleanly imports the `results.json` the job left in its experiment directory —
turning `submitted_to_slurm` into `completed`, with the same two verdict gates
a locally-executed experiment goes through.

It is a *pass*, not a wait. Blocking the pipeline on a queued job would tie a
multi-hour cluster wait to a login session, and a batch sweep would spend most
of its wall time asleep. Running separately means the reconcile can happen
tomorrow, from a different process, as many times as you like: an experiment
this pass can't resolve is left exactly as it was, so re-running it is always
safe and always cheap.

Four outcomes, and only two of them change anything:

    ACTION_COMPLETED  the job finished and its results.json is readable —
                      status becomes "completed" and the metrics are imported.
    ACTION_FAILED     the job reached a terminal state that isn't COMPLETED,
                      or finished without leaving usable results — status
                      becomes "slurm_job_failed", which the Writer reads as
                      inconclusive, same as every other non-completed status.
    ACTION_PENDING    still queued or running. Unchanged; ask again later.
    ACTION_UNKNOWN    we couldn't find out — no sacct on this machine, no
                      accounting record left, or the experiment directory
                      isn't visible from here. Unchanged, deliberately: see
                      `_import_results` for why "I can't see it" must not be
                      allowed to read as "the job failed".

Injects `state_lookup` for the same reason `CoderAgent` injects
`network_check`/`gpu_check` — it is the one dependency on actually being on a
cluster, and tests pass a dict lookup instead. Reads no settings; the caller
resolves paths and does the file I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import compute_provenance, provenance, sandbox, slurm_submit

logger = logging.getLogger(__name__)

STATUS_SUBMITTED = "submitted_to_slurm"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "slurm_job_failed"

ACTION_COMPLETED = "completed"
ACTION_FAILED = "failed"
ACTION_PENDING = "pending"
ACTION_UNKNOWN = "unknown"

StateLookup = Callable[[str], tuple[str | None, str | None]]


@dataclass
class Outcome:
    """What this pass decided about one experiment, whether or not it changed
    anything. Returned for every `submitted_to_slurm` entry so a caller can
    report "3 still running" as readily as "1 imported"."""

    hypothesis_id: str
    job_id: str
    action: str
    slurm_state: str | None
    detail: str

    @property
    def changed(self) -> bool:
        return self.action in (ACTION_COMPLETED, ACTION_FAILED)


def _import_results(experiment: dict, state: str) -> tuple[dict | None, Outcome | None]:
    """Read the results a finished job left behind, or explain why not.

    The distinction this function exists to draw: an experiment directory that
    is *missing* is not a failed job. `code_path` is an absolute path, often on
    cluster scratch, and a reconcile run from a laptop — or from a node that
    doesn't mount that filesystem — would find nothing there for a job that
    succeeded perfectly well. Marking that failed would be a lie that also
    destroys the record, since a later pass on the right machine would find the
    status no longer says `submitted_to_slurm` and skip it. So a directory we
    cannot see is ACTION_UNKNOWN and stays submitted; only a directory we *can*
    see, with no usable results in it, is a failure.
    """
    hypothesis_id = experiment.get("hypothesis_id", "")
    job_id = str(experiment.get("slurm_job_id") or "")
    code_path = experiment.get("code_path")

    if not code_path or not Path(code_path).is_dir():
        return None, Outcome(
            hypothesis_id,
            job_id,
            ACTION_UNKNOWN,
            state,
            f"job {job_id} reported {state}, but its experiment directory "
            f"({code_path!r}) is not visible from here — left as it was; "
            "re-run this from a machine that can see it.",
        )

    results, diagnosis = sandbox.read_results_json_for_diagnosis(Path(code_path))
    if results is None:
        return None, Outcome(
            hypothesis_id,
            job_id,
            ACTION_FAILED,
            state,
            f"job {job_id} reported {state} but left no usable results: {diagnosis}",
        )

    return results, None


def reconcile_experiment(
    experiment: dict, state_lookup: StateLookup
) -> tuple[dict, Outcome | None]:
    """Returns (experiment, outcome). `outcome` is None for an entry this pass
    has nothing to say about — anything that isn't a `submitted_to_slurm` entry
    carrying a job id. The experiment dict is returned unchanged unless the
    outcome `changed`."""
    if experiment.get("status") != STATUS_SUBMITTED:
        return experiment, None

    job_id = str(experiment.get("slurm_job_id") or "")
    hypothesis_id = experiment.get("hypothesis_id", "")
    if not job_id:
        return experiment, None

    state, error = state_lookup(job_id)
    if state is None:
        return experiment, Outcome(hypothesis_id, job_id, ACTION_UNKNOWN, None, error or "")

    if not slurm_submit.is_terminal(state):
        return experiment, Outcome(
            hypothesis_id, job_id, ACTION_PENDING, state, f"job {job_id} is {state}"
        )

    if state != "COMPLETED":
        updated = dict(experiment)
        updated["status"] = STATUS_FAILED
        updated["reason"] = (
            f"SLURM job {job_id} ended in state {state} — no results were produced. "
            f"Its log is beside the code in {experiment.get('code_path')}."
        )
        return updated, Outcome(
            hypothesis_id, job_id, ACTION_FAILED, state, f"job {job_id} ended {state}"
        )

    results, failure = _import_results(experiment, state)
    if results is None:
        assert failure is not None
        if failure.action == ACTION_FAILED:
            updated = dict(experiment)
            updated["status"] = STATUS_FAILED
            updated["reason"] = failure.detail
            return updated, failure
        return experiment, failure

    # The same two gates a locally-executed experiment goes through, in the
    # same order. The data one reads the document the submitting run already
    # wrote rather than re-resolving inputs here, where the staging directory
    # may not even be mounted — an absent document withholds, which is the safe
    # direction and what an old summary predating provenance gets.
    document = experiment.get("data_provenance") or {}
    results = provenance.apply_document_to_results(results, document)
    # Nothing downscaled this run: `repair.downscale` only ever rewrites code
    # in the local execution loop, and the cluster job ran whatever run.py was
    # submitted, under sbatch's own --time rather than a coder timeout. A job
    # that *did* exceed that limit is TIMEOUT above, a failure rather than a
    # truncated success, so "ran at full size" is the honest record here.
    compute_document = compute_provenance.as_document([])
    results = compute_provenance.apply_to_results(results, [])

    updated = dict(experiment)
    updated["status"] = STATUS_COMPLETED
    updated["reason"] = ""
    updated["results"] = results
    updated["compute_provenance"] = compute_document
    return updated, Outcome(
        hypothesis_id,
        job_id,
        ACTION_COMPLETED,
        state,
        f"job {job_id} completed — imported results from {experiment.get('code_path')}",
    )


def reconcile_summary(
    summary: dict, state_lookup: StateLookup | None = None
) -> tuple[dict, list[Outcome]]:
    """Reconcile every submitted job in one coder summary. Returns (summary,
    outcomes) — a new summary dict, and one Outcome per submitted experiment.

    The returned summary is a fresh object even when nothing changed, so a
    caller deciding whether to write is looking at `outcomes`, not at object
    identity."""
    lookup = state_lookup or slurm_submit.job_state

    experiments = summary.get("experiments") or []
    reconciled: list[dict] = []
    outcomes: list[Outcome] = []

    for experiment in experiments:
        if not isinstance(experiment, dict):
            reconciled.append(experiment)
            continue
        updated, outcome = reconcile_experiment(experiment, lookup)
        reconciled.append(updated)
        if outcome is not None:
            outcomes.append(outcome)
            logger.info("[%s] %s", outcome.hypothesis_id, outcome.detail)

    updated_summary = dict(summary)
    updated_summary["experiments"] = reconciled
    return updated_summary, outcomes
