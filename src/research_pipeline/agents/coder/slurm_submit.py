"""SLURM submission mechanics for generated experiment code. No LLM calls
here, and nothing imports this at module scope on the hot path — kept separate
from sandbox.py because sandbox.py must stay runnable anywhere, while these
commands only exist on a cluster login/compute node.

Submission is off by default (`CODER_AUTO_SUBMIT_SLURM`). When it is on, the
caller is responsible for checking the safety lint and the job caps first —
see CoderAgent._handle_unrunnable_locally.

`job_state` is the other half of that: submission is asynchronous, so the run
that submits a job cannot know how it ended. Asking `sacct` afterwards is what
lets `reconcile.py` turn a `submitted_to_slurm` experiment into a real result
instead of the permanent "no results are available" the Writer would otherwise
print.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SUBMIT_TIMEOUT_SECONDS = 30
QUEUE_TIMEOUT_SECONDS = 30
ACCOUNTING_TIMEOUT_SECONDS = 30

# A job in one of these will never change state again, so a reconcile pass can
# stop asking about it and record what happened. Anything else (PENDING,
# RUNNING, SUSPENDED, COMPLETING, REQUEUED) means "ask again later".
#
# PREEMPTED belongs here even though SLURM may requeue the job: on a requeue
# the *new* attempt gets its own accounting row, and `-X` below reports the
# latest one, so a requeued job reads as PENDING/RUNNING again rather than
# staying stuck on the preempted attempt. That matters on Barkla, whose free
# GPU partitions (gpu-a100-lowbig, gpu-a-lowsmall) preempt by design.
TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
    }
)

_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")

# Depth reported when the queue can't be read — larger than any sensible cap,
# so a probe failure blocks submission instead of waving it past.
_UNKNOWN_QUEUE_DEPTH = 10_000


def slurm_available() -> bool:
    return shutil.which("sbatch") is not None


def count_running_jobs(user: str | None = None) -> int:
    """How many jobs this user currently has queued or running, cluster-wide.

    Deliberately shells out rather than counting in-process: a batch run spans
    many separate processes (and many separate SLURM jobs), so an in-process
    counter would happily let each one submit its own full quota. Returns 0
    when squeue isn't present, so callers work unchanged off-cluster."""
    if shutil.which("squeue") is None:
        return 0

    user = user or os.environ.get("USER") or ""
    try:
        proc = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=QUEUE_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning(
            "Could not query squeue (%s) — treating the queue as full to stay on the safe side.",
            exc,
        )
        # A failed probe must not read as "nothing running", or the caps stop
        # capping exactly when the cluster is least responsive.
        return _UNKNOWN_QUEUE_DEPTH

    if proc.returncode != 0:
        logger.warning("squeue exited %d: %s", proc.returncode, (proc.stderr or "").strip())
        return _UNKNOWN_QUEUE_DEPTH

    return len([line for line in proc.stdout.splitlines() if line.strip()])


def submit_job(sbatch_path: Path, cwd: Path) -> tuple[str | None, str | None]:
    """Submits one job script. Returns (job_id, error) — exactly one is set."""
    if not slurm_available():
        return None, "sbatch is not on PATH — this doesn't look like a SLURM cluster"

    try:
        proc = subprocess.run(
            ["sbatch", str(sbatch_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=SUBMIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, f"sbatch did not return within {SUBMIT_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return None, f"could not run sbatch: {exc}"

    if proc.returncode != 0:
        return (
            None,
            f"sbatch exited with code {proc.returncode}: {(proc.stderr or proc.stdout or '').strip()[-500:]}",
        )

    match = _JOB_ID_RE.search(proc.stdout or "")
    if not match:
        return (
            None,
            f"could not parse a job id out of sbatch's output: {(proc.stdout or '').strip()[-200:]}",
        )

    return match.group(1), None


def accounting_available() -> bool:
    return shutil.which("sacct") is not None


def job_state(job_id: str) -> tuple[str | None, str | None]:
    """What SLURM's accounting database says became of one job. Returns
    (state, error) — exactly one is set.

    `-X` asks for the job allocation only. Without it sacct also returns a row
    per step (`<id>.batch`, `<id>.extern`), and those carry their own states —
    a `.batch` step reading COMPLETED under a job that was CANCELLED is exactly
    the kind of disagreement that would have a reconcile pass import results
    from a run that was killed.

    A state can carry a trailing qualifier ("CANCELLED by 123456"), so only the
    first word is returned; TERMINAL_STATES is written against that form.

    Two different "don't know"s, both reported as an error rather than a state,
    because the caller's only safe response to either is to leave the
    experiment as it was and ask again later:
      - sacct isn't installed (we're not on a cluster, or not on a node that
        can see the accounting database);
      - the job has no accounting row at all, which means either the id is
        wrong or SLURM's accounting retention has already purged it.
    """
    if not accounting_available():
        return None, "sacct is not on PATH — cannot ask what became of this job"

    try:
        proc = subprocess.run(
            ["sacct", "-j", str(job_id), "-X", "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
            timeout=ACCOUNTING_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, f"sacct did not return within {ACCOUNTING_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return None, f"could not run sacct: {exc}"

    if proc.returncode != 0:
        return (
            None,
            f"sacct exited with code {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()[-500:]}",
        )

    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None, f"SLURM has no accounting record for job {job_id}"

    return lines[0].split()[0], None


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES
