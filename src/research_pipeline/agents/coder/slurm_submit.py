"""SLURM submission mechanics for generated experiment code. No LLM calls
here, and nothing imports this at module scope on the hot path — kept separate
from sandbox.py because sandbox.py must stay runnable anywhere, while these
commands only exist on a cluster login/compute node.

Submission is off by default (`CODER_AUTO_SUBMIT_SLURM`). When it is on, the
caller is responsible for checking the safety lint and the job caps first —
see CoderAgent._handle_unrunnable_locally.
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
