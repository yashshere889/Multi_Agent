"""Run directories, their `run.json`, and launching the runner subprocess.

A run is a directory and nothing else. There is no database and no in-memory
registry, so the server can be restarted — or replaced by a fresh one at the
other end of a new SSH tunnel — without losing track of anything in flight.

Layout, everything a run touches under one rsync-able tree:

    <runs_dir>/<run_id>/
        run.json        question, params, status, pid, timings, final result
        events.jsonl    the progress + log stream (events.py)
        stdout.log      the runner's own stdout/stderr
        outputs/        passed to the graph as output_dir: vN.pdf, review_log.json, ...
        papers/         passed to the graph as download_dir
        experiments/    via CODER_EXPERIMENTS_DIR in the child's environment
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from research_pipeline.config import settings
from research_pipeline.webapp import events
from research_pipeline.webapp.events import utc_now

logger = logging.getLogger(__name__)

RUN_FILE = "run.json"
STDOUT_LOG = "stdout.log"

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL_STATUSES = frozenset({COMPLETED, FAILED, CANCELLED})

# Runs worth offering a resume for: they stopped without a result, so there may
# be checkpointed progress to pick up. COMPLETED is excluded because there is
# nothing left to run, and the two non-terminal statuses because the run is
# already going — relaunching one would put a second process on the same run
# directory and the same thread_id, interleaving their events and their writes.
RESUMABLE_STATUSES = frozenset({FAILED, CANCELLED})


class RunNotFound(LookupError):
    """Raised for an unknown run id, and for anything that isn't a run id at
    all — the routes map both to a 404."""


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, just owned by someone else.
        return True
    return True


class RunStore:
    """Every read and write of run state goes through here, so nothing else
    needs to know the directory layout or hold a lock."""

    def __init__(self, runs_dir: Optional[str | Path] = None) -> None:
        self.runs_dir = Path(runs_dir or settings.webapp_runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        # Runners this process launched, kept only so they can be waited on —
        # see _process_alive. Runs launched by an earlier server aren't in here
        # and don't need to be: once that server exited, its children were
        # reparented to init, which reaps them.
        self._children: dict[str, subprocess.Popen] = {}

    # --- paths -----------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        """Validating the id as a UUID is the whole path-traversal defense:
        anything that survives `uuid.UUID()` is 36 hex-and-dash characters and
        cannot escape runs_dir, so no separate sanitization is needed."""
        try:
            parsed = uuid.UUID(str(run_id))
        except (ValueError, AttributeError, TypeError):
            raise RunNotFound(f"not a run id: {run_id!r}") from None
        return self.runs_dir / str(parsed)

    def resolve_inside(self, run_id: str, path: str | Path) -> Path:
        """Resolve a path that a run claims to have produced, refusing anything
        outside its own directory. Paths in `paper_path` and friends come from
        agent output, so they're checked rather than trusted before the server
        hands a file to a browser."""
        run_dir = self.run_dir(run_id).resolve()
        candidate = Path(path)
        candidate = (run_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if run_dir != candidate and run_dir not in candidate.parents:
            raise RunNotFound(f"{path} is outside run {run_id}")
        return candidate

    # --- reading ---------------------------------------------------------

    def get(self, run_id: str) -> dict:
        run_dir = self.run_dir(run_id)
        path = run_dir / RUN_FILE
        if not path.exists():
            raise RunNotFound(run_id)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunNotFound(f"unreadable run.json for {run_id}: {exc}") from exc
        return self._reconcile(record)

    def list_runs(self) -> list[dict]:
        records = []
        for child in self.runs_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                records.append(self.get(child.name))
            except RunNotFound:
                continue
        return sorted(records, key=lambda r: r.get("created_at") or "", reverse=True)

    def active_count(self) -> int:
        return sum(1 for r in self.list_runs() if r.get("status") in (PENDING, RUNNING))

    def _process_alive(self, record: dict) -> bool:
        """Whether this run's runner is still going — and, as a side effect,
        the only thing that reaps it when it isn't.

        A child of a still-running parent that nobody waits on becomes a
        zombie, and `os.kill(pid, 0)` *succeeds* for a zombie. Relying on that
        alone would report every finished runner as still alive, which is
        exactly the case this whole reconciliation exists to catch. Popen.poll()
        does the wait, so the entry leaves the process table for good.
        """
        run_id = record["run_id"]
        child = self._children.get(run_id)
        if child is None:
            return _pid_alive(record.get("pid"))
        if child.poll() is None:
            return True
        del self._children[run_id]
        return False

    def _reconcile(self, record: dict) -> dict:
        """Recover the status of a run whose process died without reporting.

        A runner that is SIGKILLed, or whose node is pre-empted, never gets to
        write its own terminal status — without this it would show as running
        forever. The events file is trusted first (the runner appends there
        before updating run.json), and only then process liveness.
        """
        # Unconditional, and before the early return: a cancelled or completed
        # run still has a child to collect.
        alive = self._process_alive(record)
        if record.get("status") not in (PENDING, RUNNING):
            return record

        run_id = record["run_id"]
        ended = events.terminal_event(self.run_dir(run_id))
        if ended:
            status = {
                events.RUN_COMPLETED: COMPLETED,
                events.RUN_FAILED: FAILED,
                events.RUN_CANCELLED: CANCELLED,
            }[ended["type"]]
            return self.update(
                run_id,
                status=status,
                finished_at=record.get("finished_at") or ended.get("ts"),
                error=record.get("error") or ended.get("error"),
                final_result=record.get("final_result") or ended.get("final_result"),
            )

        # Only meaningful once a process has been spawned, and only on this
        # machine. For a run inherited from an earlier server this falls back to
        # a bare pid probe, where pid reuse could in principle keep a dead run
        # looking alive; the consequence is a stale row, not a wrong result.
        if record.get("status") == RUNNING and not alive:
            return self.update(
                run_id,
                status=FAILED,
                error="The runner process exited without reporting a result (killed, out of memory, or pre-empted).",
                finished_at=utc_now(),
            )
        return record

    # --- writing ---------------------------------------------------------

    def update(self, run_id: str, **fields) -> dict:
        """Merge fields into run.json under an exclusive lock. Locked because
        the server and the runner subprocess both write it — the server for
        cancellation, the runner for progress."""
        path = self.run_dir(run_id) / RUN_FILE
        if not path.exists():
            raise RunNotFound(run_id)

        with open(path, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                record = json.loads(handle.read())
                record.update(fields)
                handle.seek(0)
                handle.truncate()
                json.dump(record, handle, indent=2, ensure_ascii=False, default=str)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return record

    def create(self, question: str, params: Optional[dict] = None) -> dict:
        run_id = str(uuid.uuid4())
        run_dir = self.runs_dir / run_id
        for sub in ("outputs", "papers", "experiments"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        record = {
            "run_id": run_id,
            "question": question,
            "params": params or {},
            "status": PENDING,
            "pid": None,
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "final_result": None,
            "error": None,
        }
        (run_dir / RUN_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Created run %s: %s", run_id, question)
        return record

    # --- process control -------------------------------------------------

    def launch(self, run_id: str) -> dict:
        """Spawn the runner for a created run.

        `start_new_session=True` puts the runner in its own process group, so a
        Ctrl-C on the server — or the server exiting — doesn't take running
        pipelines down with it. That's the same property that makes the server
        restartable mid-run.
        """
        run_dir = self.run_dir(run_id)
        if not (run_dir / RUN_FILE).exists():
            raise RunNotFound(run_id)

        env = dict(os.environ)
        # Settings are read once at import, so per-run output routing can only
        # happen per process. This is the one output location the graph can't
        # pass through as an argument, and without it two concurrent runs would
        # overwrite each other's generated code: hypothesis ids are always
        # H1/H2/H3, so both would write experiments/H1/.
        env["CODER_EXPERIMENTS_DIR"] = str((run_dir / "experiments").resolve())
        env["PYTHONUNBUFFERED"] = "1"

        log_handle = open(run_dir / STDOUT_LOG, "ab")
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "research_pipeline.webapp.runner", str(run_dir)],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
        finally:
            log_handle.close()

        self._children[run_id] = process
        logger.info("Launched runner for %s as pid %d", run_id, process.pid)
        return self.update(
            run_id,
            status=RUNNING,
            pid=process.pid,
            started_at=utc_now(),
            # Cleared because this run is in flight again. A no-op for a freshly
            # created run, and what stops a resumed one from showing the failure
            # it was resumed *from* next to a running badge.
            error=None,
            finished_at=None,
        )

    def cancel(self, run_id: str) -> dict:
        """SIGTERM the runner. The runner installs a handler that records a
        `run_cancelled` event, so a cancelled run is distinguishable from a
        crashed one; if it's already gone, just record the status."""
        record = self.get(run_id)
        if record.get("status") in TERMINAL_STATUSES:
            return record

        pid = record.get("pid")
        if self._process_alive(record):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as exc:
                logger.warning("Could not signal pid %s for run %s: %s", pid, run_id, exc)

        return self.update(run_id, status=CANCELLED, finished_at=utc_now(), error="Cancelled by the user.")
