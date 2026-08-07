"""The append-only event stream a run writes as it goes.

One JSON object per line in `events.jsonl`, which is the whole interface
between the runner subprocess and the web server: the runner only appends, the
server only reads. A plain file rather than a queue or a socket because it
survives the server restarting, an SSH tunnel dropping, and the runner being
killed — all of which happen on a cluster.

Appends take an exclusive lock and reads take a shared one, the same
`fcntl.flock` discipline `batch.py` uses on its manifest, so a poll landing
mid-append never sees a half-written line.
"""

from __future__ import annotations

import fcntl
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

EVENTS_NAME = "events.jsonl"

RUN_STARTED = "run_started"
STAGE_COMPLETED = "stage_completed"
LOG = "log"
RUN_COMPLETED = "run_completed"
RUN_FAILED = "run_failed"
RUN_CANCELLED = "run_cancelled"

TERMINAL_EVENTS = frozenset({RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED})

# Long enough for a traceback tail, short enough that one pathological log line
# can't turn the events file into something the browser chokes on. Matches the
# cap the Coder Agent already puts on its fix_history error summaries.
MAX_MESSAGE_CHARS = 2000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLog:
    """Writer side. One instance per process — `seq` is kept in memory after an
    initial count, so a run with thousands of log lines doesn't re-read the
    whole file on every append."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / EVENTS_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = sum(1 for _ in read_events(run_dir))

    def append(self, event_type: str, **fields) -> dict:
        event = {"seq": self._seq, "ts": utc_now(), "type": event_type, **fields}
        self._seq += 1

        line = json.dumps(event, ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(line + "\n")
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return event

    def log(self, level: str, logger_name: str, message: str) -> dict:
        return self.append(LOG, level=level, logger=logger_name, message=message[:MAX_MESSAGE_CHARS])


def read_events(
    run_dir: str | Path,
    since: int = 0,
    types: Optional[Iterable[str]] = None,
    tail: Optional[int] = None,
) -> list[dict]:
    """Reader side. `since` is an exclusive cursor on `seq`; `tail` keeps only
    the last N matching events. A malformed line is skipped rather than raising
    — a truncated events file (a runner killed mid-write, a filesystem that
    dropped the tail) should still render every event that made it."""
    path = Path(run_dir) / EVENTS_NAME
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            raw_lines = handle.readlines()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    wanted = set(types) if types is not None else None
    events = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Skipping a malformed line in %s", path)
            continue
        if event.get("seq", 0) < since:
            continue
        if wanted is not None and event.get("type") not in wanted:
            continue
        events.append(event)

    return events[-tail:] if tail else events


def terminal_event(run_dir: str | Path) -> Optional[dict]:
    """The event that ended the run, or None while it's still going."""
    ended = read_events(run_dir, types=TERMINAL_EVENTS)
    return ended[-1] if ended else None
