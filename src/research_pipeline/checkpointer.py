"""Factory for the shared LangGraph checkpointer (and the shared node cache).

The same "one place" idea as llm.get_chat_model(): every one of the eight
compiled graphs — the seven agents plus the top-level orchestrator — calls
get_checkpointer() and gets back a BaseCheckpointSaver. No graph.py constructs
a saver of its own, so switching the whole pipeline from in-process
checkpointing to a durable file or database is one environment variable.

Why this exists at all: every graph used to hardcode `MemorySaver()`, which
records a run's progress node by node and then loses all of it when the process
ends. That is exactly the wrong shape for how this pipeline actually runs — a
pre-empted SLURM job on Barkla, a restarted Kaggle kernel, a crashed webapp
runner — and webapp/runner.py already threads a *stable* run_id through as its
thread_id, implying resumability that an empty-on-startup checkpointer could
never deliver.

Backends (CHECKPOINTER_BACKEND):
- "memory"   — MemorySaver. The default, and byte-for-byte the old behaviour.
- "sqlite"   — one file at CHECKPOINTER_SQLITE_PATH. No server, no setup.
- "postgres" — CHECKPOINTER_POSTGRES_URI, for a shared or multi-host setup.

Both non-default backends live behind an optional dependency, so a plain
`uv sync` install keeps working exactly as before and only pays for what it
uses (see pyproject.toml's checkpoint-sqlite / checkpoint-postgres extras).

Note what this module does *not* do: making storage durable makes resuming
*possible*, but nothing here resumes anything. Callers still invoke with fresh
input rather than `None`; teaching webapp/runner.py and batch.py to detect "this
thread_id already has checkpoints, continue it" is a separate behavioural
change.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import BaseCheckpointSaver

from research_pipeline.config import settings

logger = logging.getLogger(__name__)

# Built lazily on first use and reused for the rest of the process. Agent graphs
# are rebuilt per call (hypothesis_agent.py and reviewer_agent.py both call
# build_*_graph() inside .run()), so without memoization a sqlite/postgres
# backend would open a fresh connection on every agent invocation and leak them
# for the life of the run.
_checkpointer: BaseCheckpointSaver | None = None
_node_cache: BaseCache | None = None


def _missing_extra(backend: str, extra: str, exc: ImportError) -> SystemExit:
    """Same shape as cli.run_serve_cli's optional-dependency error: name the
    missing module and the exact command that installs it, rather than letting
    a bare ImportError surface from four stack frames down inside a graph
    build."""
    return SystemExit(
        f"CHECKPOINTER_BACKEND={backend} needs its optional dependencies ({exc.name}). Install them with:\n"
        f"    uv sync --extra {extra}"
    )


def _sqlite_saver() -> BaseCheckpointSaver:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise _missing_extra("sqlite", "checkpoint-sqlite", exc) from exc

    path = Path(settings.checkpointer_sqlite_path)
    # "checkpoints/pipeline.db" is the default, and nothing else creates that
    # directory — a run shouldn't fail on a missing parent it never asked for.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Deliberately not SqliteSaver.from_conn_string(...): that is a context
    # manager that closes the connection on __exit__, and this saver has to
    # outlive the call that built it — every later build_*_graph() reuses it.
    # check_same_thread=False because LangGraph runs nodes on worker threads
    # (every Send fan-out in this pipeline does), so the connection is used
    # from a different thread than the one that opened it.
    conn = sqlite3.connect(path, check_same_thread=False)
    logger.info("Checkpointing to sqlite at %s", path)
    return SqliteSaver(conn)


def _postgres_saver() -> BaseCheckpointSaver:
    # Checked before the import so a missing URI reports the missing URI, rather
    # than whichever of the two problems happens to be discovered first.
    uri = settings.checkpointer_postgres_uri
    if not uri:
        raise SystemExit(
            "CHECKPOINTER_BACKEND=postgres needs CHECKPOINTER_POSTGRES_URI to be set "
            "(e.g. postgresql://user:pass@host:5432/dbname)."
        )

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
    except ImportError as exc:
        raise _missing_extra("postgres", "checkpoint-postgres", exc) from exc

    # autocommit/prepare_threshold are what PostgresSaver.from_conn_string sets
    # up for you; we open the connection ourselves for the same reason as
    # sqlite above — from_conn_string closes it on context exit.
    conn = Connection.connect(uri, autocommit=True, prepare_threshold=0)
    saver = PostgresSaver(conn)
    # Unlike SqliteSaver, this one does not create its tables on demand.
    saver.setup()
    logger.info("Checkpointing to postgres")
    return saver


def get_checkpointer() -> BaseCheckpointSaver:
    """The one place a checkpointer is constructed. Returns a process-wide
    singleton, so all eight graphs share one connection — sharing is safe
    because checkpoints are keyed by (thread_id, checkpoint_ns) and every call
    site already mints its own thread_id (a fresh uuid4 per agent call, the
    run_id for the orchestrator)."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    backend = settings.checkpointer_backend.strip().lower()
    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        _checkpointer = MemorySaver()
    elif backend == "sqlite":
        _checkpointer = _sqlite_saver()
    elif backend == "postgres":
        _checkpointer = _postgres_saver()
    else:
        raise SystemExit(
            f"Unknown CHECKPOINTER_BACKEND={settings.checkpointer_backend!r}. "
            "Expected one of: memory, sqlite, postgres."
        )
    return _checkpointer


def get_node_cache() -> BaseCache:
    """The shared node cache backing CachePolicy on the paper-search nodes.
    A singleton for the same reason get_checkpointer() is one: graphs are
    rebuilt per call, and a fresh cache per build would never register a hit.

    InMemoryCache is the only cache backend OSS LangGraph ships, so this is
    per-process by construction — worth real time inside a long-lived process
    (the web app across several runs, an orchestrate-batch sweep across a
    question list) and worth exactly nothing to a one-shot CLI invocation."""
    global _node_cache
    if _node_cache is None:
        from langgraph.cache.memory import InMemoryCache

        _node_cache = InMemoryCache()
    return _node_cache


def reset_checkpointer() -> None:
    """Drops the memoized singletons so the next call rebuilds from current
    settings. Exists for tests (which need a genuinely fresh saver to prove a
    sqlite file outlives the connection that wrote it); nothing in the running
    pipeline changes backends mid-process."""
    global _checkpointer, _node_cache
    _checkpointer = None
    _node_cache = None
