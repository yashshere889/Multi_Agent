"""A cross-run library of fixes that actually worked, backed by a LangGraph
`Store` (cross-thread/long-term memory — distinct from the checkpointer, which
is thread-scoped and exists for resumability, not learning). Extends
`starters.py`'s idea — ground codegen in a real worked example instead of
asking the model to invent structure from scratch — with examples this
pipeline discovers itself, instead of ones hand-authored once and never
updated.

When `_node_attempt` (coder_agent.py) sees a fix_history entry's `resolved`
flip to True — the regeneration that followed a given error_source actually
got past it — `record_fix` persists whichever code sections changed between
the broken attempt and the one that resolved it, keyed by `error_source`.
`_regenerate_with_fix`'s prompt then calls `recall_fixes` for the *current*
error_source and shows the model its own past successes on that exact failure
category, via `CoderAgent._fix_pattern_block`.

Why this needs its own factory, deliberately not reusing checkpointer.py's
`get_checkpointer()`: a checkpointer's whole point is resumability *within* a
run that might crash, so "memory" (lost when the process exits) is a
reasonable default — most pipeline runs are one-shot CLI invocations that live
and die in a single process anyway. A fix-pattern library's *entire* value is
accumulating across many separate process invocations over weeks — an
in-memory default would make it silently do nothing for the pipeline's most
common usage pattern. CODER_FIX_STORE_BACKEND therefore defaults to "sqlite",
not "memory"; "memory" is still available as an explicit, informed opt-out
(tests, ephemeral/CI runs, or a user who doesn't want disk writes at all).

Two things this module deliberately does NOT do, both to stay within what's
actually needed rather than what a "memory system" could hypothetically grow
into:
- No embedding/semantic search. Retrieval is exact: `error_source` is already
  the filter (it's a closed, deterministic taxonomy — VALID_ERROR_SOURCES in
  schema.py), so ranking by embedding similarity would add a dependency and a
  cost for no real gain over the namespace lookup Store already does for free.
- No pruning of old/superseded patterns. A store that only ever grows is a
  real long-term concern (disk space, a slowly staler pool of examples), but
  building a retention policy now, before there's any evidence of it mattering
  in practice, is exactly the kind of speculative feature this codebase avoids
  elsewhere. `recall_fixes` sorts by recency and only ever shows a handful, so
  an unbounded store degrades toward "slightly stale examples ignored in
  favor of newer ones" rather than breaking anything.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langgraph.store.base import BaseStore

from research_pipeline.config import settings

logger = logging.getLogger(__name__)

NAMESPACE_PREFIX = "coder_fix_patterns"

# Real generated functions can run long; capped so one unusually large section
# can't make a fix-pattern prompt block balloon the way an uncapped one would.
# Matches ERROR_SUMMARY_MAX_CHARS's role in schema.py — same idea, different
# payload (code, not an error message).
MAX_PATTERN_CODE_CHARS = 4000

# How many candidates recall_fixes fetches before sorting by recency and
# slicing to the caller's limit — see recall_fixes' docstring for why this
# can't just be store.search(..., limit=requested_limit).
_RECALL_FETCH_LIMIT = 20

_store: BaseStore | None = None


def _truncate_code(text: str) -> str:
    if len(text) <= MAX_PATTERN_CODE_CHARS:
        return text
    return text[:MAX_PATTERN_CODE_CHARS] + "\n# ... truncated ..."


def _missing_extra(backend: str, extra: str, exc: ImportError) -> SystemExit:
    """Same shape as checkpointer.py's _missing_extra — name the missing
    module and the exact command that installs it."""
    return SystemExit(
        f"CODER_FIX_STORE_BACKEND={backend} needs its optional dependencies ({exc.name}). Install them with:\n"
        f"    uv sync --extra {extra}"
    )


def _sqlite_store() -> BaseStore:
    try:
        from langgraph.store.sqlite import SqliteStore
    except ImportError as exc:
        # Bundled with langgraph-checkpoint-sqlite, the same package
        # CHECKPOINTER_BACKEND=sqlite already depends on — one extra covers
        # both the checkpoint saver and this store.
        raise _missing_extra("sqlite", "checkpoint-sqlite", exc) from exc

    path = Path(settings.coder_fix_store_sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately not SqliteStore.from_conn_string(...): that's a
    # @contextmanager-wrapped generator, and with no `with` block holding it
    # open, the returned generator is garbage-collected almost immediately —
    # which runs its cleanup and closes the connection out from under the
    # store on the very next call. Opening the connection directly and handing
    # it to the constructor is exactly checkpointer._sqlite_saver's fix for
    # the identical problem with SqliteSaver.from_conn_string.
    # isolation_level=None (autocommit): SqliteStore manages its own
    # transactions via BEGIN/commit per batch, which conflicts with sqlite3's
    # own default implicit-transaction handling otherwise ("cannot start a
    # transaction within a transaction" on the very first write).
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()
    logger.info("Storing fix patterns in sqlite at %s", path)
    return store


def _postgres_store() -> BaseStore:
    uri = settings.coder_fix_store_postgres_uri
    if not uri:
        raise SystemExit(
            "CODER_FIX_STORE_BACKEND=postgres needs CODER_FIX_STORE_POSTGRES_URI to be set "
            "(e.g. postgresql://user:pass@host:5432/dbname)."
        )
    try:
        from langgraph.store.postgres import PostgresStore
        from psycopg import Connection
    except ImportError as exc:
        # Same package family as checkpointer.py's postgres saver — bundled
        # with langgraph-checkpoint-postgres.
        raise _missing_extra("postgres", "checkpoint-postgres", exc) from exc

    # Same reasoning as _sqlite_store above: open the connection directly
    # rather than through the from_conn_string context manager, so it isn't
    # closed by garbage collection the moment this function returns.
    conn = Connection.connect(uri, autocommit=True, prepare_threshold=0)
    store = PostgresStore(conn)
    store.setup()
    logger.info("Storing fix patterns in postgres")
    return store


def _backend() -> str:
    return settings.coder_fix_store_backend.strip().lower()


def get_store() -> BaseStore:
    """The one place this store is constructed. A process-wide singleton for
    the same reason get_checkpointer() is one — nothing here is rebuilt per
    call, so there's no per-build connection to leak."""
    global _store
    if _store is not None:
        return _store

    backend = _backend()
    if backend == "memory":
        from langgraph.store.memory import InMemoryStore

        _store = InMemoryStore()
    elif backend == "sqlite":
        _store = _sqlite_store()
    elif backend == "postgres":
        _store = _postgres_store()
    else:
        raise SystemExit(
            f"Unknown CODER_FIX_STORE_BACKEND={settings.coder_fix_store_backend!r}. "
            "Expected one of: memory, sqlite, postgres."
        )
    return _store


def reset_store() -> None:
    """Drops the memoized singleton so the next call rebuilds from current
    settings. For tests — nothing in the running pipeline changes backends
    mid-process."""
    global _store
    _store = None


def record_fix(
    store: BaseStore,
    *,
    error_source: str,
    error_summary: str,
    broken_sections: dict[str, str],
    fixed_sections: dict[str, str],
) -> None:
    """Persists whichever sections actually changed between a failing attempt
    and the regeneration that got past its error — nothing is written if
    nothing meaningfully changed (a regeneration that "fixed" `error_source`
    only because a later section it never touched happened to differ, or one
    that came back byte-identical, teaches nothing worth recalling).

    One Store item per resolved fix, not one per changed section: a fix is a
    single coherent event ("this is what changed to get past this error"),
    and _fix_pattern_block wants to show it back to the model that way rather
    than as unrelated fragments.
    """
    changed: dict[str, dict[str, str]] = {}
    for name, fixed in fixed_sections.items():
        if not fixed.strip():
            continue
        broken = broken_sections.get(name, "")
        if fixed == broken:
            continue
        changed[name] = {"before": _truncate_code(broken), "after": _truncate_code(fixed)}
    if not changed:
        return

    key = str(uuid.uuid4())
    store.put(
        (NAMESPACE_PREFIX, error_source),
        key,
        {
            "error_source": error_source,
            "error_summary": error_summary,
            "changed_sections": changed,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def recall_fixes(store: BaseStore, error_source: str, limit: int = 2) -> list[dict]:
    """Up to `limit` past fixes for this error_source, most recent first.

    Fetches a wider batch than `limit` and sorts client-side by `created_at`
    rather than trusting `store.search(..., limit=limit)`'s own ordering,
    because BaseStore makes no ordering guarantee without a `query` (which
    requires a configured embedder this module deliberately doesn't use — see
    the module docstring). _RECALL_FETCH_LIMIT bounds that fetch so recall
    stays cheap even once a namespace holds hundreds of entries.
    """
    items = store.search((NAMESPACE_PREFIX, error_source), limit=_RECALL_FETCH_LIMIT)
    ranked = sorted(items, key=lambda item: item.created_at, reverse=True)
    return [item.value for item in ranked[:limit]]
