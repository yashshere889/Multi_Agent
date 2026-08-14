"""Tests for the shared checkpointer factory.

The point of this module is durability, so the sqlite test deliberately does not
just check "a SqliteSaver came back" — it writes state through one saver, throws
that saver away, builds a second one against the same file, and reads the state
back. That's the actual behaviour the pipeline needs when a SLURM job is
pre-empted or a Kaggle kernel restarts.
"""

import builtins
from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from research_pipeline import checkpointer as checkpointer_module
from research_pipeline.checkpointer import (
    get_checkpointer,
    get_node_cache,
    pending_nodes,
    reset_checkpointer,
)


# Every test here starts from fresh singletons — see conftest.py's autouse
# fixture, which exists precisely because this module would otherwise leave a
# sqlite saver behind as every later test's checkpointer.


def _with_settings(monkeypatch, **overrides):
    monkeypatch.setattr(
        checkpointer_module,
        "settings",
        replace(checkpointer_module.settings, **overrides),
    )


def _build_counting_graph():
    """A trivial two-node graph, just something that produces checkpoints."""
    from typing import TypedDict

    class State(TypedDict, total=False):
        value: int
        doubled: int

    graph = StateGraph(State)
    graph.add_node("set_value", lambda state: {"value": 21})
    graph.add_node("double", lambda state: {"doubled": state["value"] * 2})
    graph.set_entry_point("set_value")
    graph.add_edge("set_value", "double")
    graph.add_edge("double", END)
    return graph.compile(checkpointer=get_checkpointer())


# -- default backend -------------------------------------------------------------------


def test_default_backend_is_an_in_memory_saver():
    assert isinstance(get_checkpointer(), MemorySaver)


def test_checkpointer_is_memoized_so_every_graph_shares_one():
    assert get_checkpointer() is get_checkpointer()


def test_node_cache_is_memoized_so_every_graph_shares_one():
    from langgraph.cache.memory import InMemoryCache

    cache = get_node_cache()
    assert isinstance(cache, InMemoryCache)
    assert cache is get_node_cache()


def test_reset_drops_both_singletons():
    first_saver, first_cache = get_checkpointer(), get_node_cache()
    reset_checkpointer()
    assert get_checkpointer() is not first_saver
    assert get_node_cache() is not first_cache


def test_unknown_backend_names_the_valid_options(monkeypatch):
    _with_settings(monkeypatch, checkpointer_backend="redis")
    with pytest.raises(SystemExit) as excinfo:
        get_checkpointer()
    assert "memory, sqlite, postgres" in str(excinfo.value)


def test_backend_name_is_case_and_whitespace_insensitive(monkeypatch):
    _with_settings(monkeypatch, checkpointer_backend="  MEMORY ")
    assert isinstance(get_checkpointer(), MemorySaver)


# -- sqlite durability -----------------------------------------------------------------


def test_sqlite_state_survives_a_fresh_checkpointer_on_the_same_file(monkeypatch, tmp_path):
    pytest.importorskip(
        "langgraph.checkpoint.sqlite",
        reason="optional dependency; uv sync --extra checkpoint-sqlite",
    )
    db_path = tmp_path / "nested" / "pipeline.db"
    _with_settings(monkeypatch, checkpointer_backend="sqlite", checkpointer_sqlite_path=str(db_path))

    config = {"configurable": {"thread_id": "durable-thread"}}
    first_graph = _build_counting_graph()
    assert first_graph.invoke({}, config=config)["doubled"] == 42
    # The parent directory is created for us — nothing else makes checkpoints/.
    assert db_path.exists() and db_path.stat().st_size > 0

    # Stand in for "the process died": drop the memoized saver (and close its
    # connection) so the second graph genuinely re-opens the file from scratch.
    checkpointer_module._checkpointer.conn.close()
    reset_checkpointer()

    second_graph = _build_counting_graph()
    assert second_graph is not first_graph
    restored = second_graph.get_state(config)
    assert restored.values["value"] == 21
    assert restored.values["doubled"] == 42


def test_sqlite_saver_is_memoized_rather_than_reopened_per_graph(monkeypatch, tmp_path):
    pytest.importorskip(
        "langgraph.checkpoint.sqlite",
        reason="optional dependency; uv sync --extra checkpoint-sqlite",
    )
    _with_settings(
        monkeypatch,
        checkpointer_backend="sqlite",
        checkpointer_sqlite_path=str(tmp_path / "pipeline.db"),
    )
    # Agent graphs are rebuilt per .run() call, so this is what stops a long run
    # leaking one sqlite connection per agent invocation.
    assert get_checkpointer() is get_checkpointer()


# -- missing optional dependencies -----------------------------------------------------


def _block_import(monkeypatch, blocked: str):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"No module named {blocked!r}", name=blocked)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_sqlite_dependency_names_the_uv_command(monkeypatch, tmp_path):
    _with_settings(
        monkeypatch,
        checkpointer_backend="sqlite",
        checkpointer_sqlite_path=str(tmp_path / "pipeline.db"),
    )
    _block_import(monkeypatch, "langgraph.checkpoint.sqlite")
    with pytest.raises(SystemExit) as excinfo:
        get_checkpointer()
    message = str(excinfo.value)
    assert "langgraph.checkpoint.sqlite" in message
    assert "uv sync --extra checkpoint-sqlite" in message


def test_missing_postgres_dependency_names_the_uv_command(monkeypatch):
    _with_settings(
        monkeypatch,
        checkpointer_backend="postgres",
        checkpointer_postgres_uri="postgresql://user:pass@localhost:5432/db",
    )
    _block_import(monkeypatch, "langgraph.checkpoint.postgres")
    with pytest.raises(SystemExit) as excinfo:
        get_checkpointer()
    message = str(excinfo.value)
    assert "langgraph.checkpoint.postgres" in message
    assert "uv sync --extra checkpoint-postgres" in message


def test_postgres_without_a_uri_fails_before_trying_to_connect(monkeypatch):
    _with_settings(monkeypatch, checkpointer_backend="postgres", checkpointer_postgres_uri="")
    with pytest.raises(SystemExit) as excinfo:
        get_checkpointer()
    assert "CHECKPOINTER_POSTGRES_URI" in str(excinfo.value)


# -- pending_nodes / resuming ----------------------------------------------------------


def _build_breakable_graph(calls, fail_second: bool):
    """Two sequential nodes, the second of which can be made to blow up. `calls`
    counts executions per node, which is how the resume test proves the first
    node is *not* re-run."""
    from typing import TypedDict

    class State(TypedDict, total=False):
        first: str
        second: str

    def first(_state):
        calls.append("first")
        return {"first": "done"}

    def second(_state):
        calls.append("second")
        if fail_second:
            raise RuntimeError("pre-empted")
        return {"second": "done"}

    graph = StateGraph(State)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.set_entry_point("first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    return graph.compile(checkpointer=get_checkpointer())


def test_pending_nodes_is_empty_for_a_thread_that_was_never_run():
    graph = _build_counting_graph()
    assert pending_nodes(graph, {"configurable": {"thread_id": "never-seen"}}) == ()


def test_pending_nodes_is_empty_after_a_run_reaches_the_end():
    graph = _build_counting_graph()
    config = {"configurable": {"thread_id": "finished"}}
    graph.invoke({}, config=config)
    assert pending_nodes(graph, config) == ()


def test_pending_nodes_names_the_node_a_crashed_run_stopped_before(monkeypatch, tmp_path):
    pytest.importorskip(
        "langgraph.checkpoint.sqlite",
        reason="optional dependency; uv sync --extra checkpoint-sqlite",
    )
    _with_settings(
        monkeypatch,
        checkpointer_backend="sqlite",
        checkpointer_sqlite_path=str(tmp_path / "pipeline.db"),
    )
    config = {"configurable": {"thread_id": "crashed"}}
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="pre-empted"):
        _build_breakable_graph(calls, fail_second=True).invoke({}, config=config)

    assert calls == ["first", "second"]
    assert pending_nodes(_build_breakable_graph(calls, fail_second=True), config) == ("second",)


def test_resuming_a_crashed_run_does_not_re_run_the_stage_that_finished(monkeypatch, tmp_path):
    """The whole point of durable checkpointing: a job pre-empted in stage two
    picks up at stage two, rather than paying for stage one again."""
    pytest.importorskip(
        "langgraph.checkpoint.sqlite",
        reason="optional dependency; uv sync --extra checkpoint-sqlite",
    )
    db_path = tmp_path / "pipeline.db"
    _with_settings(monkeypatch, checkpointer_backend="sqlite", checkpointer_sqlite_path=str(db_path))
    config = {"configurable": {"thread_id": "resumable"}}

    calls: list[str] = []
    with pytest.raises(RuntimeError, match="pre-empted"):
        _build_breakable_graph(calls, fail_second=True).invoke({}, config=config)

    # Stand in for the job being resubmitted as a new process.
    checkpointer_module._checkpointer.conn.close()
    reset_checkpointer()
    _with_settings(monkeypatch, checkpointer_backend="sqlite", checkpointer_sqlite_path=str(db_path))

    calls.clear()
    graph = _build_breakable_graph(calls, fail_second=False)
    assert pending_nodes(graph, config) == ("second",)
    # `None` is what resuming means: continue from the checkpoint.
    result = graph.invoke(None, config=config)

    assert calls == ["second"]  # "first" was not paid for twice
    assert result == {"first": "done", "second": "done"}
    assert pending_nodes(graph, config) == ()
