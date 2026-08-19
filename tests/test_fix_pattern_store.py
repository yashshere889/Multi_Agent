"""Tests for the Coder Agent's cross-run fix-pattern library.

Backend-selection tests follow test_checkpointer.py's pattern exactly (same
module, same "one factory" shape): settings is a frozen dataclass, so
monkeypatch.setattr replaces the *module-level* `settings` name inside
fix_pattern_store itself (dataclasses.replace(...)), not an attribute on the
frozen instance.
"""

from dataclasses import replace

import pytest
from langgraph.store.memory import InMemoryStore

from research_pipeline.agents.coder import fix_pattern_store as fix_pattern_store_module
from research_pipeline.agents.coder.fix_pattern_store import (
    MAX_PATTERN_CODE_CHARS,
    NAMESPACE_PREFIX,
    get_store,
    recall_fixes,
    record_fix,
    reset_store,
)
from research_pipeline.config import load_settings


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_store()
    yield
    reset_store()


def _with_settings(monkeypatch, **overrides):
    monkeypatch.setattr(
        fix_pattern_store_module,
        "settings",
        replace(fix_pattern_store_module.settings, **overrides),
    )


def test_record_fix_writes_only_the_sections_that_actually_changed():
    store = InMemoryStore()
    record_fix(
        store,
        error_source="run_experiment",
        error_summary="RuntimeError: boom",
        broken_sections={
            "run_experiment_function": "def run_experiment(data, model):\n    raise RuntimeError('boom')\n",
            "helpers": "# unchanged\n",
        },
        fixed_sections={
            "run_experiment_function": "def run_experiment(data, model):\n    return {'metrics': {}}\n",
            "helpers": "# unchanged\n",
        },
    )
    items = store.search((NAMESPACE_PREFIX, "run_experiment"))
    assert len(items) == 1
    changed = items[0].value["changed_sections"]
    assert list(changed) == ["run_experiment_function"]  # unchanged "helpers" excluded
    assert "raise RuntimeError" in changed["run_experiment_function"]["before"]
    assert "return {'metrics': {}}" in changed["run_experiment_function"]["after"]


def test_record_fix_writes_nothing_when_no_section_changed():
    store = InMemoryStore()
    record_fix(
        store,
        error_source="compile_check",
        error_summary="x",
        broken_sections={"helpers": "same\n"},
        fixed_sections={"helpers": "same\n"},
    )
    assert store.search((NAMESPACE_PREFIX, "compile_check")) == []


def test_record_fix_writes_nothing_when_the_fixed_section_is_blank():
    # A "fix" that just deletes the section's content teaches nothing useful
    # to recall — and an empty `after` would render as a useless example.
    store = InMemoryStore()
    record_fix(
        store,
        error_source="compile_check",
        error_summary="x",
        broken_sections={"helpers": "def f(:\n"},
        fixed_sections={"helpers": "   \n"},
    )
    assert store.search((NAMESPACE_PREFIX, "compile_check")) == []


def test_record_fix_captures_a_newly_added_section_as_before_empty():
    store = InMemoryStore()
    record_fix(
        store,
        error_source="static_lint",
        error_summary="x",
        broken_sections={},  # e.g. "helpers" didn't exist in the broken version at all
        fixed_sections={"helpers": "def sanitize(x):\n    return x\n"},
    )
    items = store.search((NAMESPACE_PREFIX, "static_lint"))
    assert items[0].value["changed_sections"]["helpers"]["before"] == ""
    assert "def sanitize" in items[0].value["changed_sections"]["helpers"]["after"]


def test_record_fix_truncates_an_oversized_section():
    store = InMemoryStore()
    huge = "x = 1\n" * (MAX_PATTERN_CODE_CHARS // 4)
    assert len(huge) > MAX_PATTERN_CODE_CHARS
    record_fix(
        store,
        error_source="run_experiment",
        error_summary="x",
        broken_sections={"helpers": ""},
        fixed_sections={"helpers": huge},
    )
    stored = store.search((NAMESPACE_PREFIX, "run_experiment"))[0].value
    after = stored["changed_sections"]["helpers"]["after"]
    assert len(after) <= MAX_PATTERN_CODE_CHARS + len("\n# ... truncated ...")
    assert after.endswith("# ... truncated ...")


def test_recall_fixes_returns_most_recent_first_and_respects_limit():
    store = InMemoryStore()
    for i in range(4):
        record_fix(
            store,
            error_source="run_experiment",
            error_summary=f"attempt {i}",
            broken_sections={"helpers": f"broken {i}\n"},
            fixed_sections={"helpers": f"fixed {i}\n"},
        )
    recalled = recall_fixes(store, "run_experiment", limit=2)
    assert len(recalled) == 2
    # Most recent (attempt 3) first, then attempt 2 — not insertion order.
    assert recalled[0]["error_summary"] == "attempt 3"
    assert recalled[1]["error_summary"] == "attempt 2"


def test_recall_fixes_is_scoped_to_its_own_error_source():
    store = InMemoryStore()
    record_fix(
        store,
        error_source="run_experiment",
        error_summary="a",
        broken_sections={"helpers": "a\n"},
        fixed_sections={"helpers": "b\n"},
    )
    record_fix(
        store,
        error_source="compile_check",
        error_summary="c",
        broken_sections={"helpers": "c\n"},
        fixed_sections={"helpers": "d\n"},
    )
    assert len(recall_fixes(store, "run_experiment")) == 1
    assert len(recall_fixes(store, "static_lint")) == 0  # never recorded


def test_recall_fixes_on_an_empty_store_returns_an_empty_list():
    assert recall_fixes(InMemoryStore(), "run_experiment") == []


def test_get_store_defaults_to_sqlite_backend(monkeypatch):
    # Not an assertion against the live process's settings singleton — the
    # test session's conftest.py deliberately forces CODER_FIX_STORE_BACKEND
    # to "memory" so a bare pytest run never writes a real sqlite file (see
    # conftest.py's docstring). This checks load_settings()'s own default,
    # the way it would resolve for a real, non-test invocation with no
    # override set.
    monkeypatch.delenv("CODER_FIX_STORE_BACKEND", raising=False)
    assert load_settings().coder_fix_store_backend == "sqlite"


def test_get_store_is_a_singleton(monkeypatch):
    _with_settings(monkeypatch, coder_fix_store_backend="memory")
    first = get_store()
    second = get_store()
    assert first is second  # not rebuilt on every call


def test_get_store_rejects_an_unknown_backend(monkeypatch):
    _with_settings(monkeypatch, coder_fix_store_backend="not-a-real-backend")
    with pytest.raises(SystemExit):
        get_store()


def test_reset_store_forces_a_fresh_instance_on_next_call(monkeypatch):
    _with_settings(monkeypatch, coder_fix_store_backend="memory")
    first = get_store()
    reset_store()
    second = get_store()
    assert first is not second


def test_sqlite_backend_persists_across_a_fresh_store_instance(monkeypatch, tmp_path):
    # The whole reason this backend defaults on: a pattern recorded by one
    # process must be recallable by a completely separate later process. Proven
    # the same way test_checkpointer.py proves sqlite durability — write with
    # one instance, throw it away, read with a second one built fresh from the
    # same file.
    db_path = tmp_path / "patterns.db"
    _with_settings(
        monkeypatch, coder_fix_store_backend="sqlite", coder_fix_store_sqlite_path=str(db_path)
    )
    first = get_store()
    record_fix(
        first,
        error_source="run_experiment",
        error_summary="boom",
        broken_sections={"helpers": "broken\n"},
        fixed_sections={"helpers": "fixed\n"},
    )
    reset_store()

    second = get_store()
    assert second is not first
    recalled = recall_fixes(second, "run_experiment")
    assert len(recalled) == 1
    assert recalled[0]["error_summary"] == "boom"
