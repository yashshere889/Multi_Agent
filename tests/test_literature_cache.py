"""Tests for CachePolicy on the literature agent's paper-search nodes.

The thing worth proving is narrow: two runs of the compiled graph over the same
research question, in one process, hit arXiv / Semantic Scholar / CORE once
between them rather than twice — and that turning ENABLE_PAPER_SEARCH_CACHE off
really does go back to hitting them every time, rather than quietly caching with
a short TTL.

The search functions are replaced at the *graph module's* namespace, not the
nodes module's: graph.py imports them by name at import time, so that's the
binding build_literature_graph() actually reads.
"""

from dataclasses import replace

import pytest

from research_pipeline.agents.literature import graph as literature_graph
from research_pipeline.agents.literature.graph import build_literature_graph


QUESTION = "does retrieval augmentation help long-context reasoning?"


# get_node_cache() is a process-wide singleton by design; conftest.py's autouse
# fixture is what gives each test here an empty one to start from.


class CountingSearch:
    """Stands in for one search node, recording how many times it really ran."""

    def __init__(self, key: str, papers: list[dict]):
        self._key = key
        self._papers = papers
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        return {self._key: self._papers}


def _stub_everything_but_search(monkeypatch, tmp_path):
    """Neutralizes the nodes around the searches: query generation would call
    the LLM, downloads would hit the network, and save_metadata would write
    outside tmp_path."""
    monkeypatch.setattr(
        literature_graph, "generate_queries", lambda state: {"search_queries": ["rag long context"]}
    )
    monkeypatch.setattr(literature_graph, "download_papers_node", lambda state: {})
    monkeypatch.setattr(literature_graph, "save_metadata_node", lambda state: {"metadata_path": str(tmp_path)})


def _install_counting_searches(monkeypatch):
    arxiv = CountingSearch("arxiv_papers", [{"title": "Paper A", "arxiv_id": "1"}])
    semantic_scholar = CountingSearch("semantic_scholar_papers", [{"title": "Paper B", "paper_id": "2"}])
    core = CountingSearch("core_papers", [])
    monkeypatch.setattr(literature_graph, "search_arxiv_node", arxiv)
    monkeypatch.setattr(literature_graph, "search_semantic_scholar_node", semantic_scholar)
    monkeypatch.setattr(literature_graph, "search_core_node", core)
    return arxiv, semantic_scholar, core


def _run_twice(tmp_path):
    """Two builds, two invocations, two thread_ids — i.e. two independent runs
    that happen to share a process, which is the only situation an in-memory
    node cache can help."""
    payload = {
        "research_question": QUESTION,
        "max_results_per_query": 3,
        "download_dir": str(tmp_path),
        "metadata_path": str(tmp_path / "metadata.json"),
    }
    for thread_id in ("run-one", "run-two"):
        build_literature_graph().invoke(payload, config={"configurable": {"thread_id": thread_id}})


def test_identical_runs_share_one_set_of_search_calls(monkeypatch, tmp_path):
    _stub_everything_but_search(monkeypatch, tmp_path)
    arxiv, semantic_scholar, core = _install_counting_searches(monkeypatch)

    _run_twice(tmp_path)

    assert (arxiv.calls, semantic_scholar.calls, core.calls) == (1, 1, 1)


def test_cache_is_shared_across_separately_built_graphs(monkeypatch, tmp_path):
    """The cache being a memoized singleton is what makes this work at all —
    agent graphs are rebuilt per call, and a per-build cache would never hit."""
    _stub_everything_but_search(monkeypatch, tmp_path)
    arxiv, _, _ = _install_counting_searches(monkeypatch)

    payload = {
        "research_question": QUESTION,
        "download_dir": str(tmp_path),
        "metadata_path": str(tmp_path / "metadata.json"),
    }
    first, second = build_literature_graph(), build_literature_graph()
    assert first is not second
    first.invoke(payload, config={"configurable": {"thread_id": "a"}})
    second.invoke(payload, config={"configurable": {"thread_id": "b"}})

    assert arxiv.calls == 1


def test_a_different_question_is_not_a_cache_hit(monkeypatch, tmp_path):
    _stub_everything_but_search(monkeypatch, tmp_path)
    arxiv, _, _ = _install_counting_searches(monkeypatch)

    graph = build_literature_graph()
    for i, question in enumerate([QUESTION, "something else entirely"]):
        graph.invoke(
            {
                "research_question": question,
                "download_dir": str(tmp_path),
                "metadata_path": str(tmp_path / "metadata.json"),
            },
            config={"configurable": {"thread_id": f"q{i}"}},
        )

    assert arxiv.calls == 2


def test_disabling_the_toggle_searches_every_time(monkeypatch, tmp_path):
    monkeypatch.setattr(
        literature_graph,
        "settings",
        replace(literature_graph.settings, enable_paper_search_cache=False),
    )
    _stub_everything_but_search(monkeypatch, tmp_path)
    arxiv, semantic_scholar, core = _install_counting_searches(monkeypatch)

    _run_twice(tmp_path)

    assert (arxiv.calls, semantic_scholar.calls, core.calls) == (2, 2, 2)
