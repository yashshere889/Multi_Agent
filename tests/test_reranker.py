"""Tests for the cross-encoder reranker.

None of these load a model. The real one is 2.2GB and needs torch, which is an
optional extra precisely because it is not installable everywhere — so the unit
that matters here is the *policy* around scoring (what gets reordered, what gets
dropped, and what happens when scoring is unavailable), which is exactly the
part that must hold on a machine that can never run the model.
"""

import sys
from dataclasses import replace

from research_pipeline import reranker
from research_pipeline.config import settings


def _enabled(monkeypatch, **overrides):
    monkeypatch.setattr(reranker, "settings", replace(settings, enable_rerank=True, **overrides))


def _papers():
    return [
        {"title": "Least relevant", "abstract": "a"},
        {"title": "Most relevant", "abstract": "b"},
        {"title": "Middling", "abstract": "c"},
    ]


def _scores(*values):
    return lambda query, documents: list(values)


def test_disabled_returns_the_pool_untouched():
    papers = _papers()
    assert reranker.rerank_papers("a question", papers) is papers


def test_reorders_by_score_and_records_it(monkeypatch):
    _enabled(monkeypatch)
    papers = _papers()

    ranked = reranker.rerank_papers("a question", papers, score_fn=_scores(0.1, 9.0, 1.0))

    assert [p["title"] for p in ranked] == ["Most relevant", "Middling", "Least relevant"]
    assert ranked[0]["rerank_score"] == 9.0
    # The caller's own dicts are left alone — the scores go on copies.
    assert papers == _papers()


def test_ties_keep_the_order_the_merge_produced(monkeypatch):
    _enabled(monkeypatch)

    ranked = reranker.rerank_papers("a question", _papers(), score_fn=_scores(1.0, 1.0, 1.0))

    assert [p["title"] for p in ranked] == ["Least relevant", "Most relevant", "Middling"]


def test_top_k_truncates_after_ordering(monkeypatch):
    _enabled(monkeypatch)

    ranked = reranker.rerank_papers(
        "a question", _papers(), top_k=2, score_fn=_scores(0.1, 9.0, 1.0)
    )

    assert [p["title"] for p in ranked] == ["Most relevant", "Middling"]


def test_top_k_zero_keeps_everything(monkeypatch):
    _enabled(monkeypatch, rerank_top_k=0)

    ranked = reranker.rerank_papers("a question", _papers(), score_fn=_scores(0.1, 9.0, 1.0))

    assert len(ranked) == 3


def test_settings_top_k_applies_when_the_caller_passes_none(monkeypatch):
    _enabled(monkeypatch, rerank_top_k=1)

    ranked = reranker.rerank_papers("a question", _papers(), score_fn=_scores(0.1, 9.0, 1.0))

    assert [p["title"] for p in ranked] == ["Most relevant"]


def test_unavailable_scorer_leaves_the_pool_untouched(monkeypatch):
    _enabled(monkeypatch)
    papers = _papers()

    assert reranker.rerank_papers("a question", papers, score_fn=lambda q, d: None) is papers


def test_a_wrong_length_score_list_is_refused_rather_than_zipped(monkeypatch):
    """zip() would silently drop the unscored tail — i.e. lose papers to a bug."""
    _enabled(monkeypatch)
    papers = _papers()

    assert reranker.rerank_papers("a question", papers, score_fn=_scores(1.0, 2.0)) is papers


def test_no_question_means_nothing_to_score_against(monkeypatch):
    _enabled(monkeypatch)
    papers = _papers()

    assert reranker.rerank_papers(None, papers, score_fn=_scores(1.0, 2.0, 3.0)) is papers
    assert reranker.rerank_papers("   ", papers, score_fn=_scores(1.0, 2.0, 3.0)) is papers


def test_a_pool_with_no_text_is_left_alone(monkeypatch):
    _enabled(monkeypatch)
    papers = [{"year": 2024}, {"year": 2025}]

    assert reranker.rerank_papers("a question", papers, score_fn=_scores(1.0, 2.0)) is papers


def test_empty_pool_short_circuits(monkeypatch):
    _enabled(monkeypatch)
    assert reranker.rerank_papers("a question", []) == []


def test_paper_text_prefers_passages_over_the_abstract():
    """Same preference agents/hypothesis/papers.py applies, so a paper found by
    passage search is judged on its passages."""
    text = reranker._paper_text({"title": "T", "abstract": "the abstract", "full_text": "the passages"})

    assert text.startswith("T")
    assert "the passages" in text
    assert "the abstract" not in text


def test_a_missing_dependency_degrades_once_rather_than_raising(monkeypatch):
    """The only path available on a machine where torch cannot be installed."""
    _enabled(monkeypatch)
    # `from transformers import ...` against a None entry raises ImportError,
    # which is what a machine without the extra installed really does.
    monkeypatch.setitem(sys.modules, "transformers", None)

    assert reranker._load() is None
    assert reranker.is_available() is False
    # And the failure is memoized: a 50-paper pool must not retry the import 50
    # times, nor log 50 warnings.
    monkeypatch.delitem(sys.modules, "transformers")
    assert reranker._load() is None
    assert reranker.score_pairs("q", ["doc"]) is None


def test_scoring_failure_is_survivable(monkeypatch):
    _enabled(monkeypatch)

    def _boom(query, documents):
        raise RuntimeError("CUDA out of memory")

    papers = _papers()
    monkeypatch.setattr(reranker, "score_pairs", _boom)

    try:
        result = reranker.rerank_papers("a question", papers)
    except RuntimeError:
        raise AssertionError("rerank_papers must not propagate a scoring failure")
    assert result is papers
