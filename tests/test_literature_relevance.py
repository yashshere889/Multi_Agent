"""Tests for the relevance screen (agents/literature/relevance.py).

Two properties matter more than the happy path and get the most attention here:
the filter never drops a paper because *it* failed, and it never returns an
empty pool. Everything downstream treats an empty paper list as a hard error, so
a screening bug is a failed run rather than a slightly worse one.
"""

from unittest.mock import MagicMock

import pytest

from research_pipeline.agents.literature.relevance import (
    DIRECT_RELEVANCE_CRITERION,
    _batches,
    _paper_line,
    _parse_scores,
    apply_threshold,
    score_papers,
)
from research_pipeline.llm_json import LLMJSONError


def _paper(title: str, **extra) -> dict:
    return {"title": title, "abstract": "an abstract", "year": 2024, **extra}


def _model_returning(*payloads: str) -> MagicMock:
    """A chat model whose successive .invoke calls return the given raw strings.
    A payload may also be an Exception, which is raised instead."""
    model = MagicMock()

    def side_effect(_messages, **_kwargs):
        payload = payloads[min(side_effect.calls, len(payloads) - 1)]
        side_effect.calls += 1
        if isinstance(payload, Exception):
            raise payload
        return MagicMock(content=payload)

    side_effect.calls = 0
    model.invoke.side_effect = side_effect
    return model


# -- digest / batching -------------------------------------------------------


def test_paper_line_prefers_tldr_over_abstract():
    line = _paper_line(0, _paper("A Paper", tldr="the one-sentence version"))
    assert "the one-sentence version" in line
    assert "an abstract" not in line


def test_paper_line_labels_with_the_pool_index():
    assert _paper_line(7, _paper("A Paper")).startswith("[P7] A Paper")


def test_batches_split_on_the_character_budget_keeping_pool_indices():
    papers = [_paper(f"Paper {i}") for i in range(6)]
    batches = _batches(papers, max_chars=len(_paper_line(0, papers[0])) * 2 + 5)

    assert len(batches) > 1
    # Indices are positions in the whole pool, not within a batch — that's what
    # lets the per-batch scores merge back without renumbering.
    assert [index for batch in batches for index, _ in batch] == list(range(6))


def test_a_single_oversized_paper_still_gets_a_batch():
    papers = [_paper("Tiny"), _paper("Huge", abstract="x" * 5000)]
    batches = _batches(papers, max_chars=50)

    assert [index for batch in batches for index, _ in batch] == [0, 1]


# -- score parsing -----------------------------------------------------------


def test_parse_scores_reads_well_formed_entries():
    raw = {"scores": [{"id": "P0", "score": 5}, {"id": "P2", "score": 0}]}
    assert _parse_scores(raw, {0, 1, 2}) == {0: 5, 2: 0}


@pytest.mark.parametrize(
    "entry",
    [
        {"id": "not-an-id", "score": 4},   # unparseable id
        {"id": "P9", "score": 4},          # not in this batch
        {"id": "P0", "score": 11},         # outside the 0-5 rubric
        {"id": "P0", "score": "high"},     # not an integer
        {"id": "P0"},                      # no score at all
        "a bare string",                   # not even an object
    ],
)
def test_parse_scores_discards_anything_malformed(entry):
    assert _parse_scores({"scores": [entry]}, {0, 1}) == {}


def test_parse_scores_tolerates_a_missing_scores_key():
    assert _parse_scores({}, {0}) == {}


# -- scoring -----------------------------------------------------------------


def test_score_papers_maps_scores_back_positionally():
    model = _model_returning('{"scores": [{"id": "P0", "score": 5}, {"id": "P1", "score": 1}]}')
    scores = score_papers(model, "a question", [_paper("A"), _paper("B")])
    assert scores == [5, 1]


def test_score_papers_returns_none_for_papers_the_model_skipped():
    model = _model_returning('{"scores": [{"id": "P0", "score": 4}]}')
    assert score_papers(model, "a question", [_paper("A"), _paper("B")]) == [4, None]


def test_score_papers_returns_all_none_when_the_llm_is_unreachable():
    """The realistic outage isn't malformed JSON, it's a connection error — and
    it must not escape, or the screen becomes the most fragile node in the graph."""
    model = _model_returning(ConnectionError("no route to host"))
    assert score_papers(model, "a question", [_paper("A"), _paper("B")]) == [None, None]


def test_score_papers_survives_an_unparseable_response():
    model = _model_returning(LLMJSONError("still not JSON"))
    assert score_papers(model, "a question", [_paper("A")]) == [None]


def test_one_failing_batch_does_not_cost_the_others_their_scores():
    papers = [_paper(f"Paper {i}") for i in range(4)]
    budget = len(_paper_line(0, papers[0])) * 2 + 5  # ~2 papers per batch
    model = _model_returning(
        ConnectionError("first batch dies"),
        '{"scores": [{"id": "P2", "score": 5}, {"id": "P3", "score": 4}]}',
    )

    scores = score_papers(model, "a question", papers, batch_max_chars=budget)

    assert scores[:2] == [None, None]
    assert scores[2:] == [5, 4]


def test_score_papers_short_circuits_on_an_empty_pool():
    model = _model_returning('{"scores": []}')
    assert score_papers(model, "a question", []) == []
    model.invoke.assert_not_called()


def test_the_criterion_reaches_the_prompt():
    model = _model_returning('{"scores": []}')
    score_papers(model, "a question", [_paper("A")], criterion=DIRECT_RELEVANCE_CRITERION)

    _system, human = model.invoke.call_args[0][0]
    assert DIRECT_RELEVANCE_CRITERION in human[1]
    assert "a question" in human[1]


# -- thresholding ------------------------------------------------------------


def test_apply_threshold_splits_on_the_minimum():
    papers = [_paper("keep"), _paper("drop")]
    kept, dropped = apply_threshold(papers, [3, 2], min_score=3, keep_min=5)

    assert [p["title"] for p in kept] == ["keep"]
    assert [p["title"] for p in dropped] == ["drop"]


def test_an_unscored_paper_is_always_kept():
    """None means 'we failed to judge this', which is never grounds to drop."""
    kept, dropped = apply_threshold([_paper("unscored")], [None], min_score=5, keep_min=1)

    assert [p["title"] for p in kept] == ["unscored"]
    assert dropped == []
    assert kept[0]["relevance_score"] is None


def test_kept_papers_carry_the_score_they_were_judged_on():
    kept, _ = apply_threshold([_paper("a")], [4], min_score=3, keep_min=1)
    assert kept[0]["relevance_score"] == 4


def test_the_pool_is_never_emptied_even_when_everything_scores_below():
    papers = [_paper("worst"), _paper("best"), _paper("middling")]
    kept, dropped = apply_threshold(papers, [0, 2, 1], min_score=4, keep_min=2)

    # The best of a bad set, in score order — an empty pool would fail the run
    # outright in the next agent.
    assert [p["title"] for p in kept] == ["best", "middling"]
    assert [p["title"] for p in dropped] == ["worst"]


def test_keep_min_zero_opts_out_of_the_rescue_entirely():
    """The Interdisciplinary Agent's screen passes keep_min=0: an empty
    cross-field result is a legitimate answer there, not a failed run."""
    kept, dropped = apply_threshold([_paper("only")], [0], min_score=4, keep_min=0)
    assert kept == []
    assert [p["title"] for p in dropped] == ["only"]


def test_apply_threshold_does_not_mutate_its_input():
    """The search nodes it filters are cached by LangGraph, so annotating their
    output in place would leak this run's scores into the next cache hit."""
    original = _paper("a")
    apply_threshold([original], [4], min_score=3, keep_min=1)
    assert "relevance_score" not in original
