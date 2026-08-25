from research_pipeline.agents.literature.nodes import _looks_like_pdf, _normalize_title, _safe_filename


def test_normalize_title_strips_non_alnum_and_lowercases():
    assert _normalize_title("RAG: A Survey!") == "ragasurvey"


def test_safe_filename_disambiguates_same_title_different_ids():
    paper_a = {"title": "Attention Is All You Need", "arxiv_id": "1706.03762"}
    paper_b = {"title": "Attention Is All You Need", "arxiv_id": "1706.03762v2"}
    assert _safe_filename(paper_a) != _safe_filename(paper_b)


def test_looks_like_pdf_trusts_content_type_header():
    assert _looks_like_pdf(b"whatever", "application/pdf; charset=binary") is True


def test_looks_like_pdf_falls_back_to_magic_bytes():
    assert _looks_like_pdf(b"%PDF-1.4 ...", "") is True
    assert _looks_like_pdf(b"<html>not a pdf</html>", "text/html") is False


# -- score_relevance_node ----------------------------------------------------
#
# The node's own job is thin — relevance.py does the scoring and thresholding,
# and is tested directly in test_literature_relevance.py. What matters here is
# that every way this node can fail leaves the pool intact rather than empty.

from dataclasses import replace  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

from research_pipeline.agents.literature import nodes as literature_nodes  # noqa: E402
from research_pipeline.agents.literature.nodes import score_relevance_node  # noqa: E402


def _filter_settings(monkeypatch, **overrides):
    monkeypatch.setattr(
        literature_nodes,
        "settings",
        replace(literature_nodes.settings, enable_relevance_filter=True, **overrides),
    )


def _state(*titles: str) -> dict:
    return {
        "research_question": "does retrieval augmentation help?",
        "merged_papers": [{"title": t, "abstract": "an abstract"} for t in titles],
    }


class _FakeModel:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, _messages, **_kwargs):
        return SimpleNamespace(content=self._content)


def test_score_relevance_node_drops_papers_below_the_threshold(monkeypatch):
    _filter_settings(monkeypatch, relevance_min_score=3, relevance_keep_min=1)
    scored = '{"scores": [{"id": "P0", "score": 5}, {"id": "P1", "score": 1}]}'

    with patch.object(literature_nodes, "get_chat_model", return_value=_FakeModel(scored)):
        result = score_relevance_node(_state("relevant", "off topic"))

    assert [p["title"] for p in result["merged_papers"]] == ["relevant"]
    assert result["papers_filtered_out"] == 1


def test_score_relevance_node_is_a_pass_through_when_disabled(monkeypatch):
    monkeypatch.setattr(
        literature_nodes, "settings", replace(literature_nodes.settings, enable_relevance_filter=False)
    )
    with patch.object(literature_nodes, "get_chat_model") as get_model:
        assert score_relevance_node(_state("a", "b")) == {}
    get_model.assert_not_called()


def test_score_relevance_node_keeps_everything_when_the_model_cannot_be_built(monkeypatch):
    """An unconfigured or unreachable LLM endpoint must not cost the run its papers."""
    _filter_settings(monkeypatch)
    with patch.object(literature_nodes, "get_chat_model", side_effect=RuntimeError("no LLM_BASE_URL")):
        assert score_relevance_node(_state("a", "b")) == {}


def test_score_relevance_node_keeps_everything_when_scoring_fails(monkeypatch):
    _filter_settings(monkeypatch, relevance_min_score=3)
    with patch.object(literature_nodes, "get_chat_model", return_value=_FakeModel("not json")):
        result = score_relevance_node(_state("a", "b"))

    assert [p["title"] for p in result["merged_papers"]] == ["a", "b"]
    assert result["papers_filtered_out"] == 0


def test_score_relevance_node_handles_an_empty_pool(monkeypatch):
    _filter_settings(monkeypatch)
    with patch.object(literature_nodes, "get_chat_model") as get_model:
        assert score_relevance_node({"research_question": "q", "merged_papers": []}) == {}
    get_model.assert_not_called()


# -- expand_citations_node ---------------------------------------------------
#
# The node's contract is narrower than expansion.py's: everything it adds must
# already have been screened, and every failure mode must add nothing rather
# than break the run.

from research_pipeline.agents.literature.nodes import expand_citations_node  # noqa: E402


def _expansion_settings(monkeypatch, **overrides):
    base = dict(
        enable_citation_expansion=True,
        enable_relevance_filter=True,
        citation_expansion_seeds=2,
        citation_expansion_per_seed=50,
        citation_expansion_max_papers=10,
        citation_expansion_directions=("references",),
        relevance_min_score=3,
    )
    base.update(overrides)
    monkeypatch.setattr(literature_nodes, "settings", replace(literature_nodes.settings, **base))


def _pool_state(*, filtered_out=0):
    return {
        "research_question": "does retrieval augmentation help?",
        "merged_papers": [{"title": "A Seed Paper", "arxiv_id": "1", "relevance_score": 5}],
        "papers_filtered_out": filtered_out,
    }


def _found(*titles):
    return [{"title": t, "doi": f"10.1/{t.replace(' ', '')}", "abstract": "x"} for t in titles]


def test_expansion_screens_what_it_finds_before_merging_it(monkeypatch):
    """The pool-wide invariant is that everything in merged_papers has been
    screened; expansion must not be the hole in it."""
    _expansion_settings(monkeypatch)
    scored = '{"scores": [{"id": "P0", "score": 5}, {"id": "P1", "score": 0}]}'

    with patch.object(literature_nodes.expansion, "expand", return_value=_found("Relevant", "Off Topic")), \
         patch.object(literature_nodes, "get_chat_model", return_value=_FakeModel(scored)):
        result = expand_citations_node(_pool_state())

    assert [p["title"] for p in result["merged_papers"]] == ["A Seed Paper", "Relevant"]
    assert result["papers_from_citations"] == 1


def test_expansion_adds_its_rejects_to_the_running_filtered_count(monkeypatch):
    _expansion_settings(monkeypatch)
    scored = '{"scores": [{"id": "P0", "score": 0}]}'

    with patch.object(literature_nodes.expansion, "expand", return_value=_found("Off Topic")), \
         patch.object(literature_nodes, "get_chat_model", return_value=_FakeModel(scored)):
        result = expand_citations_node(_pool_state(filtered_out=4))

    assert result["papers_filtered_out"] == 5


def test_expansion_keeps_candidates_when_the_screen_cannot_run(monkeypatch):
    """Same direction as everywhere else: an unreachable LLM widens the pool."""
    _expansion_settings(monkeypatch)

    with patch.object(literature_nodes.expansion, "expand", return_value=_found("Found")), \
         patch.object(literature_nodes, "get_chat_model", side_effect=RuntimeError("no LLM")):
        result = expand_citations_node(_pool_state())

    assert result["papers_from_citations"] == 1


def test_expansion_does_not_screen_when_screening_is_off(monkeypatch):
    _expansion_settings(monkeypatch, enable_relevance_filter=False)

    with patch.object(literature_nodes.expansion, "expand", return_value=_found("Found")), \
         patch.object(literature_nodes, "get_chat_model") as get_model:
        result = expand_citations_node(_pool_state())

    assert result["papers_from_citations"] == 1
    get_model.assert_not_called()


def test_expansion_never_empties_the_pool_when_every_candidate_scores_low(monkeypatch):
    """keep_min=0 here on purpose: unlike the search results there is nothing to
    rescue, since the pool it would join is already populated."""
    _expansion_settings(monkeypatch)
    scored = '{"scores": [{"id": "P0", "score": 0}, {"id": "P1", "score": 0}]}'

    with patch.object(literature_nodes.expansion, "expand", return_value=_found("A", "B")), \
         patch.object(literature_nodes, "get_chat_model", return_value=_FakeModel(scored)):
        result = expand_citations_node(_pool_state())

    assert result["papers_from_citations"] == 0
    assert [p["title"] for p in result["merged_papers"]] == ["A Seed Paper"]


def test_expansion_is_a_pass_through_when_disabled(monkeypatch):
    _expansion_settings(monkeypatch, enable_citation_expansion=False)
    with patch.object(literature_nodes.expansion, "expand") as expand:
        assert expand_citations_node(_pool_state()) == {}
    expand.assert_not_called()


def test_expansion_is_a_pass_through_when_nothing_was_found(monkeypatch):
    _expansion_settings(monkeypatch)
    with patch.object(literature_nodes.expansion, "expand", return_value=[]):
        assert expand_citations_node(_pool_state()) == {}


def test_expansion_does_nothing_with_an_empty_pool(monkeypatch):
    _expansion_settings(monkeypatch)
    with patch.object(literature_nodes.expansion, "expand") as expand:
        assert expand_citations_node({"research_question": "q", "merged_papers": []}) == {}
    expand.assert_not_called()
