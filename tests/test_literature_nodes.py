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
