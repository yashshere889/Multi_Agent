import pytest

from research_pipeline.agents.literature.nodes import (
    _looks_like_pdf,
    _normalize_title,
    _safe_filename,
    merge_and_dedupe_node,
)


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


# --- merge_and_dedupe -----------------------------------------------------------
#
# The snippet search is the only source carrying body passages and the only one
# carrying no DOI/year/PDF, so a paper found both ways has to come out of the
# merge with the richer record's metadata *and* the snippet record's text.


def _state(**sources):
    state = {"arxiv_papers": [], "semantic_scholar_papers": [], "core_papers": [], "snippet_papers": []}
    state.update(sources)
    return state


def test_merge_gives_a_duplicate_the_snippet_record_s_passages():
    arxiv = {"title": "Shared Title", "arxiv_id": "1", "abstract": "abs", "year": 2024,
             "pdf_url": "https://arxiv.org/pdf/1"}
    snippet = {"title": "Shared Title!", "paper_id": "CorpusId:9", "full_text": "a passage"}

    merged = merge_and_dedupe_node(_state(arxiv_papers=[arxiv], snippet_papers=[snippet]))["merged_papers"]

    assert len(merged) == 1
    # The arXiv record still wins: it has the PDF, the year and the id.
    assert merged[0]["arxiv_id"] == "1"
    assert merged[0]["year"] == 2024
    assert merged[0]["full_text"] == "a passage"


def test_merge_keeps_a_paper_only_the_snippet_search_found():
    snippet = {"title": "Body-Only Match", "paper_id": "CorpusId:9", "full_text": "a passage"}

    merged = merge_and_dedupe_node(_state(snippet_papers=[snippet]))["merged_papers"]

    assert [p["title"] for p in merged] == ["Body-Only Match"]


def test_merge_still_prefers_the_record_that_has_a_pdf():
    """Unchanged tie-break — the record with a PDF wins — but it now inherits
    the fields the loser had and it lacked, rather than discarding them."""
    without_pdf = {"title": "Shared Title", "paper_id": "s2-1", "abstract": "the abstract"}
    with_pdf = {"title": "Shared Title", "paper_id": "core-1", "year": 2023,
                "pdf_url": "https://core.ac.uk/1.pdf"}

    merged = merge_and_dedupe_node(
        _state(semantic_scholar_papers=[without_pdf], core_papers=[with_pdf])
    )["merged_papers"]

    assert len(merged) == 1
    assert merged[0]["paper_id"] == "core-1"
    assert merged[0]["abstract"] == "the abstract"


def test_merge_keeps_the_winner_s_own_passages_over_a_duplicate_s():
    first = {"title": "Shared", "paper_id": "a", "full_text": "first text"}
    second = {"title": "Shared", "paper_id": "b", "full_text": "second text"}

    merged = merge_and_dedupe_node(_state(snippet_papers=[first, second]))["merged_papers"]

    assert len(merged) == 1
    assert merged[0]["full_text"] == "first text"


def test_merge_works_on_state_predating_the_snippet_source():
    """snippet_papers is read with a default so a hand-built state (or a
    checkpoint written before this source existed) still merges."""
    state = {"arxiv_papers": [{"title": "A", "arxiv_id": "1"}], "semantic_scholar_papers": [], "core_papers": []}

    assert len(merge_and_dedupe_node(state)["merged_papers"]) == 1


# --- snippet search node --------------------------------------------------------


def test_snippet_node_asks_for_passages_proportional_to_max_results(monkeypatch):
    """The endpoint's limit counts passages, so --max-results has to be scaled
    up to reach a comparable number of *papers* — but it must still be honoured,
    or a deliberately small run gets one branch returning far more than the rest.
    """
    from research_pipeline.agents.literature import nodes

    captured = {}

    def _record(queries, max_results):
        captured["limit"] = max_results
        return []

    monkeypatch.setattr(nodes, "search_semantic_scholar_snippets", _record)
    nodes.search_semantic_scholar_snippets_node(
        {"search_queries": ["q"], "max_results_per_query": 2}
    )

    assert captured["limit"] == 2 * nodes.settings.snippet_passages_per_result


def test_snippet_node_contributes_an_empty_branch_when_disabled(monkeypatch):
    from dataclasses import replace

    from research_pipeline.agents.literature import nodes

    monkeypatch.setattr(nodes, "settings", replace(nodes.settings, enable_snippet_search=False))
    monkeypatch.setattr(
        nodes,
        "search_semantic_scholar_snippets",
        lambda *a, **k: pytest.fail("snippet search ran with ENABLE_SNIPPET_SEARCH off"),
    )

    assert nodes.search_semantic_scholar_snippets_node({"search_queries": ["q"]}) == {"snippet_papers": []}
