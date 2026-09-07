from dataclasses import replace
from unittest.mock import MagicMock, patch

import arxiv
import requests

from research_pipeline.agents.literature.clients import (
    MAX_SNIPPETS_PER_PAPER,
    search_arxiv,
    search_core,
    search_semantic_scholar_snippets,
)
from research_pipeline.config import settings


def _fake_result(short_id: str, title: str):
    result = MagicMock()
    result.get_short_id.return_value = short_id
    result.title = title
    result.authors = []
    result.summary = "abstract"
    result.published.year = 2024
    result.pdf_url = f"https://arxiv.org/pdf/{short_id}"
    result.doi = None
    result.entry_id = f"https://arxiv.org/abs/{short_id}"
    return result


def test_search_arxiv_skips_failing_query_and_keeps_other_results():
    good_result = _fake_result("1234.5678", "A Good Paper")

    def fake_results(search):
        if "boom" in search.query:
            raise arxiv.HTTPError(url="https://export.arxiv.org/api/query", retry=3, status=429)
        return iter([good_result])

    fake_client = MagicMock()
    fake_client.results.side_effect = fake_results

    with patch("research_pipeline.agents.literature.clients.arxiv.Client", return_value=fake_client):
        papers = search_arxiv(["boom query", "fine query"], max_results=5)

    assert len(papers) == 1
    assert papers[0]["arxiv_id"] == "1234.5678"


def test_search_arxiv_survives_connection_error():
    fake_client = MagicMock()
    fake_client.results.side_effect = requests.ConnectionError("no route to host")

    with patch("research_pipeline.agents.literature.clients.arxiv.Client", return_value=fake_client):
        papers = search_arxiv(["any query"], max_results=5)

    assert papers == []


def _fake_core_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def test_search_core_returns_empty_list_and_warns_when_key_is_unset(monkeypatch):
    monkeypatch.setattr(
        "research_pipeline.agents.literature.clients.settings", replace(settings, core_api_key="")
    )
    with patch("research_pipeline.agents.literature.clients.requests.request") as mock_request:
        papers = search_core(["query"], max_results=5)

    assert papers == []
    mock_request.assert_not_called()


def test_search_core_normalizes_core_response(monkeypatch):
    monkeypatch.setattr(
        "research_pipeline.agents.literature.clients.settings", replace(settings, core_api_key="fake-key")
    )
    data = {
        "results": [{
            "id": 12345,
            "doi": "10.1/abc",
            "title": "A CORE Paper",
            "authors": [{"name": "A. Uthor"}],
            "abstract": "abstract text",
            "yearPublished": 2023,
            "downloadUrl": "https://core.ac.uk/download/12345.pdf",
        }]
    }
    with patch(
        "research_pipeline.agents.literature.clients.requests.request",
        return_value=_fake_core_response(200, data),
    ):
        papers = search_core(["query"], max_results=5)

    assert len(papers) == 1
    paper = papers[0]
    assert paper["source"] == "core"
    assert paper["paper_id"] == "12345"
    assert paper["title"] == "A CORE Paper"
    assert paper["authors"] == ["A. Uthor"]
    assert paper["year"] == 2023
    assert paper["pdf_url"] == "https://core.ac.uk/download/12345.pdf"
    assert paper["doi"] == "10.1/abc"


def test_search_core_skips_failing_query_and_keeps_going(monkeypatch):
    monkeypatch.setattr(
        "research_pipeline.agents.literature.clients.settings", replace(settings, core_api_key="fake-key")
    )
    with patch(
        "research_pipeline.agents.literature.clients.requests.request",
        return_value=_fake_core_response(403, text="forbidden"),
    ):
        papers = search_core(["query"], max_results=5)

    assert papers == []


def test_search_core_dedupes_within_source(monkeypatch):
    monkeypatch.setattr(
        "research_pipeline.agents.literature.clients.settings", replace(settings, core_api_key="fake-key")
    )
    monkeypatch.setattr("research_pipeline.agents.literature.clients.time.sleep", lambda _seconds: None)
    data = {"results": [{"id": 1, "title": "Dup"}, {"id": 1, "title": "Dup"}]}
    with patch(
        "research_pipeline.agents.literature.clients.requests.request",
        return_value=_fake_core_response(200, data),
    ):
        papers = search_core(["q1", "q2"], max_results=5)

    assert len(papers) == 1


# --- Semantic Scholar snippet (passage) search ---------------------------------
#
# Two requests make up one snippet search: GET /snippet/search for the passages,
# then POST /paper/batch to hydrate the metadata that endpoint does not return
# (no abstract, year, DOI or PDF link). The fakes below route on URL so a test
# can fail one without failing the other.


def _snippet_match(corpus_id, text, score=0.5, title="A Snippet Paper", authors=("S. Nippet",)):
    return {
        "score": score,
        "snippet": {"text": text, "snippetKind": "body"},
        "paper": {"corpusId": corpus_id, "title": title, "authors": list(authors)},
    }


def _routing_request(snippet_response, batch_response):
    def _request(method, url, **kwargs):
        return batch_response if "paper/batch" in url else snippet_response
    return _request


def _snippet_client(monkeypatch, snippet_response, batch_response=None):
    monkeypatch.setattr(
        "research_pipeline.agents.literature.clients.settings",
        replace(settings, semantic_scholar_api_key="fake-key"),
    )
    monkeypatch.setattr("research_pipeline.agents.literature.clients.time.sleep", lambda _seconds: None)
    return patch(
        "research_pipeline.agents.literature.clients.requests.request",
        side_effect=_routing_request(snippet_response, batch_response or _fake_core_response(200, [])),
    )


def test_snippet_search_skips_entirely_when_key_is_unset(monkeypatch):
    monkeypatch.setattr(
        "research_pipeline.agents.literature.clients.settings",
        replace(settings, semantic_scholar_api_key=""),
    )
    with patch("research_pipeline.agents.literature.clients.requests.request") as mock_request:
        papers = search_semantic_scholar_snippets(["query"], max_results=20)

    assert papers == []
    mock_request.assert_not_called()


def test_snippet_search_groups_passages_into_one_hydrated_paper(monkeypatch):
    snippets = _fake_core_response(200, {"data": [
        _snippet_match("777", "first passage", score=0.9),
        _snippet_match("777", "second passage", score=0.4),
    ]})
    batch = _fake_core_response(200, [{
        "paperId": "sha-777",
        "title": "A Snippet Paper",
        "abstract": "the real abstract",
        "authors": [{"name": "S. Nippet"}],
        "year": 2024,
        "externalIds": {"DOI": "10.1/snip", "CorpusId": 777},
        "openAccessPdf": {"url": "https://example.org/777.pdf"},
        "url": "https://www.semanticscholar.org/paper/sha-777",
    }])
    with _snippet_client(monkeypatch, snippets, batch):
        papers = search_semantic_scholar_snippets(["query"], max_results=20)

    assert len(papers) == 1
    paper = papers[0]
    assert paper["source"] == "semantic_scholar_snippets"
    assert paper["paper_id"] == "sha-777"
    # Hydration fills in everything /snippet/search itself never returns.
    assert paper["abstract"] == "the real abstract"
    assert paper["year"] == 2024
    assert paper["doi"] == "10.1/snip"
    assert paper["pdf_url"] == "https://example.org/777.pdf"
    # Highest-scoring passage first, both kept, one paper.
    assert paper["full_text"] == "first passage\n\nsecond passage"


def test_snippet_search_keeps_the_paper_when_hydration_fails(monkeypatch):
    """Losing the metadata is survivable; losing the passages to the same
    failure would not be, so hydration degrades rather than raising."""
    snippets = _fake_core_response(200, {"data": [_snippet_match("777", "a passage")]})
    with _snippet_client(monkeypatch, snippets, _fake_core_response(500, text="boom")):
        papers = search_semantic_scholar_snippets(["query"], max_results=20)

    assert len(papers) == 1
    assert papers[0]["title"] == "A Snippet Paper"
    assert papers[0]["authors"] == ["S. Nippet"]
    assert papers[0]["full_text"] == "a passage"
    assert papers[0]["paper_id"] == "CorpusId:777"
    assert papers[0]["abstract"] == ""
    assert papers[0]["doi"] is None


def test_snippet_search_caps_passages_per_paper_keeping_the_best(monkeypatch):
    matches = [
        _snippet_match("777", f"passage {i}", score=i / 100)
        for i in range(MAX_SNIPPETS_PER_PAPER + 3)
    ]
    with _snippet_client(monkeypatch, _fake_core_response(200, {"data": matches})):
        papers = search_semantic_scholar_snippets(["query"], max_results=20)

    kept = papers[0]["full_text"].split("\n\n")
    assert len(kept) == MAX_SNIPPETS_PER_PAPER
    assert kept[0] == f"passage {MAX_SNIPPETS_PER_PAPER + 2}"
    assert "passage 0" not in kept


def test_snippet_search_dedupes_the_same_passage_across_queries(monkeypatch):
    with _snippet_client(
        monkeypatch, _fake_core_response(200, {"data": [_snippet_match("777", "same passage")]})
    ):
        papers = search_semantic_scholar_snippets(["q1", "q2"], max_results=20)

    assert len(papers) == 1
    assert papers[0]["full_text"] == "same passage"


def test_snippet_search_skips_a_failing_query_without_raising(monkeypatch):
    with _snippet_client(monkeypatch, _fake_core_response(429, text="rate limited")):
        papers = search_semantic_scholar_snippets(["query"], max_results=20)

    assert papers == []


def test_snippet_search_ignores_matches_with_no_paper_or_no_text(monkeypatch):
    data = {"data": [
        {"score": 0.5, "snippet": {"text": "orphan passage"}, "paper": {}},
        _snippet_match("777", "   "),
        _snippet_match("888", "a real passage"),
    ]}
    with _snippet_client(monkeypatch, _fake_core_response(200, data)):
        papers = search_semantic_scholar_snippets(["query"], max_results=20)

    assert [p["full_text"] for p in papers] == ["a real passage"]
