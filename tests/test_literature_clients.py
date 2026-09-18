from dataclasses import replace
from unittest.mock import MagicMock, patch

import arxiv
import requests

from research_pipeline.agents.literature.clients import (
    SEMANTIC_SCHOLAR_FIELDS,
    search_arxiv,
    search_core,
    search_semantic_scholar,
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
    result.categories = ["cs.CL", "cs.LG"]
    result.journal_ref = None
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


# -- bibliometric fields -----------------------------------------------------
#
# citationCount / venue / fieldsOfStudy / tldr are extra `fields` on a request
# already being made, so they cost nothing — but each has a consumer: the counts
# and venue are a deterministic quality prior, fieldsOfStudy lets the
# Interdisciplinary Agent check a cross-field paper's discipline against
# something other than the query that found it, and tldr is a denser digest than
# a truncated abstract. Coverage differs per source; the *shape* must not.


def _fake_s2_response(papers):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": papers}
    resp.text = ""
    return resp


def _search_s2(monkeypatch, paper):
    monkeypatch.setattr(
        "research_pipeline.agents.literature.clients.settings",
        replace(settings, semantic_scholar_api_key="fake-key"),
    )
    monkeypatch.setattr("research_pipeline.agents.literature.clients.time.sleep", lambda _s: None)
    with patch(
        "research_pipeline.agents.literature.clients.requests.request",
        return_value=_fake_s2_response([paper]),
    ):
        return search_semantic_scholar(["query"], max_results=5)[0]


def test_semantic_scholar_requests_the_bibliometric_fields():
    for field in ("citationCount", "influentialCitationCount", "venue", "fieldsOfStudy", "tldr"):
        assert field in SEMANTIC_SCHOLAR_FIELDS


def test_semantic_scholar_maps_the_bibliometric_fields(monkeypatch):
    paper = _search_s2(monkeypatch, {
        "paperId": "abc",
        "title": "A Paper",
        "abstract": "abstract text",
        "authors": [],
        "citationCount": 412,
        "influentialCitationCount": 37,
        "venue": "NeurIPS",
        "fieldsOfStudy": ["Computer Science", None],
        "tldr": {"model": "tldr@v2", "text": "  the short version  "},
    })

    assert paper["citation_count"] == 412
    assert paper["influential_citation_count"] == 37
    assert paper["venue"] == "NeurIPS"
    assert paper["fields_of_study"] == ["Computer Science"]  # the null is dropped
    assert paper["tldr"] == "the short version"


def test_semantic_scholar_normalizes_absent_bibliometrics_to_none(monkeypatch):
    """Every one of these is nullable in a real response — a paper with no
    venue, no tldr, or no fieldsOfStudy is common and must not propagate a
    null-ish value of a different type."""
    paper = _search_s2(monkeypatch, {
        "paperId": "abc", "title": "A Paper", "authors": [],
        "venue": "", "fieldsOfStudy": None, "tldr": None,
    })

    assert paper["citation_count"] is None
    assert paper["venue"] is None
    assert paper["fields_of_study"] == []
    assert paper["tldr"] is None


def test_arxiv_maps_categories_as_its_field_of_study_signal():
    fake_client = MagicMock()
    fake_client.results.return_value = iter([_fake_result("1234.5678", "A Paper")])

    with patch("research_pipeline.agents.literature.clients.arxiv.Client", return_value=fake_client):
        paper = search_arxiv(["query"], max_results=5)[0]

    assert paper["fields_of_study"] == ["cs.CL", "cs.LG"]
    # arXiv publishes neither, so the keys exist only to keep one shape.
    assert paper["citation_count"] is None
    assert paper["tldr"] is None


def test_core_wraps_its_single_field_of_study_in_a_list(monkeypatch):
    monkeypatch.setattr(
        "research_pipeline.agents.literature.clients.settings", replace(settings, core_api_key="fake-key")
    )
    data = {"results": [{"id": 1, "title": "A CORE Paper", "fieldOfStudy": "biology", "publisher": "Elsevier"}]}
    with patch(
        "research_pipeline.agents.literature.clients.requests.request",
        return_value=_fake_core_response(200, data),
    ):
        paper = search_core(["query"], max_results=5)[0]

    assert paper["fields_of_study"] == ["biology"]
    assert paper["venue"] == "Elsevier"
