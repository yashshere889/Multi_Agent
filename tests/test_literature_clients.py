from unittest.mock import MagicMock, patch

import arxiv
import requests

from research_pipeline.agents.literature.clients import search_arxiv


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
