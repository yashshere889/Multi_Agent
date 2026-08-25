"""Tests for citation-graph expansion.

The behaviours worth pinning down are the ones that make expansion safe to leave
on: it never re-adds a paper the pool already has, it never grows past its
budget, and every failure mode adds nothing rather than breaking the run.
Co-citation ranking gets its own attention because it is the whole reason
expansion doesn't just dump a bibliography into the pool.
"""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.agents.literature import clients, expansion
from research_pipeline.agents.literature.nodes import dedupe_key
from research_pipeline.config import settings


def _paper(title, *, arxiv_id=None, doi=None, paper_id=None, score=None, citations=None, source="arxiv"):
    paper = {"title": title, "source": source}
    if arxiv_id:
        paper["arxiv_id"] = arxiv_id
    if doi:
        paper["doi"] = doi
    if paper_id:
        paper["paper_id"] = paper_id
    if score is not None:
        paper["relevance_score"] = score
    if citations is not None:
        paper["citation_count"] = citations
    return paper


# -- identifying a seed to S2 ------------------------------------------------


def test_an_s2_paper_uses_its_own_id_needing_no_resolution():
    paper = _paper("A", paper_id="abc123", doi="10.1/x", source="semantic_scholar")
    assert clients.semantic_scholar_identifier(paper) == "abc123"


def test_a_doi_is_preferred_over_an_arxiv_id():
    """The same work often has both a preprint and a published record; the DOI
    is the unambiguous one."""
    assert clients.semantic_scholar_identifier(_paper("A", doi="10.1/x", arxiv_id="1706.03762")) == "DOI:10.1/x"


def test_an_arxiv_version_suffix_is_stripped():
    """S2 resolves arXiv:1706.03762, not arXiv:1706.03762v2."""
    assert clients.semantic_scholar_identifier(_paper("A", arxiv_id="1706.03762v2")) == "arXiv:1706.03762"


def test_a_paper_with_no_resolvable_id_returns_none():
    assert clients.semantic_scholar_identifier({"title": "A", "source": "core"}) is None


# -- seed selection ----------------------------------------------------------


def test_seeds_are_the_highest_scoring_papers():
    papers = [_paper("low", arxiv_id="1", score=1), _paper("high", arxiv_id="2", score=5),
              _paper("mid", arxiv_id="3", score=3)]
    assert [p["title"] for p in expansion.choose_seeds(papers, 2)] == ["high", "mid"]


def test_scored_papers_outrank_unscored_ones():
    """Unscored is not evidence of quality either way, but a hop is expensive
    enough to spend on the papers we do know something about."""
    papers = [_paper("unscored", arxiv_id="1"), _paper("scored low", arxiv_id="2", score=1)]
    assert [p["title"] for p in expansion.choose_seeds(papers, 1)] == ["scored low"]


def test_citation_count_breaks_a_score_tie():
    papers = [_paper("quiet", arxiv_id="1", score=4, citations=3),
              _paper("seminal", arxiv_id="2", score=4, citations=9000)]
    assert expansion.choose_seeds(papers, 1)[0]["title"] == "seminal"


def test_papers_with_no_resolvable_id_are_not_chosen_as_seeds():
    papers = [{"title": "unresolvable", "source": "core", "relevance_score": 5},
              _paper("resolvable", arxiv_id="1", score=1)]
    assert [p["title"] for p in expansion.choose_seeds(papers, 5)] == ["resolvable"]


def test_a_seed_limit_of_zero_selects_nothing():
    assert expansion.choose_seeds([_paper("a", arxiv_id="1")], 0) == []


# -- ranking -----------------------------------------------------------------


def test_candidates_are_ranked_by_how_many_seeds_cite_them():
    """The core claim: a paper cited by several independent seeds is central to
    the problem; one cited by a single seed is often that seed's own tangent."""
    hops = [
        [_paper("cited by all three", doi="10.1/a"), _paper("tangent of seed one", doi="10.1/b")],
        [_paper("cited by all three", doi="10.1/a")],
        [_paper("cited by all three", doi="10.1/a"), _paper("cited by two", doi="10.1/c")],
    ]
    hops[1].append(_paper("cited by two", doi="10.1/c"))

    ranked = expansion.rank_candidates(hops, dedupe_key, set())

    assert [p["title"] for p in ranked] == ["cited by all three", "cited by two", "tangent of seed one"]
    assert ranked[0]["cited_by_seeds"] == 3


def test_a_bibliography_listing_a_paper_twice_is_not_two_seeds_agreeing():
    hops = [[_paper("dup", doi="10.1/a"), _paper("dup", doi="10.1/a")]]
    assert expansion.rank_candidates(hops, dedupe_key, set())[0]["cited_by_seeds"] == 1


def test_papers_already_in_the_pool_are_not_returned_again():
    known = {dedupe_key(_paper("already have", doi="10.1/a"))}
    hops = [[_paper("already have", doi="10.1/a"), _paper("new one", doi="10.1/b")]]

    assert [p["title"] for p in expansion.rank_candidates(hops, dedupe_key, known)] == ["new one"]


def test_untitled_candidates_are_discarded():
    hops = [[{"title": "", "doi": "10.1/a"}, _paper("real", doi="10.1/b")]]
    assert [p["title"] for p in expansion.rank_candidates(hops, dedupe_key, set())] == ["real"]


def test_ranking_does_not_mutate_the_fetched_papers():
    original = _paper("a", doi="10.1/a")
    expansion.rank_candidates([[original]], dedupe_key, set())
    assert "cited_by_seeds" not in original


# -- the round ---------------------------------------------------------------


def _fetcher(*results):
    calls = []

    def fetch(paper_id, direction, limit):
        calls.append((paper_id, direction, limit))
        return [dict(p) for p in (results[min(len(calls) - 1, len(results) - 1)] if results else [])]

    fetch.calls = calls
    return fetch


def _expand(papers, fetch, **kwargs):
    options = {"directions": ("references",), "max_seeds": 2, "per_seed": 50, "max_new_papers": 10}
    options.update(kwargs)
    return expansion.expand(papers, dedupe_key, fetch, **options)


def test_expand_returns_papers_not_already_held():
    pool = [_paper("seed", arxiv_id="1", score=5)]
    fetch = _fetcher([_paper("found", doi="10.1/new")])

    found = _expand(pool, fetch)

    assert [p["title"] for p in found] == ["found"]
    assert fetch.calls == [("arXiv:1", "references", 50)]


def test_expand_respects_the_new_paper_budget():
    """The cap is what keeps expansion a supplement to the search rather than a
    replacement for it."""
    pool = [_paper("seed", arxiv_id="1", score=5)]
    fetch = _fetcher([_paper(f"found {i}", doi=f"10.1/{i}") for i in range(30)])

    assert len(_expand(pool, fetch, max_new_papers=4)) == 4


def test_expand_hops_once_per_seed_and_direction():
    pool = [_paper("a", arxiv_id="1", score=5), _paper("b", arxiv_id="2", score=4)]
    fetch = _fetcher([])

    _expand(pool, fetch, directions=("references", "citations"), max_seeds=2)

    assert fetch.calls == [
        ("arXiv:1", "references", 50), ("arXiv:1", "citations", 50),
        ("arXiv:2", "references", 50), ("arXiv:2", "citations", 50),
    ]


def test_expand_adds_nothing_when_no_seed_can_be_resolved():
    fetch = _fetcher([_paper("never fetched", doi="10.1/x")])
    assert _expand([{"title": "no ids", "source": "core"}], fetch) == []
    assert fetch.calls == []


def test_expand_adds_nothing_when_every_hop_comes_back_empty():
    assert _expand([_paper("seed", arxiv_id="1", score=5)], _fetcher([])) == []


def test_expand_survives_a_pool_of_one_unscored_paper():
    pool = [_paper("seed", arxiv_id="1")]
    assert [p["title"] for p in _expand(pool, _fetcher([_paper("found", doi="10.1/n")]))] == ["found"]


# -- the client hop ----------------------------------------------------------


def _response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = "{}"
    return resp


@pytest.fixture(autouse=True)
def _with_key(monkeypatch):
    monkeypatch.setattr(clients, "settings", replace(settings, semantic_scholar_api_key="fake-key"))
    monkeypatch.setattr(clients.time, "sleep", lambda _s: None)


def test_fetch_related_maps_the_cited_paper():
    payload = {"data": [{"citedPaper": {
        "paperId": "abc", "title": "A Referenced Paper", "abstract": "text",
        "authors": [{"name": "A. Uthor"}], "year": 2019, "citationCount": 500,
        "externalIds": {"DOI": "10.1/ref"},
    }}]}
    with patch.object(clients, "_request_with_retry", return_value=_response(payload)):
        papers = clients.fetch_related("arXiv:1706.03762", "references", 50)

    assert len(papers) == 1
    assert papers[0]["title"] == "A Referenced Paper"
    assert papers[0]["citation_count"] == 500
    assert papers[0]["doi"] == "10.1/ref"
    # Recorded so the eval harness can tell query hits from graph hits.
    assert papers[0]["discovered_via"] == "references"


def test_fetch_related_reads_the_citing_paper_in_the_other_direction():
    payload = {"data": [{"citingPaper": {"paperId": "x", "title": "A Citing Paper"}}]}
    with patch.object(clients, "_request_with_retry", return_value=_response(payload)) as request:
        papers = clients.fetch_related("abc", "citations", 10)

    assert papers[0]["title"] == "A Citing Paper"
    assert "citingPaper.title" in request.call_args.kwargs["params"]["fields"]


def test_fetch_related_returns_nothing_without_an_api_key(monkeypatch):
    monkeypatch.setattr(clients, "settings", replace(settings, semantic_scholar_api_key=""))
    with patch.object(clients, "_request_with_retry") as request:
        assert clients.fetch_related("abc", "references", 50) == []
    request.assert_not_called()


def test_a_paper_with_no_parsed_bibliography_is_routine_not_fatal():
    """404s here are common — many publishers' bibliographies are simply absent."""
    with patch.object(clients, "_request_with_retry", return_value=_response({}, status=404)):
        assert clients.fetch_related("abc", "references", 50) == []


def test_fetch_related_survives_a_connection_error():
    import requests

    with patch.object(clients, "_request_with_retry", side_effect=requests.ConnectionError("down")):
        assert clients.fetch_related("abc", "references", 50) == []


def test_fetch_related_survives_an_unparseable_response():
    resp = _response({})
    resp.json.side_effect = ValueError("not json")
    with patch.object(clients, "_request_with_retry", return_value=resp):
        assert clients.fetch_related("abc", "references", 50) == []


def test_fetch_related_skips_untitled_entries():
    payload = {"data": [{"citedPaper": {"title": None}}, {"citedPaper": {"paperId": "y", "title": "Real"}}, {}]}
    with patch.object(clients, "_request_with_retry", return_value=_response(payload)):
        assert [p["title"] for p in clients.fetch_related("abc", "references", 50)] == ["Real"]


def test_an_unknown_direction_is_a_programming_error():
    with pytest.raises(ValueError, match="direction must be one of"):
        clients.fetch_related("abc", "sideways", 50)
