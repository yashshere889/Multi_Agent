"""Tests for gold set loading, validation, and bootstrapping.

The bootstrap tests never reach Semantic Scholar — they stand in for the
references endpoint and check the paging, filtering, and dedupe around it.
"""

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.config import settings
from research_pipeline.eval.gold import (
    GoldSetError,
    bootstrap_from_survey,
    load_gold_set,
    validate_gold_entry,
    write_gold_entry,
)


def _entry(**overrides):
    return {
        "question": "does retrieval augmentation help long-context reasoning?",
        "papers": [{"title": "Attention Is All You Need", "arxiv_id": "1706.03762"}],
        **overrides,
    }


# -- validation --------------------------------------------------------------


def test_validate_accepts_a_well_formed_entry():
    validate_gold_entry(_entry())  # should not raise


def test_validate_rejects_a_missing_question():
    with pytest.raises(GoldSetError, match="question is missing"):
        validate_gold_entry(_entry(question="  "))


def test_validate_rejects_an_empty_paper_list():
    """A gold set with no papers can't measure recall, so it's a configuration
    error rather than a zero score."""
    with pytest.raises(GoldSetError, match="papers is empty"):
        validate_gold_entry(_entry(papers=[]))


def test_validate_rejects_a_paper_with_nothing_to_match_on():
    """Such an entry can only ever count as a miss, silently depressing recall."""
    with pytest.raises(GoldSetError, match="no title, doi, or arxiv_id"):
        validate_gold_entry(_entry(papers=[{"year": 2020}]))


def test_validate_reports_every_problem_at_once():
    with pytest.raises(GoldSetError) as exc:
        validate_gold_entry({"question": "", "papers": []})
    assert "question" in str(exc.value) and "papers" in str(exc.value)


# -- loading -----------------------------------------------------------------


def test_load_gold_set_reads_a_single_file(tmp_path):
    path = write_gold_entry(_entry(), tmp_path / "one.json")
    assert len(load_gold_set(path)) == 1


def test_load_gold_set_reads_a_directory_in_a_stable_order(tmp_path):
    write_gold_entry(_entry(question="b question"), tmp_path / "b.json")
    write_gold_entry(_entry(question="a question"), tmp_path / "a.json")

    assert [e["question"] for e in load_gold_set(tmp_path)] == ["a question", "b question"]


def test_load_gold_set_rejects_an_empty_directory(tmp_path):
    with pytest.raises(GoldSetError, match="No gold set files"):
        load_gold_set(tmp_path)


def test_load_gold_set_reports_a_missing_path(tmp_path):
    with pytest.raises(GoldSetError, match="not found"):
        load_gold_set(tmp_path / "nope.json")


def test_load_gold_set_reports_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(GoldSetError, match="not valid JSON"):
        load_gold_set(bad)


def test_write_gold_entry_validates_before_writing(tmp_path):
    with pytest.raises(GoldSetError):
        write_gold_entry(_entry(papers=[]), tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


# -- bootstrap ---------------------------------------------------------------


def _response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def _reference(title, doi=None, arxiv=None, year=2022):
    external = {}
    if doi:
        external["DOI"] = doi
    if arxiv:
        external["ArXiv"] = arxiv
    return {"citedPaper": {"title": title, "externalIds": external, "year": year}}


@pytest.fixture(autouse=True)
def _quiet_key(monkeypatch):
    monkeypatch.setattr(
        "research_pipeline.eval.gold.settings", replace(settings, semantic_scholar_api_key="fake-key")
    )


def _bootstrap(responses, **kwargs):
    with patch("research_pipeline.eval.gold._request_with_retry", side_effect=responses):
        return bootstrap_from_survey("arXiv:2312.10997", "does RAG help?", **kwargs)


def test_bootstrap_records_the_surveys_real_references():
    entry = _bootstrap([
        _response({"title": "A Survey of RAG", "year": 2023, "referenceCount": 2}),
        _response({"data": [_reference("Paper One", arxiv="1706.03762"), _reference("Paper Two", doi="10.1/x")]}),
    ])

    assert entry["question"] == "does RAG help?"
    assert [p["title"] for p in entry["papers"]] == ["Paper One", "Paper Two"]
    assert entry["papers"][0]["arxiv_id"] == "1706.03762"
    # The provenance is recorded so anyone can re-fetch and audit the list.
    assert entry["source"]["survey_title"] == "A Survey of RAG"
    assert entry["source"]["kind"] == "survey_references"


def test_bootstrap_applies_the_year_cutoff():
    entry = _bootstrap([
        _response({"title": "A Survey", "year": 2023}),
        _response({"data": [_reference("Old Work", arxiv="1", year=1994),
                            _reference("Recent Work", arxiv="2", year=2021)]}),
    ], min_year=2015)

    assert [p["title"] for p in entry["papers"]] == ["Recent Work"]


def test_bootstrap_dedupes_repeated_references():
    entry = _bootstrap([
        _response({"title": "A Survey", "year": 2023}),
        _response({"data": [_reference("Same Paper", doi="10.1/x"), _reference("Same Paper", doi="10.1/x")]}),
    ])

    assert len(entry["papers"]) == 1


def test_bootstrap_skips_references_with_no_title():
    entry = _bootstrap([
        _response({"title": "A Survey", "year": 2023}),
        _response({"data": [{"citedPaper": {"title": None}}, _reference("Real Paper", arxiv="1")]}),
    ])

    assert [p["title"] for p in entry["papers"]] == ["Real Paper"]


def test_bootstrap_raises_when_the_survey_cannot_be_resolved():
    with pytest.raises(GoldSetError, match="404"):
        _bootstrap([_response({"error": "not found"}, status=404)])


def test_bootstrap_raises_rather_than_writing_an_empty_gold_set():
    """An empty bibliography is nearly always a bad id or an unparsed
    publisher, and a silently empty gold set would score every run at zero."""
    with pytest.raises(GoldSetError, match="No usable references"):
        _bootstrap([
            _response({"title": "A Survey", "year": 2023}),
            _response({"data": []}),
        ])
