"""Tests for the deterministic eval metrics.

Paper matching gets the most attention: every recall number this harness reports
rests on it, and both failure directions are quiet. Matching too loosely inflates
recall and makes a bad change look good; matching too tightly reports misses that
were really hits and sends you chasing queries that were fine.
"""

import pytest

from research_pipeline.eval.metrics import (
    aggregate,
    duplicate_groups,
    interdisciplinary_metrics,
    literature_metrics,
    match_gold,
    paper_keys,
    pool_metrics,
    recall,
    same_paper,
)


# -- identity ----------------------------------------------------------------


def test_the_same_paper_matches_across_differing_identifiers():
    """The whole point: a gold entry recorded with a DOI must match the same
    work returned by arXiv with only an id and a title."""
    gold = {"title": "Attention Is All You Need", "doi": "10.5555/3295222"}
    returned = {"title": "Attention is all you need", "arxiv_id": "1706.03762"}
    assert same_paper(gold, returned)


def test_arxiv_versions_and_prefixes_are_one_key():
    assert same_paper({"arxiv_id": "1706.03762v2"}, {"arxiv_id": "arXiv:1706.03762"})


def test_doi_resolver_prefixes_are_stripped():
    assert same_paper({"doi": "https://doi.org/10.1/ABC"}, {"doi": "10.1/abc"})


def test_unrelated_papers_do_not_match():
    assert not same_paper(
        {"title": "Attention Is All You Need", "arxiv_id": "1706.03762"},
        {"title": "Deep Residual Learning for Image Recognition", "arxiv_id": "1512.03385"},
    )


def test_a_short_generic_title_is_not_used_as_identity_evidence():
    """'Introduction' appearing in two bibliographies is not a match, and
    treating it as one would silently inflate recall."""
    assert not same_paper({"title": "Survey"}, {"title": "Survey"})


def test_a_paper_with_no_identifiers_has_no_keys():
    assert paper_keys({"year": 2020}) == set()


# -- recall ------------------------------------------------------------------


def _gold():
    return [
        {"title": "Attention Is All You Need", "arxiv_id": "1706.03762"},
        {"title": "Deep Residual Learning for Image Recognition", "arxiv_id": "1512.03385"},
        {"title": "Batch Normalization Accelerating Deep Network Training", "arxiv_id": "1502.03167"},
    ]


def test_recall_counts_matched_gold_papers():
    returned = [{"title": "Attention Is All You Need"}, {"title": "Something Else Entirely Here"}]
    assert recall(_gold(), returned) == pytest.approx(1 / 3)


def test_recall_is_none_without_a_gold_set():
    """None, not 0.0 — 'nothing to measure against' and 'found nothing' are
    opposite findings."""
    assert recall([], [{"title": "A Paper"}]) is None


def test_match_gold_reports_which_papers_were_missed():
    found = match_gold(_gold(), [{"arxiv_id": "1706.03762"}])
    assert [p["arxiv_id"] for p in found["found"]] == ["1706.03762"]
    assert len(found["missed"]) == 2


# -- pool properties ---------------------------------------------------------


def test_duplicate_groups_catches_a_preprint_and_its_published_version():
    """Exactly what the pipeline's single doi-or-title dedupe key cannot: the
    two records key on different fields, so both survive."""
    pool = [
        {"title": "A Very Distinctive Paper Title", "arxiv_id": "2001.00001"},
        {"title": "A Very Distinctive Paper Title", "doi": "10.1/published"},
    ]
    assert len(duplicate_groups(pool)) == 1


def test_a_clean_pool_has_no_duplicate_groups():
    assert duplicate_groups([{"title": "First Distinctive Title"}, {"title": "Second Distinctive Title"}]) == []


def test_pool_metrics_reports_coverage_and_sources():
    pool = [
        {"title": "A", "abstract": "text", "source": "arxiv", "pdf_url": "u", "relevance_score": 4},
        {"title": "B", "abstract": "", "source": "core", "relevance_score": 2},
    ]
    result = pool_metrics(pool)

    assert result["pool_size"] == 2
    assert result["abstract_coverage"] == 0.5
    assert result["with_pdf"] == 0.5
    assert result["by_source"] == {"arxiv": 1, "core": 1}
    assert result["mean_relevance_score"] == 3.0


def test_pool_metrics_on_an_empty_pool_reports_none_not_zero():
    result = pool_metrics([])
    assert result["pool_size"] == 0
    assert result["abstract_coverage"] is None
    assert result["mean_relevance_score"] is None


def test_literature_metrics_carries_the_screened_out_count_through():
    output = {"merged_papers": [{"title": "Attention Is All You Need"}], "papers_filtered_out": 7}
    result = literature_metrics(_gold(), output)

    assert result["gold_found"] == 1
    assert result["screened_out"] == 7
    assert len(result["missed_titles"]) == 2


# -- interdisciplinary -------------------------------------------------------


def _inter_output(**overrides):
    core = {"title": "In Domain Retrieval Paper", "arxiv_id": "1", "fields_of_study": ["Computer Science"]}
    cross = {"title": "Ecology Rarefaction Paper", "arxiv_id": "9", "fields_of_study": ["Biology"],
             "discipline": "ecology"}
    return {
        "papers": [core, cross],
        "core_paper_ids": ["1"],
        "cross_field_papers": [cross],
        "fields_explored": [{"field": "ecology"}],
        "bridge_insights": [{"insight": "x", "supporting_paper_ids": ["9"]}],
        **overrides,
    }


def test_off_field_rate_confirms_a_cross_field_paper_really_is_cross_field():
    """The agent stamps `discipline` from the query that found it — an
    assertion. fieldsOfStudy is the independent evidence."""
    assert interdisciplinary_metrics(_inter_output())["off_field_rate"] == 1.0


def test_off_field_rate_catches_an_in_domain_paper_wearing_a_cross_field_label():
    output = _inter_output()
    output["cross_field_papers"][0]["fields_of_study"] = ["Computer Science"]
    output["papers"][1]["fields_of_study"] = ["Computer Science"]

    assert interdisciplinary_metrics(output)["off_field_rate"] == 0.0


def test_grounded_insight_rate_catches_an_insight_citing_only_in_domain_work():
    """The agent checks that cited ids *resolve*, not that they point at
    cross-field work — so an insight can cite an in-domain paper as evidence of
    a cross-field transfer and pass."""
    output = _inter_output(bridge_insights=[{"insight": "x", "supporting_paper_ids": ["1"]}])
    assert interdisciplinary_metrics(output)["grounded_insight_rate"] == 0.0


def test_field_coverage_is_none_when_no_source_reported_a_field():
    output = _inter_output()
    output["cross_field_papers"][0]["fields_of_study"] = []
    output["papers"][1]["fields_of_study"] = []

    result = interdisciplinary_metrics(output)
    assert result["field_coverage"] == 0.0
    assert result["off_field_rate"] is None


# -- aggregation -------------------------------------------------------------


def test_aggregate_skips_nones_rather_than_counting_them_as_zero():
    """A question with no gold set must not drag the mean recall down."""
    rows = [{"recall": 0.5, "gold_found": 1, "gold_total": 2}, {"recall": None, "gold_found": 0, "gold_total": 0}]
    result = aggregate(rows)

    assert result["mean_recall"] == 0.5
    assert result["questions"] == 2


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == {}


# -- citation expansion ------------------------------------------------------
#
# The eval's job for expansion is to answer whether it earns its extra API
# calls, which means separating what the queries found from what the graph did.


def _query_hit(title, arxiv_id):
    return {"title": title, "arxiv_id": arxiv_id, "source": "arxiv"}


def _graph_hit(title, arxiv_id):
    return {"title": title, "arxiv_id": arxiv_id, "source": "semantic_scholar",
            "discovered_via": "references", "cited_by_seeds": 3}


def test_gold_found_via_citations_counts_only_what_the_queries_missed():
    pool = [
        _query_hit("Attention Is All You Need", "1706.03762"),
        _graph_hit("Deep Residual Learning for Image Recognition", "1512.03385"),
    ]
    result = literature_metrics(_gold(), {"merged_papers": pool})

    assert result["gold_found"] == 2
    assert result["gold_found_via_citations"] == 1
    assert result["recall"] == pytest.approx(2 / 3)
    # What the queries alone would have scored.
    assert result["recall_without_expansion"] == pytest.approx(1 / 3)


def test_a_gold_paper_found_both_ways_is_not_credited_to_expansion():
    """Expansion re-finding something a query already had is not extra recall."""
    pool = [
        _query_hit("Attention Is All You Need", "1706.03762"),
        _graph_hit("Attention Is All You Need", "1706.03762"),
    ]
    assert literature_metrics(_gold(), {"merged_papers": pool})["gold_found_via_citations"] == 0


def test_expansion_metrics_are_zero_when_expansion_is_off():
    pool = [_query_hit("Attention Is All You Need", "1706.03762")]
    result = literature_metrics(_gold(), {"merged_papers": pool})

    assert result["from_citations"] == 0
    assert result["gold_found_via_citations"] == 0
    assert result["recall"] == result["recall_without_expansion"]


def test_pool_metrics_counts_papers_the_graph_contributed():
    pool = [_query_hit("A Paper", "1"), _graph_hit("Another Paper", "2")]
    assert pool_metrics(pool)["from_citations"] == 1


def test_aggregate_totals_the_expansion_contribution():
    rows = [
        {"from_citations": 5, "gold_found_via_citations": 2},
        {"from_citations": 3, "gold_found_via_citations": 0},
    ]
    result = aggregate(rows)

    assert result["total_from_citations"] == 8
    assert result["total_gold_via_citations"] == 2
