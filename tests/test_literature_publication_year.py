"""A year a source did supply must not print as "n.d." — and one it didn't must.

r15's reference list carried 26 "n.d." citations and r13's 35. Sources leave
their own year field empty while still filling a publication-date field, so the
year is often recoverable. What is never acceptable is inventing one: a
repository deposit date is not a publication year, and a wrong year in a
citation is a factual error where "n.d." is a true statement.
"""

from research_pipeline.agents.literature.clients import (
    paper_from_semantic_scholar,
    publication_year,
)
from research_pipeline.agents.writer.citations import year_text


def test_an_explicit_year_wins():
    assert publication_year({"year": 2021}) == 2021
    assert publication_year({"yearPublished": 2019}) == 2019


def test_a_year_given_as_a_string_is_accepted():
    assert publication_year({"year": "2020"}) == 2020
    assert publication_year({"yearPublished": " 1998 "}) == 1998


def test_a_publication_date_is_used_when_the_year_field_is_empty():
    assert publication_year({"year": None, "publishedDate": "2023-04-17T00:00:00"}) == 2023
    assert publication_year({"publicationDate": "2007-11-01"}) == 2007
    assert publication_year({"datePublished": "1996"}) == 1996


def test_a_deposit_or_creation_date_is_never_used_as_a_publication_year():
    """The dangerous direction: a fabricated year reads as authoritative."""
    assert publication_year({"depositedDate": "2024-01-01", "createdDate": "2024-02-02"}) is None


def test_nothing_datelike_stays_absent_and_prints_as_nd():
    assert publication_year({}) is None
    assert publication_year({"year": None, "title": "A 2020 retrospective"}) is None
    assert publication_year(None) is None
    assert year_text(publication_year({})) == "n.d."


def test_a_boolean_is_not_a_year():
    assert publication_year({"year": True}) is None


def test_the_semantic_scholar_mapper_recovers_a_missing_year():
    recovered = paper_from_semantic_scholar(
        {
            "paperId": "abc",
            "title": "Sequence risk in retirement",
            "authors": [{"name": "A. Researcher"}],
            "year": None,
            "publicationDate": "2018-06-01",
        }
    )
    assert recovered["year"] == 2018

    unchanged = paper_from_semantic_scholar(
        {"paperId": "d", "title": "t", "authors": [], "year": 2011}
    )
    assert unchanged["year"] == 2011
