"""A name suffix is not a surname, and treating it as one fabricates a citation.

Barkla job 10550577's paper cited Melvin Stephens Jr. and Steven Haider as
"(Jr. and Haider, ...)", and the Reviewer flagged it correctly: "the ground
truth does not list any paper by 'Jr. and Haider'. This citation appears to be
fabricated or misattributed." A name-parsing bug producing a real hallucination
in the paper and an entry in the flag count at the same time.
"""

from research_pipeline.agents.writer.citations import (
    build_surname_year_lookup,
    first_author_surname,
    surname_of,
)


def test_a_generational_suffix_is_skipped():
    assert surname_of("Melvin Stephens Jr.") == "Stephens"
    assert surname_of("John Smith III") == "Smith"
    assert surname_of("Robert Downey Sr") == "Downey"


def test_a_post_nominal_is_skipped():
    assert surname_of("A. Researcher, PhD") == "Researcher"
    assert surname_of("Jane Doe MD") == "Doe"


def test_the_surname_first_form_with_a_suffix_still_resolves():
    assert surname_of("Stephens Jr., Melvin") == "Stephens"


def test_the_forms_that_already_worked_are_unchanged():
    # CORE surname-first, and arXiv/S2 given-name-first.
    assert surname_of("Horneff, Wolfram J.") == "Horneff"
    assert surname_of("Wolfram J. Horneff") == "Horneff"
    assert surname_of("Steven Haider") == "Haider"


def test_a_bare_suffix_is_never_stripped_to_nothing():
    assert surname_of("Jr.") == "Jr."
    assert surname_of("") == "Unknown"
    assert first_author_surname([]) == "Unknown"


def test_the_grounding_lookup_keys_on_the_corrected_surname():
    """The printed citation and the check that validates it must agree."""
    index = {
        "p1": {
            "id": "p1",
            "title": "Is There a Retirement-Consumption Puzzle?",
            "authors": ["Melvin Stephens Jr.", "Steven Haider"],
            "year": 2004,
            "source": "semantic_scholar",
            "url": None,
        }
    }
    lookup = build_surname_year_lookup(index)

    assert ("stephens", "2004") in lookup
    assert ("jr.", "2004") not in lookup
