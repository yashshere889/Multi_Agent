"""A claim the Reviewer cleared is not a hallucination against the paper.

Barkla job 10522992's handed-over draft carried 51 "hallucinations", 11 of them
entries whose own issue text ended "Therefore, this is not a hallucination" —
the model returning the claims it had examined, verdict and all. Counting those
inflates the reported number, misdirects `best_iteration` (which hands over the
draft with the fewest counted issues), and sends the Writer revision feedback on
sentences the Reviewer agreed with.
"""

from research_pipeline.agents.reviewer.reviewer_agent import _ungrounded_only

# Verbatim from that run's review_20260915T035737Z.json.
CLEARED = {
    "claim": "The dynamic strategy showed lower sensitivity to early negative returns.",
    "issue": (
        "This claim is grounded in the ground truth, which states that the dynamic strategy "
        "shows significantly lower sensitivity (p=0.0000, 95% CI [0.08, 0.08]), supporting the "
        "hypothesis. Therefore, this is not a hallucination."
    ),
}
REAL = {
    "claim": "assessing the impact of delayed retirement on elderly welfare in China",
    "issue": (
        "The ground truth abstract for Butrica et al. (n.d.) does not mention anything about "
        "'assessing the impact of delayed retirement on elderly welfare in China'. The abstract "
        "focuses on the U.S. context."
    ),
}
# The shape the filter must not mistake for an exoneration: grounded in part,
# ungrounded in what the draft added on top.
PARTLY_GROUNDED = {
    "claim": "Ibrahim et al. (2021) analyse pension liabilities across nine scenarios using PUC.",
    "issue": (
        "The ground truth abstract is supported by the ground truth on the PUC method, but the "
        "draft adds a claim about salary growth dominating that the abstract does not state."
    ),
}


def test_a_claim_the_reviewer_cleared_in_prose_is_dropped():
    assert _ungrounded_only("Results", [CLEARED]) == []


def test_the_grounded_field_is_enough_on_its_own():
    entry = dict(REAL, grounded=True)
    assert _ungrounded_only("Related Work", [entry]) == []


def test_a_real_flag_survives_and_keeps_its_location():
    kept = _ungrounded_only("Related Work", [REAL])

    assert len(kept) == 1
    assert kept[0]["location"] == "Related Work"
    assert kept[0]["claim"] == REAL["claim"]
    assert "does not mention" in kept[0]["issue"]


def test_a_partly_grounded_finding_is_not_mistaken_for_an_exoneration():
    assert len(_ungrounded_only("Related Work", [PARTLY_GROUNDED])) == 1


def test_grounded_false_is_kept_and_a_missing_field_is_kept():
    entries = [dict(REAL, grounded=False), REAL]
    assert len(_ungrounded_only("Related Work", entries)) == 2


def test_the_mixed_list_from_that_run_keeps_only_the_real_finding():
    kept = _ungrounded_only("Results", [CLEARED, REAL, dict(CLEARED, grounded=True)])

    assert [entry["claim"] for entry in kept] == [REAL["claim"]]


def test_junk_entries_are_ignored_rather_than_crashing():
    assert _ungrounded_only("Results", ["nonsense", None, 3, REAL]) == [
        {"location": "Results", "claim": REAL["claim"], "issue": REAL["issue"]}
    ]
    assert _ungrounded_only("Results", None) == []
