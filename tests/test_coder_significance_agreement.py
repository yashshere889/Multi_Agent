"""Calling a difference significant has to agree with the test beside it.

Barkla job 10523041 reported `sensitivity_difference_p_value = 1.0` and wrote
"shows significantly lower sensitivity" in the same results.json. The check
added for job 10510547 passed it: a p-value existed, which was all it asked.
"""

from research_pipeline.agents.coder.sandbox import check_significance_claim

NOTES = (
    "Dynamic withdrawal strategy shows significantly lower sensitivity to early negative "
    "returns compared to fixed 4% rule."
)


def test_significance_claimed_against_p_of_one_is_rejected():
    findings = check_significance_claim(
        {"notes": NOTES, "metrics": {"sensitivity_difference_p_value": 1.0}}
    )

    assert len(findings) == 1
    assert "the test it rests on says it is not" in findings[0]
    assert "p = 1.0" in findings[0]


def test_significance_backed_by_a_real_p_value_passes():
    assert (
        check_significance_claim(
            {"notes": NOTES, "metrics": {"sensitivity_difference_p_value": 0.003}}
        )
        == []
    )


def test_a_p_value_exactly_at_the_threshold_is_not_significant():
    findings = check_significance_claim({"notes": NOTES, "metrics": {"diff_p_value": 0.05}})
    assert len(findings) == 1


def test_no_test_at_all_still_gets_the_original_complaint():
    findings = check_significance_claim({"notes": NOTES, "metrics": {"sensitivity_difference": 0.08}})
    assert "no test is among the metrics" in findings[0]


def test_a_note_that_never_claims_significance_is_left_alone():
    assert (
        check_significance_claim(
            {"notes": "The dynamic strategy showed a smaller decrease in longevity.",
             "metrics": {"diff_p_value": 1.0}}
        )
        == []
    )
