"""Calling a difference significant is a claim about a test.

Barkla job 10510547 removed the degenerate t-test its predecessor reported
rather than fixing it, kept the note "shows significantly lower sensitivity",
and the paper repeated that wording six times with nothing behind it.
"""

from research_pipeline.agents.coder import sandbox

JOB_10510547 = {
    "metrics": {
        "fixed_mean_longevity": 29.037875,
        "dynamic_mean_longevity": 28.924133,
        "fixed_variance_longevity": 9.939345,
        "dynamic_variance_longevity": 12.149299,
        "fixed_sensitivity": 1.1666666,
        "dynamic_sensitivity": 1.0833333,
        "sensitivity_difference": 0.0833333,
    },
    "notes": (
        "Dynamic withdrawal strategy shows significantly lower sensitivity to early negative "
        "returns compared to fixed 4% rule."
    ),
}


def test_the_recorded_claim_is_flagged():
    findings = sandbox.check_significance_claim(JOB_10510547)
    assert len(findings) == 1
    assert "no test is among the metrics" in findings[0]
    assert "per-path" in findings[0]


def test_a_reported_p_value_clears_it():
    results = {
        **JOB_10510547,
        "metrics": {**JOB_10510547["metrics"], "sensitivity_t_test": {"p_value": 0.03}},
    }
    assert sandbox.check_significance_claim(results) == []


def test_a_confidence_interval_clears_it():
    results = {
        **JOB_10510547,
        "metrics": {**JOB_10510547["metrics"], "difference_ci": {"lower": 0.01, "upper": 0.16}},
    }
    assert sandbox.check_significance_claim(results) == []


def test_a_standard_error_clears_it():
    results = {**JOB_10510547, "metrics": {**JOB_10510547["metrics"], "difference_stderr": 0.02}}
    assert sandbox.check_significance_claim(results) == []


def test_a_claim_that_does_not_mention_significance_is_left_alone():
    results = {**JOB_10510547, "notes": "The dynamic strategy lost 0.08 fewer years of longevity."}
    assert sandbox.check_significance_claim(results) == []


def test_the_success_notes_field_is_read_too():
    results = {"metrics": {"a_mean": 1.0, "b_mean": 2.0}, "success_notes": "A significant difference."}
    assert len(sandbox.check_significance_claim(results)) == 1


def test_no_metrics_and_no_claim_says_nothing():
    assert sandbox.check_significance_claim({}) == []
