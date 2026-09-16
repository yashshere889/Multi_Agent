"""A p-value computed from a constant is not evidence, however real the data.

Barkla job 10522992 simulated 30-year retirements on the staged Fama-French
daily factors, reduced each strategy to the *spread* of a longevity capped at
30 years, and resampled it 10,000 times. Every resample came out identical, so
`ttest_1samp` returned an infinite statistic and p = 0.0, and the paper reported
"significantly lower sensitivity (p=0.0000, 95% CI [0.08, 0.08])" under a
published verdict. Nothing flagged it: the metrics were flat, and the guard only
descended into nested dicts.
"""

from research_pipeline.agents.coder.sandbox import check_results_plausibility

# Verbatim from that run's results.json.
R12B_METRICS = {
    "fixed_mean_longevity": 29.037875,
    "dynamic_mean_longevity": 28.924133333333337,
    "fixed_success_probability": 0.8828,
    "dynamic_success_probability": 0.8857,
    "fixed_strategy_sensitivity": 1.1666666666666665,
    "dynamic_strategy_sensitivity": 1.0833333333333335,
    "sensitivity_difference": 0.08333333333333348,
    "sensitivity_difference_stderr": 0.0,
    "sensitivity_difference_ci_lower": 0.08333333333333348,
    "sensitivity_difference_ci_upper": 0.08333333333333348,
    "sensitivity_difference_pvalue": 0.0,
}


def _findings(metrics, source=""):
    return check_results_plausibility(metrics, source)


def test_a_flat_zero_variance_test_is_rejected():
    findings = _findings(R12B_METRICS, "RETIREMENT_PERIOD_YEARS = 30\n")

    assert any("no variance to test" in f for f in findings)
    complaint = next(f for f in findings if "no variance to test" in f)
    assert "'sensitivity_difference'" in complaint
    # The ceiling is the cause, so the fix prompt must name the knob that caps it.
    assert "RETIREMENT_PERIOD_YEARS" in complaint


def test_the_same_test_nested_is_rejected_the_same_way():
    findings = _findings(
        {
            "sensitivity_difference": {
                "value": 0.0833,
                "stderr": 0.0,
                "ci_lower": 0.0833,
                "ci_upper": 0.0833,
                "p_value": 0.0,
            }
        }
    )
    assert any("no variance to test" in f for f in findings)


def test_a_zero_width_interval_without_a_p_value_still_names_the_interval():
    findings = _findings({"diff_ci_lower": 0.4, "diff_ci_upper": 0.4})

    assert any("ci_lower == ci_upper" in f for f in findings)
    assert not any("no variance to test" in f for f in findings)


def test_the_single_number_comparison_signature_still_fires_when_flat():
    findings = _findings({"diff_t_statistic": 0.0, "diff_p_value": 1.0})
    assert any("statistic of exactly 0 with p = 1.0" in f for f in findings)


def test_a_real_test_with_a_real_interval_passes():
    findings = _findings(
        {
            "fixed_success_probability": 0.883,
            "dynamic_success_probability": 0.902,
            "difference": 0.019,
            "difference_stderr": 0.0062,
            "difference_ci_lower": 0.0069,
            "difference_ci_upper": 0.0311,
            "difference_pvalue": 0.0021,
        }
    )
    assert findings == []


def test_an_enormous_effect_whose_p_value_underflowed_is_not_rejected():
    """p == 0.0 alone is underflow; only p == 0 with a zero-width interval is the defect."""
    findings = _findings(
        {
            "difference": 4.2,
            "difference_stderr": 0.01,
            "difference_ci_lower": 4.18,
            "difference_ci_upper": 4.22,
            "difference_pvalue": 0.0,
        }
    )
    assert findings == []


# --- Barkla job 10523041: what the model produced once p = 0.0 was rejected ---
#
# The zero-variance guard worked — the fabricated p = 0.0 is gone and the fix
# loop ran 11 attempts instead of 6. The run then published a verdict on a
# difference of exactly 0.0 at p = 1.0, contradicting both its own arms and its
# own note, "shows significantly lower sensitivity".
R13_METRICS = {
    "fixed_mean_longevity": 29.037875,
    "dynamic_mean_longevity": 28.924133333333337,
    "fixed_success_probability": 0.8828,
    "dynamic_success_probability": 0.8857,
    "fixed_sensitivity": 1.1666666666666665,
    "dynamic_sensitivity": 1.0833333333333335,
    "sensitivity_difference": 0.0,
    "sensitivity_difference_p_value": 1.0,
}
R13_NOTES = (
    "Dynamic withdrawal strategy shows significantly lower sensitivity to early negative "
    "returns (smaller decrease in longevity with worsening early returns) compared to fixed 4% rule."
)


def test_a_zero_difference_that_contradicts_its_own_arms_is_rejected():
    findings = _findings(R13_METRICS)

    complaint = next(f for f in findings if "contradicts the arms" in f)
    assert "'sensitivity_difference' is exactly 0" in complaint
    assert "1.16667" in complaint and "1.08333" in complaint


def test_p_equals_one_fires_even_when_the_estimate_is_not_called_a_statistic():
    findings = _findings({"sensitivity_difference": 0.0, "sensitivity_difference_p_value": 1.0})
    assert any("statistic of exactly 0 with p = 1.0" in f for f in findings)


def test_identical_arms_cannot_carry_a_non_zero_difference():
    findings = _findings({"fixed_score": 0.5, "dynamic_score": 0.5, "score_difference": 0.2})
    assert any("identical arms cannot have a" in f for f in findings)


def test_a_difference_scaled_differently_from_its_arms_is_left_alone():
    """A difference reported as a percentage change is not a contradiction."""
    findings = _findings({"fixed_score": 0.40, "dynamic_score": 0.50, "score_difference": 25.0})
    assert findings == []
