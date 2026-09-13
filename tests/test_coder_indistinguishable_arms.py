"""Arms that never differed cannot carry a verdict.

Barkla job 10496137 compared a fixed 4% withdrawal rule with a dynamic one over
10,000 Monte Carlo paths of real returns, reported identical longevity, zero
variance and success probability 1.0 for both, and published a refutation.
"""

from research_pipeline.agents.coder import saturation, sandbox

JOB_10496137_METRICS = {
    "fixed_mean_longevity": 40.0,
    "dynamic_mean_longevity": 40.0,
    "fixed_variance_longevity": 0.0,
    "dynamic_variance_longevity": 0.0,
    "fixed_success_probability": 1.0,
    "dynamic_success_probability": 1.0,
    "sensitivity_ratio": 0.0,
    "statistical_significance": False,
    "p_value": 1.0,
}


def test_the_recorded_run_is_indistinguishable():
    assert saturation.indistinguishable(JOB_10496137_METRICS) == [
        "dynamic_mean_longevity",
        "dynamic_success_probability",
        "dynamic_variance_longevity",
        "fixed_mean_longevity",
        "fixed_success_probability",
        "fixed_variance_longevity",
    ]


def test_arms_that_differ_on_any_shared_measure_are_not():
    metrics = {**JOB_10496137_METRICS, "dynamic_mean_longevity": 38.5}
    assert saturation.indistinguishable(metrics) == []


def test_identical_arms_with_a_real_spread_are_a_finding_not_a_defect():
    metrics = {
        "fixed_mean_longevity": 31.0,
        "dynamic_mean_longevity": 31.0,
        "fixed_std_longevity": 4.2,
        "dynamic_std_longevity": 4.2,
    }
    assert saturation.indistinguishable(metrics) == []


def test_suffix_arm_names_are_recognised():
    metrics = {"longevity_fixed": 40.0, "longevity_dynamic": 40.0, "std_fixed": 0.0, "std_dynamic": 0.0}
    assert saturation.indistinguishable(metrics) == [
        "longevity_dynamic",
        "longevity_fixed",
        "std_dynamic",
        "std_fixed",
    ]


def test_one_dict_per_arm_is_recognised():
    metrics = {
        "deterministic_metrics": {"mean": 5.0, "std": 0.0},
        "stochastic_metrics": {"mean": 5.0, "std": 0.0},
    }
    assert saturation.indistinguishable(metrics) == [
        "deterministic_metrics.mean",
        "deterministic_metrics.std",
        "stochastic_metrics.mean",
        "stochastic_metrics.std",
    ]


def test_a_single_shared_measure_is_too_weak_to_judge():
    assert saturation.indistinguishable({"fixed_std": 0.0, "dynamic_std": 0.0}) == []


def test_unpaired_metrics_are_ignored():
    assert saturation.indistinguishable({"accuracy": 0.8, "loss_std": 0.0, "n": 100}) == []


def test_the_fix_loop_is_told_what_is_wrong():
    findings = sandbox.check_results_plausibility(JOB_10496137_METRICS)
    assert len(findings) == 1
    assert "cannot be told apart" in findings[0]
    assert "averaged across periods rather than compounded" in findings[0]
    assert "raise the cap" in findings[0]


def test_the_censoring_constant_is_named_when_the_code_has_one():
    """Barkla 10510222: every path survived to a 30-year horizon, so both arms read 30.0."""
    metrics = {
        "fixed_mean_longevity": 30.0,
        "dynamic_mean_longevity": 30.0,
        "fixed_std_longevity": 0.0,
        "dynamic_std_longevity": 0.0,
    }
    source = "SEED = 7\nRETIREMENT_PERIOD_YEARS = 30\nNUM_SIMULATIONS = 10000\n"

    findings = sandbox.check_results_plausibility(metrics, source)

    assert "(RETIREMENT_PERIOD_YEARS = 30)" in findings[0]


def test_no_constant_is_named_when_nothing_matches():
    findings = sandbox.check_results_plausibility(JOB_10496137_METRICS, "SEED = 7\n")
    assert "every path reached the cap," in findings[0]


def test_the_verdict_is_withheld_and_the_claim_kept():
    stamped = saturation.apply_to_results(
        {"metrics": JOB_10496137_METRICS, "meets_success_criteria": False}
    )
    assert stamped["meets_success_criteria"] == "unknown"
    assert stamped["model_reported_meets_success_criteria"] is False
    assert stamped["measurement_validity"] == saturation.VERDICT_INDISTINGUISHABLE
    assert "fixed_variance_longevity" in stamped["verdict_withheld_because"]
    assert stamped["metrics"] == JOB_10496137_METRICS


def test_a_saturated_result_is_reported_as_saturation_only():
    metrics = {"fixed_accuracy": 1.0, "dynamic_accuracy": 1.0, "fixed_f1": 1.0, "dynamic_f1": 1.0}
    stamped = saturation.apply_to_results({"metrics": metrics, "meets_success_criteria": False})
    assert stamped["measurement_validity"] == saturation.VERDICT_SATURATED
    assert "indistinguishable_metrics" not in stamped


def test_a_result_with_real_differences_is_untouched():
    results = {"metrics": {**JOB_10496137_METRICS, "dynamic_variance_longevity": 2.5}}
    assert saturation.apply_to_results(results) is results
