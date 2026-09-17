"""A program that ran is reported with its numbers, verdict withheld.

Barkla job 10536963 spent all 14 fix attempts alternating between
`implausible_results` and `unsupported_significance_claim` on a plan measuring
the spread of a portfolio longevity capped at 30 years. Every attempt from the
fifth on produced the same correct numbers off real staged Fama-French data —
29.04 vs 28.92 years, 0.8828 vs 0.8857 success — and the run reported
`code_generated_not_run` with `results: null`, discarding all of it. Same
resolution as the data, compute and saturation gates: regeneration cannot fix a
measure the plan chose, so report the metrics and withhold the verdict.
"""

import json

from research_pipeline.agents.coder import saturation
from research_pipeline.agents.coder.coder_agent import _RESULTS_LEVEL_ERROR_SOURCES
from research_pipeline.agents.coder.schema import VALID_ERROR_SOURCES

R14_RESULTS = {
    "metrics": {
        "fixed_mean_longevity": 29.037875,
        "dynamic_mean_longevity": 28.924133333333337,
        "fixed_success_probability": 0.8828,
        "dynamic_success_probability": 0.8857,
    },
    "meets_success_criteria": True,
    "notes": "Dynamic withdrawal strategy shows significantly lower sensitivity.",
}


def test_the_results_level_sources_are_real_error_sources():
    assert _RESULTS_LEVEL_ERROR_SOURCES <= VALID_ERROR_SOURCES


def test_withholding_keeps_the_metrics_and_records_the_model_s_own_claim():
    withheld = saturation.withhold(
        R14_RESULTS,
        saturation.VERDICT_UNDECIDABLE,
        saturation.WITHHELD_UNDECIDABLE.format(names="implausible_results"),
    )

    assert withheld["metrics"] == R14_RESULTS["metrics"]
    assert withheld["meets_success_criteria"] == "unknown"
    assert withheld["model_reported_meets_success_criteria"] is True
    assert withheld["measurement_validity"] == saturation.VERDICT_UNDECIDABLE
    assert "implausible_results" in withheld["verdict_withheld_because"]
    # The input is never mutated.
    assert R14_RESULTS["meets_success_criteria"] is True


def test_an_earlier_gate_s_record_survives_a_second_withholding():
    first = saturation.withhold(R14_RESULTS, "data_surrogate", "inputs were synthesized.")
    second = saturation.withhold(
        first, saturation.VERDICT_UNDECIDABLE, saturation.WITHHELD_UNDECIDABLE.format(names="x")
    )

    assert second["model_reported_meets_success_criteria"] is True
    assert "inputs were synthesized." in second["verdict_withheld_because"]
    assert "verdict is withheld" in second["verdict_withheld_because"]


def test_the_existing_gates_still_stamp_what_they_always_did():
    """apply_to_results was refactored onto `withhold`; its output must not move."""
    saturated = saturation.apply_to_results(
        {"metrics": {"accuracy": 1.0, "f1": 1.0}, "meets_success_criteria": True}
    )
    assert saturated["meets_success_criteria"] == "unknown"
    assert saturated["measurement_validity"] == saturation.VERDICT_SATURATED
    assert saturated["saturated_metrics"]
    assert saturated["model_reported_meets_success_criteria"] is True


def test_a_run_with_nothing_to_say_is_returned_untouched():
    clean = {"metrics": {"accuracy": 0.81, "f1": 0.77}, "meets_success_criteria": True}
    assert saturation.apply_to_results(clean) == clean


def test_provenance_is_read_off_disk_and_degrades_to_empty(tmp_path):
    from research_pipeline.agents.coder.coder_agent import _provenance_on_disk

    assert _provenance_on_disk(str(tmp_path)) == {}
    (tmp_path / "data_provenance.json").write_text("not json")
    assert _provenance_on_disk(str(tmp_path)) == {}
    (tmp_path / "data_provenance.json").write_text(json.dumps({"sources": [{"kind": "real_local"}]}))
    assert _provenance_on_disk(str(tmp_path))["sources"][0]["kind"] == "real_local"
