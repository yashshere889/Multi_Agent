"""Unit tests for agents/coder/dataset_scoring.py — the rubric itself.

The property every test here defends is the same one: **the model supplies
labels and evidence, Python supplies the number**. A change that lets a model
response influence the score by any route other than a band label is the
regression these exist to catch.
"""

import pytest

from research_pipeline.agents.coder import dataset_inspect, dataset_scoring, dataset_spec

SPEC = dataset_spec.DatasetSpec(
    task="train a coding model",
    domain="software engineering",
    languages=("python",),
    data_types=("instruction", "code"),
    desired_examples=50_000,
    minimum_quality=0.8,
    license_requirements=("permissive",),
    avoid=dataset_spec.REQUIRED_AVOID,
)


def _report(**overrides):
    defaults = {
        "rows_sampled": 100,
        "num_rows_total": 50_000,
        "duplicate_rate": 0.0,
        "empty_rate": 0.0,
        "malformed_rate": 0.0,
        "repetition_score": 0.0,
        "size_adequacy": 1.0,
    }
    defaults.update(overrides)
    return dataset_inspect.InspectionReport(**defaults)


def _scored(**overrides):
    components = overrides.pop("components", dataset_scoring.ScoreComponents())
    scored = dataset_scoring.ScoredDataset(
        dataset_id=overrides.pop("dataset_id", "acme/thing"),
        components=components,
        report=overrides.pop("report", _report()),
        **overrides,
    )
    return dataset_scoring.rescore(scored)


# -- the weighted sum --------------------------------------------------------


def test_the_weights_sum_to_one():
    # Enforced at import too, because a table that doesn't sum to 1 makes every
    # score silently incomparable to the configured minimum.
    assert sum(dataset_scoring.WEIGHTS.values()) == pytest.approx(1.0)


def test_a_hand_computed_example():
    components = dataset_scoring.ScoreComponents(
        task_relevance="related",  # 0.6 * 0.35 = 0.210
        content_relevance="exact",  # 1.0 * 0.20 = 0.200
        quality=0.8,  # 0.8 * 0.15 = 0.120
        provenance="partial",  # 0.5 * 0.10 = 0.050
        schema_fit="partial",  # 0.5 * 0.10 = 0.050
        license_fit="permitted",  # 1.0 * 0.10 = 0.100
    )

    assert dataset_scoring.score(components) == pytest.approx(0.73)


def test_the_best_and_worst_possible_candidates():
    best = dataset_scoring.ScoreComponents(
        task_relevance="exact",
        content_relevance="exact",
        quality=1.0,
        provenance="documented",
        schema_fit="expected",
        license_fit="permitted",
    )
    worst = dataset_scoring.ScoreComponents(
        task_relevance="unrelated",
        content_relevance="unrelated",
        quality=0.0,
        provenance="absent",
        schema_fit="incompatible",
        license_fit="incompatible",
    )

    assert dataset_scoring.score(best) == pytest.approx(1.0)
    assert dataset_scoring.score(worst) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("dimension", "band", "value"),
    [
        ("task_relevance", "exact", 1.0),
        ("task_relevance", "related", 0.6),
        ("task_relevance", "weak", 0.3),
        ("task_relevance", "unrelated", 0.0),
        ("content_relevance", "exact", 1.0),
        ("content_relevance", "related", 0.6),
        ("content_relevance", "weak", 0.3),
        ("content_relevance", "unrelated", 0.0),
        ("schema_fit", "expected", 1.0),
        ("schema_fit", "partial", 0.5),
        ("schema_fit", "incompatible", 0.0),
        ("license_fit", "permitted", 1.0),
        ("license_fit", "unknown", 0.3),
        ("license_fit", "incompatible", 0.0),
        ("provenance", "documented", 1.0),
        ("provenance", "partial", 0.5),
        ("provenance", "unknown", 0.3),
        ("provenance", "absent", 0.0),
    ],
)
def test_every_band_contributes_its_weight(dimension, band, value):
    """Each band, isolated: everything else at zero, so the total *is* that
    dimension's contribution."""
    zeroed = {
        "task_relevance": "unrelated",
        "content_relevance": "unrelated",
        "quality": 0.0,
        "provenance": "absent",
        "schema_fit": "incompatible",
        "license_fit": "incompatible",
    }
    components = dataset_scoring.ScoreComponents(**{**zeroed, dimension: band})

    assert dataset_scoring.score(components) == pytest.approx(
        dataset_scoring.WEIGHTS[dimension] * value
    )


# -- the model cannot supply a number ----------------------------------------


def test_a_model_supplied_score_is_discarded_on_parse():
    appraisal = dataset_scoring.coerce_appraisal(
        {
            "score": 0.99,
            "dataset_score": 1.0,
            "task_relevance": "unrelated",
            "content_relevance": "unrelated",
            "column_mapping": {},
        }
    )

    assert "score" not in appraisal
    assert appraisal["task_relevance"] == "unrelated"


def test_a_confident_claim_on_an_unrelated_dataset_still_scores_near_zero():
    appraisal = dataset_scoring.coerce_appraisal(
        {"score": 0.99, "task_relevance": "unrelated", "content_relevance": "unrelated"}
    )
    components = dataset_scoring.ScoreComponents(
        task_relevance=appraisal["task_relevance"],
        content_relevance=appraisal["content_relevance"],
        quality=0.0,
        provenance="absent",
        schema_fit="incompatible",
        license_fit="incompatible",
    )

    assert dataset_scoring.score(components) == 0.0


@pytest.mark.parametrize(
    ("answer", "band"),
    [
        ("exact match", "exact"),
        ("EXACT", "exact"),
        ("directly on task", "exact"),
        ("related task", "related"),
        ("adjacent", "related"),
        ("weakly related", "weak"),
        ("tangential at best", "weak"),
        ("unrelated", "unrelated"),
        ("not related to the spec", "unrelated"),
        ("moderately relevant, I think", "unrelated"),  # unrecognised -> pessimistic
        ("", "unrelated"),
        (None, "unrelated"),
        (0.7, "unrelated"),
    ],
)
def test_relevance_answers_are_coerced_conservatively(answer, band):
    assert dataset_scoring.coerce_label("task_relevance", answer) == band


def test_an_unrecognised_requirement_status_becomes_unknown_not_pass():
    # The prompt's whole instruction is to mark UNKNOWN rather than assume; a
    # coercion that rounded ambiguity up to `pass` would undo it.
    requirements = dataset_scoring.coerce_requirements(
        {
            "contains_python": {"status": "probably", "evidence": "the name says so"},
            "permissive_license": {"status": "fail", "evidence": "cc-by-nc"},
        }
    )

    assert requirements["contains_python"]["status"] == "unknown"
    assert requirements["permissive_license"]["status"] == "fail"


def test_a_requirement_with_no_evidence_says_so():
    requirements = dataset_scoring.coerce_requirements({"x": {"status": "pass"}})

    assert requirements["x"]["evidence"] == "No evidence was provided."


# -- license: decided here, never asked ---------------------------------------


@pytest.mark.parametrize(
    ("license_id", "band"),
    [
        ("apache-2.0", "permitted"),
        ("mit", "permitted"),
        ("cc0-1.0", "permitted"),
        ("cc-by-nc-4.0", "incompatible"),
        ("cc-by-nc-sa-4.0", "incompatible"),
        ("cc-by-nd-4.0", "incompatible"),
        ("cc-by-sa-4.0", "incompatible"),  # share-alike, and this spec asked for permissive
        ("other", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
        ("some-bespoke-license-2.1", "unknown"),  # a real id we have no opinion about
    ],
)
def test_license_bands(license_id, band):
    assert dataset_scoring.license_label(license_id, SPEC) == band


def test_a_share_alike_policy_admits_share_alike_licenses():
    spec = dataset_spec.DatasetSpec(**{**SPEC.to_dict(), "license_requirements": ["share-alike"]})

    assert dataset_scoring.license_label("cc-by-sa-4.0", spec) == "permitted"
    assert dataset_scoring.license_label("cc-by-nc-4.0", spec) == "incompatible"


def test_an_any_policy_permits_everything():
    spec = dataset_spec.DatasetSpec(**{**SPEC.to_dict(), "license_requirements": ["any"]})

    assert dataset_scoring.license_label("cc-by-nc-4.0", spec) == "permitted"


def test_an_empty_policy_is_treated_as_permissive_required():
    # Saying nothing about licensing is not the same as saying it doesn't matter.
    spec = dataset_spec.DatasetSpec(**{**SPEC.to_dict(), "license_requirements": []})

    assert spec.requires_permissive_license is True
    assert dataset_scoring.license_label("cc-by-nc-4.0", spec) == "incompatible"


# -- quality: arithmetic over measured rates ----------------------------------


def test_a_clean_ample_dataset_scores_full_quality():
    assert dataset_scoring.quality_score(_report()) == pytest.approx(1.0)


def test_each_defect_costs_its_documented_amount():
    assert dataset_scoring.quality_score(_report(duplicate_rate=0.1)) == pytest.approx(0.9)
    assert dataset_scoring.quality_score(_report(empty_rate=0.1)) == pytest.approx(0.85)
    assert dataset_scoring.quality_score(_report(malformed_rate=0.1)) == pytest.approx(0.85)


def test_each_penalty_is_capped_so_one_defect_cannot_hide_the_others():
    only_duplicates = dataset_scoring.quality_score(_report(duplicate_rate=1.0))
    both = dataset_scoring.quality_score(_report(duplicate_rate=1.0, empty_rate=1.0))

    assert only_duplicates == pytest.approx(0.6)  # capped at 0.40
    assert both == pytest.approx(0.3)  # and the empty penalty still lands


def test_ordinary_repetition_is_not_penalised():
    assert dataset_scoring.quality_score(_report(repetition_score=0.3)) == pytest.approx(1.0)
    assert dataset_scoring.quality_score(_report(repetition_score=0.5)) == pytest.approx(0.8)


def test_size_scales_quality_rather_than_deducting_from_it():
    # A spotless dataset a tenth the size asked for is usable-but-thin, which is
    # a scaling of its worth, not a defect.
    assert dataset_scoring.quality_score(_report(size_adequacy=0.0)) == pytest.approx(0.5)
    assert dataset_scoring.quality_score(_report(size_adequacy=0.5)) == pytest.approx(0.75)


def test_an_unsampleable_dataset_is_unestablished_not_zero():
    assert dataset_scoring.quality_score(_report(rows_sampled=0)) == 0.3


# -- schema fit: the model maps, Python verifies -------------------------------


def test_a_full_mapping_onto_real_columns_is_expected():
    band, resolved = dataset_scoring.schema_fit_label(
        {"instruction": "prompt", "code": "completion"}, SPEC, ["prompt", "completion", "id"]
    )

    assert band == "expected"
    assert resolved == {"instruction": "prompt", "code": "completion"}


def test_a_mapping_onto_an_invented_column_is_dropped():
    # The hallucination this arrangement exists to catch: the model names a
    # column that isn't in the schema the viewer reported.
    band, resolved = dataset_scoring.schema_fit_label(
        {"instruction": "prompt", "code": "does_not_exist"}, SPEC, ["prompt", "completion"]
    )

    assert band == "partial"
    assert resolved == {"instruction": "prompt"}


def test_no_usable_mapping_is_incompatible():
    band, resolved = dataset_scoring.schema_fit_label({}, SPEC, ["prompt", "completion"])

    assert band == "incompatible"
    assert resolved == {}


def test_a_mapping_that_is_not_a_dict_is_ignored_rather_than_crashing():
    assert dataset_scoring.schema_fit_label("prompt", SPEC, ["prompt"])[0] == "incompatible"


# -- provenance ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("info", "band"),
    [
        ({"id": "a/b", "cardData": {"citation": "@misc{}"}}, "documented"),
        ({"id": "a/b", "tags": ["arxiv:2401.00001"]}, "documented"),
        ({"id": "a/b", "cardData": {"source_datasets": ["squad"]}}, "documented"),
        ({"id": "a/b", "cardData": {"license": "mit"}}, "partial"),
        ({"id": "a/b"}, "unknown"),
        ({}, "absent"),
        (None, "absent"),
    ],
)
def test_provenance_bands(info, band):
    assert dataset_scoring.provenance_label(info) == band


# -- the critic ---------------------------------------------------------------


def test_a_hard_fail_vetoes_whatever_the_score_was():
    scored = _scored(
        components=dataset_scoring.ScoreComponents(
            task_relevance="exact",
            content_relevance="exact",
            quality=1.0,
            provenance="documented",
            schema_fit="expected",
            license_fit="permitted",
        )
    )
    assert scored.score == pytest.approx(1.0)

    dataset_scoring.apply_critic(
        scored,
        [
            dataset_scoring.CriticFinding(
                "evaluation_contamination", "rows quote the held-out split"
            )
        ],
        SPEC,
    )
    dataset_scoring.decide(scored, 0.75)

    assert scored.decision == "reject"
    assert "evaluation_contamination" in scored.reasons_for_rejection[0]


def test_soft_findings_subtract_their_fixed_penalty():
    scored = _scored(
        components=dataset_scoring.ScoreComponents(
            task_relevance="exact",
            content_relevance="exact",
            quality=1.0,
            provenance="documented",
            schema_fit="expected",
            license_fit="permitted",
        )
    )
    dataset_scoring.apply_critic(
        scored,
        [
            dataset_scoring.CriticFinding("substantial_duplication", "a third are repeats"),
            dataset_scoring.CriticFinding("suspiciously_tiny", "only 40 rows"),
        ],
        SPEC,
    )

    assert scored.base_score == pytest.approx(1.0)
    assert scored.score == pytest.approx(0.8)  # 0.10 + 0.10
    assert scored.decision != "reject"


def test_synthetic_only_vetoes_when_the_spec_avoids_it():
    scored = _scored()
    dataset_scoring.apply_critic(
        scored, [dataset_scoring.CriticFinding("synthetic_only", "card says GPT-generated")], SPEC
    )

    assert scored.decision == "reject"


def test_synthetic_only_is_a_penalty_when_the_spec_wants_synthetic_data():
    spec = dataset_spec.DatasetSpec(**{**SPEC.to_dict(), "avoid": ["duplicates"]})
    scored = _scored(
        components=dataset_scoring.ScoreComponents(
            task_relevance="exact",
            content_relevance="exact",
            quality=1.0,
            provenance="documented",
            schema_fit="expected",
            license_fit="permitted",
        )
    )
    dataset_scoring.apply_critic(
        scored, [dataset_scoring.CriticFinding("synthetic_only", "card says GPT-generated")], spec
    )

    assert scored.decision != "reject"
    assert scored.score == pytest.approx(0.85)


def test_findings_outside_the_vocabulary_are_discarded():
    findings = dataset_scoring.coerce_findings(
        {
            "findings": [
                {"code": "evaluation_contamination", "evidence": "e"},
                {"code": "i_just_dont_like_it", "evidence": "vibes"},
                {"code": "EVALUATION-CONTAMINATION", "evidence": "duplicate"},
            ]
        }
    )

    assert [finding.code for finding in findings] == ["evaluation_contamination"]


def test_findings_survive_a_bare_list_and_bare_strings():
    findings = dataset_scoring.coerce_findings(["personal_information", "nonsense"])

    assert [finding.code for finding in findings] == ["personal_information"]


def test_a_malformed_critic_response_finds_nothing():
    assert dataset_scoring.coerce_findings("no findings, looks fine") == []


# -- the threshold ------------------------------------------------------------


def test_a_candidate_below_the_threshold_is_rejected_with_the_number_named():
    scored = _scored(
        components=dataset_scoring.ScoreComponents(
            task_relevance="weak",
            content_relevance="weak",
            quality=0.5,
            provenance="unknown",
            schema_fit="partial",
            license_fit="permitted",
        )
    )
    dataset_scoring.decide(scored, 0.75)

    assert scored.decision == "reject"
    assert "below the 0.75 threshold" in scored.reasons_for_rejection[-1]


def test_an_incompatible_license_is_rejected_even_above_the_threshold():
    scored = _scored(
        components=dataset_scoring.ScoreComponents(
            task_relevance="exact",
            content_relevance="exact",
            quality=1.0,
            provenance="documented",
            schema_fit="expected",
            license_fit="incompatible",
        )
    )
    assert scored.score == pytest.approx(0.9)

    dataset_scoring.decide(scored, 0.75)

    assert scored.decision == "reject"
    assert "License requirement not satisfied" in scored.reasons_for_rejection


# -- the prefilter ------------------------------------------------------------


def test_the_prefilter_drops_incompatible_licenses_before_they_cost_a_call():
    candidates = [
        {"id": "a/permissive", "cardData": {"license": "mit"}, "downloads": 100},
        {"id": "b/noncommercial", "cardData": {"license": "cc-by-nc-4.0"}, "downloads": 999_999},
    ]

    ranked = dataset_scoring.prefilter(candidates, SPEC, limit=5)

    assert [entry["id"] for entry in ranked] == ["a/permissive"]


def test_the_prefilter_prefers_a_name_match_over_raw_popularity():
    candidates = [
        {"id": "someone/python-instruction-code", "cardData": {"license": "mit"}, "downloads": 100},
        {"id": "someone/unrelated-images", "cardData": {"license": "mit"}, "downloads": 5_000_000},
    ]

    ranked = dataset_scoring.prefilter(candidates, SPEC, limit=2)

    assert ranked[0]["id"] == "someone/python-instruction-code"
    assert ranked[0]["prefilter_score"] > ranked[1]["prefilter_score"]


def test_the_prefilter_honours_its_limit():
    candidates = [
        {"id": f"x/d{index}", "cardData": {"license": "mit"}, "downloads": index}
        for index in range(10)
    ]

    assert len(dataset_scoring.prefilter(candidates, SPEC, limit=3)) == 3


def test_size_categories_stand_in_for_a_row_count_before_size_is_known():
    assert dataset_scoring.rows_from_size_categories({"size_categories": ["10K<n<100K"]}) == 10_000
    assert dataset_scoring.rows_from_size_categories({"size_categories": "n<1K"}) == 0
    assert dataset_scoring.rows_from_size_categories({}) == 0


# -- serialization ------------------------------------------------------------


def test_a_candidate_round_trips_through_state():
    original = _scored(
        components=dataset_scoring.ScoreComponents(
            task_relevance="exact",
            content_relevance="related",
            quality=0.9,
            provenance="documented",
            schema_fit="expected",
            license_fit="permitted",
            evidence={"task_relevance": "the card says so"},
        ),
        revision="abc123",
        license="mit",
    )
    original.findings = [dataset_scoring.CriticFinding("suspiciously_tiny", "40 rows")]

    restored = dataset_scoring.from_state(dataset_scoring.to_state(original))

    assert restored.dataset_id == original.dataset_id
    assert restored.score == original.score
    assert restored.components == original.components
    assert restored.report == original.report
    assert restored.findings == original.findings


def test_state_written_by_an_older_revision_still_loads():
    # A checkpoint resumed after this module gained or lost a field must not
    # crash with an unexpected keyword.
    restored = dataset_scoring.from_state(
        {
            "dataset_id": "a/b",
            "score": 0.5,
            "a_field_from_the_future": 1,
            "components": {"task_relevance": "exact", "gone": 2},
            "report": {"rows_sampled": 3},
        }
    )

    assert restored.dataset_id == "a/b"
    assert restored.components.task_relevance == "exact"
    assert restored.report.rows_sampled == 3


def test_the_record_carries_everything_needed_to_re_derive_the_score():
    scored = _scored(
        components=dataset_scoring.ScoreComponents(
            task_relevance="exact",
            content_relevance="exact",
            quality=1.0,
            provenance="documented",
            schema_fit="expected",
            license_fit="permitted",
            evidence={"task_relevance": "the card describes exactly this task"},
        ),
        revision="abc123",
        license="apache-2.0",
        size_bytes=1234,
    )
    dataset_scoring.decide(scored, 0.75)
    record = scored.as_record()

    assert record["decision"] == "accept"
    assert record["revision"] == "abc123"
    assert record["weights"] == dataset_scoring.WEIGHTS
    assert set(record["evidence"]) == set(dataset_scoring.WEIGHTS)
    assert record["bands"]["task_relevance"] == "exact"
    assert record["evidence_notes"]["task_relevance"]
    assert record["inspection"]["rows_sampled"] == 100
    # The record's own numbers reproduce the score it reports.
    assert sum(
        dataset_scoring.WEIGHTS[name] * value for name, value in record["evidence"].items()
    ) == pytest.approx(record["score"])
