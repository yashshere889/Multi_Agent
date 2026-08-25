"""Unit tests for agents/coder/dataset_spec.py.

Two properties matter here and both are about not trusting the draft: every
field the model returns is coerced or dropped, and a spec exists even when the
model never produced one.
"""

import pytest

from research_pipeline.agents.coder import dataset_spec

PLAN = {
    "hypothesis_id": "H1",
    "objective": "fine-tune a Python code model on instruction data",
    "design": "comparative benchmark",
    "data_requirements": {
        "source": "public code corpora",
        "description": "python functions paired with natural-language instructions",
        "preprocessing_steps": ["deduplicate"],
    },
    "methods": [{"name": "LoRA fine-tuning", "description": "d", "reused_from_literature": True}],
    "evaluation": {"metrics": ["pass@1"], "baseline": "base model", "success_criteria": "> 0.4"},
}


def test_a_well_formed_draft_is_kept():
    spec = dataset_spec.validate_spec(
        {
            "task": "train/evaluate a coding model",
            "domain": "software engineering",
            "languages": ["Python"],
            "data_types": ["instruction", "code", "explanation"],
            "desired_examples": 50_000,
            "minimum_quality": 0.8,
            "license_requirements": ["permissive"],
            "avoid": ["unrelated programming languages"],
        },
        PLAN,
    )

    assert spec.task == "train/evaluate a coding model"
    assert spec.languages == ("python",)  # lowercased
    assert spec.data_types == ("instruction", "code", "explanation")
    assert spec.desired_examples == 50_000
    assert spec.minimum_quality == 0.8


def test_the_required_avoid_floor_cannot_be_dropped():
    # A spec that forgot to mention duplicates would quietly disable the
    # duplicate penalty for that plan, so the floor is unioned in regardless.
    spec = dataset_spec.validate_spec({"avoid": ["unrelated languages"]}, PLAN)

    assert "unrelated languages" in spec.avoid
    for required in dataset_spec.REQUIRED_AVOID:
        assert required in spec.avoid


def test_an_unrecognised_license_policy_is_dropped():
    # dataset_scoring.license_label switches on these exact strings, so an
    # invented policy would behave like no policy at all.
    spec = dataset_spec.validate_spec({"license_requirements": ["whatever-is-fine"]}, PLAN)

    assert spec.license_requirements == ("permissive",)
    assert spec.requires_permissive_license is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 1), (-5, 1), (10**12, dataset_spec.MAX_DESIRED_EXAMPLES), ("2000", 2000), (1500.7, 1500)],
)
def test_desired_examples_is_clamped_and_coerced(value, expected):
    assert (
        dataset_spec.validate_spec({"desired_examples": value}, PLAN).desired_examples == expected
    )


@pytest.mark.parametrize(("value", "expected"), [(-1, 0.0), (5, 1.0), ("0.6", 0.6)])
def test_minimum_quality_is_clamped(value, expected):
    assert dataset_spec.validate_spec({"minimum_quality": value}, PLAN).minimum_quality == expected


def test_a_comma_separated_string_is_accepted_where_a_list_was_asked_for():
    spec = dataset_spec.validate_spec({"data_types": "instruction, code"}, PLAN)

    assert spec.data_types == ("instruction", "code")


def test_partial_drafts_fall_back_field_by_field():
    # Three fields right and the rest dropped still contributes those three.
    spec = dataset_spec.validate_spec({"task": "sort things", "desired_examples": 42}, PLAN)

    assert spec.task == "sort things"
    assert spec.desired_examples == 42
    assert spec.data_types  # from the plan, not empty


def test_a_non_dict_payload_falls_back_entirely():
    assert dataset_spec.validate_spec("nope", PLAN) == dataset_spec.fallback_spec(PLAN)
    assert dataset_spec.validate_spec(None, PLAN) == dataset_spec.fallback_spec(PLAN)


def test_the_fallback_spec_reads_the_plan():
    spec = dataset_spec.fallback_spec(PLAN)

    assert "instruction" in spec.data_types
    assert "code" in spec.data_types
    assert "python" in spec.languages
    assert spec.license_requirements == ("permissive",)


def test_the_fallback_spec_still_produces_something_for_an_empty_plan():
    spec = dataset_spec.fallback_spec({})

    assert spec.data_types == ("text",)
    assert spec.languages == ("en",)
    assert spec.desired_examples == dataset_spec.DEFAULT_DESIRED_EXAMPLES


def test_search_queries_are_built_from_the_structured_fields():
    spec = dataset_spec.validate_spec(
        {
            "task": "train/evaluate a coding model",
            "domain": "software engineering",
            "data_types": ["instruction", "code"],
            "languages": ["python"],
        },
        PLAN,
    )
    queries = dataset_spec.search_queries(spec, PLAN)

    assert queries
    assert all(len(query.split()) <= 5 for query in queries)
    assert any("coding" in query or "software" in query for query in queries)
    assert any("instruction" in query for query in queries)
    assert len(queries) == len(set(queries))  # no duplicate work


def test_search_queries_keep_the_raw_description_as_a_last_resort():
    # A plan whose prose names the dataset outright must still match through the
    # keyword path when the structured fields miss.
    plan = {
        **PLAN,
        "data_requirements": {**PLAN["data_requirements"], "description": "SQuAD v2 questions"},
    }
    queries = dataset_spec.search_queries(
        dataset_spec.validate_spec({"task": "x", "domain": "y"}, plan), plan
    )

    assert any("squad" in query for query in queries)


def test_tokenize_drops_stopwords_and_bare_numbers():
    assert dataset_spec.tokenize("a survey of 500 undergraduate students in 2024") == [
        "survey",
        "undergraduate",
        "students",
    ]


def test_a_spec_round_trips_through_its_dict_form():
    original = dataset_spec.validate_spec(
        {"task": "t", "domain": "d", "data_types": ["code"], "desired_examples": 99}, PLAN
    )

    assert dataset_spec.from_dict(original.to_dict()) == original


def test_from_dict_tolerates_a_checkpoint_missing_keys():
    spec = dataset_spec.from_dict({"task": "just this"})

    assert spec.task == "just this"
    assert spec.data_types == ("text",)
    assert spec.license_requirements == ("permissive",)


def test_experiment_vocabulary_is_stripped_before_pairing():
    """Barkla job 10334327 pooled ONE candidate — an AIME maths corpus — because
    `train evaluate` was the only query that returned anything. A plan is full
    of words no dataset is named after, and they sit wherever they sit: leading
    pairs grab the verbs, and one filler word at the end ("... stock price
    forecasting models") shifts a trailing window off the topic."""
    spec = dataset_spec.validate_spec(
        {
            "task": "train/evaluate stock price forecasting models",
            "domain": "financial time series",
            "data_types": ["datetime", "adjusted_close_price"],
        },
        PLAN,
    )
    queries = dataset_spec.search_queries(spec, PLAN)

    assert "stock price" in queries
    assert queries[0] == "stock price"  # the topic leads
    assert not any("train" in query or "evaluate" in query for query in queries)
    assert not any("models" in query for query in queries)


def test_column_shaped_data_types_do_not_lead_the_search():
    # "datetime adjusted" matches no dataset name; the topical phrases do.
    spec = dataset_spec.validate_spec(
        {
            "task": "forecast equity prices",
            "domain": "finance",
            "data_types": ["datetime", "adjusted_close_price"],
        },
        PLAN,
    )
    queries = dataset_spec.search_queries(spec, PLAN)

    assert queries[0] != "datetime adjusted"


def test_a_topical_phrase_survives_filler_stripping():
    # The stripping must not eat the words that carry the subject.
    assert dataset_spec._topical("train and evaluate LSTM models for stock price forecasting") == [
        "lstm",
        "stock",
        "price",
        "forecasting",
    ]
