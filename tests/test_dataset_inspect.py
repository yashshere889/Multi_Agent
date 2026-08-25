"""Unit tests for agents/coder/dataset_inspect.py.

Pure module, pure tests: no fakes, no network, no model. Every number this
module produces is arithmetic over a list of dicts, and these pin that
arithmetic — because `dataset_scoring.quality_score` and the critic's hard
fails both read it, so a silent change here quietly changes which datasets a
run is willing to use.
"""

from research_pipeline.agents.coder import dataset_inspect

COLUMNS = [{"name": "instruction", "type": "string"}, {"name": "code", "type": "string"}]


def _clean(count=10):
    return [
        {"instruction": f"write function number {index}", "code": f"def f{index}(): pass"}
        for index in range(count)
    ]


def test_a_clean_sample_measures_clean():
    report = dataset_inspect.inspect_rows(_clean(), COLUMNS)

    assert report.rows_sampled == 10
    assert report.duplicate_rate == 0.0
    assert report.empty_rate == 0.0
    assert report.malformed_rate == 0.0
    assert report.contamination_hits == 0
    assert report.pii_hits == 0
    assert report.notes == []
    assert report.column_fill == {"instruction": 1.0, "code": 1.0}


def test_exact_duplicates_are_counted():
    # The four repeats are copies of row 0, which is already in the first six:
    # six distinct records across ten rows.
    rows = _clean(6) + [_clean(1)[0]] * 4
    report = dataset_inspect.inspect_rows(rows, COLUMNS)

    assert report.duplicate_rate == 0.4
    assert any("duplicates" in note for note in report.notes)


def test_whitespace_only_differences_still_count_as_duplicates():
    # A record differing only in trailing newlines is the duplicate it looks
    # like; normalizing before hashing is what makes the rate honest.
    rows = [{"instruction": "do the thing", "code": "pass"} for _ in range(4)]
    rows[1]["code"] = "pass\n"
    rows[2]["instruction"] = "do  the   thing"

    assert dataset_inspect.inspect_rows(rows, COLUMNS).duplicate_rate == 0.75


def test_empty_and_malformed_are_measured_separately():
    # A row present-but-null is empty; a row missing a declared column is
    # malformed. Different defects, scored separately on purpose.
    rows = _clean(6)
    rows.append({"instruction": "", "code": None})  # empty
    rows.append({"instruction": "only this one"})  # malformed: no `code` key

    report = dataset_inspect.inspect_rows(rows, COLUMNS)

    assert report.empty_rate == 0.125
    assert report.malformed_rate == 0.125
    assert report.column_fill["code"] == 0.75


def test_templated_rows_score_as_repetitive():
    rows = [
        {
            "instruction": "Below is an instruction. Write a response that completes it properly.",
            "code": f"return {index}",
        }
        for index in range(10)
    ]
    report = dataset_inspect.inspect_rows(rows, COLUMNS)

    assert report.repetition_score == 1.0
    assert any("repetitive" in note for note in report.notes)


def test_a_short_column_is_not_mistaken_for_repetition():
    # Two filled cells always share a 50%+ mode. That is noise, not a defect,
    # and penalising it would fail every genuinely tiny sample.
    rows = [{"instruction": "a", "code": "x"}, {"instruction": "b", "code": "x"}]

    assert dataset_inspect.inspect_rows(rows, COLUMNS).repetition_score == 0.0


def test_benchmark_mentions_are_counted_not_judged():
    rows = _clean(8)
    rows[0]["instruction"] = "Solve this HumanEval task"
    rows[1]["code"] = "# taken from the MMLU test set"

    report = dataset_inspect.inspect_rows(rows, COLUMNS)

    assert report.contamination_hits == 2
    assert any("evaluation benchmark" in note for note in report.notes)


def test_personal_data_shapes_are_detected():
    rows = _clean(6)
    rows[0]["instruction"] = "email me at someone@example.com"
    rows[1]["instruction"] = "his ssn is 123-45-6789"

    assert dataset_inspect.inspect_rows(rows, COLUMNS).pii_hits == 2


def test_ordinary_prose_does_not_trip_the_pii_patterns():
    rows = [
        {"instruction": f"released in 2024 with version 3.11.{index}", "code": "pass"}
        for index in range(8)
    ]

    assert dataset_inspect.inspect_rows(rows, COLUMNS).pii_hits == 0


def test_an_unexpected_script_is_flagged_against_the_requested_language():
    rows = [{"instruction": "привет как дела сегодня", "code": "pass"} for _ in range(5)]

    report = dataset_inspect.inspect_rows(rows, COLUMNS, languages=("en",))

    assert report.dominant_script == "cyrillic"
    assert report.unexpected_language is True


def test_no_language_expectation_means_no_language_complaint():
    # A programming-language or multilingual spec implies no script at all, so
    # suspicion is the wrong default — "python" maps to no expected script.
    rows = [{"instruction": "привет как дела сегодня", "code": "pass"} for _ in range(5)]

    assert (
        dataset_inspect.inspect_rows(rows, COLUMNS, languages=("python",)).unexpected_language
        is False
    )


def test_size_adequacy_compares_the_real_total_not_the_sample():
    report = dataset_inspect.inspect_rows(
        _clean(10), COLUMNS, desired_examples=1000, num_rows_total=250
    )

    assert report.size_adequacy == 0.25
    assert any("250 rows available against 1000 requested" in note for note in report.notes)


def test_an_ample_dataset_saturates_at_one():
    report = dataset_inspect.inspect_rows(
        _clean(10), COLUMNS, desired_examples=100, num_rows_total=50_000
    )

    assert report.size_adequacy == 1.0


def test_no_rows_reports_nothing_established_rather_than_something_clean():
    report = dataset_inspect.inspect_rows([], COLUMNS)

    assert report.rows_sampled == 0
    assert report.notes == ["no rows could be sampled from this dataset"]


def test_non_dict_entries_are_ignored_rather_than_crashing():
    report = dataset_inspect.inspect_rows([*_clean(4), "junk", None, 7], COLUMNS)

    assert report.rows_sampled == 4


def test_plain_column_names_are_accepted_as_well_as_viewer_features():
    report = dataset_inspect.inspect_rows(_clean(6), ["instruction", "code"])

    assert report.column_fill == {"instruction": 1.0, "code": 1.0}


def test_the_report_serializes_to_the_record_shape():
    payload = dataset_inspect.inspect_rows(_clean(6), COLUMNS).to_dict()

    assert payload["rows_sampled"] == 6
    assert payload["duplicates_estimate"] == 0.0
    assert payload["invalid_rows"] == 0.0
    assert "script_mix" in payload


def test_a_constant_categorical_column_is_not_templated_content():
    # The live-Hub regression: m-a-p/CodeFeedback-Filtered-Instruction carries a
    # `lang` column that is "python" on every row. That is a fact about the
    # schema, not generated filler, and scoring it as maximal repetition cost a
    # genuinely good 156k-example dataset 0.2 of its quality.
    rows = [
        {
            "instruction": f"write a function that does thing number {index} correctly",
            "code": f"def f{index}(): return {index}",
            "lang": "python",
        }
        for index in range(20)
    ]
    columns = [{"name": "instruction"}, {"name": "code"}, {"name": "lang"}]

    # 1/20 is the floor when every content value is distinct — the constant
    # `lang` column contributed nothing.
    assert dataset_inspect.inspect_rows(rows, columns).repetition_score == 0.05


def test_a_constant_content_column_still_counts_as_repetition():
    # The converse: the same shape, but the constant column is long-form text.
    rows = [
        {
            "instruction": "Below is an instruction that describes a task. Write a response.",
            "code": f"def f{index}(): return {index}",
            "lang": "python",
        }
        for index in range(20)
    ]
    columns = [{"name": "instruction"}, {"name": "code"}, {"name": "lang"}]

    assert dataset_inspect.inspect_rows(rows, columns).repetition_score == 1.0
