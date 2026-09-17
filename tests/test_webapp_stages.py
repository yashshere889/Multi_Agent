"""What the progress view predicts, for full runs and for partial ones.

`summarize()` is covered end to end in test_webapp_runner.py (every stage's
delta, through the real event stream). This file is about the other half:
which rows `build_progress` invents for a run that has not finished yet, which
is the half that a custom start/end stage changes.
"""

from __future__ import annotations

from research_pipeline.webapp import stages


def _events(*stage_names) -> list[dict]:
    return [{"stage": name, "summary": {}, "ts": "2026-01-01T00:00:00+00:00"} for name in stage_names]


def _keys(rows, status=None) -> list[str]:
    return [r["key"] for r in rows if status is None or r["status"] == status]


# -- the default full run is unchanged ---------------------------------------------------


def test_a_default_run_still_predicts_the_whole_fixed_chain():
    rows = stages.build_progress(_events(stages.LITERATURE), is_active=True)

    assert _keys(rows) == [
        stages.LITERATURE,
        stages.INTERDISCIPLINARY,
        stages.HYPOTHESIS,
        stages.EXPERIMENT_PLANNER,
        stages.CODER,
        stages.DRAFT_OR_REVISE,
        stages.FINALIZE,
    ]
    assert _keys(rows, stages.RUNNING) == [stages.INTERDISCIPLINARY]


def test_a_finished_run_predicts_nothing():
    rows = stages.build_progress(_events(stages.LITERATURE, stages.FINALIZE), is_active=False)

    assert _keys(rows) == [stages.LITERATURE, stages.FINALIZE]
    assert _keys(rows, stages.PENDING) == []


# -- include_interdisciplinary=False -----------------------------------------------------


def test_the_cross_field_stage_is_not_predicted_when_it_is_disabled():
    upcoming = stages._next_stage(stages.LITERATURE, {}, None, include_interdisciplinary=False)

    assert upcoming == stages.HYPOTHESIS


def test_the_cross_field_stage_is_never_listed_when_it_is_disabled():
    rows = stages.build_progress(_events(stages.LITERATURE), is_active=True, include_interdisciplinary=False)

    assert stages.INTERDISCIPLINARY not in _keys(rows)
    assert _keys(rows, stages.RUNNING) == [stages.HYPOTHESIS]
    assert _keys(rows, stages.PENDING) == [
        stages.EXPERIMENT_PLANNER,
        stages.CODER,
        stages.DRAFT_OR_REVISE,
        stages.FINALIZE,
    ]


def test_a_run_that_has_not_started_still_omits_the_disabled_stage():
    rows = stages.build_progress([], is_active=True, include_interdisciplinary=False)

    assert _keys(rows, stages.RUNNING) == [stages.LITERATURE]
    assert stages.INTERDISCIPLINARY not in _keys(rows)


# -- end_stage short of the full pipeline ------------------------------------------------


def test_the_end_stage_predicts_finalize_rather_than_the_next_stage():
    upcoming = stages._next_stage(stages.HYPOTHESIS, {}, None, end_stage="hypothesis")

    assert upcoming == stages.FINALIZE


def test_nothing_past_the_end_stage_is_listed_as_pending():
    rows = stages.build_progress(_events(stages.LITERATURE), is_active=True, end_stage="hypothesis")

    assert _keys(rows) == [stages.LITERATURE, stages.INTERDISCIPLINARY, stages.HYPOTHESIS, stages.FINALIZE]
    assert _keys(rows, stages.RUNNING) == [stages.INTERDISCIPLINARY]
    # A run stopping at the Hypothesis Agent never drafts a paper.
    assert stages.DRAFT_OR_REVISE not in _keys(rows)
    assert stages.CODER not in _keys(rows)


def test_the_stage_after_the_end_stage_is_only_finalize():
    rows = stages.build_progress(_events(stages.LITERATURE, stages.INTERDISCIPLINARY, stages.HYPOTHESIS), True, end_stage="hypothesis")

    assert _keys(rows, stages.RUNNING) == [stages.FINALIZE]
    assert _keys(rows, stages.PENDING) == []


def test_ending_at_the_coder_keeps_every_upstream_stage_but_drops_the_write_up():
    rows = stages.build_progress(_events(stages.LITERATURE), is_active=True, end_stage="coder")

    assert _keys(rows) == [
        stages.LITERATURE,
        stages.INTERDISCIPLINARY,
        stages.HYPOTHESIS,
        stages.EXPERIMENT_PLANNER,
        stages.CODER,
        stages.FINALIZE,
    ]
    assert stages.DRAFT_OR_REVISE not in _keys(rows)


def test_ending_at_a_disabled_stage_resolves_back_to_the_stage_before_it():
    """end_stage="interdisciplinary_literature" with that stage switched off is
    a run that stops after literature — the same resolution the orchestrator's
    own router applies, so the prediction cannot disagree with the routing."""
    upcoming = stages._next_stage(
        stages.LITERATURE, {}, None, end_stage="interdisciplinary_literature", include_interdisciplinary=False
    )

    assert upcoming == stages.FINALIZE

    rows = stages.build_progress(
        _events(stages.LITERATURE), is_active=True, end_stage="interdisciplinary_literature", include_interdisciplinary=False
    )
    assert _keys(rows) == [stages.LITERATURE, stages.FINALIZE]


def test_an_explicit_full_end_stage_matches_the_default():
    default = stages.build_progress(_events(stages.CODER), is_active=True)
    explicit = stages.build_progress(_events(stages.CODER), is_active=True, end_stage="writer_reviewer")

    assert default == explicit
    assert _keys(explicit, stages.RUNNING) == [stages.DRAFT_OR_REVISE]


# -- the writer/reviewer cycle is still the orchestrator's decision -----------------------


def test_the_review_stage_still_routes_on_the_orchestrators_own_condition():
    another = stages._next_stage(stages.REVIEW, {"converged": False, "iteration": 1}, 3, end_stage="writer_reviewer")
    done = stages._next_stage(stages.REVIEW, {"converged": True, "iteration": 1}, 3, end_stage="writer_reviewer")

    assert another == stages.DRAFT_OR_REVISE
    assert done == stages.FINALIZE


def test_the_seeded_literature_node_is_displayed_as_the_literature_stage():
    assert stages.stage_for_node("seed_literature") == stages.LITERATURE
    assert stages.stage_for_node(stages.CODER) == stages.CODER
