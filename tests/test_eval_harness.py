"""Tests for the eval harness and the LLM judge.

Nothing here reaches the network or an LLM: the literature graph is stubbed and
the judge gets a canned chat model. What's being checked is the harness's own
contracts — that a run is replayable offline, that one failed question doesn't
cost the others theirs, and that the threshold sweep says what it claims to.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from research_pipeline.eval import harness, judge, report


# -- the judge ---------------------------------------------------------------


class _FakeModel:
    def __init__(self, *contents):
        self._contents = list(contents)
        self.calls = []

    def invoke(self, messages, **_kwargs):
        self.calls.append(messages)
        return SimpleNamespace(content=self._contents[min(len(self.calls) - 1, len(self._contents) - 1)])


def _papers(*titles):
    return [{"title": t, "abstract": "an abstract"} for t in titles]


def test_judge_returns_a_verdict_per_paper():
    model = _FakeModel('{"verdicts": [{"id": "P0", "keep": true}, {"id": "P1", "keep": false}]}')
    assert judge.judge_pool(model, "a question", _papers("A", "B")) == [True, False]


def test_judge_leaves_unanswered_papers_unjudged():
    model = _FakeModel('{"verdicts": [{"id": "P0", "keep": true}]}')
    assert judge.judge_pool(model, "a question", _papers("A", "B")) == [True, None]


def test_judge_survives_an_unreachable_model():
    model = _FakeModel("")
    model.invoke = lambda *_a, **_k: (_ for _ in ()).throw(ConnectionError("down"))
    assert judge.judge_pool(model, "a question", _papers("A")) == [None]


def test_judge_discards_a_non_boolean_verdict():
    model = _FakeModel('{"verdicts": [{"id": "P0", "keep": "yes"}]}')
    assert judge.judge_pool(model, "a question", _papers("A")) == [None]


def test_precision_ignores_unjudged_papers():
    """Guessing either way would bias the one number the harness exists to report."""
    assert judge.precision([True, False, None]) == 0.5


def test_precision_is_none_when_nothing_was_judged():
    """None, not 0.0 — 'the judge was unreachable' and 'every paper was junk'
    must not render as the same number."""
    assert judge.precision([None, None]) is None


def test_the_judge_does_not_reuse_the_screens_rubric():
    """If these ever converge the eval stops being independent of the thing it
    evaluates — a screen that is confidently wrong would score perfectly."""
    from research_pipeline.agents.literature import relevance

    assert judge.SYSTEM_PROMPT != relevance.SYSTEM_PROMPT
    assert "0 to 5" not in judge.JUDGE_PROMPT
    assert "keep" in judge.JUDGE_PROMPT


def test_agreement_compares_the_screen_against_the_judge():
    papers = [
        {"title": "A", "relevance_score": 5},   # screen keeps, judge keeps  -> agree
        {"title": "B", "relevance_score": 1},   # screen drops, judge drops  -> agree
        {"title": "C", "relevance_score": 4},   # screen keeps, judge bins   -> disagree
        {"title": "D", "relevance_score": 0},   # screen drops, judge keeps  -> disagree
    ]
    result = judge.agreement_with_screen(papers, [True, False, False, True], min_score=3)

    assert result["compared"] == 4
    assert result["agreement"] == 0.5
    assert result["screen_kept_judge_binned"] == 1
    assert result["screen_dropped_judge_kept"] == 1


def test_agreement_is_none_without_overlapping_evidence():
    assert judge.agreement_with_screen([{"title": "A"}], [True], min_score=3) is None


# -- running -----------------------------------------------------------------


def _gold_entry(question="does RAG help?", papers=None):
    return {
        "question": question,
        "papers": papers or [{"title": "Attention Is All You Need", "arxiv_id": "1706.03762"}],
    }


def _stub_graph(monkeypatch, result=None, error=None):
    class _Graph:
        def invoke(self, _payload, config=None):
            if error:
                raise error
            return result or {"merged_papers": [], "search_queries": []}

    monkeypatch.setattr(harness, "build_literature_graph", lambda: _Graph())


def test_run_gold_set_records_the_whole_returned_pool(monkeypatch, tmp_path):
    """The raw pool is what makes score_run replayable without re-searching."""
    pool = [{"title": "Attention Is All You Need", "arxiv_id": "1706.03762", "source": "arxiv"}]
    _stub_graph(monkeypatch, {"merged_papers": pool, "search_queries": ["attention"]})

    run = harness.run_gold_set([_gold_entry()], name="baseline", output_dir=tmp_path)

    assert run["name"] == "baseline"
    assert run["questions"][0]["literature_output"]["merged_papers"] == pool
    assert run["questions"][0]["gold_total"] == 1
    # Self-describing: two runs are only comparable if these match.
    assert run["config"]["max_results_per_query"]


def test_one_failed_question_does_not_cost_the_others_their_results(monkeypatch, tmp_path):
    calls = {"n": 0}

    class _Graph:
        def invoke(self, _payload, config=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("rate limited")
            return {"merged_papers": [{"title": "A Paper"}], "search_queries": []}

    monkeypatch.setattr(harness, "build_literature_graph", lambda: _Graph())

    run = harness.run_gold_set(
        [_gold_entry("first"), _gold_entry("second")], name="r", output_dir=tmp_path
    )

    assert "rate limited" in run["questions"][0]["error"]
    assert run["questions"][1]["literature_output"]["merged_papers"] == [{"title": "A Paper"}]


def test_eval_mode_restores_what_it_patched(monkeypatch, tmp_path):
    """It disables the in-pipeline screen and PDF downloading for the duration
    of a run; leaving either patched would silently change the real pipeline."""
    from research_pipeline.agents.literature import graph as lit_graph
    from research_pipeline.agents.literature import nodes as lit_nodes

    before_settings, before_download = lit_nodes.settings, lit_graph.download_papers_node
    _stub_graph(monkeypatch)

    harness.run_gold_set([_gold_entry()], name="r", output_dir=tmp_path)

    assert lit_nodes.settings is before_settings
    assert lit_graph.download_papers_node is before_download


def test_eval_mode_restores_even_when_a_run_raises(monkeypatch):
    from research_pipeline.agents.literature import nodes as lit_nodes

    before = lit_nodes.settings
    with pytest.raises(RuntimeError):
        with harness._eval_mode():
            assert lit_nodes.settings.enable_relevance_filter is False
            raise RuntimeError("boom")

    assert lit_nodes.settings is before


# -- scoring -----------------------------------------------------------------


def _run_with(pool, question="does RAG help?"):
    return {
        "name": "r",
        "config": {},
        "questions": [{
            "question": question,
            "gold_total": 1,
            "literature_output": {"merged_papers": pool, "search_queries": ["q"]},
        }],
    }


def test_score_run_needs_no_network_when_screening_is_off():
    pool = [{"title": "Attention Is All You Need", "arxiv_id": "1706.03762", "abstract": "a"}]
    with patch.object(harness, "get_chat_model", side_effect=AssertionError("must not build a model")):
        scored = harness.score_run(
            _run_with(pool), {"does RAG help?": _gold_entry()["papers"]}, screen=False, judge=False
        )

    assert scored["questions"][0]["recall"] == 1.0
    assert scored["aggregate"]["mean_recall"] == 1.0


def test_score_run_replays_the_screen_over_a_saved_pool():
    pool = [
        {"title": "Attention Is All You Need", "arxiv_id": "1706.03762", "abstract": "a"},
        {"title": "A Totally Unrelated Paper", "abstract": "b"},
    ]
    model = _FakeModel('{"scores": [{"id": "P0", "score": 5}, {"id": "P1", "score": 0}]}')

    scored = harness.score_run(
        _run_with(pool), {"does RAG help?": _gold_entry()["papers"]}, screen=True, chat_model=model
    )

    # Screening annotates but must not drop — the sweep decides what a
    # threshold would remove.
    assert scored["questions"][0]["pool_size"] == 2
    assert scored["questions"][0]["mean_relevance_score"] == 2.5


def test_the_sweep_shows_what_each_threshold_costs():
    pool = [
        {"title": "Attention Is All You Need", "arxiv_id": "1706.03762", "abstract": "a"},
        {"title": "A Totally Unrelated Paper", "abstract": "b"},
    ]
    model = _FakeModel('{"scores": [{"id": "P0", "score": 5}, {"id": "P1", "score": 0}]}')

    scored = harness.score_run(
        _run_with(pool), {"does RAG help?": _gold_entry()["papers"]},
        screen=True, thresholds=(0, 3), chat_model=model,
    )
    sweep = {row["threshold"]: row for row in scored["questions"][0]["sweep"]}

    # Threshold 0 is the pre-screen baseline; 3 drops the unrelated paper
    # without costing the gold hit.
    assert sweep[0]["kept"] == 2 and sweep[0]["recall"] == 1.0
    assert sweep[3]["kept"] == 1 and sweep[3]["recall"] == 1.0


def test_the_sweep_reports_good_papers_a_threshold_would_lose():
    pool = [{"title": "A Good Paper", "abstract": "a"}, {"title": "Another Good One", "abstract": "b"}]
    model = _FakeModel(
        '{"scores": [{"id": "P0", "score": 5}, {"id": "P1", "score": 1}]}',   # screen
        '{"verdicts": [{"id": "P0", "keep": true}, {"id": "P1", "keep": true}]}',  # judge
    )

    scored = harness.score_run(
        _run_with(pool), {}, screen=True, judge=True, thresholds=(0, 3), chat_model=model
    )
    sweep = {row["threshold"]: row for row in scored["questions"][0]["sweep"]}

    # The judge would keep both, so a threshold of 3 costs one good paper.
    assert sweep[0]["lost_good_papers"] == 0
    assert sweep[3]["lost_good_papers"] == 1


def test_an_unscreened_paper_survives_every_threshold():
    """Consistent with the production screen: an unscored paper is never dropped
    on the strength of the screen's own failure."""
    pool = [{"title": "Unscored Paper", "abstract": "a"}]
    scored = harness.score_run(_run_with(pool), {}, screen=False, thresholds=(0, 5))

    assert all(row["kept"] == 1 for row in scored["questions"][0]["sweep"])


# -- comparison and reporting ------------------------------------------------


def test_compare_reports_per_metric_deltas():
    before = {"name": "a", "questions": [{"question": "q", "recall": 0.4, "pool_size": 40}]}
    after = {"name": "b", "questions": [{"question": "q", "recall": 0.5, "pool_size": 25}]}

    result = harness.compare(before, after)

    assert result["questions"][0]["recall"]["delta"] == pytest.approx(0.1)
    assert result["questions"][0]["pool_size"]["delta"] == -15


def test_compare_flags_questions_with_no_baseline():
    result = harness.compare({"questions": []}, {"questions": [{"question": "new"}]})
    assert result["unmatched"] == ["new"]


def test_the_report_renders_a_run_with_failures_and_no_judge():
    """Total by construction: a missing number prints as an em dash, never 0.0."""
    scored = {
        "name": "r",
        "questions": [{"question": "q", "error": "boom", "pool_size": 0, "gold_found": 0,
                       "gold_total": 3, "recall": 0.0, "abstract_coverage": None,
                       "mean_relevance_score": None, "sweep": []}],
        "aggregate": {"questions": 1, "mean_recall": 0.0, "total_gold_found": 0, "total_gold": 3},
    }
    text = report.format_run(scored)

    assert "FAILED" in text
    assert "—" in text


def test_the_report_renders_an_empty_run():
    assert report.format_run({"name": "empty", "questions": [], "aggregate": {}})


def test_run_files_round_trip_through_disk(tmp_path):
    run = _run_with([{"title": "A"}])
    path = harness.write_run(run, tmp_path / "r.json")
    assert harness.read_run(path) == run
    assert json.loads(path.read_text())["name"] == "r"
