"""Covers agents/coder/benchmark.py — the scorer, the comparator, and the
frozen plan corpus.

The corpus is kept continuously valid here the same way test_coder_starters.py
keeps the starter library valid: every plan file must parse and satisfy the
Experiment Planner's own output contract, because the Coder Agent validates its
input against exactly that before doing anything, so a malformed case would fail
at run time as a benchmark harness bug rather than as a result.
"""

import json

import pytest

from research_pipeline.agents.coder import benchmark
from research_pipeline.agents.experiment_planner.schema import validate_output

CORPUS = benchmark.corpus_cases()


# -- the frozen corpus ---------------------------------------------------------------------


def test_the_corpus_is_not_empty():
    assert CORPUS, "a benchmark with no cases silently measures nothing"


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_every_case_satisfies_the_planner_output_contract(path):
    validate_output(json.loads(path.read_text()))


def test_every_case_has_a_globally_unique_hypothesis_id():
    """Cases are matched across runs by hypothesis_id, so two cases sharing one
    would collapse into a single entry and silently shrink the corpus."""
    ids = [
        plan["hypothesis_id"]
        for path in CORPUS
        for plan in json.loads(path.read_text())["experiment_plans"]
    ]
    assert len(ids) == len(set(ids)), sorted(ids)


def test_the_corpus_covers_every_execution_route():
    """The point of a fixed corpus is that it exercises the branches in
    _attempt_once. Losing coverage of one is how a regression there goes
    unnoticed while the headline number stays flat."""
    plans = [plan for path in CORPUS for plan in json.loads(path.read_text())["experiment_plans"]]
    complexities = {plan["estimated_complexity"] for plan in plans}
    assert complexities == {"low", "medium", "high"}
    assert any(not plan["feasible"] for plan in plans), "no case exercises the skipped path"
    assert any(plan["feasible"] for plan in plans)
    sources = " ".join(plan["data_requirements"]["source"].lower() for plan in plans)
    assert "synthetic" in sources, "no case exercises the surrogate-data path"
    assert "hugging face" in sources, "no case exercises the dataset-lookup path"


# -- scoring -------------------------------------------------------------------------------


def _experiment(
    hid, status="completed", meets=True, attempts=0, history=None, real=True, full_size=True
):
    return {
        "hypothesis_id": hid,
        "status": status,
        "reason": "" if status == "completed" else "because",
        "code_path": f"experiments/{hid}",
        "assumptions_made": [],
        "results": {"metrics": {"accuracy": 0.9}, "meets_success_criteria": meets, "notes": ""}
        if status == "completed"
        else None,
        "fix_attempts": attempts,
        "fix_history": history or [],
        "slurm_job_id": None,
        "starter_used": "",
        "data_provenance": {"all_inputs_real": real},
        "compute_provenance": {"ran_at_full_size": full_size},
    }


def _summary(tmp_path, name, experiments):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "coder_agent_summary_20260905T000000Z.json").write_text(
        json.dumps(
            {
                "experiments": experiments,
                "shared_infrastructure_path": "experiments/_shared",
                "source_hypothesis_ids": [e["hypothesis_id"] for e in experiments],
                "generated_at": "2026-09-05T00:00:00+00:00",
                "model": "test",
            }
        )
    )
    return directory


def test_a_withheld_verdict_is_completed_but_not_interpretable(tmp_path):
    """The distinction the whole scorer is built around. An experiment that ran
    and had its verdict withheld — synthetic inputs, or a run truncated to fit
    its budget — produced no finding a paper can state, and counting it as a
    success is how you optimise toward a pipeline that always finishes and never
    concludes anything."""
    root = _summary(
        tmp_path,
        "run",
        [
            _experiment("H1", meets=True),
            _experiment("H2", meets="unknown", real=False),
            _experiment("H3", meets=False),
        ],
    )
    result = benchmark.score(root)

    assert result.total == 3
    assert result.completed == 3
    # A real refutation counts: it is a finding. "unknown" is not.
    assert result.interpretable == 2
    assert result.real_data == 2


def test_scoring_counts_each_non_completed_status_separately(tmp_path):
    root = _summary(
        tmp_path,
        "run",
        [
            _experiment("H1"),
            _experiment("H2", status="skipped"),
            _experiment("H3", status="code_generated_not_run"),
            _experiment("H4", status="submitted_to_slurm"),
            _experiment("H5", status="slurm_job_failed"),
        ],
    )
    result = benchmark.score(root)

    assert (result.completed, result.skipped, result.deferred, result.failed) == (1, 1, 2, 1)


def test_a_truncated_run_is_counted_even_though_it_completed(tmp_path):
    root = _summary(
        tmp_path, "run", [_experiment("H1", meets="unknown", full_size=False), _experiment("H2")]
    )
    result = benchmark.score(root)
    assert result.truncated == 1
    assert result.interpretable == 1


def test_a_summary_predating_compute_provenance_is_not_counted_as_truncated(tmp_path):
    # Otherwise every comparison against an older run would report a phantom
    # improvement in a metric that simply did not exist before.
    old = _experiment("H1")
    del old["compute_provenance"]
    result = benchmark.score(_summary(tmp_path, "run", [old]))
    assert result.truncated == 0


def test_attempts_to_green_averages_only_the_experiments_that_finished(tmp_path):
    """Averaging in the ones that exhausted the budget measures how often the
    budget is exhausted — which `completed` already reports — and pins this
    number near max_fix_attempts however cheap the successes get."""
    root = _summary(
        tmp_path,
        "run",
        [
            _experiment("H1", attempts=0),
            _experiment("H2", attempts=2),
            _experiment("H3", status="code_generated_not_run", attempts=3),
        ],
    )
    result = benchmark.score(root)
    assert result.attempts_to_green == 1.0
    assert result.total_fix_attempts == 5


def test_the_unresolved_error_is_the_last_one_that_was_never_cleared(tmp_path):
    """An earlier failure a regeneration fixed is not what cost the experiment;
    counting it would rank the noisiest category above the one ending runs."""
    history = [
        {
            "attempt": 1,
            "error_source": "compile_check",
            "error_summary": "",
            "code_path": "",
            "resolved": True,
        },
        {
            "attempt": 2,
            "error_source": "run_experiment",
            "error_summary": "",
            "code_path": "",
            "resolved": False,
        },
    ]
    root = _summary(
        tmp_path,
        "run",
        [
            _experiment("H1", status="code_generated_not_run", history=history),
            # Completed in the end, so nothing is still standing against it.
            _experiment("H2", history=history),
        ],
    )
    result = benchmark.score(root)
    assert dict(result.unresolved_errors) == {"run_experiment": 1}


def test_scoring_skips_a_corrupt_summary_rather_than_aborting(tmp_path):
    """One interrupted run's half-written file must not hide every other run's
    data — the same rule scripts/analyze_coder_fix_history.py follows."""
    root = _summary(tmp_path, "run", [_experiment("H1")])
    (root / "coder_agent_summary_broken.json").write_text("{not json")
    (root / "coder_agent_summary_empty.json").write_text('{"no": "experiments"}')

    result = benchmark.score(root)
    assert result.total == 1


def test_scoring_walks_a_whole_tree_of_runs(tmp_path):
    _summary(tmp_path / "a", "case1", [_experiment("H1")])
    _summary(tmp_path / "b", "case2", [_experiment("H2")])
    assert benchmark.score(tmp_path).total == 2


def test_scoring_accepts_a_single_summary_file(tmp_path):
    root = _summary(tmp_path, "run", [_experiment("H1")])
    only = next(root.glob("coder_agent_summary_*.json"))
    assert benchmark.score(only).total == 1


# -- comparing -----------------------------------------------------------------------------


def test_comparison_names_which_cases_moved_and_which_way(tmp_path):
    before = benchmark.score(
        _summary(
            tmp_path,
            "before",
            [
                _experiment("H1", status="code_generated_not_run"),
                _experiment("H2", meets="unknown"),
                _experiment("H3"),
            ],
        )
    )
    after = benchmark.score(
        _summary(
            tmp_path,
            "after",
            [
                _experiment("H1"),  # not run -> interpretable
                _experiment("H2", meets="unknown"),  # unchanged
                _experiment("H3", status="skipped"),  # regressed
            ],
        )
    )

    changes = {c.hypothesis_id: c for c in benchmark.compare(before, after)}
    assert set(changes) == {"H1", "H3"}, "an unchanged case should not be reported"
    assert changes["H1"].direction == "better"
    assert changes["H1"].after == "interpretable"
    assert changes["H3"].direction == "worse"


def test_a_case_present_in_only_one_run_is_reported_not_dropped(tmp_path):
    """It usually means the corpus itself changed, which invalidates the
    comparison — that has to be visible rather than silently absorbed."""
    before = benchmark.score(_summary(tmp_path, "before", [_experiment("H1")]))
    after = benchmark.score(_summary(tmp_path, "after", [_experiment("H1"), _experiment("H9")]))
    changes = benchmark.compare(before, after)
    assert [(c.hypothesis_id, c.before) for c in changes] == [("H9", "(absent)")]


def test_the_comparison_report_warns_when_the_two_runs_are_different_sizes(tmp_path):
    before = benchmark.score(_summary(tmp_path, "before", [_experiment("H1")]))
    after = benchmark.score(_summary(tmp_path, "after", [_experiment("H1"), _experiment("H2")]))
    report = benchmark.format_comparison(before, after)
    assert "not comparable as rates" in report


def test_the_comparison_report_says_what_one_case_is_worth(tmp_path):
    """A benchmark that launders one flaky run into a confident percentage is
    worse than no benchmark, so the resolution is printed, not assumed."""
    experiments = [_experiment(f"H{i}") for i in range(4)]
    before = benchmark.score(_summary(tmp_path, "before", experiments))
    after = benchmark.score(_summary(tmp_path, "after", experiments))
    report = benchmark.format_comparison(before, after)
    assert "One case is 25%" in report
    assert "none — every case landed the same way" in report


def test_the_score_report_leads_with_interpretable(tmp_path):
    root = _summary(tmp_path, "run", [_experiment("H1"), _experiment("H2", meets="unknown")])
    report = benchmark.format_score(benchmark.score(root))
    assert "interpretable      1/2 (50%)" in report
    assert report.index("interpretable") < report.index("completed")


def test_the_score_report_handles_an_empty_directory(tmp_path):
    assert "no coder summaries found" in benchmark.format_score(benchmark.score(tmp_path))
