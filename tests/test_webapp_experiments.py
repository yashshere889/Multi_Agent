import json

from research_pipeline.webapp import experiments


def _summary(run_dir, experiments_list, name="coder_agent_summary_20260820T120000Z.json"):
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / name).write_text(
        json.dumps(
            {
                "experiments": experiments_list,
                "shared_infrastructure_path": str(run_dir / "experiments" / "_shared"),
                "source_hypothesis_ids": [e["hypothesis_id"] for e in experiments_list],
                "generated_at": "2026-08-20T12:00:00+00:00",
                "model": "qwen3-coder",
            }
        )
    )
    return outputs / name


def _completed_experiment(run_dir, hypothesis_id="H1"):
    code_dir = run_dir / "experiments" / hypothesis_id
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "run.py").write_text("def main():\n    pass\n")
    (code_dir / "results.json").write_text('{"metrics": {"f1": 0.71}}')
    (code_dir / "requirements.txt").write_text("numpy\n")
    return {
        "hypothesis_id": hypothesis_id,
        "status": "completed",
        "reason": "",
        "code_path": str(code_dir),
        "assumptions_made": ["treated missing labels as negatives"],
        "results": {"metrics": {"f1": 0.71, "auc": 0.83}, "meets_success_criteria": True, "notes": "ran clean"},
        "fix_attempts": 0,
        "fix_history": [],
        "slurm_job_id": None,
        "starter_used": "sklearn_classification",
        "data_provenance": {
            "inputs": [{"name": "IMDB reviews", "kind": "real_download", "uri": "hf://imdb", "reason": ""}],
            "methodological_validity": "real data — findings are interpretable as evidence for the hypothesis",
            "all_inputs_real": True,
            "surrogate_count": 0,
        },
    }


def test_no_summary_means_nothing_to_inspect(tmp_path):
    assert experiments.load_experiments(tmp_path) == ([], None)


def test_loads_a_completed_experiment(tmp_path):
    _summary(tmp_path, [_completed_experiment(tmp_path)])

    views, summary_rel = experiments.load_experiments(tmp_path)

    assert len(views) == 1
    view = views[0]
    assert view.hypothesis_id == "H1"
    assert view.ok
    assert view.metrics == {"f1": 0.71, "auc": 0.83}
    assert view.verdict is True
    assert not view.verdict_withheld
    assert view.starter_used == "sklearn_classification"
    assert summary_rel == "outputs/coder_agent_summary_20260820T120000Z.json"


def test_the_newest_summary_wins(tmp_path):
    """The agent writes an interim summary as well as a final one."""
    _summary(tmp_path, [_completed_experiment(tmp_path, "H1")], "coder_agent_summary_20260820T120000Z.json")
    _summary(tmp_path, [_completed_experiment(tmp_path, "H2")], "coder_agent_summary_20260820T130000Z.json")

    views, _ = experiments.load_experiments(tmp_path)

    assert [v.hypothesis_id for v in views] == ["H2"]


def test_a_withheld_verdict_is_surfaced_with_its_reason(tmp_path):
    """provenance.apply_to_results turns meets_success_criteria into the string
    "unknown" when any input is synthetic — the distinction that stops the
    Writer publishing a refutation off invented numbers."""
    experiment = _completed_experiment(tmp_path)
    experiment["results"] = {
        "metrics": {"rmse": 2.1},
        "meets_success_criteria": "unknown",
        "model_reported_meets_success_criteria": False,
        "methodological_validity": "synthetic surrogate data — the pipeline is exercised but ...",
        "verdict_withheld_because": "One or more inputs are synthetic surrogates, so these metrics ...",
        "notes": "",
    }
    experiment["data_provenance"] = {
        "inputs": [{"name": "CMS claims", "kind": "synthetic_surrogate", "reason": "requires a Data Use Agreement"}],
        "methodological_validity": "synthetic surrogate data — ...",
        "all_inputs_real": False,
        "surrogate_count": 1,
    }
    _summary(tmp_path, [experiment])

    view = experiments.load_experiments(tmp_path)[0][0]

    assert view.verdict_withheld
    assert view.model_reported_verdict is False
    assert view.surrogate_count == 1
    assert "synthetic surrogates" in view.verdict_withheld_because
    assert view.inputs[0]["kind"] == "synthetic_surrogate"


def test_fix_history_carries_each_attempt_and_its_snapshot(tmp_path):
    experiment = _completed_experiment(tmp_path)
    attempts_dir = tmp_path / "experiments" / "H1" / "fix_attempts"
    for n in (1, 2):
        (attempts_dir / f"attempt_{n}").mkdir(parents=True)
    experiment["fix_attempts"] = 2
    experiment["fix_history"] = [
        {
            "attempt": 1,
            "error_source": "missing_data_fallback",
            "error_summary": "load_data reads reviews.csv with nothing to fall back on",
            "code_path": str(attempts_dir / "attempt_1"),
            "resolved": True,
            "same_error_streak": 1,
        },
        {
            "attempt": 2,
            "error_source": "run_experiment",
            "error_summary": "Traceback ...\nZeroDivisionError",
            "code_path": str(attempts_dir / "attempt_2"),
            "resolved": False,
            "same_error_streak": 1,
        },
    ]
    _summary(tmp_path, [experiment])

    view = experiments.load_experiments(tmp_path)[0][0]

    assert [f.attempt for f in view.fix_attempts] == [1, 2]
    assert [f.error_source for f in view.fix_attempts] == ["missing_data_fallback", "run_experiment"]
    assert [f.resolved for f in view.fix_attempts] == [True, False]
    assert view.fix_attempts[0].code_rel == "experiments/H1/fix_attempts/attempt_1"


def test_a_fix_entry_written_before_the_streak_counter_defaults_to_one(tmp_path):
    experiment = _completed_experiment(tmp_path)
    experiment["fix_history"] = [
        {"attempt": 1, "error_source": "compile_check", "error_summary": "x", "code_path": "", "resolved": True}
    ]
    _summary(tmp_path, [experiment])

    view = experiments.load_experiments(tmp_path)[0][0]

    assert view.fix_attempts[0].same_error_streak == 1
    assert view.fix_attempts[0].code_rel is None


def test_only_the_experiment_files_that_exist_are_offered(tmp_path):
    _summary(tmp_path, [_completed_experiment(tmp_path)])

    view = experiments.load_experiments(tmp_path)[0][0]

    names = [name for name, _why, _rel in view.files]
    assert names == ["run.py", "results.json", "requirements.txt"]
    assert view.files[0][2] == "experiments/H1/run.py"


def test_code_written_outside_the_run_directory_gets_no_link(tmp_path):
    """A CLI run's experiments/ copied in later: the browser cannot serve it,
    and resolve_inside would refuse the link anyway."""
    experiment = _completed_experiment(tmp_path)
    experiment["code_path"] = "/somewhere/else/experiments/H1"
    _summary(tmp_path, [experiment])

    view = experiments.load_experiments(tmp_path)[0][0]

    assert view.code_rel is None
    assert view.files == []


def test_an_experiment_that_was_never_run_has_no_metrics(tmp_path):
    experiment = _completed_experiment(tmp_path)
    experiment["status"] = "code_generated_not_run"
    experiment["reason"] = "plan needs a GPU and none was detected"
    experiment["results"] = None
    _summary(tmp_path, [experiment])

    view = experiments.load_experiments(tmp_path)[0][0]

    assert not view.ok
    assert view.metrics == {}
    assert view.verdict is None
    assert not view.verdict_withheld
    assert "GPU" in view.reason


def test_a_torn_summary_file_degrades_instead_of_raising(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "coder_agent_summary_20260820T120000Z.json").write_text('{"experiments": [{"hyp')

    assert experiments.load_experiments(tmp_path) == ([], None)


def test_find_experiment_by_hypothesis_id(tmp_path):
    _summary(tmp_path, [_completed_experiment(tmp_path, "H1"), _completed_experiment(tmp_path, "H2")])

    views, _ = experiments.load_experiments(tmp_path)

    assert experiments.find_experiment(views, "H2").hypothesis_id == "H2"
    assert experiments.find_experiment(views, "H9") is None
