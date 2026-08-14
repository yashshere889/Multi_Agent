"""Keeps agents/coder/templates/starters/*.sections continuously pre-validated:
every entry in starters.STARTERS must parse, splice into a real run.py via the
same sandbox.render_experiment_template every model-generated experiment goes
through, pass every check a generated experiment is checked against, and
actually execute to a successful results.json. This is what makes "pre-
validated" true on every commit, not just at authoring time.

Also covers starters.select_starter's keyword matching directly, since
test_coder_agent.py only exercises the wiring (that a matching starter's code
reaches the prompt), not the matching logic itself.
"""

import json
import subprocess
import sys

import pytest

from research_pipeline.agents.coder import sandbox, starters

STARTER_IDS = sorted(starters.STARTERS)


@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_starter_sections_are_present_and_non_empty(starter_id):
    sections = starters.STARTERS[starter_id]["sections"]
    for name in (
        "load_data_function",
        "build_model_function",
        "run_experiment_function",
        "evaluate_function",
    ):
        assert sections[name].strip(), f"{starter_id}.{name} is empty"


@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_starter_defines_the_required_function_names(starter_id):
    sections = starters.STARTERS[starter_id]["sections"]
    assert sandbox.check_required_function_names(sections) == []


@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_starter_load_data_has_no_unguarded_local_read(starter_id):
    sections = starters.STARTERS[starter_id]["sections"]
    assert sandbox.check_data_fallback(sections["load_data_function"]) == []


def _render(starter_id: str) -> str:
    sections = starters.STARTERS[starter_id]["sections"]
    return sandbox.render_experiment_template(
        hypothesis_id=f"TEST_{starter_id}",
        objective="test objective",
        design="test design",
        data_description="synthetic",
        baseline="n/a",
        success_criteria="n/a",
        agent_imports=sections.get("imports", ""),
        agent_configuration=sections.get("configuration", ""),
        load_data_function=sections["load_data_function"],
        build_model_function=sections["build_model_function"],
        run_experiment_function=sections["run_experiment_function"],
        evaluate_function=sections["evaluate_function"],
        agent_helpers=sections.get("helpers", ""),
    )


@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_starter_compiles_clean(starter_id):
    run_py = _render(starter_id)
    _, compile_error = sandbox.lenient_compile_check(run_py, "run.py")
    assert compile_error is None


@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_starter_passes_static_safety_check(starter_id):
    run_py = _render(starter_id)
    assert sandbox.static_safety_check(run_py) == []


@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_starter_runs_end_to_end_and_writes_a_successful_results_json(starter_id, tmp_path):
    """Runs the fully-spliced run.py with the system interpreter (starters are
    pure standard library — no venv/uv provisioning needed) and checks the
    same results.json shape a completed experiment's execution produces."""
    run_py = _render(starter_id)
    experiment_dir = tmp_path / starter_id
    experiment_dir.mkdir()
    (experiment_dir / "run.py").write_text(run_py)

    proc = subprocess.run(
        [sys.executable, "run.py"],
        cwd=experiment_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"{starter_id} run.py failed:\n{proc.stdout}\n{proc.stderr}"

    results = json.loads((experiment_dir / "results.json").read_text())
    assert results["status"] == "success"
    assert results["meets_success_criteria"] in (True, False, "unknown")
    assert results["metrics"]


@pytest.mark.parametrize(
    ("design", "method_name", "expected_id"),
    [
        ("comparative benchmark", "logistic regression", "classification"),
        ("comparative benchmark", "random forest classifier", "classification"),
        ("predictive modeling", "linear regression", "regression"),
        ("predictive modeling", "ridge regression", "regression"),
        ("cluster analysis", "k-means clustering", "clustering"),
        ("cluster analysis", "DBSCAN", "clustering"),
        ("NLP sentiment analysis", "TF-IDF + logistic regression", "text_classification"),
        ("text classification of product reviews", "naive bayes", "text_classification"),
    ],
)
def test_select_starter_matches_expected_archetype(design, method_name, expected_id):
    plan = {
        "design": design,
        "methods": [{"name": method_name}],
        "data_requirements": {"source": "synthetic", "description": ""},
    }
    selected = starters.select_starter(plan)
    assert selected is not None
    assert selected["id"] == expected_id


def test_select_starter_falls_back_to_general_when_nothing_matches():
    plan = {
        "design": "Monte Carlo simulation",
        "methods": [{"name": "custom agent-based model"}],
        "data_requirements": {"source": "synthetic", "description": ""},
    }
    assert starters.select_starter(plan) is None


def test_select_starter_prioritizes_text_classification_over_plain_classification():
    """A plan that mentions both "classification" and NLP-specific signals
    should match text_classification, not fall through to the generic tabular
    classifier — this is the reason for _MATCH_KEYWORDS' fixed check order."""
    plan = {
        "design": "text classification of movie reviews",
        "methods": [{"name": "logistic regression classifier"}],
        "data_requirements": {"source": "synthetic", "description": "review corpus"},
    }
    selected = starters.select_starter(plan)
    assert selected is not None
    assert selected["id"] == "text_classification"


def test_starters_dir_has_no_untracked_extra_files():
    """Every .sections file under templates/starters/ should be registered in
    starters.STARTERS — an orphaned file would silently never be validated or
    selectable."""
    on_disk = {p.stem for p in starters.STARTERS_DIR.glob("*.sections")}
    assert on_disk == set(starters.STARTERS)
