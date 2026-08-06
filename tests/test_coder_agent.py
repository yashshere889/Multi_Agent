import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_pipeline.agents.coder import sandbox
from research_pipeline.agents.coder.coder_agent import CoderAgent, CoderAgentError
from research_pipeline.agents.coder.schema import SchemaValidationError, validate_output


# -- schema.py: output validation ------------------------------------------------------


def _valid_experiment(hid: str, status: str = "completed") -> dict:
    base = {"hypothesis_id": hid, "status": status, "reason": "", "assumptions_made": []}
    if status == "skipped":
        return {**base, "reason": "infeasible", "code_path": None, "results": None}
    if status == "code_generated_not_run":
        return {**base, "reason": "needs GPU", "code_path": f"experiments/{hid}", "results": None}
    return {
        **base,
        "code_path": f"experiments/{hid}",
        "results": {"metrics": {"accuracy": 0.9}, "meets_success_criteria": True, "notes": "ok"},
    }


def _valid_output() -> dict:
    return {
        "experiments": [_valid_experiment("H1"), _valid_experiment("H2", "skipped")],
        "shared_infrastructure_path": "experiments/_shared",
        "source_hypothesis_ids": ["H1", "H2"],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def test_validate_output_accepts_well_formed_result():
    validate_output(_valid_output())  # should not raise


def test_validate_output_rejects_bad_status():
    data = _valid_output()
    data["experiments"][0]["status"] = "in_progress"
    with pytest.raises(SchemaValidationError, match="status should be one of"):
        validate_output(data)


def test_validate_output_requires_results_when_completed():
    data = _valid_output()
    data["experiments"][0]["results"] = None
    with pytest.raises(SchemaValidationError, match="results is required"):
        validate_output(data)


def test_validate_output_requires_null_code_path_when_skipped():
    data = _valid_output()
    data["experiments"][1]["code_path"] = "experiments/H2"
    with pytest.raises(SchemaValidationError, match="code_path should be null"):
        validate_output(data)


def test_validate_output_requires_reason_for_skipped():
    data = _valid_output()
    data["experiments"][1]["reason"] = ""
    with pytest.raises(SchemaValidationError, match="reason is required"):
        validate_output(data)


def test_validate_output_expected_ids_catches_missing_entry():
    data = _valid_output()
    data["experiments"] = data["experiments"][:1]
    with pytest.raises(SchemaValidationError, match="missing entries"):
        validate_output(data, expected_hypothesis_ids=["H1", "H2"])


# -- sandbox.py: execution mechanics (local/offline only) ------------------------------


def test_missing_packages_flags_nonexistent_and_ignores_present():
    result = sandbox.missing_packages(["os", "definitely_not_a_real_package_xyz==1.0", ""])
    assert result == ["definitely_not_a_real_package_xyz"]


def test_compile_check_passes_valid_file(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("def main():\n    return 1\n")
    assert sandbox.compile_check([good]) is None


def test_compile_check_reports_syntax_error(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def main(:\n    pass\n")
    error = sandbox.compile_check([bad])
    assert error is not None
    assert "bad.py" in error


def test_render_sbatch_template_includes_hypothesis_id():
    text = sandbox.render_sbatch_template("H1", has_requirements=True)
    assert "H1" in text
    assert "python run.py" in text
    assert "sbatch" in text.lower()


def test_ensure_experiment_env_returns_current_interpreter_when_nothing_missing(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("os\n")  # stdlib, never "missing"
    python_exec, error = sandbox.ensure_experiment_env(tmp_path, requirements, network_available=True)
    assert error is None
    assert python_exec == Path(sys.executable)


def test_ensure_experiment_env_reports_missing_package_without_network(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely_not_a_real_package_xyz\n")
    python_exec, error = sandbox.ensure_experiment_env(tmp_path, requirements, network_available=False)
    assert python_exec is None
    assert "definitely_not_a_real_package_xyz" in error
    assert "network" in error


def test_run_experiment_success(tmp_path):
    script = tmp_path / "run.py"
    script.write_text("print('ok')\n")
    ok, message = sandbox.run_experiment(Path(sys.executable), script, tmp_path, timeout_seconds=10)
    assert ok is True
    assert message == ""


def test_run_experiment_reports_nonzero_exit(tmp_path):
    script = tmp_path / "run.py"
    script.write_text("import sys\nsys.exit(1)\n")
    ok, message = sandbox.run_experiment(Path(sys.executable), script, tmp_path, timeout_seconds=10)
    assert ok is False
    assert "exited with code 1" in message


def test_run_experiment_reports_timeout(tmp_path):
    script = tmp_path / "run.py"
    script.write_text("import time\ntime.sleep(5)\n")
    ok, message = sandbox.run_experiment(Path(sys.executable), script, tmp_path, timeout_seconds=1)
    assert ok is False
    assert "timed out" in message


# -- coder_agent.py: orchestration, with a fake chat model and no real network/uv ------


class FakeChatModel:
    """Returns canned JSON responses looked up by a keyword found in the prompt."""

    def __init__(self, response_by_keyword: dict[str, str]):
        self._response_by_keyword = response_by_keyword
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        prompt_text = messages[-1][1]
        for keyword, response in self._response_by_keyword.items():
            if keyword in prompt_text:
                return SimpleNamespace(content=response)
        raise AssertionError(f"No fake response configured for prompt: {prompt_text[:300]!r}")


# Matches the four required run_py_sections keys the Coder Agent now asks the
# model for — run.py's metadata block + orchestration are a fixed template
# (see sandbox.render_experiment_template), not model output.
GOOD_SECTIONS = {
    "imports": "",
    "configuration": "",
    "load_data_function": "def load_data():\n    return None\n",
    "build_model_function": "def build_model():\n    return None\n",
    "run_experiment_function": "def run_experiment(data, model):\n    return {}\n",
    "evaluate_function": (
        "def evaluate(experiment_output):\n"
        '    return {"accuracy": 0.9, "meets_success_criteria": True, "success_notes": "ok"}\n'
    ),
    "helpers": "",
}


def _codegen_response(sections=None, readme="# Test experiment\n", requirements="", assumptions=None, needs_gpu=False) -> str:
    return json.dumps({
        "run_py_sections": sections or GOOD_SECTIONS,
        "readme": readme,
        "requirements_txt": requirements,
        "assumptions_made": assumptions or [],
        "needs_network": False,
        "needs_gpu": needs_gpu,
    })


def _plan(hid: str, feasible=True, complexity="low") -> dict:
    return {
        "hypothesis_id": hid,
        "feasible": feasible,
        "feasibility_notes": "not enough compute" if not feasible else "fits in a single job",
        "objective": f"test objective for {hid}",
        "variables": {"independent": ["x"], "dependent": ["y"]},
        "design": "comparative benchmark",
        "data_requirements": {"source": "synthetic", "description": "d", "preprocessing_steps": []},
        "methods": [{"name": "baseline", "description": "d", "reused_from_literature": True}],
        "evaluation": {"metrics": ["accuracy"], "baseline": "random", "success_criteria": "accuracy > 0.5"},
        "implementation_steps": [{"step": 1, "description": "do it"}],
        "estimated_complexity": complexity,
        "risks": ["none"],
    }


def _planner_output(plans: list[dict], shared_infrastructure=None) -> dict:
    return {
        "experiment_plans": plans,
        "shared_infrastructure": shared_infrastructure or [],
        "priority_order": [{"hypothesis_id": p["hypothesis_id"], "rank": i + 1, "justification": "j"} for i, p in enumerate(plans)],
        "source_hypothesis_ids": [p["hypothesis_id"] for p in plans],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def test_run_skips_infeasible_plan_without_calling_llm(tmp_path):
    fake_model = FakeChatModel({})  # no responses configured — any call fails the test
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=tmp_path / "experiments", output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", feasible=False)]))

    exp = result["experiments"][0]
    assert exp["status"] == "skipped"
    assert exp["code_path"] is None
    assert "infeasible" in exp["reason"].lower()
    assert fake_model.calls == []


def test_run_completes_low_complexity_feasible_plan(tmp_path):
    fake_model = FakeChatModel({'"hypothesis_id": "H1"': _codegen_response()})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=experiments_dir, output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["results"]["metrics"] == {"accuracy": 0.9}
    assert exp["results"]["meets_success_criteria"] is True
    assert (experiments_dir / "H1" / "run.py").exists()
    assert (experiments_dir / "H1" / "results.json").exists()

    written = list((tmp_path / "outputs").glob("coder_agent_summary_*.json"))
    assert len(written) == 1


def test_run_marks_high_complexity_as_not_run_and_generates_sbatch(tmp_path):
    fake_model = FakeChatModel({'"hypothesis_id": "H2"': _codegen_response()})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=experiments_dir, output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H2", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "high" in exp["reason"]
    assert (experiments_dir / "H2" / "run.sbatch").exists()
    assert not (experiments_dir / "H2" / "results.json").exists()  # never executed


def test_run_detects_syntax_error_and_never_executes(tmp_path):
    broken_sections = {**GOOD_SECTIONS, "evaluate_function": "def evaluate(experiment_output:\n    pass\n"}
    fake_model = FakeChatModel({'"hypothesis_id": "H1"': _codegen_response(broken_sections)})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=experiments_dir, output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "syntax error" in exp["reason"].lower()
    assert not (experiments_dir / "H1" / "results.json").exists()


def test_run_reports_execution_failure(tmp_path):
    # run.py's fixed orchestration catches this, writes a "failure" results.json,
    # and exits 1 — sandbox.run_experiment sees the nonzero exit.
    failing_sections = {**GOOD_SECTIONS, "run_experiment_function": "def run_experiment(data, model):\n    raise RuntimeError('boom')\n"}
    fake_model = FakeChatModel({'"hypothesis_id": "H1"': _codegen_response(failing_sections)})
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=tmp_path / "experiments", output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "execution failed" in exp["reason"].lower()


def test_run_reports_missing_required_code_section(tmp_path):
    incomplete_sections = {**GOOD_SECTIONS, "evaluate_function": ""}  # model omitted a required section
    fake_model = FakeChatModel({'"hypothesis_id": "H1"': _codegen_response(incomplete_sections)})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=experiments_dir, output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "evaluate_function" in exp["reason"]
    assert not (experiments_dir / "H1" / "run.py").exists()  # never written — incomplete response


def test_run_skips_execution_when_gpu_needed_but_unavailable(tmp_path):
    fake_model = FakeChatModel({'"hypothesis_id": "H1"': _codegen_response(needs_gpu=True)})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=experiments_dir, output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "gpu" in exp["reason"].lower()
    assert (experiments_dir / "H1" / "run.sbatch").exists()


def test_run_missing_package_without_network_skips_execution(tmp_path):
    fake_model = FakeChatModel({
        '"hypothesis_id": "H1"': _codegen_response(requirements="definitely_not_a_real_package_xyz\n"),
    })
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=tmp_path / "experiments", output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "definitely_not_a_real_package_xyz" in exp["reason"]
    assert "network" in exp["reason"]


def test_run_sets_up_shared_infrastructure_exactly_once(tmp_path):
    fake_model = FakeChatModel({
        "Shared infrastructure items": json.dumps({"files": {"data_utils.py": "def load():\n    pass\n", "README.md": "shared"}}),
        '"hypothesis_id": "H1"': _codegen_response(),
        '"hypothesis_id": "H2"': _codegen_response(),
    })
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=experiments_dir, output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1"), _plan("H2")], shared_infrastructure=["shared eval harness"]))

    shared_calls = [c for c in fake_model.calls if "Shared infrastructure items" in c[-1][1]]
    assert len(shared_calls) == 1
    assert (experiments_dir / "_shared" / "data_utils.py").exists()
    assert (experiments_dir / "_shared" / "__init__.py").exists()
    assert (experiments_dir / "__init__.py").exists()
    assert result["shared_infrastructure_path"] == str(experiments_dir / "_shared")
    assert all(e["status"] == "completed" for e in result["experiments"])


def test_run_respects_priority_order(tmp_path):
    fake_model = FakeChatModel({
        '"hypothesis_id": "H1"': _codegen_response(),
        '"hypothesis_id": "H2"': _codegen_response(),
    })
    agent = CoderAgent(
        chat_model=fake_model, experiments_dir=tmp_path / "experiments", output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    planner_output = _planner_output([_plan("H1"), _plan("H2")])
    # reverse priority: H2 should be processed (and appear) before H1
    planner_output["priority_order"] = [
        {"hypothesis_id": "H2", "rank": 1, "justification": "j"},
        {"hypothesis_id": "H1", "rank": 2, "justification": "j"},
    ]
    result = agent.run(planner_output)
    assert [e["hypothesis_id"] for e in result["experiments"]] == ["H2", "H1"]


def test_run_rejects_malformed_planner_input(tmp_path):
    agent = CoderAgent(
        chat_model=FakeChatModel({}), experiments_dir=tmp_path / "experiments", output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False,
    )
    with pytest.raises(CoderAgentError, match="Experiment Planner's output schema"):
        agent.run({"experiment_plans": "not a list"})


# -- CoderAgent._read_results_json: direct unit tests -----------------------------------
# (the fixed run.py template always writes a well-formed results.json on any exit path,
# so these malformed/missing cases can no longer be reached by driving the full agent
# through a fake LLM response — testing the method directly instead)


def test_read_results_json_returns_none_when_file_missing(tmp_path):
    assert CoderAgent._read_results_json(tmp_path) is None


def test_read_results_json_returns_none_when_not_valid_json(tmp_path):
    (tmp_path / "results.json").write_text("not json")
    assert CoderAgent._read_results_json(tmp_path) is None


def test_read_results_json_returns_none_when_missing_required_keys(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"foo": "bar"}))
    assert CoderAgent._read_results_json(tmp_path) is None


def test_read_results_json_accepts_well_formed_file(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True}))
    data = CoderAgent._read_results_json(tmp_path)
    assert data["metrics"] == {"accuracy": 0.9}
    assert data["notes"] == ""  # defaulted


# -- sandbox.render_experiment_template: brace-safety ------------------------------------


def test_render_experiment_template_handles_braces_in_generated_code(tmp_path):
    # str.format() would choke on this (dict literal + f-string); plain
    # string replacement must not.
    run_py = sandbox.render_experiment_template(
        hypothesis_id="H1",
        objective="test",
        design="benchmark",
        data_description="d",
        baseline="b",
        success_criteria="s",
        agent_imports="",
        agent_configuration='CONFIG = {"a": 1, "b": {"nested": 2}}',
        load_data_function='def load_data():\n    return {"x": 1}\n',
        build_model_function="def build_model():\n    return None\n",
        run_experiment_function='def run_experiment(data, model):\n    label = f"{data[\'x\']}"\n    return {"label": label}\n',
        evaluate_function='def evaluate(experiment_output):\n    return {"meets_success_criteria": "unknown", "success_notes": f"got {experiment_output}"}\n',
        agent_helpers="",
    )
    assert 'CONFIG = {"a": 1, "b": {"nested": 2}}' in run_py
    assert "__AGENT_" not in run_py  # every marker was substituted

    rendered = tmp_path / "run.py"
    rendered.write_text(run_py)
    assert sandbox.compile_check([rendered]) is None  # the whole splice is syntactically valid
