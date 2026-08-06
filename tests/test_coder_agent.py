import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_pipeline.agents.coder import sandbox, slurm_submit
from research_pipeline.agents.coder.coder_agent import CoderAgent, CoderAgentError
from research_pipeline.agents.coder.schema import ERROR_SUMMARY_MAX_CHARS, SchemaValidationError, validate_output


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


# -- sandbox.read_results_json_for_diagnosis: direct unit tests --------------------------
# (the fixed run.py template always writes a well-formed results.json on any exit path,
# so these malformed/missing cases can no longer be reached by driving the full agent
# through a fake LLM response — testing the function directly instead)


def test_read_results_json_reports_missing_file(tmp_path):
    results, diagnosis = sandbox.read_results_json_for_diagnosis(tmp_path)
    assert results is None
    assert "did not write results.json" in diagnosis


def test_read_results_json_reports_invalid_json(tmp_path):
    (tmp_path / "results.json").write_text("not json")
    results, diagnosis = sandbox.read_results_json_for_diagnosis(tmp_path)
    assert results is None
    assert "not valid JSON" in diagnosis


def test_read_results_json_reports_missing_required_keys(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"foo": "bar"}))
    results, diagnosis = sandbox.read_results_json_for_diagnosis(tmp_path)
    assert results is None
    assert "missing required key" in diagnosis


def test_read_results_json_surfaces_the_traceback_run_py_recorded(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"error": "Traceback...\nValueError: bad shape"}))
    results, diagnosis = sandbox.read_results_json_for_diagnosis(tmp_path)
    assert results is None
    assert "ValueError: bad shape" in diagnosis


def test_read_results_json_accepts_well_formed_file(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True}))
    data, diagnosis = sandbox.read_results_json_for_diagnosis(tmp_path)
    assert diagnosis is None
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


# -- coder_agent.py: the fix loop --------------------------------------------------------


class ScriptedChatModel:
    """Serves responses by prompt kind, so a test can make the first codegen
    fail and the fix that follows succeed. Each kind takes a list consumed in
    order; the last entry repeats once exhausted."""

    KINDS = {
        "The code you generated for hypothesis": "fix",
        "Review the experiment code below": "self_review",
        "Shared infrastructure items": "shared_infra",
    }

    def __init__(self, **responses_by_kind: list[str]):
        self._responses = responses_by_kind
        self.calls_by_kind: dict[str, int] = {}

    def _kind(self, prompt: str) -> str:
        for marker, kind in self.KINDS.items():
            if marker in prompt:
                return kind
        return "codegen"

    def invoke(self, messages):
        prompt = messages[-1][1]
        kind = self._kind(prompt)
        index = self.calls_by_kind.get(kind, 0)
        self.calls_by_kind[kind] = index + 1
        responses = self._responses.get(kind)
        if not responses:
            raise AssertionError(f"No {kind!r} response configured for prompt: {prompt[:200]!r}")
        return SimpleNamespace(content=responses[min(index, len(responses) - 1)])


BROKEN_SYNTAX_SECTIONS = {**GOOD_SECTIONS, "evaluate_function": "def evaluate(experiment_output:\n    pass\n"}
RAISING_SECTIONS = {
    **GOOD_SECTIONS,
    "run_experiment_function": "def run_experiment(data, model):\n    raise RuntimeError('boom')\n",
}


def _agent(tmp_path, model, **kwargs):
    return CoderAgent(
        chat_model=model, experiments_dir=tmp_path / "experiments", output_dir=tmp_path / "outputs",
        network_check=lambda: False, gpu_check=lambda: False, **kwargs,
    )


def test_fix_loop_recovers_after_a_failed_execution(tmp_path):
    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)],
        fix=[_codegen_response(GOOD_SECTIONS)],
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["fix_attempts"] == 1
    assert exp["fix_history"][0]["error_source"] == "run_experiment"
    assert exp["fix_history"][0]["resolved"] is True
    assert model.calls_by_kind["fix"] == 1


def test_fix_loop_recovers_after_a_syntax_error(tmp_path):
    model = ScriptedChatModel(
        codegen=[_codegen_response(BROKEN_SYNTAX_SECTIONS)],
        fix=[_codegen_response(GOOD_SECTIONS)],
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["fix_history"][0]["error_source"] == "compile_check"


def test_fix_loop_gives_up_after_max_attempts(tmp_path):
    model = ScriptedChatModel(codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(RAISING_SECTIONS)])
    result = _agent(tmp_path, model, max_fix_attempts=2).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert exp["fix_attempts"] == 2
    assert len(exp["fix_history"]) == 2
    assert "2 fix attempt(s)" in exp["reason"]
    assert model.calls_by_kind["fix"] == 2


def test_fix_loop_records_nothing_when_the_first_attempt_works(tmp_path):
    model = ScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["fix_attempts"] == 0
    assert exp["fix_history"] == []
    assert "fix" not in model.calls_by_kind


def test_fix_loop_snapshots_the_code_that_failed(tmp_path):
    model = ScriptedChatModel(codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    snapshot = Path(result["experiments"][0]["fix_history"][0]["code_path"])
    assert snapshot.exists()
    assert "raise RuntimeError('boom')" in snapshot.read_text()  # the failing version, preserved
    assert "raise RuntimeError('boom')" not in (tmp_path / "experiments" / "H1" / "run.py").read_text()


def test_fix_loop_truncates_a_long_error_summary(tmp_path):
    noisy = {**GOOD_SECTIONS, "run_experiment_function": "def run_experiment(data, model):\n    raise RuntimeError('x' * 5000)\n"}
    model = ScriptedChatModel(codegen=[_codegen_response(noisy)], fix=[_codegen_response(GOOD_SECTIONS)])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    summary = result["experiments"][0]["fix_history"][0]["error_summary"]
    assert 0 < len(summary) <= ERROR_SUMMARY_MAX_CHARS


def test_env_error_is_not_retried_through_the_llm(tmp_path):
    # A missing package is an environment problem — regenerating the code can't
    # fix it, so it must not burn fix attempts.
    model = ScriptedChatModel(codegen=[_codegen_response(requirements="definitely_not_a_real_package_xyz\n")])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert exp["fix_attempts"] == 0
    assert "fix" not in model.calls_by_kind


# -- sandbox.static_safety_check ---------------------------------------------------------


@pytest.mark.parametrize(
    "code, expected",
    [
        ("result = eval(user_input)", "eval"),
        ("exec(compile(src, '<s>', 'exec'))", "exec"),
        ("subprocess.run(cmd, shell=True)", "shell=True"),
        ("os.system('rm -rf /tmp/x')", "os.system"),
        ("shutil.rmtree(path)", "rmtree"),
        ("os.remove(path)", "deletion"),
        ("import pickle\npickle.loads(blob)", "pickle"),
        ("key = os.environ['AWS_SECRET_ACCESS_KEY']", "credential"),
    ],
)
def test_static_safety_check_flags_dangerous_code(code, expected):
    findings = sandbox.static_safety_check(code)
    assert findings, f"expected {code!r} to be flagged"
    assert any(expected in f for f in findings)


def test_static_safety_check_passes_ordinary_experiment_code():
    code = "\n".join(GOOD_SECTIONS[k] for k in ("load_data_function", "build_model_function", "evaluate_function"))
    assert sandbox.static_safety_check(code) == []


def test_lint_failure_routes_through_the_fix_loop(tmp_path):
    unsafe = {**GOOD_SECTIONS, "helpers": "def cleanup(p):\n    import shutil\n    shutil.rmtree(p)\n"}
    model = ScriptedChatModel(codegen=[_codegen_response(unsafe)], fix=[_codegen_response(GOOD_SECTIONS)])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["fix_history"][0]["error_source"] == "static_lint"
    assert "rmtree" in exp["fix_history"][0]["error_summary"]


# -- coder_agent.py: gated SLURM auto-submit ---------------------------------------------


def _patch_settings(monkeypatch, **overrides):
    """Settings is a frozen dataclass, so swap in a copy rather than mutating."""
    from research_pipeline.agents.coder import coder_agent as coder_agent_module

    monkeypatch.setattr(
        coder_agent_module, "settings", dataclasses.replace(coder_agent_module.settings, **overrides)
    )


@pytest.fixture
def auto_submit(monkeypatch):
    """Turns auto-submission on and stubs out every real SLURM call. Yields a
    record of what would have been submitted."""
    submitted = []
    _patch_settings(
        monkeypatch,
        coder_auto_submit_slurm=True,
        coder_max_concurrent_slurm_jobs=4,
        coder_max_slurm_jobs_per_run=10,
    )
    monkeypatch.setattr(slurm_submit, "count_running_jobs", lambda user=None: 0)

    def fake_submit(sbatch_path, cwd):
        submitted.append(Path(sbatch_path))
        return f"999{len(submitted)}", None

    monkeypatch.setattr(slurm_submit, "submit_job", fake_submit)
    return submitted


def _clean_review() -> str:
    return json.dumps({"looks_correct": True, "concerns": []})


def test_auto_submit_off_by_default_leaves_sbatch_for_review(tmp_path, monkeypatch):
    monkeypatch.setattr(
        slurm_submit, "submit_job", lambda *a, **k: pytest.fail("submitted with auto-submit off")
    )
    model = ScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert exp["slurm_job_id"] is None
    assert "submit it yourself" in exp["reason"]
    assert (tmp_path / "experiments" / "H1" / "run.sbatch").exists()


def test_auto_submit_submits_when_enabled_and_clean(tmp_path, auto_submit):
    model = ScriptedChatModel(codegen=[_codegen_response()], self_review=[_clean_review()])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "submitted_to_slurm"
    assert exp["slurm_job_id"] == "9991"
    assert exp["results"] is None
    assert len(auto_submit) == 1


def test_auto_submit_never_submits_code_the_lint_keeps_flagging(tmp_path, auto_submit):
    # The lint runs before the submit branch, so unsafe code goes through the
    # fix loop first. When the fix doesn't clean it up, nothing is submitted.
    unsafe = {**GOOD_SECTIONS, "helpers": "def wipe(p):\n    import os\n    os.system('rm -rf ' + p)\n"}
    model = ScriptedChatModel(
        codegen=[_codegen_response(unsafe)], fix=[_codegen_response(unsafe)], self_review=[_clean_review()]
    )
    result = _agent(tmp_path, model, max_fix_attempts=1).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "os.system" in exp["reason"]
    assert auto_submit == []  # never reached sbatch


def test_auto_submit_fixes_code_the_self_review_flags(tmp_path, auto_submit):
    model = ScriptedChatModel(
        codegen=[_codegen_response()],
        self_review=[json.dumps({"looks_correct": False, "concerns": ["load_data ignores the plan's dataset"]}), _clean_review()],
        fix=[_codegen_response(GOOD_SECTIONS)],
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "submitted_to_slurm"
    assert exp["fix_history"][0]["error_source"] == "self_review"
    assert "ignores the plan's dataset" in exp["fix_history"][0]["error_summary"]


def test_auto_submit_respects_the_concurrent_job_cap(tmp_path, auto_submit, monkeypatch):
    monkeypatch.setattr(slurm_submit, "count_running_jobs", lambda user=None: 4)  # already at the cap
    model = ScriptedChatModel(codegen=[_codegen_response()], self_review=[_clean_review()])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "already queued" in exp["reason"]
    assert auto_submit == []


def test_auto_submit_respects_the_per_run_budget(tmp_path, auto_submit, monkeypatch):
    _patch_settings(monkeypatch, coder_auto_submit_slurm=True, coder_max_slurm_jobs_per_run=1)
    model = ScriptedChatModel(codegen=[_codegen_response()], self_review=[_clean_review()])
    result = _agent(tmp_path, model).run(
        _planner_output([_plan("H1", complexity="high"), _plan("H2", complexity="high")])
    )

    statuses = {e["hypothesis_id"]: e["status"] for e in result["experiments"]}
    assert statuses["H1"] == "submitted_to_slurm"
    assert statuses["H2"] == "code_generated_not_run"
    assert len(auto_submit) == 1
    second = next(e for e in result["experiments"] if e["hypothesis_id"] == "H2")
    assert "CODER_MAX_SLURM_JOBS_PER_RUN" in second["reason"]


def test_auto_submit_failure_falls_back_to_manual_review(tmp_path, auto_submit, monkeypatch):
    monkeypatch.setattr(slurm_submit, "submit_job", lambda *a, **k: (None, "sbatch exited with code 1: invalid partition"))
    model = ScriptedChatModel(codegen=[_codegen_response()], self_review=[_clean_review()])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "invalid partition" in exp["reason"]
    assert exp["slurm_job_id"] is None
