import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_pipeline.agents.coder import huggingface_client, prompts, sandbox, slurm_submit
from research_pipeline.agents.coder.coder_agent import (
    CoderAgent,
    CoderAgentError,
    _compact_json,
    _estimate_tokens,
    _parse_assumptions,
    _parse_bool_text,
)
from research_pipeline.agents.coder.schema import (
    ERROR_SUMMARY_MAX_CHARS,
    SchemaValidationError,
    validate_output,
)
from research_pipeline.llm_sections import render_sections

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


def test_lenient_compile_check_passes_clean_source():
    source = "def main():\n    return 1\n"
    fixed, error = sandbox.lenient_compile_check(source, "good.py")
    assert error is None
    assert fixed == source


def test_lenient_compile_check_repairs_redundant_trailing_backslash():
    # A trailing '\' after the closing ')' is redundant (the call already
    # closed) and, left in, force-continues onto the next statement —
    # exactly the pattern observed from small quantized models.
    source = (
        "def build_model():\n"
        "    model = dict(\n"
        "        a=1, \\\n"
        "        b=2,\n"
        "    )\\\n"
        "    return model\n"
    )
    fixed, error = sandbox.lenient_compile_check(source, "bad.py")
    assert error is None
    assert "\\" not in fixed
    compile(fixed, "bad.py", "exec")  # doesn't raise


def test_lenient_compile_check_reports_original_error_when_repair_does_not_help():
    source = "def main(:\n    pass\n"
    fixed, error = sandbox.lenient_compile_check(source, "bad.py")
    assert error is not None
    assert "bad.py" in error
    assert fixed == source  # unmodified — the repair didn't fix a different bug


def test_lenient_compile_check_repairs_literal_escaped_newline():
    # Reproduces the JSON double-escaping bug: a newline the model intended
    # inside a run_py_sections value arrives as the two literal characters
    # '\' + 'n' instead of a real line break, which the tokenizer reads as an
    # invalid explicit line continuation.
    source = (
        "def load_data():\\n    import pandas as pd\n    df = pd.read_csv('x.csv')\n    return df\n"
    )
    fixed, error = sandbox.lenient_compile_check(source, "bad.py")
    assert error is None
    assert fixed == (
        "def load_data():\n    import pandas as pd\n    df = pd.read_csv('x.csv')\n    return df\n"
    )
    compile(fixed, "bad.py", "exec")  # doesn't raise


def test_lenient_compile_check_literal_newline_repair_is_surgical():
    # A valid string literal elsewhere in the file that legitimately
    # contains "\n" must survive untouched — this can't be a blanket
    # find/replace across the whole source.
    source = 'MESSAGE = "line one\\nline two"\ndef load_data():\\n    return MESSAGE\n'
    fixed, error = sandbox.lenient_compile_check(source, "bad.py")
    assert error is None
    assert 'MESSAGE = "line one\\nline two"' in fixed
    compile(fixed, "bad.py", "exec")  # doesn't raise


def test_check_shared_infra_files_passes_clean_files():
    files = {"utils.py": "def load():\n    return 1\n", "README.md": "docs"}
    repaired, problem = sandbox.check_shared_infra_files(files)
    assert problem == ""
    assert repaired == files


def test_check_shared_infra_files_repairs_backslash_and_reports_clean():
    files = {"utils.py": "def load():\n    x = dict(a=1)\\\n    return x\n"}
    repaired, problem = sandbox.check_shared_infra_files(files)
    assert problem == ""
    compile(repaired["utils.py"], "utils.py", "exec")  # doesn't raise


def test_check_shared_infra_files_reports_syntax_error():
    files = {"utils.py": "def load(:\n    pass\n"}
    _, problem = sandbox.check_shared_infra_files(files)
    assert "syntax error" in problem


def test_check_shared_infra_files_reports_unsafe_pattern():
    files = {"utils.py": "import os\ndef wipe():\n    os.system('rm -rf /')\n"}
    _, problem = sandbox.check_shared_infra_files(files)
    assert "static safety check" in problem


def test_render_sbatch_template_includes_hypothesis_id():
    text = sandbox.render_sbatch_template("H1", has_requirements=True)
    assert "H1" in text
    assert "python run.py" in text
    assert "sbatch" in text.lower()


def test_ensure_experiment_env_returns_current_interpreter_when_nothing_missing(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("os\n")  # stdlib, never "missing"
    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=True
    )
    assert error is None
    assert python_exec == Path(sys.executable)


def test_ensure_experiment_env_reports_missing_package_without_network(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely_not_a_real_package_xyz\n")
    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=False
    )
    assert python_exec is None
    assert "definitely_not_a_real_package_xyz" in error
    assert "network" in error


def test_ensure_experiment_env_prefers_uv_when_available(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely_not_a_real_package_xyz\n")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    recorded_cmds = []
    venv_python = tmp_path / ".venv" / "bin" / "python"

    def fake_run(cmd, **kwargs):
        recorded_cmds.append(cmd)
        # A real `uv venv` materializes the interpreter; simulate that so this
        # test exercises the code path the existence check now guards, same
        # as it would on a real filesystem.
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=True
    )
    assert error is None
    assert python_exec == tmp_path / ".venv" / "bin" / "python"
    assert all(cmd[0] == "uv" for cmd in recorded_cmds)


def test_ensure_experiment_env_falls_back_to_pip_when_uv_missing(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely_not_a_real_package_xyz\n")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    recorded_cmds = []
    venv_python = tmp_path / ".venv" / "bin" / "python"

    def fake_run(cmd, **kwargs):
        recorded_cmds.append(cmd)
        # A real `venv` module materializes the interpreter; simulate that so
        # this test exercises the code path the existence check now guards.
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=True
    )
    assert error is None
    assert python_exec == venv_python
    assert recorded_cmds == [
        [sys.executable, "-m", "venv", str(tmp_path / ".venv")],
        [str(venv_python), "-m", "pip", "install", "-r", str(requirements)],
    ]
    assert not any("uv" in cmd for cmd in recorded_cmds)


def test_ensure_experiment_env_pip_failure_is_terminal_with_network(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely_not_a_real_package_xyz\n")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)

    def fake_run(cmd, **kwargs):
        if "pip" in cmd:
            raise sandbox.subprocess.CalledProcessError(
                1,
                cmd,
                stderr="No matching distribution found for definitely_not_a_real_package_xyz",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=True
    )
    assert python_exec is None
    assert "pip" in error
    assert "definitely_not_a_real_package_xyz" in error


def test_ensure_experiment_env_reports_error_when_interpreter_is_missing_after_a_successful_provision(
    tmp_path, monkeypatch
):
    # Regression test for the 2026-08-12 production crash: `uv venv` + `uv pip
    # install` both exited 0 (no CalledProcessError, no TimeoutExpired) on an
    # Apptainer container over a network filesystem, but .venv/bin/python
    # still didn't exist afterward. Without an explicit existence check here,
    # this Path is trusted on faith and run_experiment's subprocess.run raises
    # a bare, uncaught FileNotFoundError that crashes the whole orchestrator
    # run instead of degrading to the same handled "couldn't provision an
    # environment" result every other failure in this function produces.
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely_not_a_real_package_xyz\n")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")

    def fake_run(cmd, **kwargs):
        # Reports success but never actually creates .venv/bin/python.
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=True
    )
    assert python_exec is None
    assert error is not None
    assert "no interpreter exists" in error
    assert str(tmp_path / ".venv" / "bin" / "python") in error


def test_ensure_experiment_env_clears_stale_venv_before_recreating(tmp_path, monkeypatch):
    # Regression test: a prior fix attempt can leave a partial .venv behind
    # (e.g. venv created but the install step failed), and both `uv venv` and
    # the stdlib `venv` module refuse to populate an existing directory. Without
    # clearing it first, every subsequent fix attempt would fail on venv
    # creation itself regardless of what changed in the generated code.
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely_not_a_real_package_xyz\n")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "stale_marker").write_text("left behind by a previous fix attempt\n")

    def fake_run(cmd, **kwargs):
        # Simulates a real `uv venv` recreating the directory from scratch —
        # if the stale one wasn't actually cleared first, stale_marker would
        # still be sitting alongside this.
        venv_python = venv_dir / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=True
    )
    assert error is None
    # The stale marker is gone even though the directory itself now exists
    # again (recreated by the mocked provisioning call) — proving the stale
    # one was cleared rather than reused.
    assert not (venv_dir / "stale_marker").exists()


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


def test_run_experiment_resolves_relative_script_path(tmp_path, monkeypatch):
    # Regression test: CODER_EXPERIMENTS_DIR defaults to a relative
    # "experiments", so callers can pass a relative run_script alongside a
    # cwd equal to that same relative directory. Passing the relative script
    # straight through to subprocess would make the interpreter re-resolve
    # it against its own (now-different) cwd, doubling the directory
    # (experiments/H1/experiments/H1/run.py) instead of finding the file.
    experiment_dir = tmp_path / "experiments" / "H1"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "run.py").write_text("print('ok')\n")
    monkeypatch.chdir(tmp_path)
    relative_run_script = Path("experiments") / "H1" / "run.py"

    recorded_cmds = []

    def fake_run(cmd, **kwargs):
        recorded_cmds.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    ok, message = sandbox.run_experiment(
        Path(sys.executable), relative_run_script, experiment_dir, timeout_seconds=10
    )
    assert ok is True
    script_arg = Path(recorded_cmds[0][1])
    assert script_arg.is_absolute()
    assert script_arg == (experiment_dir / "run.py").resolve()


def test_run_experiment_resolves_relative_python_executable(tmp_path, monkeypatch):
    # Regression test for a 2026-08-13 production crash: ensure_experiment_env
    # returns a venv interpreter path built from experiment_dir, which is
    # relative whenever CODER_EXPERIMENTS_DIR is (its default). That relative
    # python_executable was passed straight to subprocess.run alongside
    # cwd=experiment_dir (also relative) — undoubled, unlike run_script above
    # — so the OS re-resolved it against the subprocess's new cwd and raised
    # FileNotFoundError for a venv interpreter that did exist, just not under
    # that doubled path.
    experiment_dir = tmp_path / "experiments" / "H1"
    venv_bin = experiment_dir / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.symlink_to(sys.executable)
    (experiment_dir / "run.py").write_text("print('ok')\n")
    monkeypatch.chdir(tmp_path)
    relative_python_executable = Path("experiments") / "H1" / ".venv" / "bin" / "python"
    relative_run_script = Path("experiments") / "H1" / "run.py"

    ok, message = sandbox.run_experiment(
        relative_python_executable, relative_run_script, experiment_dir, timeout_seconds=10
    )
    assert ok is True, message


# -- coder_agent.py: orchestration, with a fake chat model and no real network/uv ------


class FakeChatModel:
    """Returns canned responses looked up by a keyword found in the prompt."""

    def __init__(self, response_by_keyword: dict[str, str]):
        self._response_by_keyword = response_by_keyword
        self.calls = []
        # The Coder Agent passes per-call max_tokens (and temperature on the fix
        # paths) through invoke_json/invoke_sections, so **kwargs is required
        # here — and recording them is what lets a test assert on the override.
        self.call_kwargs = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
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


def _codegen_response(
    sections=None, readme="# Test experiment\n", requirements="", assumptions=None, needs_gpu=False
) -> str:
    """Builds a codegen response in llm_sections.py's delimited format (no
    escaping of any kind — generated code is carried verbatim between markers).

    This is the one seam nearly every test in this file goes through, so the
    transport can change here rather than in each test. Note what it no longer
    has to do: json.dumps used to escape every newline and backslash in the
    generated code, which is precisely the round-trip a small quantized model
    kept getting wrong in production."""
    fields = {
        **(sections or GOOD_SECTIONS),
        "readme": readme,
        "requirements_txt": requirements,
        "assumptions_made": "\n".join(f"- {item}" for item in (assumptions or [])),
        "needs_network": "false",
        "needs_gpu": "true" if needs_gpu else "false",
    }
    return render_sections(fields)


def _unparseable_sections_response() -> str:
    """A response the delimited parser can't use: it opens a section and never
    closes it, and every later field is missing. The format-failure equivalent of
    the old '{"run_py_sections": {BROKEN' JSON fixture — this is what a
    completion truncated mid-answer actually looks like."""
    return (
        "===BEGIN imports===\nimport json\n===END imports===\n"
        "===BEGIN load_data_function===\ndef load_data():\n    return None\n"
    )


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
        "evaluation": {
            "metrics": ["accuracy"],
            "baseline": "random",
            "success_criteria": "accuracy > 0.5",
        },
        "implementation_steps": [{"step": 1, "description": "do it"}],
        "estimated_complexity": complexity,
        "risks": ["none"],
    }


def _planner_output(plans: list[dict], shared_infrastructure=None) -> dict:
    return {
        "experiment_plans": plans,
        "shared_infrastructure": shared_infrastructure or [],
        "priority_order": [
            {"hypothesis_id": p["hypothesis_id"], "rank": i + 1, "justification": "j"}
            for i, p in enumerate(plans)
        ],
        "source_hypothesis_ids": [p["hypothesis_id"] for p in plans],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def test_run_skips_infeasible_plan_without_calling_llm(tmp_path):
    fake_model = FakeChatModel({})  # no responses configured — any call fails the test
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=tmp_path / "experiments",
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", feasible=False)]))

    exp = result["experiments"][0]
    assert exp["status"] == "skipped"
    assert exp["code_path"] is None
    assert "infeasible" in exp["reason"].lower()
    assert fake_model.calls == []


def test_run_completes_low_complexity_feasible_plan(tmp_path):
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response()})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=experiments_dir,
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
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
    fake_model = FakeChatModel({'"hypothesis_id":"H2"': _codegen_response()})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=experiments_dir,
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H2", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "high" in exp["reason"]
    assert (experiments_dir / "H2" / "run.sbatch").exists()
    assert not (experiments_dir / "H2" / "results.json").exists()  # never executed


def test_run_executes_high_complexity_synchronously_when_gpu_flag_enabled(tmp_path, monkeypatch):
    _patch_settings(
        monkeypatch,
        coder_run_high_complexity_when_gpu_available=True,
        coder_high_complexity_timeout_seconds=300,
    )
    fake_model = FakeChatModel({'"hypothesis_id":"H2"': _codegen_response()})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=experiments_dir,
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: True,
    )
    result = agent.run(_planner_output([_plan("H2", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["results"]["meets_success_criteria"] is True
    assert (experiments_dir / "H2" / "results.json").exists()
    assert not (experiments_dir / "H2" / "run.sbatch").exists()  # ran instead of being deferred


def test_run_still_defers_high_complexity_without_gpu_even_with_flag_enabled(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_run_high_complexity_when_gpu_available=True)
    fake_model = FakeChatModel({'"hypothesis_id":"H2"': _codegen_response()})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=experiments_dir,
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,  # no GPU actually present
    )
    result = agent.run(_planner_output([_plan("H2", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert (experiments_dir / "H2" / "run.sbatch").exists()
    assert not (experiments_dir / "H2" / "results.json").exists()


def test_run_detects_syntax_error_and_never_executes(tmp_path):
    broken_sections = {
        **GOOD_SECTIONS,
        "evaluate_function": "def evaluate(experiment_output:\n    pass\n",
    }
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response(broken_sections)})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=experiments_dir,
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "syntax error" in exp["reason"].lower()
    assert not (experiments_dir / "H1" / "results.json").exists()


def test_run_reports_execution_failure(tmp_path):
    # run.py's fixed orchestration catches this, writes a "failure" results.json,
    # and exits 1 — sandbox.run_experiment sees the nonzero exit.
    failing_sections = {
        **GOOD_SECTIONS,
        "run_experiment_function": "def run_experiment(data, model):\n    raise RuntimeError('boom')\n",
    }
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response(failing_sections)})
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=tmp_path / "experiments",
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "execution failed" in exp["reason"].lower()


def test_run_reports_missing_required_code_section(tmp_path):
    incomplete_sections = {
        **GOOD_SECTIONS,
        "evaluate_function": "",
    }  # model omitted a required section
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response(incomplete_sections)})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=experiments_dir,
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "evaluate_function" in exp["reason"]
    assert not (experiments_dir / "H1" / "run.py").exists()  # never written — incomplete response


def test_run_skips_execution_when_gpu_needed_but_unavailable(tmp_path):
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response(needs_gpu=True)})
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=experiments_dir,
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "gpu" in exp["reason"].lower()
    assert (experiments_dir / "H1" / "run.sbatch").exists()


def test_run_missing_package_without_network_skips_execution(tmp_path):
    fake_model = FakeChatModel(
        {
            '"hypothesis_id":"H1"': _codegen_response(
                requirements="definitely_not_a_real_package_xyz\n"
            ),
        }
    )
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=tmp_path / "experiments",
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "definitely_not_a_real_package_xyz" in exp["reason"]
    assert "network" in exp["reason"]


def test_run_sets_up_shared_infrastructure_exactly_once(tmp_path):
    fake_model = FakeChatModel(
        {
            "Shared infrastructure items": render_sections(
                {"data_utils.py": "def load():\n    pass\n", "README.md": "shared"}
            ),
            '"hypothesis_id":"H1"': _codegen_response(),
            '"hypothesis_id":"H2"': _codegen_response(),
        }
    )
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=experiments_dir,
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
    )
    result = agent.run(
        _planner_output([_plan("H1"), _plan("H2")], shared_infrastructure=["shared eval harness"])
    )

    shared_calls = [c for c in fake_model.calls if "Shared infrastructure items" in c[-1][1]]
    assert len(shared_calls) == 1
    assert (experiments_dir / "_shared" / "data_utils.py").exists()
    assert (experiments_dir / "_shared" / "__init__.py").exists()
    assert (experiments_dir / "__init__.py").exists()
    assert result["shared_infrastructure_path"] == str(experiments_dir / "_shared")
    assert all(e["status"] == "completed" for e in result["experiments"])


def test_run_respects_priority_order(tmp_path):
    fake_model = FakeChatModel(
        {
            '"hypothesis_id":"H1"': _codegen_response(),
            '"hypothesis_id":"H2"': _codegen_response(),
        }
    )
    agent = CoderAgent(
        chat_model=fake_model,
        experiments_dir=tmp_path / "experiments",
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
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
        chat_model=FakeChatModel({}),
        experiments_dir=tmp_path / "experiments",
        output_dir=tmp_path / "outputs",
        network_check=lambda: False,
        gpu_check=lambda: False,
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
    (tmp_path / "results.json").write_text(
        json.dumps({"error": "Traceback...\nValueError: bad shape"})
    )
    results, diagnosis = sandbox.read_results_json_for_diagnosis(tmp_path)
    assert results is None
    assert "ValueError: bad shape" in diagnosis


def test_read_results_json_accepts_well_formed_file(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
    )
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
        # Per-call invoke kwargs, per kind — how a test checks that a fix
        # regeneration carried temperature=0.0 and an initial generation didn't.
        self.kwargs_by_kind: dict[str, list[dict]] = {}

    def _kind(self, prompt: str) -> str:
        for marker, kind in self.KINDS.items():
            if marker in prompt:
                return kind
        return "codegen"

    def invoke(self, messages, **kwargs):
        prompt = messages[-1][1]
        kind = self._kind(prompt)
        self.kwargs_by_kind.setdefault(kind, []).append(kwargs)
        index = self.calls_by_kind.get(kind, 0)
        self.calls_by_kind[kind] = index + 1
        responses = self._responses.get(kind)
        if not responses:
            raise AssertionError(f"No {kind!r} response configured for prompt: {prompt[:200]!r}")
        return SimpleNamespace(content=responses[min(index, len(responses) - 1)])


BROKEN_SYNTAX_SECTIONS = {
    **GOOD_SECTIONS,
    "evaluate_function": "def evaluate(experiment_output:\n    pass\n",
}
RAISING_SECTIONS = {
    **GOOD_SECTIONS,
    "run_experiment_function": "def run_experiment(data, model):\n    raise RuntimeError('boom')\n",
}


def _agent(tmp_path, model, **kwargs):
    # network/GPU default to absent (no test may touch either), but both are
    # overridable so the Hugging Face lookup tests below can turn the network on.
    kwargs.setdefault("network_check", lambda: False)
    kwargs.setdefault("gpu_check", lambda: False)
    return CoderAgent(
        chat_model=model,
        experiments_dir=tmp_path / "experiments",
        output_dir=tmp_path / "outputs",
        **kwargs,
    )


class RecordingScriptedChatModel(ScriptedChatModel):
    """ScriptedChatModel, but also keeps the full prompt text per kind so a
    test can assert on what a later call was actually shown."""

    def __init__(self, **responses_by_kind: list[str]):
        super().__init__(**responses_by_kind)
        self.prompts_by_kind: dict[str, list[str]] = {}

    def invoke(self, messages, **kwargs):
        prompt = messages[-1][1]
        self.prompts_by_kind.setdefault(self._kind(prompt), []).append(prompt)
        return super().invoke(messages, **kwargs)


def test_compact_json_has_no_indentation_whitespace():
    compact = _compact_json({"a": 1, "b": [1, 2, {"c": 3}]})
    assert compact == '{"a":1,"b":[1,2,{"c":3}]}'
    assert "\n" not in compact
    assert "  " not in compact


def test_codegen_prompt_sent_to_model_uses_compact_json(tmp_path):
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    _agent(tmp_path, model).run(_planner_output([_plan("H1")]))

    prompt = model.prompts_by_kind["codegen"][0]
    # A pretty-printed plan_block would contain '"hypothesis_id": "H1"' (space
    # after the colon) and newline-indented structure; compact JSON has neither.
    assert '"hypothesis_id":"H1"' in prompt
    assert '"hypothesis_id": "H1"' not in prompt


# One delimited section per generated file, named after the file — the
# shared-infra call is the one place the section names aren't known in advance.
BROKEN_SHARED_FILES = {"utils.py": "def load(:\n    pass\n"}
GOOD_SHARED_FILES = {"utils.py": "def load():\n    return 1\n"}


def test_shared_infra_fix_loop_recovers_after_a_broken_first_generation(tmp_path):
    model = ScriptedChatModel(
        shared_infra=[render_sections(BROKEN_SHARED_FILES), render_sections(GOOD_SHARED_FILES)],
        codegen=[_codegen_response()],
    )
    experiments_dir = tmp_path / "experiments"
    agent = _agent(tmp_path, model)
    result = agent.run(
        _planner_output([_plan("H1")], shared_infrastructure=["shared eval harness"])
    )

    assert model.calls_by_kind["shared_infra"] == 2
    written = (experiments_dir / "_shared" / "utils.py").read_text()
    compile(written, "utils.py", "exec")  # the repaired/regenerated version was kept
    assert result["experiments"][0]["status"] == "completed"


def test_shared_infra_still_broken_after_max_attempts_warns_downstream_experiments(tmp_path):
    model = RecordingScriptedChatModel(
        shared_infra=[
            render_sections(BROKEN_SHARED_FILES)
        ],  # never recovers — same response every retry
        codegen=[_codegen_response()],
    )
    agent = _agent(tmp_path, model, max_fix_attempts=2)
    agent.run(_planner_output([_plan("H1")], shared_infrastructure=["shared eval harness"]))

    # initial generation + max_fix_attempts regenerations, never more
    assert model.calls_by_kind["shared_infra"] == 3
    codegen_prompt = model.prompts_by_kind["codegen"][0]
    assert "WARNING: shared infrastructure still fails a check" in codegen_prompt


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
    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(RAISING_SECTIONS)]
    )
    result = _agent(tmp_path, model, max_fix_attempts=2).run(
        _planner_output([_plan("H1", complexity="low")])
    )

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
    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    snapshot = Path(result["experiments"][0]["fix_history"][0]["code_path"])
    assert snapshot.exists()
    assert "raise RuntimeError('boom')" in snapshot.read_text()  # the failing version, preserved
    assert (
        "raise RuntimeError('boom')" not in (tmp_path / "experiments" / "H1" / "run.py").read_text()
    )


def test_fix_loop_truncates_a_long_error_summary(tmp_path):
    noisy = {
        **GOOD_SECTIONS,
        "run_experiment_function": "def run_experiment(data, model):\n    raise RuntimeError('x' * 5000)\n",
    }
    model = ScriptedChatModel(
        codegen=[_codegen_response(noisy)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    summary = result["experiments"][0]["fix_history"][0]["error_summary"]
    assert 0 < len(summary) <= ERROR_SUMMARY_MAX_CHARS


def test_env_error_is_not_retried_through_the_llm(tmp_path):
    # A missing package is an environment problem — regenerating the code can't
    # fix it, so it must not burn fix attempts.
    model = ScriptedChatModel(
        codegen=[_codegen_response(requirements="definitely_not_a_real_package_xyz\n")]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert exp["fix_attempts"] == 0
    assert "fix" not in model.calls_by_kind


def test_unparseable_format_from_initial_generation_routes_through_the_fix_loop_instead_of_crashing(
    tmp_path,
):
    # Regression test: a codegen response the transport can't parse (surviving
    # invoke_sections' own repair retry) used to raise CoderAgentError straight
    # out of process_current_plan and crash the whole multi-plan run. It must
    # instead be treated like any other per-plan failure the fix loop can
    # regenerate against.
    model = ScriptedChatModel(
        codegen=[
            _unparseable_sections_response()
        ],  # same broken text feeds the internal repair retry too
        fix=[_codegen_response(GOOD_SECTIONS)],
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["fix_attempts"] == 1
    assert exp["fix_history"][0]["error_source"] == "invalid_format"
    assert exp["fix_history"][0]["resolved"] is True
    assert model.calls_by_kind["fix"] == 1


def test_unparseable_format_from_regeneration_still_counts_against_the_fix_budget(tmp_path):
    # The regeneration call (_regenerate_with_fix) goes through the same
    # _call_sections path and can fail the same way — it must be caught too, not
    # just the initial generation call. invoke_sections' internal repair retry
    # sends a fresh prompt with no "fix" marker text, so ScriptedChatModel
    # routes that retry to the "codegen" bucket — a second, broken "codegen"
    # entry keeps both the fix call and its repair retry failing.
    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS), _unparseable_sections_response()],
        fix=[_unparseable_sections_response()],
    )
    result = _agent(tmp_path, model, max_fix_attempts=2).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert exp["fix_attempts"] == 2
    assert exp["fix_history"][0]["error_source"] == "run_experiment"
    assert exp["fix_history"][1]["error_source"] == "invalid_format"


# -- coder_agent.py: the delimited code transport ----------------------------------------


def test_generated_code_with_regex_escapes_survives_the_transport_verbatim(tmp_path):
    # The whole point of the delimited transport. As a JSON string value this
    # source has to arrive as "r\"\\\\d+\"" — four backslashes to mean one — and
    # a small quantized model reliably writes two, producing either a JSON parse
    # error or a run.py with a corrupted pattern. Between delimiters there is no
    # encoding step at all, so what the model writes is what lands on disk.
    helpers = 'import re\nTOKEN_RE = re.compile(r"\\d+\\s*\\w+")\nSEP = "a\\tb"\n'
    model = ScriptedChatModel(codegen=[_codegen_response({**GOOD_SECTIONS, "helpers": helpers})])
    experiments_dir = tmp_path / "experiments"
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    assert result["experiments"][0]["status"] == "completed"
    written = (experiments_dir / "H1" / "run.py").read_text()
    assert 'TOKEN_RE = re.compile(r"\\d+\\s*\\w+")' in written  # single backslashes, unchanged
    assert "\\\\d" not in written  # never doubled on the way through


def test_assumptions_and_needs_gpu_are_parsed_out_of_raw_section_text(tmp_path):
    # needs_gpu/assumptions_made arrive as text now, so Python decides what that
    # text means: a dash-prefixed list becomes a real list, and a "true" with
    # trailing prose after it still reads as True.
    model = ScriptedChatModel(
        codegen=[
            render_sections(
                {
                    **GOOD_SECTIONS,
                    "readme": "# readme\n",
                    "requirements_txt": "",
                    "assumptions_made": "- used a synthetic sample\n\n- capped epochs at 5\n",
                    "needs_network": "false",
                    "needs_gpu": "true — the plan's model needs one",
                }
            )
        ]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["assumptions_made"] == ["used a synthetic sample", "capped epochs at 5"]
    # needs_gpu read as True, and no GPU here — so it was deferred, never run.
    assert exp["status"] == "code_generated_not_run"
    assert "gpu" in exp["reason"].lower()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("true", True),
        ("  TRUE  ", True),
        ("true — this needs a GPU", True),
        ("yes", True),
        ("false", False),
        ("false, CPU only", False),
        ("", False),
        ("unknown", False),  # unrecognisable defaults to "attempt it locally"
    ],
)
def test_parse_bool_text(text, expected):
    assert _parse_bool_text(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("- a\n- b", ["a", "b"]),
        ("a\nb", ["a", "b"]),  # dashes are optional
        ("* a\n", ["a"]),
        ("", []),
        ("- none", []),  # a "nothing to report" placeholder isn't an assumption
        ("None.\n", []),
        ("- kept the default seed\n\n\n", ["kept the default seed"]),
    ],
)
def test_parse_assumptions(text, expected):
    assert _parse_assumptions(text) == expected


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
    code = "\n".join(
        GOOD_SECTIONS[k]
        for k in ("load_data_function", "build_model_function", "evaluate_function")
    )
    assert sandbox.static_safety_check(code) == []


# -- sandbox.check_data_fallback ----------------------------------------------------------


GUARDED_LOAD_DATA = (
    "def load_data():\n"
    "    try:\n"
    "        return pd.read_csv(CSV_PATH)\n"
    "    except Exception as exc:\n"
    "        logger.warning('falling back to synthetic data: %s', exc)\n"
    "        return _synthesize()\n"
)
BARE_LOAD_DATA = "def load_data():\n    return pd.read_csv('survey_data.csv')\n"


def test_check_data_fallback_passes_a_guarded_read():
    assert sandbox.check_data_fallback(GUARDED_LOAD_DATA) == []


def test_check_data_fallback_flags_a_bare_read():
    # The real 2026-08 failure: a plan requiring *new* data collection produced
    # code that simply assumed survey_data.csv would be sitting there.
    findings = sandbox.check_data_fallback(BARE_LOAD_DATA)
    assert len(findings) == 1
    assert "read_csv" in findings[0]
    assert "synthesized stand-in" in findings[0]


@pytest.mark.parametrize(
    "body",
    [
        "    with open('data.txt') as handle:\n        return handle.read()\n",
        "    return np.load('embeddings.npy')\n",
        "    return numpy.loadtxt(DATA_PATH)\n",
        "    return pd.read_parquet(PATH)\n",
        "    return pd.read_json('rows.json')\n",
    ],
)
def test_check_data_fallback_flags_every_bare_read_flavour(body):
    assert sandbox.check_data_fallback(f"def load_data():\n{body}") != []


def test_check_data_fallback_does_not_treat_an_except_branch_as_a_guard():
    # The fallback path needs its own guard: if *it* assumes a file exists, the
    # experiment still dies on a FileNotFoundError.
    source = (
        "def load_data():\n"
        "    try:\n"
        "        return pd.read_csv(PRIMARY)\n"
        "    except FileNotFoundError:\n"
        "        return pd.read_csv(BACKUP)\n"
    )
    findings = sandbox.check_data_fallback(source)
    assert len(findings) == 1
    assert "line 5" in findings[0]  # the backup read, not the guarded primary


def test_check_data_fallback_passes_a_dataset_viewer_fetch():
    # The sanctioned remote path from the codegen prompt: parsing the response
    # body with read_csv is normal there, and there is no local file to miss.
    source = (
        "def load_data():\n"
        "    url = 'https://datasets-server.huggingface.co/rows?dataset=acme%2Fsleep'\n"
        "    response = requests.get(url, timeout=30)\n"
        "    return pd.read_json(response.text)\n"
    )
    assert sandbox.check_data_fallback(source) == []


def test_check_data_fallback_exempts_a_function_that_fetches_from_the_dataset_viewer():
    # Documented rule: a load_data on the sanctioned remote path is exempt as a
    # whole, so caching the response to disk and reading it back doesn't fail the
    # check even though that read takes a plain path.
    source = (
        "def load_data():\n"
        "    response = requests.get('https://datasets-server.huggingface.co/rows?dataset=x')\n"
        "    Path(CACHE_PATH).write_text(response.text)\n"
        "    return pd.read_csv(CACHE_PATH)\n"
    )
    assert sandbox.check_data_fallback(source) == []


def test_check_data_fallback_passes_a_read_of_a_fetched_response_body():
    source = "def load_data():\n    response = _fetch()\n    return pd.read_json(response.text)\n"
    assert sandbox.check_data_fallback(source) == []


def test_check_data_fallback_passes_a_read_of_an_in_memory_buffer():
    source = (
        "def load_data():\n    body = _fetch_rows()\n    return pd.read_csv(io.StringIO(body))\n"
    )
    assert sandbox.check_data_fallback(source) == []


def test_check_data_fallback_passes_a_purely_synthetic_load_data():
    assert sandbox.check_data_fallback(GOOD_SECTIONS["load_data_function"]) == []


def test_check_data_fallback_ignores_unparseable_source():
    # The compile check owns syntax errors (with real line numbers); this check
    # must not double-report them.
    assert sandbox.check_data_fallback("def load_data(:\n    pass\n") == []


def test_missing_data_fallback_routes_through_the_fix_loop(tmp_path):
    # End to end: an unguarded first generation is never executed, the concrete
    # finding goes back to the model, and a guarded second generation runs.
    unguarded = {**GOOD_SECTIONS, "load_data_function": BARE_LOAD_DATA}
    guarded = {**GOOD_SECTIONS, "load_data_function": "def load_data():\n    return None\n"}
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(unguarded)], fix=[_codegen_response(guarded)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["fix_attempts"] == 1
    assert exp["fix_history"][0]["error_source"] == "missing_data_fallback"
    assert exp["fix_history"][0]["resolved"] is True
    assert "read_csv" in exp["fix_history"][0]["error_summary"]
    # The model was told what to fix, in the fix prompt's error slot.
    assert "missing_data_fallback" in model.prompts_by_kind["fix"][0]
    # The unguarded version was preserved but never executed — the guarded one is
    # what ended up on disk and ran.
    snapshot = Path(exp["fix_history"][0]["code_path"])
    assert "survey_data.csv" in snapshot.read_text()
    assert "survey_data.csv" not in (tmp_path / "experiments" / "H1" / "run.py").read_text()


def test_missing_data_fallback_gives_up_without_ever_executing(tmp_path):
    unguarded = {**GOOD_SECTIONS, "load_data_function": BARE_LOAD_DATA}
    model = ScriptedChatModel(
        codegen=[_codegen_response(unguarded)], fix=[_codegen_response(unguarded)]
    )
    result = _agent(tmp_path, model, max_fix_attempts=1).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "assumes its data will be present" in exp["reason"]
    assert not (tmp_path / "experiments" / "H1" / "results.json").exists()


def test_lint_failure_routes_through_the_fix_loop(tmp_path):
    unsafe = {
        **GOOD_SECTIONS,
        "helpers": "def cleanup(p):\n    import shutil\n    shutil.rmtree(p)\n",
    }
    model = ScriptedChatModel(
        codegen=[_codegen_response(unsafe)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["fix_history"][0]["error_source"] == "static_lint"
    assert "rmtree" in exp["fix_history"][0]["error_summary"]


# -- sandbox.check_required_function_names ------------------------------------------------


def test_check_required_function_names_passes_correctly_named_sections():
    assert sandbox.check_required_function_names(GOOD_SECTIONS) == []


def test_check_required_function_names_flags_a_wrongly_named_function():
    # The defect this catches: compiles, is safe, guards its reads — and then
    # dies on a NameError only after a full venv provision, the most expensive
    # step in the loop.
    sections = {
        **GOOD_SECTIONS,
        "load_data_function": "def load_the_dataset():\n    return None\n",
    }
    findings = sandbox.check_required_function_names(sections)
    assert len(findings) == 1
    assert "load_data_function" in findings[0]
    assert "load_data" in findings[0]
    assert "load_the_dataset" in findings[0]


def test_check_required_function_names_flags_a_section_with_no_function_at_all():
    sections = {**GOOD_SECTIONS, "evaluate_function": "RESULT = {}\n"}
    findings = sandbox.check_required_function_names(sections)
    assert len(findings) == 1
    assert "no top-level function" in findings[0]


def test_check_required_function_names_ignores_a_nested_definition():
    # Nested in a class, so not callable as the bare global name run.py's
    # orchestration uses — which is why the check reads tree.body, not ast.walk.
    sections = {
        **GOOD_SECTIONS,
        "build_model_function": "class Factory:\n    def build_model(self):\n        return None\n",
    }
    findings = sandbox.check_required_function_names(sections)
    assert len(findings) == 1
    assert "build_model_function" in findings[0]


def test_check_required_function_names_skips_an_unparseable_section():
    # compile_check owns syntax errors (with real line numbers, on the whole
    # rendered run.py); this check must not double-report them.
    sections = {**GOOD_SECTIONS, "run_experiment_function": "def run_experiment(:\n    pass\n"}
    assert sandbox.check_required_function_names(sections) == []


def test_wrong_function_name_routes_through_the_fix_loop_without_provisioning(tmp_path):
    # End to end: caught before render/compile/venv, fed back to the model, and
    # recorded under an error_source the output schema accepts.
    wrong = {**GOOD_SECTIONS, "load_data_function": "def load_the_dataset():\n    return None\n"}
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(wrong)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["fix_history"][0]["error_source"] == "missing_required_function"
    assert exp["fix_history"][0]["resolved"] is True
    assert "missing_required_function" in model.prompts_by_kind["fix"][0]


# -- coder_agent.py: bounding max_tokens against the context window ----------------------


def _bounding_agent(tmp_path):
    return _agent(tmp_path, FakeChatModel({}))  # never invoked by these tests


def test_estimate_tokens_uses_the_documented_char_ratio():
    assert _estimate_tokens("x" * 4000) == 1000


def test_bounded_max_tokens_returns_the_configured_max_when_the_prompt_is_small(
    tmp_path, monkeypatch
):
    _patch_settings(monkeypatch, llm_context_window=32768, llm_max_tokens=8192)
    assert _bounding_agent(tmp_path)._bounded_max_tokens("write me an experiment") == 8192


def test_bounded_max_tokens_shrinks_to_the_remaining_headroom(tmp_path, monkeypatch):
    # The 2026-08-11 case: a fix prompt large enough that prompt + a fixed
    # max_tokens overruns the window, which the server answers with a 400 rather
    # than a completion.
    _patch_settings(monkeypatch, llm_context_window=8192, llm_max_tokens=8192)
    prompt = "x" * 4000
    expected = 8192 - (_estimate_tokens(prompts.SYSTEM_PROMPT) + _estimate_tokens(prompt)) - 512
    assert expected < 8192  # the headroom is what binds here, not llm_max_tokens
    assert _bounding_agent(tmp_path)._bounded_max_tokens(prompt) == expected


def test_bounded_max_tokens_raises_an_informative_error_when_the_prompt_is_too_large(
    tmp_path, monkeypatch
):
    _patch_settings(monkeypatch, llm_context_window=4096, llm_max_tokens=8192)
    prompt = "x" * 20000
    with pytest.raises(CoderAgentError) as excinfo:
        _bounding_agent(tmp_path)._bounded_max_tokens(prompt)

    message = str(excinfo.value)
    # Debuggable from the log line alone: prompt size, headroom, floor, window.
    prompt_tokens = _estimate_tokens(prompts.SYSTEM_PROMPT) + _estimate_tokens(prompt)
    assert str(prompt_tokens) in message
    assert str(4096 - prompt_tokens - 512) in message  # the (negative) headroom
    assert "2048" in message  # the minimum a usable completion needs
    assert "4096" in message  # the configured context window


def test_generation_calls_carry_the_bounded_max_tokens(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, llm_context_window=32768, llm_max_tokens=8192)
    model = ScriptedChatModel(codegen=[_codegen_response()])
    _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    assert model.kwargs_by_kind["codegen"][0]["max_tokens"] == 8192


def test_an_oversized_prompt_fails_one_plan_instead_of_crashing_the_run(tmp_path, monkeypatch):
    # _bounded_max_tokens raises CoderAgentError, which the generation nodes
    # already convert into a normal fix-loop outcome — no separate bypass path.
    _patch_settings(monkeypatch, llm_context_window=1024)
    model = ScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, max_fix_attempts=1).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert exp["fix_history"][0]["error_source"] == "invalid_format"
    assert "context window" in exp["fix_history"][0]["error_summary"]


# -- coder_agent.py: fix-attempt regeneration temperature --------------------------------


def test_fix_regeneration_runs_at_temperature_zero_and_initial_generation_does_not(tmp_path):
    # Full-section regeneration at a nonzero temperature has been observed
    # introducing a *different* bug each attempt; the fix turn wants the model's
    # most confident completion, not a fresh sample.
    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    assert model.kwargs_by_kind["fix"][0]["temperature"] == 0.0
    # Initial generation keeps the constructor's temperature — no override sent.
    assert model.kwargs_by_kind["codegen"][0].get("temperature") is None


def test_shared_infra_repair_runs_at_temperature_zero(tmp_path):
    model = ScriptedChatModel(
        shared_infra=[render_sections(BROKEN_SHARED_FILES), render_sections(GOOD_SHARED_FILES)],
        codegen=[_codegen_response()],
    )
    _agent(tmp_path, model).run(
        _planner_output([_plan("H1")], shared_infrastructure=["shared eval harness"])
    )

    first, repair = model.kwargs_by_kind["shared_infra"]
    assert first.get("temperature") is None
    assert repair["temperature"] == 0.0


# -- coder_agent.py: configurable low/medium execution timeouts --------------------------


def _spy_on_run_experiment(monkeypatch):
    """Records the timeout each execution was given, still running it for real
    so the plan reaches a normal terminal result."""
    real_run_experiment = sandbox.run_experiment
    timeouts = []

    def spy(python_executable, run_script, cwd, timeout_seconds):
        timeouts.append(timeout_seconds)
        return real_run_experiment(python_executable, run_script, cwd, timeout_seconds)

    monkeypatch.setattr(sandbox, "run_experiment", spy)
    return timeouts


def test_low_complexity_timeout_comes_from_settings(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_low_complexity_timeout_seconds=77)
    timeouts = _spy_on_run_experiment(monkeypatch)
    model = ScriptedChatModel(codegen=[_codegen_response()])
    _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    assert timeouts == [77]


def test_medium_complexity_timeout_comes_from_settings(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_medium_complexity_timeout_seconds=88)
    timeouts = _spy_on_run_experiment(monkeypatch)
    model = ScriptedChatModel(codegen=[_codegen_response()])
    _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="medium")]))

    assert timeouts == [88]


def test_low_and_medium_timeouts_default_to_the_previously_hardcoded_values():
    from research_pipeline.config import settings as real_settings

    assert real_settings.coder_low_complexity_timeout_seconds == 120
    assert real_settings.coder_medium_complexity_timeout_seconds == 300


# -- coder_agent.py: the Hugging Face dataset lookup -------------------------------------
# No test here touches the network: the lookup is a single injected function
# (huggingface_lookup_fn), exactly like network_check/gpu_check.


HF_DATASET_MATCH = {
    "dataset_id": "acme/sleep-survey",
    "config": "default",
    "split": "train",
    "columns": [{"name": "hours_slept", "type": "float32"}, {"name": "score", "type": "int64"}],
    "sample_rows": [{"hours_slept": 7.5, "score": 88}],
}


def _recording_lookup(result):
    """A fake huggingface_lookup_fn that records the queries it was asked."""
    queries: list[str] = []

    def lookup(query):
        queries.append(query)
        return result

    return lookup, queries


def test_matched_hf_dataset_is_offered_to_the_model_with_a_rest_url(tmp_path):
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    lookup, queries = _recording_lookup(HF_DATASET_MATCH)
    _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    # Queried with the plan's own data description, not the objective.
    assert queries == ["d"]
    prompt = model.prompts_by_kind["codegen"][0]
    assert "acme/sleep-survey" in prompt
    assert "hours_slept (float32)" in prompt  # real column names and dtypes
    assert '"hours_slept":7.5' in prompt  # real sample rows
    # The exact REST endpoint, url-encoded, so no `datasets` package is needed.
    assert "datasets-server.huggingface.co/rows?dataset=acme%2Fsleep-survey" in prompt
    # And the instruction that keeps Phase C's check satisfiable.
    assert "fall back to a small synthesized stand-in dataset" in prompt


def test_no_dataset_block_when_nothing_matched(tmp_path):
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    lookup, queries = _recording_lookup(None)
    result = _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    assert queries == ["d"]
    assert result["experiments"][0]["status"] == "completed"
    # The prompt reads exactly as it did before this lookup existed.
    assert "Dataset Viewer" not in model.prompts_by_kind["codegen"][0]


def test_a_raising_lookup_never_blocks_code_generation(tmp_path):
    # The lookup is an enhancement to a prompt, never a dependency: an injected
    # function that raises must not cost the experiment its code.
    def boom(query):
        raise RuntimeError("hub is down")

    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=boom).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert "Dataset Viewer" not in model.prompts_by_kind["codegen"][0]


def test_lookup_is_skipped_without_network(tmp_path):
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    lookup, queries = _recording_lookup(HF_DATASET_MATCH)
    _agent(tmp_path, model, huggingface_lookup_fn=lookup).run(_planner_output([_plan("H1")]))

    assert queries == []  # never attempted — the runtime probe said no network


def test_lookup_is_skipped_when_the_setting_is_off(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_enable_hf_dataset_search=False)
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    lookup, queries = _recording_lookup(HF_DATASET_MATCH)
    _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    assert queries == []
    assert "Dataset Viewer" not in model.prompts_by_kind["codegen"][0]


def test_infeasible_plan_never_reaches_the_lookup(tmp_path):
    lookup, queries = _recording_lookup(HF_DATASET_MATCH)
    _agent(
        tmp_path, FakeChatModel({}), network_check=lambda: True, huggingface_lookup_fn=lookup
    ).run(_planner_output([_plan("H1", feasible=False)]))

    assert queries == []  # skipped before the lookup node, like the LLM call


def test_the_fix_prompt_reuses_the_dataset_found_once_for_the_plan(tmp_path):
    # The dataset is usually exactly what a data-loading failure needs to be
    # fixed *with* — and looking it up once per plan (not once per attempt) keeps
    # a three-attempt fix loop from spending three more searches on it.
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    lookup, queries = _recording_lookup(HF_DATASET_MATCH)
    result = _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert len(queries) == 1
    assert "acme/sleep-survey" in model.prompts_by_kind["fix"][0]


def test_each_plan_gets_its_own_lookup(tmp_path):
    model = ScriptedChatModel(codegen=[_codegen_response()])
    lookup, queries = _recording_lookup(HF_DATASET_MATCH)
    _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1"), _plan("H2")])
    )

    assert len(queries) == 2


# -- huggingface_client.py: against faked requests responses -----------------------------


class _FakeResponse:
    def __init__(self, payload, status_code=200, valid_json=True):
        self._payload = payload
        self._valid_json = valid_json
        self.status_code = status_code
        self.text = "response body"

    def json(self):
        if not self._valid_json:
            raise ValueError("not JSON")
        return self._payload


def _fake_hf(monkeypatch, routes, recorder=None):
    """Routes huggingface_client's requests.get by URL substring. A route value
    that is an exception instance is raised instead of returned."""

    def fake_get(url, params=None, headers=None, timeout=None):
        if recorder is not None:
            recorder.append((url, dict(params or {})))
        for marker, response in routes.items():
            if marker in url:
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(huggingface_client.requests, "get", fake_get)


def _viewer_routes(search_hits):
    return {
        "api/datasets": _FakeResponse(search_hits),
        "is-valid": _FakeResponse({"viewer": True, "preview": True}),
        "splits": _FakeResponse(
            {
                "splits": [
                    {"dataset": "acme/sleep", "config": "default", "split": "test"},
                    {"dataset": "acme/sleep", "config": "default", "split": "train"},
                ]
            }
        ),
        "first-rows": _FakeResponse(
            {
                "features": [
                    {"name": "hours_slept", "type": {"dtype": "float32", "_type": "Value"}},
                    {"name": "label", "type": {"_type": "ClassLabel"}},
                ],
                "rows": [
                    {"row_idx": 0, "row": {"hours_slept": 7.5, "label": 1}},
                    {"row_idx": 1, "row": {"hours_slept": 6.0, "label": 0}},
                ],
            }
        ),
    }


def test_search_datasets_returns_hits(monkeypatch):
    _fake_hf(monkeypatch, {"api/datasets": _FakeResponse([{"id": "acme/sleep"}, {"id": "b/c"}])})
    assert huggingface_client.search_datasets("sleep", limit=2) == [
        {"id": "acme/sleep"},
        {"id": "b/c"},
    ]


def test_search_datasets_degrades_on_a_transport_error(monkeypatch):
    _fake_hf(
        monkeypatch,
        {"api/datasets": huggingface_client.requests.RequestException("connection reset")},
    )
    assert huggingface_client.search_datasets("sleep") == []


def test_search_datasets_degrades_on_a_bad_status(monkeypatch):
    _fake_hf(monkeypatch, {"api/datasets": _FakeResponse(None, status_code=429)})
    assert huggingface_client.search_datasets("sleep") == []


def test_search_datasets_degrades_on_a_non_json_body(monkeypatch):
    _fake_hf(monkeypatch, {"api/datasets": _FakeResponse(None, valid_json=False)})
    assert huggingface_client.search_datasets("sleep") == []


def test_search_datasets_skips_the_request_for_an_empty_query(monkeypatch):
    _fake_hf(monkeypatch, {})  # any request would fail the test
    assert huggingface_client.search_datasets("   ") == []


def test_find_dataset_for_experiment_describes_the_first_servable_match(monkeypatch):
    calls: list[tuple[str, dict]] = []
    _fake_hf(monkeypatch, _viewer_routes([{"id": "acme/sleep"}]), recorder=calls)

    found = huggingface_client.find_dataset_for_experiment(
        "A survey of 500 undergraduate students measuring sleep quality and exam scores"
    )

    assert found == {
        "dataset_id": "acme/sleep",
        "config": "default",
        "split": "train",  # preferred over the "test" split listed first
        "columns": [
            {"name": "hours_slept", "type": "float32"},
            {"name": "label", "type": "ClassLabel"},  # nested type falls back to _type
        ],
        "sample_rows": [{"hours_slept": 7.5, "label": 1}, {"hours_slept": 6.0, "label": 0}],
    }
    # A prose description is reduced to a keyword query — the Hub's `search`
    # matches dataset names, so the full sentence would match nothing.
    search_params = next(params for url, params in calls if "api/datasets" in url)
    assert search_params["search"] == "survey undergraduate students measuring"


def test_find_dataset_for_experiment_returns_none_when_the_viewer_cannot_serve_it(monkeypatch):
    routes = _viewer_routes([{"id": "acme/sleep"}])
    routes["is-valid"] = _FakeResponse({"viewer": False, "preview": False})
    _fake_hf(monkeypatch, routes)

    assert huggingface_client.find_dataset_for_experiment("sleep quality survey") is None


def test_find_dataset_for_experiment_returns_none_when_search_finds_nothing(monkeypatch):
    _fake_hf(monkeypatch, {"api/datasets": _FakeResponse([])})
    assert huggingface_client.find_dataset_for_experiment("sleep quality survey") is None


def test_find_dataset_for_experiment_skips_a_candidate_with_no_usable_splits(monkeypatch):
    routes = _viewer_routes([{"id": "acme/sleep"}])
    routes["splits"] = _FakeResponse({"splits": []})
    _fake_hf(monkeypatch, routes)

    assert huggingface_client.find_dataset_for_experiment("sleep quality survey") is None


def test_find_dataset_for_experiment_truncates_a_huge_cell(monkeypatch):
    routes = _viewer_routes([{"id": "acme/sleep"}])
    routes["first-rows"] = _FakeResponse(
        {
            "features": [{"name": "text", "type": {"dtype": "string"}}],
            "rows": [{"row_idx": 0, "row": {"text": "x" * 5000}}],
        }
    )
    _fake_hf(monkeypatch, routes)

    found = huggingface_client.find_dataset_for_experiment("sleep quality survey")
    assert found is not None
    cell = found["sample_rows"][0]["text"]
    assert len(cell) <= huggingface_client.MAX_CELL_CHARS + 1  # + the ellipsis


def test_find_dataset_for_experiment_never_raises_on_an_unexpected_payload(monkeypatch):
    # Every network/decode failure is absorbed by _get_json; this covers the
    # remaining class — a 200 whose body has the wrong shape entirely.
    _fake_hf(monkeypatch, {"api/datasets": _FakeResponse({"unexpected": "shape"})})
    assert huggingface_client.find_dataset_for_experiment("sleep quality survey") is None


# -- coder_agent.py: gated SLURM auto-submit ---------------------------------------------


def _patch_settings(monkeypatch, **overrides):
    """Settings is a frozen dataclass, so swap in a copy rather than mutating."""
    from research_pipeline.agents.coder import coder_agent as coder_agent_module

    monkeypatch.setattr(
        coder_agent_module,
        "settings",
        dataclasses.replace(coder_agent_module.settings, **overrides),
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
    unsafe = {
        **GOOD_SECTIONS,
        "helpers": "def wipe(p):\n    import os\n    os.system('rm -rf ' + p)\n",
    }
    model = ScriptedChatModel(
        codegen=[_codegen_response(unsafe)],
        fix=[_codegen_response(unsafe)],
        self_review=[_clean_review()],
    )
    result = _agent(tmp_path, model, max_fix_attempts=1).run(
        _planner_output([_plan("H1", complexity="high")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "os.system" in exp["reason"]
    assert auto_submit == []  # never reached sbatch


def test_auto_submit_fixes_code_the_self_review_flags(tmp_path, auto_submit):
    model = ScriptedChatModel(
        codegen=[_codegen_response()],
        self_review=[
            json.dumps(
                {"looks_correct": False, "concerns": ["load_data ignores the plan's dataset"]}
            ),
            _clean_review(),
        ],
        fix=[_codegen_response(GOOD_SECTIONS)],
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "submitted_to_slurm"
    assert exp["fix_history"][0]["error_source"] == "self_review"
    assert "ignores the plan's dataset" in exp["fix_history"][0]["error_summary"]


def test_auto_submit_respects_the_concurrent_job_cap(tmp_path, auto_submit, monkeypatch):
    monkeypatch.setattr(
        slurm_submit, "count_running_jobs", lambda user=None: 4
    )  # already at the cap
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
    monkeypatch.setattr(
        slurm_submit,
        "submit_job",
        lambda *a, **k: (None, "sbatch exited with code 1: invalid partition"),
    )
    model = ScriptedChatModel(codegen=[_codegen_response()], self_review=[_clean_review()])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "invalid partition" in exp["reason"]
    assert exp["slurm_job_id"] is None
