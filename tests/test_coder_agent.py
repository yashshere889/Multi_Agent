import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from langgraph.store.memory import InMemoryStore

from research_pipeline.agents.coder import (
    dataset_scoring,
    diagnose,
    fix_pattern_store,
    huggingface_client,
    prompts,
    provenance,
    repair,
    sandbox,
    schema,
    slurm_submit,
)
from research_pipeline.agents.coder.coder_agent import (
    _CHARS_PER_TOKEN_ESTIMATE,
    _CONTEXT_SAFETY_MARGIN,
    CoderAgent,
    CoderAgentError,
    _compact_json,
    _consecutive_error_streak,
    _default_slurm_review_prompt,
    _estimate_tokens,
    _identical_failure_streak,
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


def test_extract_third_party_imports_excludes_stdlib_and_experiments_package():
    source = (
        "import sys\n"
        "import json\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "from experiments._shared import data_utils\n"
        "import sklearn.linear_model\n"
    )
    assert sandbox.extract_third_party_imports(source) == {"pandas", "sklearn"}


def test_extract_third_party_imports_returns_empty_set_for_syntax_error():
    assert sandbox.extract_third_party_imports("def load(:\n    pass\n") == set()


def test_extract_third_party_imports_returns_empty_set_for_no_imports():
    assert sandbox.extract_third_party_imports("def load():\n    return 1\n") == set()


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


def test_a_usable_venv_survives_the_next_fix_attempt(tmp_path, monkeypatch):
    """Barkla job 10410771: ensure_experiment_env runs once per fix attempt and
    wiped the venv each time, throwing away every package the previous attempt's
    env repairs had discovered. Three attempts each started bare, each
    rediscovered what it needed one ModuleNotFoundError at a time, and each
    spent its whole CODER_MAX_ENV_REPAIRS budget doing it — numpy six times,
    then pandas six times, then numpy six times. The experiment never ran."""
    created: list[list[str]] = []

    def fake_run(command, **kwargs):
        if command[:2] == ["uv", "venv"]:
            created.append(list(command))
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True, exist_ok=True)
            (target / "bin" / "python").write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy\n")

    sandbox.ensure_experiment_env(tmp_path, requirements, network_available=True)
    venv = tmp_path / ".venv"
    # Stand in for what an env repair installed into the venv last attempt.
    repaired = venv / "lib" / "site-packages" / "pandas"
    repaired.mkdir(parents=True)

    sandbox.ensure_experiment_env(tmp_path, requirements, network_available=True)

    assert repaired.exists(), "the previous attempt's env repair was wiped"
    assert len(created) == 1, "a usable venv should be reused, not recreated"


def test_a_half_built_venv_is_still_wiped(tmp_path, monkeypatch):
    """The recovery the wipe was there for in the first place: a venv directory
    with no interpreter in it (creation succeeded, the install step died) must
    be rebuilt, or every later attempt fails on "already exists"."""
    created: list[list[str]] = []

    def fake_run(command, **kwargs):
        if command[:2] == ["uv", "venv"]:
            created.append(list(command))
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True, exist_ok=True)
            (target / "bin" / "python").write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy\n")

    # A directory, but no bin/python — exactly the partial state.
    (tmp_path / ".venv" / "lib").mkdir(parents=True)

    sandbox.ensure_experiment_env(tmp_path, requirements, network_available=True)

    assert len(created) == 1
    assert not (tmp_path / ".venv" / "lib").exists(), "the partial venv should have been rebuilt"


def test_module_importable_resolves_the_interpreter_like_run_experiment(tmp_path, monkeypatch):
    """The two must agree about which interpreter they are talking about.
    CODER_EXPERIMENTS_DIR defaults to a relative "experiments", so a venv python
    under it is a relative path and `cwd` is that same relative directory — an
    unresolved path gets re-resolved against the subprocess's own cwd, looking
    under experiments/H1/experiments/H1/. A repair that actually worked then
    reads as "installed but still not importable", which ends the attempt."""
    seen: list[str] = []

    def fake_run(command, **kwargs):
        seen.append(command[0])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    experiment_dir = Path("experiments") / "H1"
    experiment_dir.mkdir(parents=True)
    venv_python = experiment_dir / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")

    assert sandbox.module_importable(venv_python, "numpy", experiment_dir)

    # Absolute, so the subprocess's cwd cannot change what it points at.
    assert Path(seen[0]).is_absolute()
    assert "experiments/H1/experiments" not in seen[0]


def test_the_experiment_venv_is_built_with_the_requested_python(tmp_path, monkeypatch):
    """Barkla job 10334394 generated a valid TensorFlow experiment and could not
    provision it: the pipeline runs on 3.14 and uv reported "all versions of
    tensorflow have no wheels with a matching Python ABI tag (cp314) ... we only
    found cp310, cp311, cp312, cp313". Nothing was wrong with the code, and
    env-provisioning failures are terminal by design, so every torch/tensorflow
    experiment on that host died at the same gate."""
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[:2] == ["uv", "venv"]:
            # uv venv creates the interpreter the install step then targets.
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True, exist_ok=True)
            (target / "bin" / "python").write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("tensorflow\n")

    sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=True, python_version="3.12"
    )

    venv_call = next(call for call in calls if call[:2] == ["uv", "venv"])
    assert venv_call[2:4] == ["--python", "3.12"]


def test_no_requested_python_leaves_the_venv_call_as_it_was(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[:2] == ["uv", "venv"]:
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True, exist_ok=True)
            (target / "bin" / "python").write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("tensorflow\n")

    sandbox.ensure_experiment_env(tmp_path, requirements, network_available=True)

    venv_call = next(call for call in calls if call[:2] == ["uv", "venv"])
    assert "--python" not in venv_call


def test_ensure_experiment_env_returns_current_interpreter_when_nothing_missing(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("os\n")  # stdlib, never "missing"
    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=True
    )
    assert error is None
    assert python_exec == Path(sys.executable)


def test_ensure_experiment_env_ignores_extra_requirements_already_present(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("os\n")  # stdlib, never "missing"
    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path,
        requirements,
        network_available=True,
        extra_requirements=["sys"],  # also stdlib
    )
    assert error is None
    assert python_exec == Path(sys.executable)


def test_ensure_experiment_env_falls_back_to_venv_when_bare_interpreter_import_fails(
    tmp_path, monkeypatch
):
    """Regression test for job 10271093: find_spec() said the requirement is
    importable in this process (trivially true here, 'os' is stdlib), but a
    subprocess launched with the same interpreter genuinely fails to import
    it — reproducing the gap where find_spec's in-process answer (likely
    fooled on the real run by an HPC module-loaded site-packages path) didn't
    match what run_experiment's subprocess could actually do. 3 fix attempts
    on that run regenerated code against a failure that was never about the
    code. ensure_experiment_env must fall through to real venv provisioning
    rather than trusting the bare interpreter on faith."""
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("os\n")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    recorded_cmds = []

    def fake_run(cmd, **kwargs):
        recorded_cmds.append(cmd)
        if "-c" in cmd:
            return SimpleNamespace(
                returncode=1, stdout="", stderr="ImportError: no module named os"
            )
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=True
    )
    assert error is None
    assert python_exec == venv_python
    assert any("-c" in cmd for cmd in recorded_cmds)
    assert any(cmd[0] == "uv" and "venv" in cmd for cmd in recorded_cmds)


def test_ensure_experiment_env_reports_missing_when_bare_interpreter_import_fails_without_network(
    tmp_path, monkeypatch
):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("os\n")

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="ImportError")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path, requirements, network_available=False
    )
    assert python_exec is None
    assert "os" in error
    assert "network" in error


def test_ensure_experiment_env_provisions_for_extra_requirements_not_in_requirements_txt(
    tmp_path, monkeypatch
):
    # Reproduces job 10229968: requirements.txt is empty (the model never
    # declared it), but experiments/_shared/'s own import of a package makes
    # it genuinely missing — extra_requirements is how that gets surfaced.
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    recorded_cmds = []
    venv_python = tmp_path / ".venv" / "bin" / "python"

    def fake_run(cmd, **kwargs):
        recorded_cmds.append(cmd)
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    python_exec, error = sandbox.ensure_experiment_env(
        tmp_path,
        requirements,
        network_available=True,
        extra_requirements=["definitely_not_a_real_package_xyz"],
    )
    assert error is None
    assert python_exec == venv_python

    # The install command must point at a requirements file that actually
    # contains the extra requirement — installing from the original (empty)
    # requirements.txt would silently install nothing.
    install_cmd = recorded_cmds[-1]
    assert "pip" in install_cmd and "install" in install_cmd
    resolved_path = Path(install_cmd[install_cmd.index("-r") + 1])
    assert "definitely_not_a_real_package_xyz" in resolved_path.read_text()
    # The original requirements.txt on disk documents only what the model
    # itself declared, and is left untouched.
    assert requirements.read_text() == ""


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
    # The fixture plan declares data_requirements.source == "synthetic", so the
    # provenance gate withholds the verdict: the model's own claim is preserved
    # under model_reported_meets_success_criteria, and the Writer sees "unknown"
    # (which it maps to "inconclusive") rather than a bool it would publish as
    # supported or refuted. See provenance.apply_to_results.
    assert exp["results"]["meets_success_criteria"] == "unknown"
    assert exp["results"]["model_reported_meets_success_criteria"] is True
    assert exp["data_provenance"]["all_inputs_real"] is False
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
    # The fixture plan declares data_requirements.source == "synthetic", so the
    # provenance gate withholds the verdict: the model's own claim is preserved
    # under model_reported_meets_success_criteria, and the Writer sees "unknown"
    # (which it maps to "inconclusive") rather than a bool it would publish as
    # supported or refuted. See provenance.apply_to_results.
    assert exp["results"]["meets_success_criteria"] == "unknown"
    assert exp["results"]["model_reported_meets_success_criteria"] is True
    assert exp["data_provenance"]["all_inputs_real"] is False
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


# -- sandbox.check_results_plausibility ---------------------------------------------------


def test_check_results_plausibility_passes_a_real_looking_result():
    assert sandbox.check_results_plausibility({"accuracy": 0.87, "f1": 0.81}) == []


def test_check_results_plausibility_flags_empty_metrics():
    findings = sandbox.check_results_plausibility({})
    assert len(findings) == 1
    assert "no metrics at all" in findings[0]


def test_check_results_plausibility_flags_nan():
    findings = sandbox.check_results_plausibility({"accuracy": float("nan")})
    assert len(findings) == 1
    assert "accuracy" in findings[0]


def test_check_results_plausibility_flags_infinity():
    findings = sandbox.check_results_plausibility({"loss": float("inf")})
    assert len(findings) == 1
    assert "loss" in findings[0]


def test_check_results_plausibility_flags_all_zero_metrics():
    findings = sandbox.check_results_plausibility({"accuracy": 0, "f1": 0.0})
    assert len(findings) == 1
    assert "exactly 0" in findings[0]


def test_check_results_plausibility_passes_one_zero_among_real_metrics():
    # Only an *all*-zero metrics set is suspicious — a single legitimately
    # zero metric alongside real ones (e.g. a perfectly separable toy split)
    # is not.
    assert sandbox.check_results_plausibility({"accuracy": 0.9, "false_positives": 0}) == []


def test_check_results_plausibility_flags_placeholder_string():
    findings = sandbox.check_results_plausibility({"accuracy": "N/A"})
    assert len(findings) == 1
    assert "accuracy" in findings[0]


def test_check_results_plausibility_ignores_bool_metrics_for_the_zero_check():
    # bool is a subclass of int in Python — a genuine boolean flag must not
    # be treated as a numeric-zero finding.
    assert sandbox.check_results_plausibility({"converged": False}) == []


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
        "State what data this experiment actually needs": "dataset_spec",
        "Determine whether this dataset satisfies": "dataset_evidence",
        "Assume this dataset is bad": "dataset_critic",
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


def _codegen_calls(model):
    """Only the calls that asked for experiment code. Dataset selection makes
    its own model calls before generation, so a bare len(model.calls) no longer
    answers "how many times was the model asked for code?"."""
    return [call for call in model.calls if "Fill in the experiment template" in call[-1][1]]


def _agent(tmp_path, model, **kwargs):
    # network/GPU default to absent (no test may touch either), but both are
    # overridable so the Hugging Face lookup tests below can turn the network on.
    kwargs.setdefault("network_check", lambda: False)
    kwargs.setdefault("gpu_check", lambda: False)
    # A fresh, isolated in-memory store per call — never the real
    # fix_pattern_store.get_store() singleton (CODER_FIX_STORE_BACKEND
    # defaults to sqlite, which would write a real file). A test that wants
    # two agents to share recorded fix patterns passes its own fix_store=
    # explicitly, overriding this default.
    kwargs.setdefault("fix_store", InMemoryStore())
    # Every dataset seam defaults to inert here, not just in the tests that care.
    # `network_check` alone is not enough of a guard: a test that turns the
    # network on to exercise something unrelated would otherwise reach the real
    # Hugging Face API through whichever of these it left at its default.
    kwargs.setdefault("dataset_search_fn", lambda query, limit: [])
    kwargs.setdefault("dataset_describe_fn", lambda dataset_id: {})
    kwargs.setdefault("dataset_rows_fn", lambda dataset_id, config, split, limit: [])
    kwargs.setdefault("dataset_card_fn", lambda dataset_id: "")
    kwargs.setdefault("dataset_download_fn", lambda dataset_id, revision, dest: None)
    kwargs.setdefault("dataset_normalize_fn", lambda source, dest, max_rows: {})
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


MISSING_REQUIREMENT_SECTIONS = {
    **GOOD_SECTIONS,
    "imports": "import definitely_not_a_real_package_xyz\n",
}


def test_own_code_import_missing_from_requirements_is_caught_before_execution(tmp_path):
    """Regression test for a 2026-08-17 production run (job 10247173): the
    model's `imports` section imported a package it never listed in
    `requirements_txt`, and this plan had no shared_infrastructure at all — so
    the job-10229968 guard (which only ever inspected shared_files) never even
    ran. ensure_experiment_env saw nothing missing, handed back the bare
    interpreter, and execution failed with the same ModuleNotFoundError on all
    3 fix attempts, since nothing forced the regenerated requirements_txt to
    include the package either. coder_agent.py now also extracts run_py's own
    imports, so this should be caught deterministically before execution
    rather than only surfacing as a run_experiment failure the fix loop can't
    reliably resolve — no `fix` response is configured, so this test fails
    loudly (via ScriptedChatModel's AssertionError) if the old code path
    (execute first, discover the gap only via a crash) is ever reintroduced."""
    model = ScriptedChatModel(codegen=[_codegen_response(MISSING_REQUIREMENT_SECTIONS)])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "definitely_not_a_real_package_xyz" in exp["reason"]
    assert "no network access" in exp["reason"]
    assert exp["fix_attempts"] == 0  # caught up front, never needed a regeneration
    assert model.calls_by_kind["codegen"] == 1  # never had to run generated code at all


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


def test_consecutive_error_streak_counts_trailing_same_source_entries():
    assert _consecutive_error_streak([]) == 0
    assert _consecutive_error_streak([{"error_source": "run_experiment"}]) == 1
    assert (
        _consecutive_error_streak(
            [{"error_source": "run_experiment"}, {"error_source": "run_experiment"}]
        )
        == 2
    )
    # A different error_source anywhere in the trailing run stops the count —
    # only entries that stayed on the same failure category as the most recent
    # one contribute.
    assert (
        _consecutive_error_streak(
            [
                {"error_source": "compile_check"},
                {"error_source": "run_experiment"},
                {"error_source": "run_experiment"},
            ]
        )
        == 2
    )
    assert (
        _consecutive_error_streak(
            [{"error_source": "run_experiment"}, {"error_source": "compile_check"}]
        )
        == 1
    )


def test_stuck_block_is_empty_below_streak_two():
    assert CoderAgent._stuck_block(0, "") == ""
    assert CoderAgent._stuck_block(1, "") == ""


def test_stuck_block_quotes_the_previous_failure_from_streak_two():
    block = CoderAgent._stuck_block(2, "Execution failed: boom")
    assert "STUCK WARNING" in block
    assert "Execution failed: boom" in block
    assert "Do not resubmit" in block


def test_fix_loop_escalates_prompt_after_repeated_same_error(tmp_path):
    # RAISING_SECTIONS always fails the same way (RuntimeError in
    # run_experiment), so a fix that keeps returning it never clears the
    # "run_experiment" error_source — every regeneration after the first
    # should see a STUCK WARNING quoting the immediately preceding attempt's
    # own error_summary, so the model is told not to just repeat itself.
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(RAISING_SECTIONS)]
    )
    result = _agent(tmp_path, model, max_fix_attempts=3).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["fix_attempts"] == 3
    assert [entry["same_error_streak"] for entry in exp["fix_history"]] == [1, 2, 3]

    fix_prompts = model.prompts_by_kind["fix"]
    assert "STUCK WARNING" not in fix_prompts[0]  # first failure — nothing to escalate yet
    assert "STUCK WARNING" in fix_prompts[1]
    assert "STUCK WARNING" in fix_prompts[2]
    # The escalated prompt quotes the *previous* attempt's own summary, not the
    # current one — it's telling the model what it already tried and failed.
    assert exp["fix_history"][0]["error_summary"] in fix_prompts[1]
    assert exp["fix_history"][1]["error_summary"] in fix_prompts[2]


def test_fix_loop_does_not_escalate_when_the_error_category_changes(tmp_path):
    # First fix response has a different failure shape (a syntax error, not a
    # runtime one) before the second one repeats the original — the streak
    # must reset rather than keep counting across unrelated failure kinds.
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)],
        fix=[_codegen_response(BROKEN_SYNTAX_SECTIONS), _codegen_response(RAISING_SECTIONS)],
    )
    result = _agent(tmp_path, model, max_fix_attempts=3).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert [entry["error_source"] for entry in exp["fix_history"]] == [
        "run_experiment",
        "compile_check",
        "run_experiment",
    ]
    assert [entry["same_error_streak"] for entry in exp["fix_history"]] == [1, 1, 1]
    assert "STUCK WARNING" not in model.prompts_by_kind["fix"][0]
    assert "STUCK WARNING" not in model.prompts_by_kind["fix"][1]


def test_resolved_fix_is_recorded_to_the_fix_pattern_store(tmp_path):
    # RAISING_SECTIONS' run_experiment_function raises; GOOD_SECTIONS' doesn't
    # — the section that actually differs between broken and fixed.
    store = InMemoryStore()
    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    _agent(tmp_path, model, fix_store=store).run(_planner_output([_plan("H1", complexity="low")]))

    recalled = fix_pattern_store.recall_fixes(store, "run_experiment")
    assert len(recalled) == 1
    changed = recalled[0]["changed_sections"]
    assert "run_experiment_function" in changed
    assert "raise RuntimeError" in changed["run_experiment_function"]["before"]
    assert "raise RuntimeError" not in changed["run_experiment_function"]["after"]


def test_unresolved_fix_is_not_recorded_to_the_fix_pattern_store(tmp_path):
    # A regeneration that fails the *same* check again taught nothing — only a
    # resolution (the check that failed no longer does) should be recorded.
    store = InMemoryStore()
    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(RAISING_SECTIONS)]
    )
    _agent(tmp_path, model, fix_store=store, max_fix_attempts=1).run(
        _planner_output([_plan("H1", complexity="low")])
    )
    assert fix_pattern_store.recall_fixes(store, "run_experiment") == []


def test_fix_prompt_shows_a_past_fix_recorded_for_the_same_error_source(tmp_path):
    # Simulates a pattern recorded by an earlier, separate run against the
    # same (shared) store — this run's own fix loop never has to rediscover
    # this fix; it should just be shown it.
    store = InMemoryStore()
    fix_pattern_store.record_fix(
        store,
        error_source="run_experiment",
        error_summary="RuntimeError: boom",
        broken_sections={
            "run_experiment_function": "def run_experiment(data, model):\n    raise RuntimeError('boom')\n"
        },
        fixed_sections={
            "run_experiment_function": "def run_experiment(data, model):\n    return {'metrics_from_a_past_fix': True}\n"
        },
    )
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    _agent(tmp_path, model, fix_store=store).run(_planner_output([_plan("H1", complexity="low")]))

    fix_prompt = model.prompts_by_kind["fix"][0]
    assert "metrics_from_a_past_fix" in fix_prompt
    assert "Real fixes that resolved this exact error_source" in fix_prompt


def test_fix_prompt_omits_the_pattern_block_when_the_store_is_disabled(tmp_path, monkeypatch):
    import research_pipeline.agents.coder.coder_agent as coder_agent_module

    monkeypatch.setattr(
        coder_agent_module,
        "settings",
        dataclasses.replace(coder_agent_module.settings, coder_enable_fix_pattern_store=False),
    )
    store = InMemoryStore()
    fix_pattern_store.record_fix(
        store,
        error_source="run_experiment",
        error_summary="RuntimeError: boom",
        broken_sections={"run_experiment_function": "raise\n"},
        fixed_sections={"run_experiment_function": "return {'should_not_appear': True}\n"},
    )
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    _agent(tmp_path, model, fix_store=store).run(_planner_output([_plan("H1", complexity="low")]))

    assert "should_not_appear" not in model.prompts_by_kind["fix"][0]
    # The pre-seeded entry is still the only one — this run's own resolved fix
    # was not additionally recorded while the store is disabled.
    recalled = fix_pattern_store.recall_fixes(store, "run_experiment")
    assert len(recalled) == 1
    assert recalled[0]["error_summary"] == "RuntimeError: boom"


def test_a_broken_fix_pattern_store_degrades_gracefully_instead_of_crashing_the_run(tmp_path):
    class ExplodingStore:
        def search(self, *args, **kwargs):
            raise RuntimeError("store is down")

        def put(self, *args, **kwargs):
            raise RuntimeError("store is down")

    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    result = _agent(tmp_path, model, fix_store=ExplodingStore()).run(
        _planner_output([_plan("H1", complexity="low")])
    )
    assert result["experiments"][0]["status"] == "completed"


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


@pytest.mark.parametrize(
    "code",
    [
        "model.eval()",
        # As it actually appears in generated code: indented inside a function,
        # which is what makes the source parseable in the first place.
        "def build_model():\n    model = load()\n    model.eval()  # inference mode\n    return model",
        "with torch.no_grad():\n    model.eval()\n    out = model(x)",
        "self.model.eval()",
        "cursor.exec(query)",
        "estimator.eval(data)",
    ],
)
def test_static_safety_check_allows_methods_that_share_a_builtin_name(code):
    """`model.eval()` is PyTorch's switch to inference mode, not the builtin.

    The old `\\beval\\s*\\(` matched it — `.` before `eval` is a word boundary —
    which blocked essentially every experiment this pipeline generates, and was
    unfixable by regeneration because the code was already correct: a real run
    spent all three fix attempts being told to remove a line that had to stay.
    """
    assert sandbox.static_safety_check(code) == []


@pytest.mark.parametrize(
    "code, expected",
    [
        ("result = eval(user_input)", "eval"),
        ("exec(src)", "exec"),
        ("import builtins\nbuiltins.eval(x)", "eval"),
        ("__import__('os').system('x')", "__import__"),
    ],
)
def test_static_safety_check_still_flags_the_real_builtin(code, expected):
    findings = sandbox.static_safety_check(code)
    assert any(expected in f for f in findings), findings


def test_static_safety_check_reports_where_the_finding_is():
    """ "eval() call" alone gave the fix loop nothing to act on — it could not
    tell the model which line to change."""
    code = "import os\nmodel.eval()\nscore = eval(expr)\n"

    findings = sandbox.static_safety_check(code)

    assert len(findings) == 1
    assert "line 3" in findings[0]
    assert "score = eval(expr)" in findings[0]


def test_static_safety_check_falls_back_to_regex_on_unparseable_source():
    """Both callers check compilation first, so this is the path a direct caller
    takes. Imprecise on purpose — better a false positive than a silent pass.

    The `model.eval()` false positive survives here and that is fine: source
    this branch sees does not compile, so in the pipeline compile_check has
    already failed it and the model gets a syntax error, never a safety finding.
    """
    findings = sandbox.static_safety_check("def broken(:\n    eval(x)")

    assert any("eval" in f for f in findings)


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


# -- sandbox.py: check_nontrivial_function_bodies ---------------------------------------


# The exact shape a real Kaggle run reported as `status: "completed",
# fix_attempts: 0`: every required function is a comment (echoing the
# planner's/prompt's own instruction text back) followed by a bare `pass`.
HOLLOW_SECTIONS = {
    **GOOD_SECTIONS,
    "load_data_function": (
        "def load_data():\n"
        "    # Load the survey data, clean missing values, and normalize columns\n"
        "    pass\n"
    ),
    "build_model_function": (
        "def build_model():\n    # Build a logistic regression classifier\n    pass\n"
    ),
    "run_experiment_function": (
        "def run_experiment(data, model):\n    # Train the model and collect predictions\n    pass\n"
    ),
    "evaluate_function": (
        "def evaluate(experiment_output):\n    # Compute accuracy and F1 against the baseline\n    pass\n"
    ),
}


def test_check_nontrivial_function_bodies_flags_bare_pass_with_echoed_comments():
    findings = sandbox.check_nontrivial_function_bodies(HOLLOW_SECTIONS)
    assert len(findings) == 4
    assert "load_data_function" in findings[0]


def test_check_nontrivial_function_bodies_flags_ellipsis_body():
    sections = {"evaluate_function": "def evaluate(experiment_output):\n    ...\n"}
    findings = sandbox.check_nontrivial_function_bodies(sections)
    assert len(findings) == 1
    assert "evaluate" in findings[0]


def test_check_nontrivial_function_bodies_flags_docstring_only_body():
    sections = {
        "build_model_function": ('def build_model():\n    """Builds and returns the model."""\n')
    }
    assert len(sandbox.check_nontrivial_function_bodies(sections)) == 1


def test_check_nontrivial_function_bodies_flags_bare_not_implemented_raise():
    sections = {
        "run_experiment_function": (
            "def run_experiment(data, model):\n    raise NotImplementedError\n"
        )
    }
    assert len(sandbox.check_nontrivial_function_bodies(sections)) == 1


def test_check_nontrivial_function_bodies_passes_a_docstring_plus_real_code():
    sections = {
        "load_data_function": (
            'def load_data():\n    """Loads the dataset."""\n    return pd.read_csv(PATH)\n'
        )
    }
    assert sandbox.check_nontrivial_function_bodies(sections) == []


def test_check_nontrivial_function_bodies_passes_good_sections():
    assert sandbox.check_nontrivial_function_bodies(GOOD_SECTIONS) == []


def test_check_nontrivial_function_bodies_skips_a_section_that_does_not_parse():
    # compile_check's job to report a syntax error; this check should not also
    # collect a spurious finding for a section it can't even parse.
    sections = {"evaluate_function": "def evaluate(experiment_output:\n    pass\n"}
    assert sandbox.check_nontrivial_function_bodies(sections) == []


def test_run_reports_hollow_stub_functions_and_never_executes(tmp_path):
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response(HOLLOW_SECTIONS)})
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
    assert "no real implementation" in exp["reason"].lower()
    assert not (experiments_dir / "H1" / "results.json").exists()  # never executed


def test_run_reports_implausible_results_and_does_not_report_completed(tmp_path):
    # evaluate() is a real, non-trivial function (so check_nontrivial_function_bodies
    # and check_hf_dataset_usage both pass) that always returns an all-zero
    # result regardless of its input — the run genuinely executes, but the
    # output it produces is exactly the hollow tail check_results_plausibility
    # exists to catch.
    hollow_metrics_sections = {
        **GOOD_SECTIONS,
        "evaluate_function": (
            "def evaluate(experiment_output):\n"
            "    _ = experiment_output\n"
            '    return {"accuracy": 0, "f1": 0.0, "meets_success_criteria": False}\n'
        ),
    }
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response(hollow_metrics_sections)})
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
    assert "exactly 0" in exp["reason"]
    # It really did run — this is a plausibility rejection, not a compile/lint one.
    assert (experiments_dir / "H1" / "results.json").exists()


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
    assert _estimate_tokens("x" * 4000) == 4000 // _CHARS_PER_TOKEN_ESTIMATE


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
    expected = (
        8192
        - (_estimate_tokens(prompts.SYSTEM_PROMPT) + _estimate_tokens(prompt))
        - _CONTEXT_SAFETY_MARGIN
    )
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
    assert str(4096 - prompt_tokens - _CONTEXT_SAFETY_MARGIN) in message  # the (negative) headroom
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


# -- coder_agent.py: dataset selection ---------------------------------------------------
# No test here touches the network. The appraisal path's six network calls are
# six injected functions (dataset_search_fn, dataset_describe_fn,
# dataset_rows_fn, dataset_card_fn, dataset_download_fn, dataset_normalize_fn),
# exactly like network_check/gpu_check, and `_agent` defaults every one of them
# to inert so a test can't reach the Hub by forgetting one.


HF_HIT = {
    "id": "acme/sleep-survey",
    "downloads": 50_000,
    "likes": 42,
    "tags": ["task_categories:tabular-regression", "arxiv:2401.00001"],
    "cardData": {"license": "apache-2.0", "size_categories": ["1K<n<10K"], "citation": "@misc{}"},
}

# What describe_candidate returns: the viewer-confirmed schema plus the Hub
# record the deterministic dimensions are read off.
HF_CANDIDATE = {
    "dataset_id": "acme/sleep-survey",
    "config": "default",
    "split": "train",
    "columns": [{"name": "hours_slept", "type": "float32"}, {"name": "score", "type": "int64"}],
    "sample_rows": [{"hours_slept": 7.5, "score": 88}],
    "info": {
        "id": "acme/sleep-survey",
        "sha": "abc123def",
        "cardData": HF_HIT["cardData"],
        "tags": HF_HIT["tags"],
        "downloads": 50_000,
    },
    "cardData": HF_HIT["cardData"],
    "tags": HF_HIT["tags"],
    "downloads": 50_000,
    "likes": 42,
    "revision": "abc123def",
    "license": "apache-2.0",
    "num_rows": 5_000,
    "num_bytes": 2_048,
}

# The state dict shape the prompt block and check_hf_dataset_usage consume,
# which is what `HF_DATASET_MATCH` was before scoring existed. Kept minimal on
# purpose — these two consumers read only the handful of keys named here.
HF_DATASET_MATCH = {
    "dataset_id": "acme/sleep-survey",
    "config": "default",
    "split": "train",
    "columns": HF_CANDIDATE["columns"],
    "sample_rows": HF_CANDIDATE["sample_rows"],
}


def _rows(count=40):
    """A clean sample: distinct rows, every declared column filled. Scores at
    the top of the quality band, so a test that wants a *rejection* has to
    introduce the defect it is testing rather than getting one for free."""
    return [{"hours_slept": 5.0 + index * 0.1, "score": 50 + index} for index in range(count)]


def _spec_response(**overrides):
    payload = {
        "task": "predict sleep quality",
        "domain": "sleep research",
        "languages": ["en"],
        "data_types": ["hours_slept", "score"],
        "desired_examples": 1000,
        "minimum_quality": 0.7,
        "license_requirements": ["permissive"],
        "avoid": ["synthetic-only"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _evidence_response(**overrides):
    payload = {
        "requirements": {
            "contains_hours_slept": {"status": "pass", "evidence": "Sampled rows carry it."}
        },
        "task_relevance": "exact",
        "task_relevance_evidence": "The card describes sleep-quality measurements.",
        "content_relevance": "exact",
        "content_relevance_evidence": "Sampled rows are hours and scores.",
        "column_mapping": {"hours_slept": "hours_slept", "score": "score"},
    }
    payload.update(overrides)
    return json.dumps(payload)


def _critic_response(*findings):
    return json.dumps({"findings": list(findings)})


def _dataset_stack(
    hits=(HF_HIT,),
    candidate=HF_CANDIDATE,
    rows=None,
    card="# Sleep survey\nCollected from a sleep lab.",
    downloads=True,
):
    """Fakes for all six dataset seams, plus the record of what each was asked.

    `downloads=False` models huggingface_hub not being installed — the download
    function returning None, which is exactly what the real one does in that
    case.
    """
    calls: dict[str, list] = {"search": [], "describe": [], "rows": [], "card": [], "download": []}

    def search(query, limit):
        calls["search"].append(query)
        return list(hits)

    def describe(dataset_id):
        calls["describe"].append(dataset_id)
        return dict(candidate) if candidate else {}

    def fetch_rows(dataset_id, config, split, limit):
        calls["rows"].append((dataset_id, config, split, limit))
        return list(rows if rows is not None else _rows())

    def get_card(dataset_id):
        calls["card"].append(dataset_id)
        return card

    def download(dataset_id, revision, dest):
        calls["download"].append((dataset_id, revision, str(dest)))
        if not downloads:
            return None
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "train.jsonl").write_text('{"hours_slept": 7.5, "score": 88}\n', encoding="utf-8")
        return dest

    def normalize(source, dest, max_rows):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('{"hours_slept": 7.5, "score": 88}\n', encoding="utf-8")
        return {"rows_written": 1, "columns": ["hours_slept", "score"], "source_files": ["t.jsonl"]}

    kwargs = {
        "dataset_search_fn": search,
        "dataset_describe_fn": describe,
        "dataset_rows_fn": fetch_rows,
        "dataset_card_fn": get_card,
        "dataset_download_fn": download,
        "dataset_normalize_fn": normalize,
    }
    return kwargs, calls


def _dataset_model(codegen_sections=None, **overrides):
    """A ScriptedChatModel wired for the whole dataset path plus one codegen."""
    responses = {
        "dataset_spec": [_spec_response()],
        "dataset_evidence": [_evidence_response()],
        "dataset_critic": [_critic_response()],
        "codegen": [_codegen_response(codegen_sections or GOOD_SECTIONS_WITH_LOCAL_DATASET)],
    }
    responses.update(overrides)
    return RecordingScriptedChatModel(**responses)


# A load_data() that reads the downloaded, normalized local file — the default
# path now that CODER_DATASET_DOWNLOAD is on. Guarded, as check_data_fallback
# requires.
GOOD_SECTIONS_WITH_LOCAL_DATASET = {
    **GOOD_SECTIONS,
    "load_data_function": (
        "def load_data():\n"
        "    try:\n"
        "        with open(DATA_PATH, encoding='utf-8') as handle:\n"
        "            return [json.loads(line) for line in handle if line.strip()]\n"
        "    except Exception:\n"
        "        return [{'hours_slept': 7.5, 'score': 88}]\n"
    ),
    "configuration": "DATA_PATH = 'data.jsonl'\n",
}

# The REST variant, for the no-download path.
GOOD_SECTIONS_WITH_HF_DATASET = {
    **GOOD_SECTIONS,
    "load_data_function": (
        "def load_data():\n"
        "    try:\n"
        "        response = requests.get(\n"
        "            'https://datasets-server.huggingface.co/rows"
        "?dataset=acme%2Fsleep-survey&config=default&split=train&offset=0&length=100',\n"
        "            timeout=30,\n"
        "        )\n"
        "        response.raise_for_status()\n"
        "        return [entry['row'] for entry in response.json()['rows']]\n"
        "    except Exception:\n"
        "        return [{'hours_slept': 7.5, 'score': 88}]\n"
    ),
}


def test_accepted_dataset_is_downloaded_and_offered_as_a_local_file(tmp_path):
    model = _dataset_model()
    stack, calls = _dataset_stack()
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    # Searched from the spec's structured fields, not the first four words of prose.
    assert any("sleep" in query for query in calls["search"])
    assert calls["download"] and calls["download"][0][1] == "abc123def"  # pinned revision

    prompt = model.prompts_by_kind["codegen"][0]
    assert "acme/sleep-survey" in prompt
    assert "revision: abc123def" in prompt
    assert "data.jsonl" in prompt
    assert "hours_slept (float32)" in prompt  # real column names and dtypes
    # The local path, not the REST URL — and no instruction to fetch anything.
    assert "datasets-server.huggingface.co/rows" not in prompt
    assert "fall back to a small synthesized stand-in dataset" in prompt


def test_the_dataset_record_is_written_beside_the_experiment(tmp_path):
    stack, _ = _dataset_stack()
    _agent(tmp_path, _dataset_model(), network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    record = json.loads(
        (tmp_path / "experiments" / "H1" / "dataset_provenance.json").read_text(encoding="utf-8")
    )
    assert record["dataset"] == "acme/sleep-survey"
    assert record["revision"] == "abc123def"
    assert record["decision"] == "accept"
    assert record["license"] == "apache-2.0"
    assert record["downloaded_at"]
    # Every dimension, its band, and the evidence behind it — enough to
    # re-derive the score without re-running anything.
    assert set(record["evidence"]) == set(dataset_scoring.WEIGHTS)
    assert record["bands"]["license_fit"] == "permitted"
    assert record["evidence_notes"]["task_relevance"]
    assert record["inspection"]["rows_sampled"] == 40
    assert record["weights"] == dataset_scoring.WEIGHTS


def test_the_model_cannot_invent_the_score(tmp_path):
    # The load-bearing test for the whole rubric. The evidence response claims a
    # 0.99 score on a dataset it simultaneously labels unrelated on every
    # dimension it is allowed to label. Python's arithmetic must win.
    model = _dataset_model(
        dataset_evidence=[
            _evidence_response(
                score=0.99,
                task_relevance="unrelated",
                content_relevance="unrelated",
                column_mapping={},
            )
        ],
        codegen=[_codegen_response()],
    )
    stack, calls = _dataset_stack()
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(_planner_output([_plan("H1")]))

    # Scored far below the 0.75 threshold, so nothing was accepted, nothing was
    # downloaded, and the prompt reads as it does with no dataset at all.
    assert calls["download"] == []
    assert "acme/sleep-survey" not in model.prompts_by_kind["codegen"][0]
    # And the critic was never asked: a candidate already failing on score has
    # nothing an adversarial pass could change.
    assert "dataset_critic" not in model.calls_by_kind


def test_an_incompatible_license_is_decided_in_python_not_by_the_model(tmp_path):
    # The evidence response insists everything is fine; the Hub says cc-by-nc.
    # license_fit is never asked of the model, so the claim is irrelevant.
    candidate = {**HF_CANDIDATE, "license": "cc-by-nc-4.0"}
    stack, calls = _dataset_stack(
        hits=({**HF_HIT, "cardData": {**HF_HIT["cardData"], "license": "cc-by-nc-4.0"}},),
        candidate=candidate,
    )
    model = _dataset_model(codegen=[_codegen_response()])
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(_planner_output([_plan("H1")]))

    # Dropped by the prefilter before it could cost an appraisal call at all.
    assert calls["describe"] == []
    assert calls["download"] == []
    assert "acme/sleep-survey" not in model.prompts_by_kind["codegen"][0]


def test_a_critic_veto_falls_through_to_the_next_candidate(tmp_path):
    first = {**HF_CANDIDATE, "dataset_id": "acme/leaky", "revision": "r1"}
    second = {**HF_CANDIDATE, "dataset_id": "acme/clean", "revision": "r2"}
    described = iter([first, second])

    stack, calls = _dataset_stack(
        hits=({**HF_HIT, "id": "acme/leaky"}, {**HF_HIT, "id": "acme/clean"})
    )
    stack["dataset_describe_fn"] = lambda dataset_id: next(described, {})

    model = _dataset_model(
        dataset_evidence=[_evidence_response(), _evidence_response()],
        dataset_critic=[
            _critic_response(
                {"code": "evaluation_contamination", "evidence": "Rows quote the held-out split."}
            ),
            _critic_response(),
        ],
        codegen=[_codegen_response(GOOD_SECTIONS_WITH_LOCAL_DATASET)],
    )
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(_planner_output([_plan("H1")]))

    # The vetoed candidate was never downloaded; the next one was.
    assert [call[0] for call in calls["download"]] == ["acme/clean"]
    record = json.loads(
        (tmp_path / "experiments" / "H1" / "dataset_provenance.json").read_text(encoding="utf-8")
    )
    assert record["dataset"] == "acme/clean"


def test_every_candidate_vetoed_means_no_dataset_is_offered(tmp_path):
    model = _dataset_model(
        dataset_critic=[
            _critic_response({"code": "personal_information", "evidence": "Rows carry emails."})
        ],
        codegen=[_codegen_response()],
    )
    stack, calls = _dataset_stack()
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert calls["download"] == []
    assert "acme/sleep-survey" not in model.prompts_by_kind["codegen"][0]


def test_a_dataset_over_the_size_budget_falls_back_to_the_rest_url(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_dataset_max_download_gb=0.000001)
    stack, calls = _dataset_stack()
    model = _dataset_model(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert calls["download"] == []  # checked against /size *before* fetching
    prompt = model.prompts_by_kind["codegen"][0]
    assert "datasets-server.huggingface.co/rows?dataset=acme%2Fsleep-survey" in prompt


def test_the_per_run_download_cap_stops_later_plans(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_dataset_max_accepted_per_run=1)
    stack, calls = _dataset_stack()
    # The REST spelling names the dataset for both plans — the second one has no
    # local copy to point at, since the cap is what this test is about.
    model = _dataset_model(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1"), _plan("H2")])
    )

    # Both plans selected a dataset; only the first was allowed to download it.
    assert len(calls["download"]) == 1
    second = json.loads(
        (tmp_path / "experiments" / "H2" / "dataset_provenance.json").read_text(encoding="utf-8")
    )
    assert second["decision"] == "accept"
    assert second["local_path"] == ""


def test_the_appraisal_budget_bounds_the_llm_calls(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_dataset_max_appraisals=2, coder_dataset_max_inspections=4)
    hits = tuple({**HF_HIT, "id": f"acme/d{index}"} for index in range(4))
    stack, calls = _dataset_stack(hits=hits)

    def describe(dataset_id):
        calls["describe"].append(dataset_id)
        return {**HF_CANDIDATE, "dataset_id": dataset_id}

    stack["dataset_describe_fn"] = describe

    model = _dataset_model()
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(_planner_output([_plan("H1")]))

    # Four shortlisted and described (cheap), but only two appraised (a model
    # call each), and one critic call for the leader.
    assert len(calls["describe"]) == 4
    assert model.calls_by_kind["dataset_evidence"] == 2
    assert model.calls_by_kind["dataset_critic"] == 1


def test_every_query_contributes_its_full_page_to_the_pool(tmp_path, monkeypatch):
    """Barkla job 10334321: the pool cap was divided by the query count, so five
    queries took 4 hits each and pooled **6** candidates where the same queries
    at a full page pool 47. The run then appraised the least-bad of six
    irrelevant datasets and accepted none. Every query now contributes its whole
    page, interleaved round-robin up to the cap."""
    _patch_settings(monkeypatch, coder_dataset_max_candidates=40, coder_dataset_max_inspections=40)

    # Each query returns a distinct full page, so a starved take shows up
    # directly as a smaller pool.
    def search(query, limit):
        token = query.replace(" ", "-")
        return [{**HF_HIT, "id": f"{token}/d{index}"} for index in range(limit)]

    stack, calls = _dataset_stack()
    stack["dataset_search_fn"] = search

    def describe(dataset_id):
        calls["describe"].append(dataset_id)
        return {**HF_CANDIDATE, "dataset_id": dataset_id}

    stack["dataset_describe_fn"] = describe
    model = _dataset_model()
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(_planner_output([_plan("H1")]))

    # The cap is reached, not a fraction of it — under the old division this
    # would have been 40 // (number of queries).
    assert len(calls["describe"]) == 40
    # And every query is represented, rather than the first one filling the pool.
    contributing = {name.split("/")[0] for name in calls["describe"]}
    assert len(contributing) > 1


def test_a_failed_spec_call_degrades_to_a_plan_derived_spec(tmp_path):
    # The spec call returns junk. The search still runs, against a spec derived
    # from the plan — a coarser requirement, not no requirement.
    model = _dataset_model(dataset_spec=["not json at all"])
    stack, calls = _dataset_stack()
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert calls["search"]  # searched anyway, against the derived spec
    assert model.calls_by_kind["dataset_evidence"] == 1  # and appraised what it found


def test_a_raising_search_never_blocks_code_generation(tmp_path):
    def boom(query, limit):
        raise RuntimeError("hub is down")

    stack, _ = _dataset_stack()
    stack["dataset_search_fn"] = boom
    model = _dataset_model(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert "Dataset Viewer" not in model.prompts_by_kind["codegen"][0]


def test_a_failed_evidence_call_scores_the_candidate_down_rather_than_crashing(tmp_path):
    model = _dataset_model(dataset_evidence=["}{ not json"], codegen=[_codegen_response()])
    stack, calls = _dataset_stack()
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    # Both model-supplied labels fell to their pessimistic band, so the score
    # can't clear the threshold and nothing was downloaded.
    assert calls["download"] == []


def test_no_candidates_generates_exactly_as_before(tmp_path):
    stack, _ = _dataset_stack(hits=())
    model = _dataset_model(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert "Dataset Viewer" not in model.prompts_by_kind["codegen"][0]
    # Nothing to appraise means no model calls were spent on appraising.
    assert "dataset_evidence" not in model.calls_by_kind


def test_viewer_failures_are_walked_past_rather_than_costing_a_slot(tmp_path, monkeypatch):
    """Barkla job 10334376 pooled 27 candidates, passed the top 5 to describe,
    lost 3 to the Dataset Viewer, and inspected 2 — while the dataset that
    actually fitted the plan sat at rank 6 and was never looked at. The budget
    bounds how many candidates get *inspected*, not how many get attempted."""
    _patch_settings(monkeypatch, coder_dataset_max_inspections=3)
    hits = tuple({**HF_HIT, "id": f"acme/d{index}"} for index in range(12))
    stack, calls = _dataset_stack(hits=hits)

    # The first three the prefilter reaches are unservable, exactly as a viewer
    # 500 / viewer:false / no-splits looks from here.
    unservable = {"acme/d0", "acme/d1", "acme/d2"}

    def describe(dataset_id):
        calls["describe"].append(dataset_id)
        return {} if dataset_id in unservable else {**HF_CANDIDATE, "dataset_id": dataset_id}

    stack["dataset_describe_fn"] = describe
    model = _dataset_model()
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(_planner_output([_plan("H1")]))

    # Six attempted (three dead, three live) rather than three attempted and one
    # surviving — the budget is filled, not merely offered.
    assert len(calls["describe"]) == 6
    assert model.calls_by_kind["dataset_evidence"] == 3


def test_the_prefilter_window_is_still_bounded(tmp_path, monkeypatch):
    # Walking past failures must not turn into walking the whole pool.
    _patch_settings(monkeypatch, coder_dataset_max_inspections=2)
    hits = tuple({**HF_HIT, "id": f"acme/d{index}"} for index in range(40))
    stack, calls = _dataset_stack(hits=hits)
    stack["dataset_describe_fn"] = lambda dataset_id: calls["describe"].append(dataset_id) or {}

    _agent(
        tmp_path, _dataset_model(codegen=[_codegen_response()]), network_check=lambda: True, **stack
    ).run(_planner_output([_plan("H1")]))

    # Nothing is servable, so it exhausts the window and stops — not the pool.
    assert len(calls["describe"]) == 2 * 4


def test_an_unservable_candidate_is_dropped(tmp_path):
    stack, calls = _dataset_stack(candidate={})
    model = _dataset_model(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert calls["describe"] == ["acme/sleep-survey"]
    assert "dataset_evidence" not in model.calls_by_kind


def test_download_is_skipped_when_the_setting_is_off(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_dataset_download=False)
    stack, calls = _dataset_stack()
    model = _dataset_model(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert calls["download"] == []
    assert "datasets-server.huggingface.co/rows" in model.prompts_by_kind["codegen"][0]


def test_a_missing_huggingface_hub_degrades_to_the_rest_url(tmp_path):
    # downloads=False is what the real download_dataset returns when
    # huggingface_hub isn't installed.
    stack, calls = _dataset_stack(downloads=False)
    model = _dataset_model(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert calls["download"]  # attempted
    assert "datasets-server.huggingface.co/rows" in model.prompts_by_kind["codegen"][0]


def test_selection_is_skipped_without_network(tmp_path):
    stack, calls = _dataset_stack()
    model = _dataset_model(codegen=[_codegen_response()])
    _agent(tmp_path, model, **stack).run(_planner_output([_plan("H1")]))

    assert calls["search"] == []  # never attempted — the runtime probe said no network
    assert "dataset_spec" not in model.calls_by_kind


def test_selection_is_skipped_when_the_setting_is_off(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_enable_hf_dataset_search=False)
    stack, calls = _dataset_stack()
    model = _dataset_model(codegen=[_codegen_response()])
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(_planner_output([_plan("H1")]))

    assert calls["search"] == []
    assert "Dataset Viewer" not in model.prompts_by_kind["codegen"][0]


def test_infeasible_plan_never_reaches_selection(tmp_path):
    stack, calls = _dataset_stack()
    _agent(tmp_path, FakeChatModel({}), network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1", feasible=False)])
    )

    assert calls["search"] == []  # skipped before the first dataset node, like the LLM call


def test_the_fix_prompt_reuses_the_dataset_selected_once_for_the_plan(tmp_path):
    # The dataset is usually exactly what a data-loading failure needs to be
    # fixed *with* — and selecting it once per plan (not once per attempt) keeps
    # a three-attempt fix loop from re-running the whole appraisal.
    model = _dataset_model(
        codegen=[_codegen_response(RAISING_SECTIONS)],
        fix=[_codegen_response(GOOD_SECTIONS_WITH_LOCAL_DATASET)],
    )
    stack, calls = _dataset_stack()
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert model.calls_by_kind["dataset_evidence"] == 1
    assert len(calls["download"]) == 1
    assert "acme/sleep-survey" in model.prompts_by_kind["fix"][0]


def test_each_plan_gets_its_own_selection(tmp_path):
    stack, calls = _dataset_stack()
    model = _dataset_model()
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1"), _plan("H2")])
    )

    assert model.calls_by_kind["dataset_spec"] == 2
    # The second plan reuses the cached download rather than re-fetching it —
    # the cache is keyed by (repo id, revision) and shared across plans.
    assert len(calls["download"]) == 1


def test_a_used_local_dataset_earns_a_real_data_verdict(tmp_path):
    # The regression this whole path exists for: _provenance_for used to read
    # hf_dataset["dataset"], a key the client has never returned, so a real
    # dataset genuinely read by generated code was never counted as a real
    # input and the verdict was withheld as "unknown".
    stack, _ = _dataset_stack()
    # A plan that declares a real source: _plan()'s default is "synthetic",
    # which would make the mixed verdict correct for a reason unrelated to the
    # bug this covers.
    plan = {
        **_plan("H1", complexity="low"),
        "data_requirements": {
            "source": "Hugging Face",
            "description": "sleep survey rows",
            "preprocessing_steps": [],
        },
    }
    result = _agent(tmp_path, _dataset_model(), network_check=lambda: True, **stack).run(
        _planner_output([plan])
    )

    experiment = result["experiments"][0]
    assert experiment["data_provenance"]["all_inputs_real"] is True
    assert experiment["results"]["meets_success_criteria"] != "unknown"
    inputs = experiment["data_provenance"]["inputs"]
    assert inputs[0]["kind"] == "real_local"
    assert inputs[0]["local_path"].endswith("data.jsonl")
    assert inputs[0]["name"] == "Hugging Face dataset acme/sleep-survey"


def test_the_real_normalizer_carries_a_downloaded_dataset_to_a_real_verdict(tmp_path):
    """The seam six Barkla runs never got through end to end.

    Every other dataset test fakes `normalize_to_jsonl`, so the join between
    what the Hub actually hands over (a snapshot directory of CSV/parquet plus a
    README) and the provenance credit at the far end was only ever asserted in
    halves. This runs the real normalizer over a realistic snapshot — the shape
    Ammok/apple_stock_price_from_1980-2021 actually has, which is what run 6
    downloaded — and follows it all the way to `all_inputs_real`.

    Job 10410771 reached this point on the cluster and died provisioning
    TensorFlow, which has nothing to do with any of it.
    """
    snapshot_rows = [
        "Date,Open,High,Low,Close,Adj Close,Volume",
        "1980-12-12,0.128348,0.128906,0.128348,0.128348,0.100178,469033600",
        "1980-12-15,0.122210,0.122210,0.121652,0.121652,0.094952,175884800",
        "1980-12-16,0.113281,0.113281,0.112723,0.112723,0.087983,105728000",
    ]

    def download(dataset_id, revision, dest):
        # What snapshot_download leaves behind: the data file at the repo root,
        # alongside the card. Not a JSONL — that is the normalizer's job.
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "AAPL.csv").write_text("\n".join(snapshot_rows) + "\n", encoding="utf-8")
        (dest / "README.md").write_text("# Apple stock price\n", encoding="utf-8")
        return dest

    stack, calls = _dataset_stack()
    stack["dataset_download_fn"] = download
    # The real one, not a fake. This is the whole point of the test.
    stack["dataset_normalize_fn"] = huggingface_client.normalize_to_jsonl

    plan = {
        **_plan("H1", complexity="low"),
        "data_requirements": {
            "source": "Hugging Face",
            "description": "daily stock closing prices",
            "preprocessing_steps": [],
        },
    }
    model = _dataset_model()
    result = _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([plan])
    )

    # 1. The real normalizer turned the snapshot CSV into stdlib-readable JSONL.
    record = json.loads(
        (tmp_path / "experiments" / "H1" / "dataset_provenance.json").read_text(encoding="utf-8")
    )
    local_path = Path(record["local_path"])
    assert local_path.name == "data.jsonl"
    rows = [json.loads(line) for line in local_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["Date"] == "1980-12-12"
    assert rows[0]["Close"] == "0.128348"
    assert record["rows"] == 3  # taken from what was written, not from the card

    # 2. The codegen prompt pointed at that exact file, via the local-file
    #    variant of the dataset block rather than the REST one. Asserted on the
    #    variant's own wording, not on the viewer hostname: the *provenance*
    #    block names that host too, because this plan's declared source is
    #    "Hugging Face" and provenance.OPEN_SOURCES maps it to the /rows URI.
    prompt = model.prompts_by_kind["codegen"][0]
    assert str(local_path) in prompt
    assert "already been downloaded and flattened to JSON Lines" in prompt
    assert "It was not downloaded, so read it over HTTP" not in prompt

    # 3. And the generated code naming it earns a real verdict — the branch that
    #    read hf_dataset["dataset"], a key the client has never returned, so a
    #    real dataset was never once counted as a real input.
    experiment = result["experiments"][0]
    provenance_doc = experiment["data_provenance"]
    assert provenance_doc["all_inputs_real"] is True
    assert provenance_doc["surrogate_count"] == 0
    assert experiment["results"]["meets_success_criteria"] != "unknown"
    source = provenance_doc["inputs"][0]
    assert source["kind"] == "real_local"
    assert source["local_path"] == str(local_path)
    assert "acme/sleep-survey" in source["name"]


def test_a_dataset_whose_normalization_yields_nothing_is_not_credited_as_real(tmp_path):
    """The converse, and the reason the assertion above is worth having: a
    download that produces no readable rows must leave the dataset uncredited
    rather than pointing generated code at an empty file."""

    def download(dataset_id, revision, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("# card only, no data\n", encoding="utf-8")
        return dest

    stack, _ = _dataset_stack()
    stack["dataset_download_fn"] = download
    stack["dataset_normalize_fn"] = huggingface_client.normalize_to_jsonl

    model = _dataset_model(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    _agent(tmp_path, model, network_check=lambda: True, **stack).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    record = json.loads(
        (tmp_path / "experiments" / "H1" / "dataset_provenance.json").read_text(encoding="utf-8")
    )
    assert record["local_path"] == ""
    # Falls back to the REST URL that was the only path before downloading
    # existed, rather than to a file that isn't there.
    prompt = model.prompts_by_kind["codegen"][0]
    assert "It was not downloaded, so read it over HTTP" in prompt
    assert "already been downloaded and flattened to JSON Lines" not in prompt


def test_an_offered_dataset_the_code_ignores_is_still_not_real(tmp_path):
    # The converse, and the reason the check above is worth having: an offered
    # dataset the generated code never names must not be counted as evidence.
    stack, _ = _dataset_stack()
    result = _agent(
        tmp_path,
        _dataset_model(codegen=[_codegen_response(assumptions=["acme/sleep-survey lacks labels"])]),
        network_check=lambda: True,
        **stack,
    ).run(_planner_output([_plan("H1", complexity="low")]))

    experiment = result["experiments"][0]
    assert experiment["data_provenance"]["all_inputs_real"] is False
    assert experiment["results"]["meets_success_criteria"] == "unknown"


def test_run_reports_a_silently_ignored_offered_dataset(tmp_path):
    # The offered dataset is real, but load_data() (via the default
    # GOOD_SECTIONS fixture) neither reads it nor declines it in
    # assumptions_made — the exact silent-third-option check_hf_dataset_usage
    # exists to catch.
    stack, _ = _dataset_stack()
    experiments_dir = tmp_path / "experiments"
    result = _agent(
        tmp_path,
        # The same dataset-ignoring code every time, so the fix budget runs out
        # and the run has to report the offer was never engaged with.
        _dataset_model(codegen=[_codegen_response()], fix=[_codegen_response()]),
        network_check=lambda: True,
        **stack,
    ).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "acme/sleep-survey" in exp["reason"]
    assert "not used" in exp["reason"].lower() or "no sign of it" in exp["reason"].lower()
    assert not (experiments_dir / "H1" / "results.json").exists()


def test_run_accepts_an_explicitly_declined_offered_dataset(tmp_path):
    # The usage note explicitly sanctions ignoring the offered dataset when it
    # doesn't fit, provided the model says so in assumptions_made — that
    # documented escape hatch must not be flagged as hollow.
    stack, _ = _dataset_stack()
    result = _agent(
        tmp_path,
        _dataset_model(
            codegen=[
                _codegen_response(
                    assumptions=["acme/sleep-survey doesn't include the label column this needs"]
                )
            ]
        ),
        network_check=lambda: True,
        **stack,
    ).run(_planner_output([_plan("H1", complexity="low")]))

    assert result["experiments"][0]["status"] == "completed"


def test_check_hf_dataset_usage_passes_when_nothing_was_offered():
    assert sandbox.check_hf_dataset_usage("", "def load_data():\n    return None\n", [], {}) == []


def test_check_hf_dataset_usage_flags_a_silently_unused_dataset():
    findings = sandbox.check_hf_dataset_usage(
        "", "def load_data():\n    return None\n", [], HF_DATASET_MATCH
    )
    assert len(findings) == 1
    assert "acme/sleep-survey" in findings[0]


def test_check_hf_dataset_usage_passes_a_reference_in_load_data():
    source = (
        "def load_data():\n"
        "    url = 'https://datasets-server.huggingface.co/rows?dataset=acme%2Fsleep-survey'\n"
        "    return requests.get(url, timeout=30).json()\n"
    )
    assert sandbox.check_hf_dataset_usage("", source, [], HF_DATASET_MATCH) == []


def test_check_hf_dataset_usage_passes_a_reference_in_configuration():
    # A dataset id stashed in a module-level constant and referenced by name
    # inside load_data() is still a legitimate use, not an unused offer.
    config = "DATASET_ID = 'acme/sleep-survey'\n"
    source = "def load_data():\n    return _fetch(DATASET_ID)\n"
    assert sandbox.check_hf_dataset_usage(config, source, [], HF_DATASET_MATCH) == []


def test_check_hf_dataset_usage_passes_a_local_file_by_full_path():
    # On the download path the repo id never appears in the code at all —
    # insisting on it there would fail correct code.
    offered = {**HF_DATASET_MATCH, "local_path": "/scratch/hf/acme__sleep-survey/data.jsonl"}
    source = (
        "def load_data():\n"
        "    with open('/scratch/hf/acme__sleep-survey/data.jsonl') as handle:\n"
        "        return [json.loads(line) for line in handle]\n"
    )
    assert sandbox.check_hf_dataset_usage("", source, [], offered) == []


def test_check_hf_dataset_usage_passes_a_local_file_by_bare_name():
    offered = {**HF_DATASET_MATCH, "local_path": "/scratch/hf/acme__sleep-survey/data.jsonl"}
    config = "DATA_PATH = Path(__file__).parent / 'data.jsonl'\n"
    source = "def load_data():\n    return _read(DATA_PATH)\n"
    assert sandbox.check_hf_dataset_usage(config, source, [], offered) == []


def test_check_hf_dataset_usage_still_flags_a_downloaded_dataset_nothing_reads():
    offered = {**HF_DATASET_MATCH, "local_path": "/scratch/hf/acme__sleep-survey/data.jsonl"}
    findings = sandbox.check_hf_dataset_usage(
        "", "def load_data():\n    return None\n", [], offered
    )
    assert len(findings) == 1


def test_check_hf_dataset_usage_passes_when_declined_in_assumptions():
    assumptions = ["Declined acme/sleep-survey — wrong schema for this task"]
    findings = sandbox.check_hf_dataset_usage(
        "", "def load_data():\n    return None\n", assumptions, HF_DATASET_MATCH
    )
    assert findings == []


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


@pytest.fixture
def slurm_submit_stubs(monkeypatch):
    """Same real-SLURM-call stubbing as `auto_submit`, but leaves
    coder_auto_submit_slurm alone — for the interactive-review tests below,
    which reach the exact same submit path via a human's "submit" answer
    instead of the auto-submit flag."""
    submitted = []
    _patch_settings(
        monkeypatch,
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


# -- coder_agent.py: interactive SLURM review (interrupt()/Command) ---------------------


def test_interactive_review_off_by_default_never_calls_the_prompt(tmp_path):
    # Same shape as test_auto_submit_off_by_default_leaves_sbatch_for_review:
    # with neither flag set, a plan that can't run here goes straight to
    # manual review — the reviewer callable must never even be reached.
    def fail_if_called(payload):
        pytest.fail("interactive review prompt was called with interactive review off")

    model = ScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, slurm_review_prompt=fail_if_called).run(
        _planner_output([_plan("H1", complexity="high")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "submit it yourself" in exp["reason"]


def test_interactive_review_submits_when_reviewer_approves(tmp_path, slurm_submit_stubs):
    model = ScriptedChatModel(codegen=[_codegen_response()], self_review=[_clean_review()])
    result = _agent(
        tmp_path,
        model,
        interactive_slurm_review=True,
        slurm_review_prompt=lambda payload: "submit",
    ).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "submitted_to_slurm"
    assert exp["slurm_job_id"] == "9991"
    assert len(slurm_submit_stubs) == 1


def test_interactive_review_leaves_for_manual_review_when_reviewer_declines(
    tmp_path, slurm_submit_stubs
):
    model = ScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(
        tmp_path,
        model,
        interactive_slurm_review=True,
        slurm_review_prompt=lambda payload: "skip",
    ).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "reviewer chose not to submit" in exp["reason"]
    assert slurm_submit_stubs == []  # never reached submit_job


def test_interactive_review_treats_any_non_submit_answer_as_skip(tmp_path, slurm_submit_stubs):
    # slurm_review_prompt's contract is "submit", or anything else means
    # skip — a stray keystroke must never be read as approval.
    model = ScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(
        tmp_path,
        model,
        interactive_slurm_review=True,
        slurm_review_prompt=lambda payload: "please wait",
    ).run(_planner_output([_plan("H1", complexity="high")]))

    assert result["experiments"][0]["status"] == "code_generated_not_run"
    assert slurm_submit_stubs == []


def test_interactive_review_approval_still_goes_through_the_self_review_gate(
    tmp_path, slurm_submit_stubs
):
    # A human saying "submit" is one more gate, not a bypass of the existing
    # ones — a self-review concern must still route through the fix loop
    # exactly like it does under CODER_AUTO_SUBMIT_SLURM.
    model = ScriptedChatModel(
        codegen=[_codegen_response()],
        self_review=[
            json.dumps({"looks_correct": False, "concerns": ["ignores the plan's dataset"]}),
            _clean_review(),
        ],
        fix=[_codegen_response(GOOD_SECTIONS)],
    )
    result = _agent(
        tmp_path,
        model,
        interactive_slurm_review=True,
        slurm_review_prompt=lambda payload: "submit",
    ).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "submitted_to_slurm"
    assert exp["fix_history"][0]["error_source"] == "self_review"


def test_interactive_review_approval_still_respects_the_concurrent_job_cap(
    tmp_path, slurm_submit_stubs, monkeypatch
):
    monkeypatch.setattr(slurm_submit, "count_running_jobs", lambda user=None: 4)
    model = ScriptedChatModel(codegen=[_codegen_response()], self_review=[_clean_review()])
    result = _agent(
        tmp_path,
        model,
        interactive_slurm_review=True,
        slurm_review_prompt=lambda payload: "submit",
    ).run(_planner_output([_plan("H1", complexity="high")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "already queued" in exp["reason"]
    assert slurm_submit_stubs == []


def test_interactive_review_prompt_sees_the_generated_code_and_hypothesis_id(
    tmp_path, slurm_submit_stubs
):
    seen = {}

    def capture(payload):
        seen.update(payload)
        return "skip"

    model = ScriptedChatModel(codegen=[_codegen_response()])
    _agent(tmp_path, model, interactive_slurm_review=True, slurm_review_prompt=capture).run(
        _planner_output([_plan("H1", complexity="high")])
    )

    assert seen["hypothesis_id"] == "H1"
    assert "estimated_complexity is 'high'" in seen["why_unrunnable"]
    assert "def load_data" in seen["run_py"]
    assert seen["sbatch_path"].endswith("run.sbatch")


def test_default_slurm_review_prompt_shows_the_code_and_reads_submit_from_stdin(
    monkeypatch, capsys
):
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    decision = _default_slurm_review_prompt(
        {
            "hypothesis_id": "H1",
            "why_unrunnable": "experiment needs a GPU, none detected in this environment",
            "run_py": "\n".join(f"line {i}" for i in range(60)),
            "code_path": "experiments/H1",
            "sbatch_path": "experiments/H1/run.sbatch",
        }
    )
    assert decision == "submit"
    out = capsys.readouterr().out
    assert "H1" in out
    assert "more lines" in out  # 60 lines exceeds the preview cap, so it's truncated


def test_default_slurm_review_prompt_treats_a_blank_answer_as_skip(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    decision = _default_slurm_review_prompt(
        {
            "hypothesis_id": "H1",
            "why_unrunnable": "x",
            "run_py": "one line\n",
            "code_path": "experiments/H1",
            "sbatch_path": "experiments/H1/run.sbatch",
        }
    )
    assert decision == "skip"


# -- Starter-program selection threading into the codegen/fix prompts and output --
# starters.select_starter is unit-tested directly in test_coder_starters.py; these
# tests only check the wiring: that a matching starter's reference code reaches the
# model and that its id is recorded on the finished result.


def _classification_plan(hid: str) -> dict:
    return {
        **_plan(hid),
        "design": "comparative benchmark",
        "methods": [
            {"name": "logistic regression", "description": "d", "reused_from_literature": True}
        ],
    }


def test_codegen_prompt_includes_matching_starter_reference(tmp_path):
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    _agent(tmp_path, model).run(_planner_output([_classification_plan("H1")]))

    prompt = model.prompts_by_kind["codegen"][0]
    assert "pre-validated reference program" in prompt
    assert "Supervised classification on structured/tabular features" in prompt
    assert "_train_logistic_regression" in prompt


def test_fix_prompt_includes_the_same_starter_reference(tmp_path):
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)],
        fix=[_codegen_response(GOOD_SECTIONS)],
    )
    _agent(tmp_path, model).run(_planner_output([_classification_plan("H1")]))

    fix_prompt = model.prompts_by_kind["fix"][0]
    assert "pre-validated reference program" in fix_prompt
    assert "Supervised classification on structured/tabular features" in fix_prompt


def test_no_matching_starter_omits_the_reference_block(tmp_path):
    """_plan()'s default design/methods don't match any starter's keywords —
    the prompt should read exactly as it did before this feature existed."""
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    _agent(tmp_path, model).run(_planner_output([_plan("H1")]))

    prompt = model.prompts_by_kind["codegen"][0]
    assert "pre-validated reference program" not in prompt


def test_starter_used_recorded_on_completed_result(tmp_path):
    model = ScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model).run(_planner_output([_classification_plan("H1")]))

    assert result["experiments"][0]["starter_used"] == "classification"


def test_starter_used_is_empty_string_when_nothing_matches(tmp_path):
    model = ScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1")]))

    assert result["experiments"][0]["starter_used"] == ""


# -- execution-failure classification and the repairs it routes to -----------------------
#
# The failure these exist for is in coder_agent_summary_20260819T172608Z.json:
# three fix attempts, three regenerations, and `ModuleNotFoundError: No module
# named 'pandas'` returned verbatim each time. Regenerating the source could not
# have installed anything, and nothing in the loop knew the difference.


def test_error_source_lists_stay_in_sync():
    """schema.VALID_ERROR_SOURCES and coder_agent._ERROR_STAGE_ORDER must match.

    _cleared_previous_error reads the order list as the definition of "a later
    stage", so a source present in one and missing from the other silently
    scores every regeneration as having made no progress — a failure that shows
    up as a degraded fix loop rather than as an error. AGENTS.md calls this out;
    this test is what actually enforces it.
    """
    from research_pipeline.agents.coder.coder_agent import _ERROR_STAGE_ORDER

    assert set(schema.VALID_ERROR_SOURCES) == set(_ERROR_STAGE_ORDER)
    assert len(_ERROR_STAGE_ORDER) == len(set(_ERROR_STAGE_ORDER)), "no duplicates"


def test_missing_package_is_classified_as_an_environment_problem():
    d = diagnose.classify_execution_failure(
        "run.py exited with code 1: Traceback (most recent call last):\n"
        '  File "/experiments/H1/run.py", line 30, in <module>\n'
        "    import pandas as pd\n"
        "ModuleNotFoundError: No module named 'pandas'"
    )
    assert d.error_source == "missing_dependency"
    assert d.route == diagnose.ROUTE_ENV
    assert d.module == "pandas" and d.package == "pandas"
    assert d.route != diagnose.ROUTE_REGENERATE, "the code was never the problem"


def test_import_alias_resolves_to_its_distribution():
    d = diagnose.classify_execution_failure("ModuleNotFoundError: No module named 'sklearn'")
    assert d.package == "scikit-learn"


def test_dead_import_goes_to_the_model_and_never_to_the_installer():
    """Installing `pymc` cannot satisfy `import pymc3`.

    The install succeeds, the code is re-run unchanged, and the identical error
    returns — an install that reports success while the loop makes no progress.
    Only new code can fix this, so it must carry the replacement API with it.
    """
    d = diagnose.classify_execution_failure("ModuleNotFoundError: No module named 'pymc3'")

    assert d.error_source == "obsolete_dependency"
    assert d.route == diagnose.ROUTE_REGENERATE
    assert d.package == "pymc"
    assert "import pymc as pm" in d.guidance


def test_missing_native_library_is_terminal():
    d = diagnose.classify_execution_failure(
        "OSError: libcudart.so.12: cannot open shared object file: No such file or directory"
    )
    assert d.error_source == "missing_system_library"
    assert d.route == diagnose.ROUTE_TERMINAL


@pytest.mark.parametrize(
    "message",
    [
        "execution timed out after 300s",
        "run.py exited with code 1: torch.cuda.OutOfMemoryError: CUDA out of memory.",
        "run.py exited with code 1: MemoryError",
    ],
)
def test_resource_failures_route_to_downscaling(message):
    d = diagnose.classify_execution_failure(message)
    assert d.error_source == "resource_limit"
    assert d.route == diagnose.ROUTE_DOWNSCALE


def test_an_ordinary_bug_is_still_a_plain_run_experiment_failure():
    """The new branches must not swallow the case they sit in front of."""
    d = diagnose.classify_execution_failure(
        "run.py exited with code 1: Traceback (most recent call last):\n"
        "IndexError: index 5 is out of bounds for axis 0 with size 3"
    )
    assert d.error_source == "run_experiment"
    assert d.route == diagnose.ROUTE_REGENERATE


def test_downscale_halves_cost_knobs_and_leaves_other_numbers_alone():
    code = "draws = 4000\nchains = 8\nbatch_size = 256\nYEAR = 2020\nthreshold = 0.05\n"
    shrunk, changes = repair.downscale(code)

    assert "draws = 2000" in shrunk
    assert "chains = 4" in shrunk
    assert "batch_size = 128" in shrunk
    assert "YEAR = 2020" in shrunk, "a year is not a cost knob"
    assert "threshold = 0.05" in shrunk
    assert len(changes) == 3


def test_downscale_respects_its_floors():
    shrunk, changes = repair.downscale("chains = 2\ndraws = 250\n")
    assert changes == [] and shrunk == "chains = 2\ndraws = 250\n"


def test_install_into_env_prefers_uv_and_reports_failure_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="No matching distribution found")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    ok, detail = sandbox.install_into_env(tmp_path / "bin" / "python", ["nope"])

    assert ok is False
    assert "No matching distribution" in detail
    assert calls[0][:3] == ["uv", "pip", "install"]


def test_install_for_falls_back_to_the_raw_import_name(tmp_path, monkeypatch):
    """The alias table has gaps; one failed install is a cheap way to cover them."""
    attempted = []

    def fake_install(python_executable, packages):
        attempted.append(list(packages))
        return (packages != ["scikit-learn"], "detail")

    monkeypatch.setattr(sandbox, "install_into_env", fake_install)
    d = diagnose.classify_execution_failure("ModuleNotFoundError: No module named 'sklearn'")

    ok, detail = repair.install_for(tmp_path / "python", d)

    assert attempted == [["scikit-learn"], ["sklearn"]]
    assert ok and "sklearn" in detail


def test_a_missing_package_is_installed_and_the_code_is_never_regenerated(tmp_path, monkeypatch):
    """The regression for coder_agent_summary_20260819T172608Z.json.

    One codegen call, one install, and the second execution runs the *same*
    file. Any regeneration here would mean the loop still believes a missing
    package is a defect in the generated source.
    """
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response()})
    agent = _agent(tmp_path, fake_model, network_check=lambda: True)

    executed_sources = []
    installed = []
    attempts = {"n": 0}

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        attempts["n"] += 1
        executed_sources.append(Path(run_script).read_text())
        if attempts["n"] == 1:
            return False, (
                "run.py exited with code 1: Traceback (most recent call last):\n"
                "    import pandas as pd\n"
                "ModuleNotFoundError: No module named 'pandas'"
            )
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    def fake_install(python_executable, packages):
        installed.append(list(packages))
        return True, "installed"

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(sandbox, "install_into_env", fake_install)
    # A real install makes the module importable; this fake stands in for that,
    # which the agent verifies rather than trusting the installer's exit code.
    monkeypatch.setattr(sandbox, "module_importable", lambda *a, **k: True)

    result = agent.run(_planner_output([_plan("H1", complexity="low")]))
    exp = result["experiments"][0]

    assert exp["status"] == "completed", exp.get("reason")
    assert installed == [["pandas"]], "the missing package was installed, once"
    assert attempts["n"] == 2, "one failed run, one re-run"
    assert executed_sources[0] == executed_sources[1], "the re-run used the SAME code"
    assert exp["fix_attempts"] == 0, "an install is not a fix attempt"
    assert len(_codegen_calls(fake_model)) == 1, "the model was asked for code once and never again"


def test_a_missing_package_with_no_network_says_so_instead_of_regenerating(tmp_path, monkeypatch):
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response()})
    agent = _agent(tmp_path, fake_model, network_check=lambda: False)

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        return False, "ModuleNotFoundError: No module named 'geopandas'"

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)

    result = agent.run(_planner_output([_plan("H1", complexity="low")]))
    exp = result["experiments"][0]

    sources = [entry["error_source"] for entry in exp["fix_history"]]
    assert sources and all(s == "missing_dependency" for s in sources)
    assert "No network access" in exp["fix_history"][0]["error_summary"]


def test_a_timed_out_experiment_is_shrunk_before_the_model_is_asked(tmp_path, monkeypatch):
    sections = {**GOOD_SECTIONS, "configuration": "draws = 4000\nchains = 8\n"}
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response(sections=sections)})
    agent = _agent(tmp_path, fake_model, network_check=lambda: True)

    seen = []

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        source = Path(run_script).read_text()
        seen.append(source)
        if "draws = 4000" in source:
            return False, "execution timed out after 120s"
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)

    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    assert result["experiments"][0]["status"] == "completed", result["experiments"][0].get("reason")
    assert len(seen) == 2, "one timed-out run, one shrunk re-run"
    assert "draws = 2000" in seen[1] and "chains = 4" in seen[1]
    assert len(_codegen_calls(fake_model)) == 1, "downscaling costs no model call"


def test_plausibility_catches_a_nan_nested_inside_a_credible_interval():
    """A metric is often a container — [low, high] for an interval, a dict per group.

    The scalar checks cannot reach inside one, so a posterior mean could look
    perfectly healthy beside an interval whose bounds are NaN.
    """
    findings = sandbox.check_results_plausibility(
        {"posterior_mean": -0.108, "credible_interval": [float("nan"), 1.2]}
    )
    assert any("credible_interval" in f for f in findings)


def test_plausibility_catches_an_infinity_nested_in_a_per_group_dict():
    findings = sandbox.check_results_plausibility(
        {"beta": 0.4, "by_region": {"north": 0.2, "south": float("inf")}}
    )
    assert any("by_region" in f for f in findings)


def test_plausibility_still_accepts_healthy_nested_metrics():
    findings = sandbox.check_results_plausibility(
        {"posterior_mean": -0.108, "credible_interval": [-0.31, 0.09], "n": 400}
    )
    assert findings == []


# -- the provenance gate ------------------------------------------------------------------
#
# A generated experiment that invents its inputs still produces metrics and a
# meets_success_criteria flag, and writer_agent maps False to "refuted" — so
# without this gate a run on synthesized data gets written up as a refutation of
# a hypothesis that was never tested.


def test_restricted_sources_become_labelled_surrogates():
    sources = provenance.resolve(
        ["CMS Medicare Hospital Claims for cardiovascular admissions"], network_available=True
    )
    assert sources[0].kind == provenance.KIND_SURROGATE
    assert "Data Use Agreement" in sources[0].reason


def test_a_source_needing_an_api_key_is_a_surrogate_until_the_key_exists(monkeypatch):
    """Public is not the same as fetchable.

    EPA AQS is open data behind free registration; without the key every request
    is a 401, which regenerating the code cannot fix. Offering it as fetchable
    would spend fix attempts on an unfixable failure.
    """
    monkeypatch.delenv("AQS_EMAIL", raising=False)
    monkeypatch.delenv("AQS_KEY", raising=False)
    sources = provenance.resolve(["EPA Air Quality System PM2.5"], network_available=True)

    assert sources[0].kind == provenance.KIND_SURROGATE
    assert "AQS_KEY" in sources[0].reason and "signup" in sources[0].reason


def test_the_same_source_is_real_once_its_key_is_set(monkeypatch):
    monkeypatch.setenv("AQS_EMAIL", "someone@example.org")
    monkeypatch.setenv("AQS_KEY", "test-key")
    sources = provenance.resolve(["EPA Air Quality System PM2.5"], network_available=True)

    assert sources[0].kind == provenance.KIND_REAL_DOWNLOAD
    block = provenance.prompt_block(sources)
    assert "os.environ['AQS_KEY']" in block
    assert "test-key" not in block, "the value itself must never reach a prompt"


def test_a_staged_file_beats_a_restricted_source(tmp_path):
    staging = tmp_path / "staged"
    staging.mkdir()
    (staging / "medicare_claims_2020.csv").write_text("a,b\n1,2\n")
    sources = provenance.resolve(
        ["CMS Medicare claims for admissions"], staging_dir=staging, network_available=True
    )
    assert sources[0].kind == provenance.KIND_REAL_LOCAL


def test_the_verdict_is_withheld_as_unknown_not_false(monkeypatch):
    """The distinction the Writer actually reads.

    writer_agent maps False to "refuted" and "unknown" to "inconclusive", so
    returning False on synthetic data would publish a refutation.
    """
    stamped = provenance.apply_to_results(
        {"metrics": {"beta": 0.42}, "meets_success_criteria": True},
        [provenance.DataSource(name="cms", kind=provenance.KIND_SURROGATE)],
    )
    assert stamped["meets_success_criteria"] == "unknown"
    assert stamped["meets_success_criteria"] is not False
    assert stamped["model_reported_meets_success_criteria"] is True
    assert stamped["metrics"] == {"beta": 0.42}, "the numbers are still reported"


def test_the_writer_reads_a_withheld_verdict_as_inconclusive():
    """Asserts the far end of the contract, not just the Coder's field."""
    from research_pipeline.agents.writer import writer_agent as writer

    stamped = provenance.apply_to_results(
        {"metrics": {"beta": 0.42}, "meets_success_criteria": True},
        [provenance.DataSource(name="cms", kind=provenance.KIND_SURROGATE)],
    )
    experiment = {"hypothesis_id": "H1", "status": "completed", "results": stamped}
    verdict, reason = writer.compute_hypothesis_verdict("H1", {"H1": experiment})

    assert verdict == "inconclusive"
    assert "unknown" in reason

    # And the contrast that makes the choice of "unknown" load-bearing: had the
    # gate returned False, the same function would publish a refutation.
    refuting = {**stamped, "meets_success_criteria": False}
    assert (
        writer.compute_hypothesis_verdict("H1", {"H1": {**experiment, "results": refuting}})[0]
        == "refuted"
    )


def test_a_fully_real_input_set_keeps_its_verdict():
    results = {"metrics": {"beta": 0.42}, "meets_success_criteria": True}
    stamped = provenance.apply_to_results(
        results, [provenance.DataSource(name="acs", kind=provenance.KIND_REAL_DOWNLOAD)]
    )
    assert stamped is results, "a real-data run is passed through untouched"


def test_no_resolvable_inputs_is_not_evidence_of_real_ones():
    assert provenance.verdict([]) == provenance.VERDICT_SURROGATE


def test_the_description_is_not_split_into_a_phantom_input():
    """data_requirements.description names the *derived* dataset, not an input."""
    parts = provenance.split_requirements(
        "EPA AQS for PM2.5; CMS claims for admissions; US Census ACS for deprivation",
        "Integrated neighbourhood-level panel linking monitors to tracts",
    )
    assert len(parts) == 3
    assert not any("Integrated neighbourhood" in p for p in parts)


# -- the no-progress stop -----------------------------------------------------------------


def test_identical_failures_are_counted_ignoring_line_numbers_and_paths():
    """Same bug, different run: the line number and the temp path move, the failure doesn't."""
    history = [
        {
            "error_source": "run_experiment",
            "error_summary": "/tmp/a/run.py line 30: boom at 0x7f01",
        },
        {
            "error_source": "run_experiment",
            "error_summary": "/tmp/b/run.py line 47: boom at 0x9e22",
        },
    ]
    assert _identical_failure_streak(history) == 2


def test_different_bugs_at_the_same_stage_are_not_a_streak():
    """error_source alone is too coarse to stop a run on.

    A model fixing one bug into a different one is making progress, even though
    both surface as `run_experiment`.
    """
    history = [
        {"error_source": "run_experiment", "error_summary": "ZeroDivisionError: division by zero"},
        {"error_source": "run_experiment", "error_summary": "IndexError: index out of range"},
    ]
    assert _identical_failure_streak(history) == 1


def test_the_fix_loop_gives_up_after_three_identical_failures(tmp_path):
    """Rather than spending the rest of the budget re-deriving the same error."""
    agent = _agent(tmp_path, FakeChatModel({}))
    entry = {"error_source": "run_experiment", "error_summary": "ValueError: shapes do not align"}

    state = {
        "current_outcome": {"error_source": "run_experiment"},
        "current_attempt": 1,
        "current_fix_history": [entry, entry, entry],
    }
    assert agent._route_after_attempt(state) == "give_up"

    state["current_fix_history"] = [entry, entry]
    assert agent._route_after_attempt(state) == "regenerate", "two is a repeat, not yet a loop"


def test_venv_root_places_the_venv_off_the_quota_bearing_filesystem(tmp_path, monkeypatch):
    """On Barkla this is localscratch: no inode quota, node-local, disposable.

    A venv is thousands of small files and scratch/fastscratch cap inodes at
    300k/500k, so one per experiment is a real cost on a shared filesystem that
    also holds the results worth keeping.
    """
    experiment_dir = tmp_path / "results" / "H1"
    experiment_dir.mkdir(parents=True)
    requirements = experiment_dir / "requirements.txt"
    requirements.write_text("some-package\n")
    venv_root = tmp_path / "localscratch"

    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    created = []

    def fake_run(cmd, **kwargs):
        if "-c" in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="ImportError")
        if "venv" in cmd:
            created.append(cmd[-1])
            Path(cmd[-1], "bin").mkdir(parents=True, exist_ok=True)
            Path(cmd[-1], "bin", "python").touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    python_exec, error = sandbox.ensure_experiment_env(
        experiment_dir, requirements, network_available=True, venv_root=venv_root
    )

    assert error is None
    assert python_exec == venv_root / "H1" / ".venv" / "bin" / "python"
    assert not (experiment_dir / ".venv").exists(), "the results directory stays free of venv files"


def test_a_missing_store_dependency_degrades_instead_of_ending_the_run(monkeypatch, caplog):
    """The fix-pattern store is an enhancement; losing it must not cost a run.

    Barkla job 10279024 completed Literature, Hypothesis and the Planner, then
    exited at the Coder because this optional feature's optional dependency was
    not installed — discarding twenty minutes of upstream work already on disk.
    """
    import logging

    from langgraph.store.memory import InMemoryStore

    from research_pipeline.agents.coder import fix_pattern_store

    monkeypatch.setattr(fix_pattern_store, "_backend", lambda: "sqlite")
    monkeypatch.setattr(
        fix_pattern_store,
        "_sqlite_store",
        lambda: (_ for _ in ()).throw(
            fix_pattern_store.MissingStoreDependency("needs its optional dependencies (foo)")
        ),
    )
    fix_pattern_store.reset_store()

    with caplog.at_level(logging.WARNING):
        store = fix_pattern_store.get_store()

    assert isinstance(store, InMemoryStore), "the run continues, without persistence"
    assert "will not persist" in caplog.text
    assert "optional dependencies" in caplog.text, "the remedy is named, not swallowed"
    fix_pattern_store.reset_store()


def test_a_mistyped_backend_name_still_fails_loudly(monkeypatch):
    """Degrading is for a packaging gap, not for a configuration mistake.

    An unknown backend is a typo the caller can fix in seconds; falling back
    silently would hide it behind a store that quietly never persists.
    """
    from research_pipeline.agents.coder import fix_pattern_store

    monkeypatch.setattr(fix_pattern_store, "_backend", lambda: "sqlyte")
    fix_pattern_store.reset_store()
    with pytest.raises(SystemExit, match="Unknown CODER_FIX_STORE_BACKEND"):
        fix_pattern_store.get_store()
    fix_pattern_store.reset_store()


def test_import_names_are_mapped_before_a_package_installer_sees_them():
    """Barkla job 10279165 died here.

    extract_third_party_imports returns *import* names by design, and those were
    installed verbatim. That is fine for numpy and pandas and fatal for sklearn:
    the `sklearn` distribution on PyPI is a deprecation shim that fails the
    install on purpose. uv printed the answer in its own hint.
    """
    assert sandbox.installable_name("sklearn") == "scikit-learn"
    assert sandbox.installable_name("cv2") == "opencv-python-headless"
    assert sandbox.installable_name("PIL") == "pillow"
    assert sandbox.installable_name("numpy") == "numpy", "unmapped names pass through"
    assert sandbox.installable_name("") == "", "blank lines survive untouched"
    assert sandbox.installable_name("# a comment") == "# a comment"


def test_mapping_a_requirement_keeps_its_version_pin():
    assert sandbox.installable_name("sklearn>=1.3") == "scikit-learn>=1.3"
    assert sandbox.installable_name("sklearn==1.3.0") == "scikit-learn==1.3.0"


def test_extracted_imports_reach_the_installer_as_distribution_names(tmp_path, monkeypatch):
    """The end-to-end version: what actually lands in the file uv installs from."""
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("")  # the model declared nothing, as in job 10279165
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/uv")
    venv_python = tmp_path / ".venv" / "bin" / "python"

    def fake_run(cmd, **kwargs):
        if "-c" in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="ImportError")
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    _python, error = sandbox.ensure_experiment_env(
        tmp_path,
        requirements,
        network_available=True,
        extra_requirements=["numpy", "pandas", "sklearn"],
    )

    assert error is None
    installed_from = (tmp_path / ".resolved_requirements.txt").read_text().split()
    assert "scikit-learn" in installed_from
    assert "sklearn" not in installed_from, "the shim name must never reach the installer"
    assert "numpy" in installed_from and "pandas" in installed_from


def test_an_install_that_does_not_make_the_import_work_is_not_retried(tmp_path, monkeypatch):
    """An installer's exit code is not proof of repair.

    Barkla job 10279290: `uv pip install pandas` returned 0 six times in a row —
    it believed pandas was already present for that interpreter — while the
    experiment went on failing to import it. Trusting the exit code turned one
    unfixable environment into six wasted installs per attempt, then spent the
    code budget on source that was never wrong.
    """
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response()})
    agent = _agent(tmp_path, fake_model, network_check=lambda: True)

    installs = []
    runs = []

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        runs.append(1)
        return False, "ModuleNotFoundError: No module named 'pandas'"

    def fake_install(python_executable, packages):
        installs.append(list(packages))
        return True, "installed"  # succeeds, and changes nothing

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(sandbox, "install_into_env", fake_install)
    monkeypatch.setattr(sandbox, "module_importable", lambda *a, **k: False)

    result = agent.run(_planner_output([_plan("H1", complexity="low")]))
    exp = result["experiments"][0]

    # The invariant that matters: one install attempt per execution, never a
    # loop within one. Before this check the agent spent its whole
    # CODER_MAX_ENV_REPAIRS budget — six installs — on every single execution.
    assert len(installs) == len(runs), f"{len(installs)} installs for {len(runs)} runs: {installs}"
    assert len(installs) <= agent.max_fix_attempts + 1

    summaries = " ".join(e["error_summary"] for e in exp["fix_history"])
    assert "still not importable" in summaries, "the message must name the real problem"
    assert exp["status"] != "completed"


# -- huggingface_client.py: the appraisal endpoints --------------------------------------


def test_search_datasets_asks_for_the_full_record(monkeypatch):
    # The prefilter ranks on tags/downloads/cardData, so the search has to ask
    # for them — the old lookup used only "id" and threw the rest away.
    seen: list = []
    _fake_hf(monkeypatch, {"api/datasets": _FakeResponse([{"id": "a/b"}])}, recorder=seen)

    assert huggingface_client.search_datasets("sleep") == [{"id": "a/b"}]
    assert seen[0][1]["full"] == "true"


def test_dataset_info_returns_the_hub_record(monkeypatch):
    payload = {"id": "a/b", "sha": "deadbeef", "cardData": {"license": "mit"}, "downloads": 7}
    _fake_hf(monkeypatch, {"api/datasets/a/b": _FakeResponse(payload)})

    assert huggingface_client.dataset_info("a/b")["sha"] == "deadbeef"


def test_dataset_info_degrades_to_an_empty_dict(monkeypatch):
    _fake_hf(monkeypatch, {"api/datasets": _FakeResponse(None, status_code=503)})

    assert huggingface_client.dataset_info("a/b") == {}


def test_dataset_size_reads_the_budget_numbers(monkeypatch):
    payload = {"size": {"dataset": {"num_rows": 5000, "num_bytes_original_files": 1234}}}
    _fake_hf(monkeypatch, {"/size": _FakeResponse(payload)})

    assert huggingface_client.dataset_size("a/b") == {"num_rows": 5000, "num_bytes": 1234}


def test_dataset_size_falls_back_through_the_byte_fields(monkeypatch):
    payload = {"size": {"dataset": {"num_rows": 10, "num_bytes_parquet_files": 99}}}
    _fake_hf(monkeypatch, {"/size": _FakeResponse(payload)})

    assert huggingface_client.dataset_size("a/b")["num_bytes"] == 99


def test_dataset_size_degrades_on_an_unexpected_shape(monkeypatch):
    _fake_hf(monkeypatch, {"/size": _FakeResponse({"size": "surprise"})})

    assert huggingface_client.dataset_size("a/b") == {}


def test_dataset_card_is_truncated(monkeypatch):
    long_card = SimpleNamespace(status_code=200, text="x" * 20_000)
    monkeypatch.setattr(huggingface_client.requests, "get", lambda *a, **k: long_card)

    card = huggingface_client.dataset_card("a/b")
    assert len(card) == huggingface_client.MAX_CARD_CHARS


def test_a_missing_card_is_an_empty_string(monkeypatch):
    monkeypatch.setattr(
        huggingface_client.requests,
        "get",
        lambda *a, **k: SimpleNamespace(status_code=404, text="not found"),
    )

    assert huggingface_client.dataset_card("a/b") == ""


def test_fetch_rows_pages_until_the_limit(monkeypatch):
    seen: list = []
    page = {"rows": [{"row": {"a": index}} for index in range(100)]}
    _fake_hf(monkeypatch, {"/rows": _FakeResponse(page)}, recorder=seen)

    rows = huggingface_client.fetch_rows("a/b", "default", "train", 200)

    assert len(rows) == 200
    assert [call[1]["offset"] for call in seen] == [0, 100]


def test_fetch_rows_stops_early_on_a_short_page(monkeypatch):
    # A short page means the split ran out; asking again just re-reads the tail.
    page = {"rows": [{"row": {"a": index}} for index in range(7)]}
    _fake_hf(monkeypatch, {"/rows": _FakeResponse(page)})

    assert len(huggingface_client.fetch_rows("a/b", "default", "train", 200)) == 7


def test_fetch_rows_degrades_to_an_empty_sample(monkeypatch):
    _fake_hf(monkeypatch, {"/rows": requests.RequestException("no route")})

    assert huggingface_client.fetch_rows("a/b", "default", "train", 50) == []


def test_describe_candidate_gathers_schema_and_metadata(monkeypatch):
    _fake_hf(
        monkeypatch,
        {
            "/is-valid": _FakeResponse({"viewer": True}),
            "/splits": _FakeResponse({"splits": [{"config": "default", "split": "train"}]}),
            "/first-rows": _FakeResponse(
                {
                    "features": [{"name": "text", "type": {"dtype": "string"}}],
                    "rows": [{"row": {"text": "hello"}}],
                }
            ),
            "/size": _FakeResponse({"size": {"dataset": {"num_rows": 5, "num_bytes": 64}}}),
            "api/datasets/a/b": _FakeResponse(
                {"id": "a/b", "sha": "cafe", "cardData": {"license": "mit"}, "tags": ["t"]}
            ),
        },
    )

    described = huggingface_client.describe_candidate("a/b")

    assert described["revision"] == "cafe"
    assert described["license"] == "mit"
    assert described["num_rows"] == 5
    assert described["columns"] == [{"name": "text", "type": "string"}]


def test_describe_candidate_gives_up_on_an_unservable_dataset(monkeypatch):
    _fake_hf(monkeypatch, {"/is-valid": _FakeResponse({"viewer": False, "preview": False})})

    assert huggingface_client.describe_candidate("a/b") == {}


def test_a_list_valued_license_takes_its_first_entry(monkeypatch):
    # cardData.license is a list about as often as it is a string.
    _fake_hf(
        monkeypatch,
        {
            "/is-valid": _FakeResponse({"viewer": True}),
            "/splits": _FakeResponse({"splits": [{"config": "default", "split": "train"}]}),
            "/first-rows": _FakeResponse(
                {"features": [{"name": "t", "type": "string"}], "rows": []}
            ),
            "/size": _FakeResponse({}),
            "api/datasets/a/b": _FakeResponse({"id": "a/b", "cardData": {"license": ["mit"]}}),
        },
    )

    assert huggingface_client.describe_candidate("a/b")["license"] == "mit"


def test_rows_url_percent_encodes_the_namespace_slash():
    url = huggingface_client.rows_url_for("acme/sleep-survey", "default", "train")

    assert "dataset=acme%2Fsleep-survey" in url
    assert "&config=default&split=train" in url


def test_download_degrades_when_huggingface_hub_is_absent(monkeypatch, tmp_path):
    # The real failure on a plain `uv sync`: the extra isn't installed.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    assert huggingface_client.download_dataset("a/b", "rev", tmp_path / "dest") is None


def test_normalize_reads_jsonl_and_csv_into_one_file(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "train.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    (source / "extra.csv").write_text("a\n3\n", encoding="utf-8")
    dest = tmp_path / "data.jsonl"

    summary = huggingface_client.normalize_to_jsonl(source, dest, max_rows=10)

    assert summary["rows_written"] == 3
    assert summary["columns"] == ["a"]
    assert dest.read_text(encoding="utf-8").count("\n") == 3


def test_normalize_reads_a_json_array_as_well_as_json_lines(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "data.json").write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
    dest = tmp_path / "out.jsonl"

    assert huggingface_client.normalize_to_jsonl(source, dest, max_rows=10)["rows_written"] == 2


def test_normalize_honours_the_row_cap(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "train.jsonl").write_text("".join(f'{{"a": {i}}}\n' for i in range(50)), "utf-8")
    dest = tmp_path / "out.jsonl"

    assert huggingface_client.normalize_to_jsonl(source, dest, max_rows=5)["rows_written"] == 5


def test_normalize_with_nothing_readable_returns_empty(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "README.md").write_text("just a card", encoding="utf-8")

    assert huggingface_client.normalize_to_jsonl(source, tmp_path / "out.jsonl", 10) == {}
