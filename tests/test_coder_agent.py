import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.store.memory import InMemoryStore

from research_pipeline.agents.coder import (
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
    _ERROR_STAGE_ORDER,
    _STRUCTURAL_ERROR_SOURCES,
    CoderAgent,
    CoderAgentError,
    _best_candidate,
    _compact_json,
    _consecutive_error_streak,
    _default_slurm_review_prompt,
    _estimate_tokens,
    _failure_signature,
    _identical_failure_streak,
    _parse_assumptions,
    _parse_bool_text,
    _recursion_limit_for,
)
from research_pipeline.agents.coder.schema import (
    ERROR_SUMMARY_MAX_CHARS,
    SchemaValidationError,
    validate_output,
)
from research_pipeline.config import settings
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


def _venv_layout(root: Path) -> tuple[Path, Path]:
    """A venv whose bin/python is a symlink to a base interpreter, which is what
    both `uv venv` and the stdlib `venv` module actually create."""
    base = root / "base" / "bin" / "python3.12"
    base.parent.mkdir(parents=True)
    base.write_text("")
    venv_python = root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base)
    return venv_python, base


def test_the_interpreter_path_is_absolute_without_dereferencing_the_venv(tmp_path):
    """Barkla jobs 10410771 and 10410847 both ended on "the package installed
    successfully but 'numpy' is still not importable". A venv's bin/python is a
    symlink to the base interpreter, and Path.resolve() follows it — handing
    back an interpreter that has no idea the venv exists and cannot see
    anything installed into it. Measured on Barkla:

        ./.venv/bin/python -c "import numpy"        -> OK 2.5.2
        $(resolve .venv/bin/python) -c "import numpy" -> ModuleNotFoundError

    Absolute is required (cwd is the experiment dir); dereferenced is fatal.
    """
    venv_python, base = _venv_layout(tmp_path)
    path = sandbox._interpreter_path(venv_python)

    assert Path(path).is_absolute()
    assert path == str(venv_python), "the venv symlink itself must be preserved"
    assert path != str(base), "resolving to the base interpreter discards the venv"


def test_both_subprocess_callers_use_the_same_undereferenced_path(tmp_path, monkeypatch):
    """module_importable and run_experiment must agree, and both must go through
    the venv — if they disagree, the check answers about a different interpreter
    than the one that runs the code."""
    seen: list[str] = []

    def fake_run(command, **kwargs):
        seen.append(command[0])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    venv_python, base = _venv_layout(tmp_path)
    experiment_dir = tmp_path / "H1"
    experiment_dir.mkdir()
    (experiment_dir / "run.py").write_text("")

    sandbox.module_importable(venv_python, "numpy", experiment_dir)
    sandbox.run_experiment(venv_python, experiment_dir / "run.py", experiment_dir, 60)

    assert seen[0] == seen[1] == str(venv_python)
    assert str(base) not in seen


def test_a_relative_interpreter_path_is_still_made_absolute(tmp_path, monkeypatch):
    # The reason absolute paths were wanted in the first place:
    # CODER_EXPERIMENTS_DIR defaults to a relative "experiments", and cwd below
    # is that same relative directory, so a relative path would be re-resolved
    # against the subprocess's own cwd — experiments/H1/experiments/H1/...
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

    sandbox.module_importable(venv_python, "numpy", experiment_dir)

    assert Path(seen[0]).is_absolute()
    assert "experiments/H1/experiments" not in seen[0]


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
    # A fresh, isolated in-memory store per call — never the real
    # fix_pattern_store.get_store() singleton (CODER_FIX_STORE_BACKEND
    # defaults to sqlite, which would write a real file). A test that wants
    # two agents to share recorded fix patterns passes its own fix_store=
    # explicitly, overriding this default.
    kwargs.setdefault("fix_store", InMemoryStore())
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
    #
    # The syntax error is put in run_experiment_function rather than in
    # evaluate_function so that the targeted regeneration it triggers asks for
    # run_experiment_function back — which is the section carrying the raising
    # body the third attempt needs. Targeting evaluate_function would keep the
    # working run_experiment from the attempt before and the run would succeed,
    # testing nothing about streaks.
    broken_run_experiment = {
        **GOOD_SECTIONS,
        "run_experiment_function": "def run_experiment(data, model:\n    pass\n",
    }
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)],
        fix=[_codegen_response(broken_run_experiment), _codegen_response(RAISING_SECTIONS)],
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


def test_unparseable_format_from_regeneration_is_caught_and_ends_the_plan(tmp_path):
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
    # invalid_format is structural, so it draws on CODER_MAX_STRUCTURAL_RETRIES
    # (2 by default) rather than on the fix budget, and exhausting that budget
    # ends the plan — it does not go on to spend the fix attempts too. Neither
    # the run nor the plan is lost, which is what this test exists to guard.
    assert [entry["error_source"] for entry in exp["fix_history"]] == [
        "run_experiment",
        "invalid_format",
        "invalid_format",
    ]
    assert [entry["attempt"] for entry in exp["fix_history"]] == [1, 2, 3]


# ── The structural budget ─────────────────────────────────────────────────────
# A malformed or hollow response is the model failing to return a program, not a
# defect in one. Nothing was rendered, provisioned or executed, so it draws on
# its own budget rather than on the one that exists for debugging code.
# See coder_agent._STRUCTURAL_ERROR_SOURCES.

INCOMPLETE_SECTIONS = {**GOOD_SECTIONS, "evaluate_function": ""}
RAISING_EVALUATE_SECTIONS = {
    **GOOD_SECTIONS,
    "evaluate_function": "def evaluate(experiment_output):\n    raise RuntimeError('boom')\n",
}


def test_structural_failures_do_not_consume_the_fix_budget(tmp_path):
    # Two malformed responses, then a real bug, then the fix for it — with a fix
    # budget of exactly one. The two structural failures have to cost nothing
    # from that budget or the real bug never gets its attempt, which is the
    # production failure this exists for: a run that spent all three attempts on
    # the model omitting a section had none left for the defect underneath.
    model = ScriptedChatModel(
        codegen=[_codegen_response(INCOMPLETE_SECTIONS)],
        fix=[
            _codegen_response(INCOMPLETE_SECTIONS),
            _codegen_response(RAISING_EVALUATE_SECTIONS),
            _codegen_response(GOOD_SECTIONS),
        ],
    )
    result = _agent(tmp_path, model, max_fix_attempts=1, max_structural_retries=2).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert [entry["error_source"] for entry in exp["fix_history"]] == [
        "missing_sections",
        "missing_sections",
        "run_experiment",
    ]


def test_the_old_single_budget_falls_out_when_structural_retries_are_zero(tmp_path):
    # The same scenario with the budget switched off: the first structural
    # failure spends the one fix attempt, and the second ends the plan.
    model = ScriptedChatModel(
        codegen=[_codegen_response(INCOMPLETE_SECTIONS)],
        fix=[
            _codegen_response(INCOMPLETE_SECTIONS),
            _codegen_response(RAISING_EVALUATE_SECTIONS),
            _codegen_response(GOOD_SECTIONS),
        ],
    )
    result = _agent(tmp_path, model, max_fix_attempts=1, max_structural_retries=0).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert [entry["error_source"] for entry in exp["fix_history"]] == ["missing_sections"]


def test_a_structural_failure_past_its_budget_ends_the_plan(tmp_path):
    # Exhausting the structural budget stops there rather than going on to spend
    # the fix budget as well. The fix budget measures attempts at fixing code,
    # and a model that will not return a program has produced no code to fix —
    # falling through would spend max_structural_retries + max_fix_attempts
    # regenerations to produce nothing, with the identical-failure stop unable
    # to catch it (an invalid_format summary embeds the model's own varying
    # output, so repeats don't look identical).
    model = ScriptedChatModel(
        codegen=[_codegen_response(INCOMPLETE_SECTIONS)],
        fix=[_codegen_response(INCOMPLETE_SECTIONS), _codegen_response(GOOD_SECTIONS)],
    )
    result = _agent(tmp_path, model, max_fix_attempts=2, max_structural_retries=1).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    # One structural retry, then the budget is spent and the plan ends — the two
    # fix attempts are never touched.
    assert [entry["error_source"] for entry in exp["fix_history"]] == ["missing_sections"]


def test_a_plan_that_never_returns_a_program_is_bounded_without_the_streak(tmp_path):
    # The reason exhausting the structural budget ends the plan rather than
    # falling through: an invalid_format summary embeds the model's own output,
    # so three malformed responses whose garbage *differs* are three different
    # failure signatures and the identical-failure stop never fires. Nothing but
    # the budget itself bounds this run.
    def garbage(marker: str) -> str:
        return f"===BEGIN imports===\nimport {marker}\ndef load_data(: pass  # {marker}\n"

    # Distinct garbage on every call, "codegen" included: invoke_sections' own
    # repair retry sends a prompt with no fix marker, so ScriptedChatModel routes
    # it to that bucket — and it is the repair retry's response that ends up
    # quoted in the error the summary is built from.
    model = ScriptedChatModel(
        codegen=[garbage(f"a{i}") for i in range(6)],
        fix=[garbage(f"f{i}") for i in range(3)],
    )
    result = _agent(tmp_path, model, max_fix_attempts=3, max_structural_retries=2).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert [entry["error_source"] for entry in exp["fix_history"]] == ["invalid_format"] * 2
    # Every summary differs, so no two share a signature — the streak stop could
    # not have been what stopped this.
    assert len({_failure_signature(entry) for entry in exp["fix_history"]}) == len(
        exp["fix_history"]
    )
    # Bounded at max_structural_retries + 1 attempts, not + max_fix_attempts.
    assert exp["fix_attempts"] == 2


def test_a_code_defect_never_draws_on_the_structural_budget(tmp_path):
    # compile_check is deliberately not structural: a syntax error is a real
    # defect in a real answer, and it is the failure most likely to repeat.
    model = ScriptedChatModel(
        codegen=[_codegen_response(BROKEN_SYNTAX_SECTIONS)],
        fix=[_codegen_response(BROKEN_SYNTAX_SECTIONS), _codegen_response(GOOD_SECTIONS)],
    )
    result = _agent(tmp_path, model, max_fix_attempts=1, max_structural_retries=2).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert [entry["error_source"] for entry in exp["fix_history"]] == ["compile_check"]


def test_structural_error_sources_are_a_subset_of_the_valid_ones():
    assert _STRUCTURAL_ERROR_SOURCES < schema.VALID_ERROR_SOURCES
    # A structural failure is one where nothing was rendered, provisioned or
    # executed — so every member must sort before compile_check, the first check
    # that reads a rendered run.py.
    order = _ERROR_STAGE_ORDER
    assert all(order.index(s) < order.index("compile_check") for s in _STRUCTURAL_ERROR_SOURCES)


def test_a_streak_does_not_abandon_the_attempt_that_broke_it(tmp_path):
    # Three identical compile_check failures, then a response that finally
    # produces a runnable program. fix_history describes the attempts *before*
    # the current one, so its streak is stale by exactly one — and abandoning the
    # plan on the attempt that just cleared the failure is the opposite of what
    # the no-progress stop is for.
    #
    # compile_check rather than a structural failure, deliberately: a structural
    # one would hit its own budget and end the plan before a streak of three
    # could ever form, so it could not exercise this at all.
    model = ScriptedChatModel(
        codegen=[_codegen_response(BROKEN_SYNTAX_SECTIONS)],
        fix=[
            _codegen_response(BROKEN_SYNTAX_SECTIONS),
            _codegen_response(BROKEN_SYNTAX_SECTIONS),
            _codegen_response(RAISING_EVALUATE_SECTIONS),
            _codegen_response(GOOD_SECTIONS),
        ],
    )
    result = _agent(tmp_path, model, max_fix_attempts=9).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert [entry["error_source"] for entry in exp["fix_history"]] == [
        "compile_check",
        "compile_check",
        "compile_check",
        "run_experiment",
    ]
    # The streak really did reach the stop's threshold — without the `resolved`
    # guard this run ends at the fourth entry instead of recovering.
    assert exp["fix_history"][2]["same_error_streak"] == 3
    assert exp["fix_history"][2]["resolved"] is True


def test_a_streak_that_keeps_repeating_still_stops_the_plan(tmp_path):
    # The other half of the same rule: when the current outcome did *not* clear
    # the streak, three identical failures still end the plan rather than
    # spending the rest of the budget re-deriving them.
    model = ScriptedChatModel(
        codegen=[_codegen_response(INCOMPLETE_SECTIONS)],
        fix=[_codegen_response(INCOMPLETE_SECTIONS)],
    )
    result = _agent(tmp_path, model, max_fix_attempts=9, max_structural_retries=9).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert exp["fix_attempts"] == 3, "stopped by the streak, not by either budget"


def test_recursion_limit_leaves_room_for_both_budgets():
    # Structural retries are real trips around the same cycle: unaccounted for,
    # a plan that spends them stops on the recursion limit rather than on a
    # check's verdict.
    assert _recursion_limit_for(3, 3, 2) > _recursion_limit_for(3, 3, 0)


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
    # `import pandas as pd` matters: without it the rendered run.py uses a name
    # it never binds, and sandbox.check_undefined_names — which runs earlier —
    # would (correctly) fail this on undefined_name before the data-fallback
    # check ever saw it.
    unguarded = {
        **GOOD_SECTIONS,
        "imports": "import pandas as pd",
        "load_data_function": BARE_LOAD_DATA,
    }
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
    unguarded = {
        **GOOD_SECTIONS,
        "imports": "import pandas as pd",
        "load_data_function": BARE_LOAD_DATA,
    }
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

# A load_data() that actually uses HF_DATASET_MATCH's dataset id (matching
# check_hf_dataset_usage) rather than GOOD_SECTIONS' `return None` — the tests
# below script a model that took the offered dataset, not one that silently
# ignored it (a separate scenario covered by test_coder_agent's
# ignored_available_dataset tests).
# Names the offered dataset — which is all sandbox.check_hf_dataset_usage asks
# for — without performing the fetch itself. The tests using this are about what
# the prompt offered and what the lookup was asked; the fetch is deliberately
# elided because no test in this file touches the network. (It used to call
# `requests` without importing it, which had the same effect only by accident:
# the NameError was swallowed by the fallback's own `except Exception`. Now that
# sandbox.check_undefined_names exists, that accident is a caught defect.)
GOOD_SECTIONS_WITH_HF_DATASET = {
    **GOOD_SECTIONS,
    "configuration": (
        "DATASET_ROWS_URL = (\n"
        "    'https://datasets-server.huggingface.co/rows"
        "?dataset=acme%2Fsleep-survey&config=default&split=train&offset=0&length=100'\n"
        ")\n"
    ),
    "load_data_function": (
        "def load_data():\n"
        "    # A real generated program fetches DATASET_ROWS_URL here, guarded, and\n"
        "    # falls back to rows in this shape when the fetch fails.\n"
        "    return [{'hours_slept': 7.5, 'score': 88}]\n"
    ),
}


def _recording_lookup(result):
    """A fake huggingface_lookup_fn that records the queries it was asked."""
    queries: list[str] = []

    def lookup(query):
        queries.append(query)
        return result

    return lookup, queries


def test_matched_hf_dataset_is_offered_to_the_model_with_a_rest_url(tmp_path):
    model = RecordingScriptedChatModel(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
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
        codegen=[_codegen_response(RAISING_SECTIONS)],
        fix=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)],
    )
    lookup, queries = _recording_lookup(HF_DATASET_MATCH)
    result = _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert len(queries) == 1
    assert "acme/sleep-survey" in model.prompts_by_kind["fix"][0]


def test_each_plan_gets_its_own_lookup(tmp_path):
    model = ScriptedChatModel(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    lookup, queries = _recording_lookup(HF_DATASET_MATCH)
    _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1"), _plan("H2")])
    )

    assert len(queries) == 2


def test_run_reports_a_silently_ignored_offered_dataset(tmp_path):
    # The offered dataset is real ("acme/sleep-survey"), but load_data() (via
    # the default GOOD_SECTIONS fixture) neither reads it nor declines it in
    # assumptions_made — the exact silent-third-option check_hf_dataset_usage
    # exists to catch.
    model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response()})
    lookup, _ = _recording_lookup(HF_DATASET_MATCH)
    experiments_dir = tmp_path / "experiments"
    agent = CoderAgent(
        chat_model=model,
        experiments_dir=experiments_dir,
        output_dir=tmp_path / "outputs",
        network_check=lambda: True,
        gpu_check=lambda: False,
        huggingface_lookup_fn=lookup,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert "acme/sleep-survey" in exp["reason"]
    assert "not used" in exp["reason"].lower() or "no sign of it" in exp["reason"].lower()
    assert not (experiments_dir / "H1" / "results.json").exists()


def test_run_accepts_an_explicitly_declined_offered_dataset(tmp_path):
    # HF_DATASET_USAGE_NOTE explicitly sanctions ignoring the offered dataset
    # when it doesn't fit, provided the model says so in assumptions_made —
    # that documented escape hatch must not be flagged as hollow.
    model = FakeChatModel(
        {
            '"hypothesis_id":"H1"': _codegen_response(
                assumptions=["acme/sleep-survey doesn't include the label column this task needs"]
            )
        }
    )
    lookup, _ = _recording_lookup(HF_DATASET_MATCH)
    agent = CoderAgent(
        chat_model=model,
        experiments_dir=tmp_path / "experiments",
        output_dir=tmp_path / "outputs",
        network_check=lambda: True,
        gpu_check=lambda: False,
        huggingface_lookup_fn=lookup,
    )
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

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
    agent = _agent(
        tmp_path, fake_model, network_check=lambda: True, huggingface_lookup_fn=lambda *a, **k: None
    )

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
    assert len(fake_model.calls) == 1, "the model was asked for code once and never again"


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
    agent = _agent(
        tmp_path, fake_model, network_check=lambda: True, huggingface_lookup_fn=lambda *a, **k: None
    )

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
    # Three executions: the smoke run (knobs pinned to their floor, clean), the
    # real run at full size (times out), and the halved re-run. The smoke run
    # passing is exactly what it is allowed to do — it can end an attempt early
    # but never let one skip the real execution.
    assert len(seen) == 3, "smoke run, timed-out full run, shrunk re-run"
    assert "draws = 250" in seen[0] and "chains = 2" in seen[0]
    assert "draws = 4000" in seen[1]
    assert "draws = 2000" in seen[2] and "chains = 4" in seen[2]
    assert len(fake_model.calls) == 1, "downscaling costs no model call"


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
    agent = _agent(
        tmp_path, fake_model, network_check=lambda: True, huggingface_lookup_fn=lambda *a, **k: None
    )

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


# ── Section line spans, and the undefined-name check they localize ────────────
# render_experiment_with_spans is what turns a line-numbered finding about the
# rendered run.py back into the name of the section that has to change.


def _render(sections):
    return sandbox.render_experiment_with_spans(
        hypothesis_id="H1",
        objective="o",
        design="d",
        data_description="dd",
        baseline="b",
        success_criteria="sc",
        agent_imports=sections.get("imports", ""),
        agent_configuration=sections.get("configuration", ""),
        load_data_function=sections["load_data_function"],
        build_model_function=sections["build_model_function"],
        run_experiment_function=sections["run_experiment_function"],
        evaluate_function=sections["evaluate_function"],
        agent_helpers=sections.get("helpers", ""),
    )


def test_render_with_spans_is_byte_identical_to_the_plain_renderer():
    # The two must never drift: the spans describe the file the other one
    # returns, and render_experiment_template now delegates precisely so there
    # is one splicing implementation rather than two that agree by inspection.
    rendered, _ = _render(GOOD_SECTIONS)
    assert rendered == sandbox.render_experiment_template(
        hypothesis_id="H1",
        objective="o",
        design="d",
        data_description="dd",
        baseline="b",
        success_criteria="sc",
        agent_imports=GOOD_SECTIONS["imports"],
        agent_configuration=GOOD_SECTIONS["configuration"],
        load_data_function=GOOD_SECTIONS["load_data_function"],
        build_model_function=GOOD_SECTIONS["build_model_function"],
        run_experiment_function=GOOD_SECTIONS["run_experiment_function"],
        evaluate_function=GOOD_SECTIONS["evaluate_function"],
        agent_helpers=GOOD_SECTIONS["helpers"],
    )


def test_spans_point_at_the_lines_each_section_actually_occupies():
    rendered, spans = _render(GOOD_SECTIONS)
    lines = rendered.split("\n")
    for section, (start, end) in spans.items():
        block = "\n".join(lines[start - 1 : end])
        expected = GOOD_SECTIONS[section].strip()
        # Empty sections are rendered as the template's own placeholder comment.
        assert block == expected or (not expected and block.startswith("# ("))


def test_section_for_line_returns_none_for_the_fixed_template():
    _, spans = _render(GOOD_SECTIONS)
    assert sandbox.section_for_line(spans, 1) is None  # the docstring header
    assert sandbox.section_for_line(spans, spans["evaluate_function"][0]) == "evaluate_function"


def test_check_undefined_names_flags_a_helper_that_was_never_written():
    rendered, spans = _render(
        {
            **GOOD_SECTIONS,
            "load_data_function": "def load_data():\n    return _read_the_survey()\n",
        }
    )
    findings = sandbox.check_undefined_names(rendered)
    assert len(findings) == 1
    line, message = findings[0]
    assert "_read_the_survey" in message
    assert sandbox.section_for_line(spans, line) == "load_data_function"


def test_check_undefined_names_is_clean_on_the_template_itself():
    # The fixed orchestration defines `logger` *after* the model's sections, and
    # a module-level binding is a module-level binding wherever it appears — a
    # check that got this wrong would fail every correct experiment.
    rendered, _ = _render(
        {
            **GOOD_SECTIONS,
            "load_data_function": 'def load_data():\n    logger.info("x")\n    return 1\n',
        }
    )
    assert sandbox.check_undefined_names(rendered) == []


def test_check_undefined_names_stays_quiet_on_a_star_import():
    # pyflakes reports ImportStarUsage rather than UndefinedName once a star
    # import is in scope, so an unresolvable name degrades to silence instead of
    # to a wall of findings against code that may well be correct.
    rendered, _ = _render(
        {
            **GOOD_SECTIONS,
            "imports": "from math import *",
            "load_data_function": "def load_data():\n    return sqrt(4) + tau\n",
        }
    )
    assert sandbox.check_undefined_names(rendered) == []


def test_check_undefined_names_leaves_a_syntax_error_to_compile_check():
    assert sandbox.check_undefined_names("def load_data(:\n    pass\n") == []


def test_undefined_name_routes_through_the_fix_loop_before_anything_is_provisioned(tmp_path):
    broken = {**GOOD_SECTIONS, "evaluate_function": "def evaluate(o):\n    return score(o)\n"}
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(broken)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["fix_history"][0]["error_source"] == "undefined_name"
    assert "score" in exp["fix_history"][0]["error_summary"]
    assert not (tmp_path / "experiments" / "H1" / ".venv").exists()


def test_compile_error_line_reads_lenient_compile_checks_own_format():
    _, error = sandbox.lenient_compile_check("def f(:\n    pass\n", "run.py")
    assert sandbox.compile_error_line(error) is not None
    assert sandbox.compile_error_line("run.py: something with no line number") is None


# ── The smoke pass ────────────────────────────────────────────────────────────
# A shrunken first execution that can fail an attempt early but can never let
# one skip the real run. See CoderAgent._smoke_failure.

SLOW_SECTIONS = {**GOOD_SECTIONS, "configuration": "n_rows = 100000\nepochs = 40\n"}


def test_smoke_run_ends_the_attempt_on_a_scale_independent_failure(tmp_path, monkeypatch):
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(SLOW_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    agent = _agent(tmp_path, model)
    runs = []

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        source = Path(run_script).read_text()
        runs.append((Path(run_script).name, timeout_seconds))
        if "epochs = 1\n" in source:  # the smoke copy — epochs pinned to its floor
            return False, (
                "run.py exited with code 1: Traceback (most recent call last):\n"
                '  File "run.py", line 60, in load_data\n'
                "AttributeError: 'NoneType' object has no attribute 'shape'"
            )
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    # The first attempt never reached a full-size execution at all.
    assert runs[0] == ("run_smoke.py", settings.coder_smoke_timeout_seconds)
    assert "run.py" not in [name for name, _ in runs[:1]]
    assert exp["fix_history"][0]["error_source"] == "run_experiment"
    assert "cost knobs at their minimum" in exp["fix_history"][0]["error_summary"]


def test_a_smoke_failure_the_shrinking_could_have_caused_is_re_run_at_full_size(
    tmp_path, monkeypatch
):
    # A ValueError about sample sizes is exactly what pinning a row count
    # produces. Reporting it would spend a fix attempt rewriting correct code,
    # so the real run happens anyway and decides.
    model = RecordingScriptedChatModel(codegen=[_codegen_response(SLOW_SECTIONS)])
    agent = _agent(tmp_path, model)
    runs = []

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        source = Path(run_script).read_text()
        runs.append(Path(run_script).name)
        if "epochs = 1\n" in source:  # the smoke copy
            return False, (
                "run.py exited with code 1: Traceback (most recent call last):\n"
                "ValueError: n_splits=5 cannot be greater than the number of members in each class."
            )
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)
    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    assert runs == ["run_smoke.py", "run.py"]
    assert result["experiments"][0]["status"] == "completed"
    assert result["experiments"][0]["fix_attempts"] == 0, "no model call was spent on it"


def test_a_missing_package_found_by_the_smoke_run_still_goes_to_the_installer(
    tmp_path, monkeypatch
):
    # An environment failure is not the smoke pass's to answer: installing and
    # re-running belongs to the execution loop, in one place.
    model = RecordingScriptedChatModel(codegen=[_codegen_response(SLOW_SECTIONS)])
    agent = _agent(
        tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lambda *a, **k: None
    )
    runs = []

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        runs.append(Path(run_script).name)
        if len(runs) <= 2:  # the smoke copy, then the first full run
            return False, "run.py exited with code 1:\nModuleNotFoundError: No module named 'numpy'"
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(sandbox, "install_into_env", lambda *a, **k: (True, "installed"))
    monkeypatch.setattr(sandbox, "module_importable", lambda *a, **k: True)

    result = agent.run(_planner_output([_plan("H1", complexity="low")]))

    assert runs == ["run_smoke.py", "run.py", "run.py"]
    assert result["experiments"][0]["status"] == "completed"
    assert result["experiments"][0]["fix_attempts"] == 0


def test_smoke_run_is_skipped_when_there_is_nothing_to_shrink(tmp_path, monkeypatch):
    # GOOD_SECTIONS declares no known cost knob, so a smoke run would be the
    # real run — pure duplicated cost, and a timeout on it would mean nothing.
    model = RecordingScriptedChatModel(codegen=[_codegen_response(GOOD_SECTIONS)])
    runs = []

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        runs.append(Path(run_script).name)
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)
    _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    assert runs == ["run.py"]


def test_smoke_run_can_be_turned_off(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_enable_smoke_run=False)
    model = RecordingScriptedChatModel(codegen=[_codegen_response(SLOW_SECTIONS)])
    runs = []

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        runs.append(Path(run_script).name)
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)
    _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    assert runs == ["run.py"]


def test_smoke_run_leaves_neither_its_script_nor_its_results_behind(tmp_path, monkeypatch):
    # Its numbers came from a run shrunk past the point of meaning anything, so
    # they must not be mistaken for the experiment's own result.
    model = RecordingScriptedChatModel(codegen=[_codegen_response(SLOW_SECTIONS)])
    seen_during_run = {}

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        (Path(cwd) / "results.json").write_text(json.dumps({"metrics": {"accuracy": 0.1}}))
        if Path(run_script).name == "run_smoke.py":
            seen_during_run["smoke_script_existed"] = Path(run_script).exists()
            return True, ""
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    assert seen_during_run["smoke_script_existed"] is True
    assert not (tmp_path / "experiments" / "H1" / "run_smoke.py").exists()
    # The reported metrics are the full run's, never the smoke run's.
    assert result["experiments"][0]["results"]["metrics"] == {"accuracy": 0.9}


def test_smoke_variant_pins_to_the_floor_where_downscale_only_halves():
    code = "epochs = 40\nn_rows = 100000\nyear = 2020\n"
    pinned, changes = repair.smoke_variant(code)
    assert "epochs = 1" in pinned and "n_rows = 1000" in pinned
    assert "year = 2020" in pinned, "a value that is not a known knob is left alone"
    assert repair.downscale(code)[0] != pinned
    assert changes


def test_smoke_variant_reports_nothing_to_shrink_when_no_knob_is_present():
    assert repair.smoke_variant("alpha = 3\n") == ("alpha = 3\n", [])


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Traceback:\nNameError: name 'helper' is not defined", True),
        ("Traceback:\nModuleNotFoundError: No module named 'pandas'", True),
        ("Traceback:\nValueError: n_splits=5 cannot be greater than the number of members", False),
        ("Traceback:\nZeroDivisionError: division by zero", False),
        # The exception that actually propagated is the last one named.
        ("TypeError: bad operand\n\nDuring handling...\n\nValueError: too few samples", False),
        ("execution timed out after 60s", False),
        ("", False),
    ],
)
def test_is_scale_independent(message, expected):
    assert diagnose.is_scale_independent(message) is expected


# ── Targeted regeneration ─────────────────────────────────────────────────────
# A failing check that names its own section asks only for that section back;
# everything else is reused verbatim. See CoderAgent._target_sections.


def test_a_localized_failure_asks_only_for_the_section_it_came_from(tmp_path):
    broken = {**GOOD_SECTIONS, "evaluate_function": "def evaluate(experiment_output:\n    pass\n"}
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(broken)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    # Asserted against the format example's placeholders, not against bare
    # BEGIN markers: the prompt also quotes the previous attempt's sections in
    # that same format, so a marker alone says nothing about what was requested.
    fix_prompt = model.prompts_by_kind["fix"][0]
    placeholders = dict(prompts.EXPERIMENT_SECTION_PLACEHOLDERS)
    assert placeholders["evaluate_function"] in fix_prompt
    # The long sections the syntax error cannot have come from are not asked
    # for — this is the whole saving, and the whole reduction in risk.
    assert placeholders["load_data_function"] not in fix_prompt
    assert placeholders["run_experiment_function"] not in fix_prompt
    assert placeholders["readme"] not in fix_prompt
    assert "do NOT return the others" in fix_prompt
    assert result["experiments"][0]["fix_history"][0]["regenerated_sections"] == [
        "imports",
        "configuration",
        "evaluate_function",
        "helpers",
    ]


def test_targeted_regeneration_keeps_the_sections_it_never_asked_for(tmp_path):
    # The response carries every section, but only the requested ones are read
    # back; the rest must survive verbatim from the previous attempt.
    distinctive = "def load_data():\n    return {'kept': 'from the first attempt'}\n"
    broken = {
        **GOOD_SECTIONS,
        "load_data_function": distinctive,
        "evaluate_function": "def evaluate(experiment_output:\n    pass\n",
    }
    replacement = {
        **GOOD_SECTIONS,
        "load_data_function": "def load_data():\n    return 'clobbered'\n",
    }
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(broken)], fix=[_codegen_response(replacement)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    assert result["experiments"][0]["status"] == "completed"
    run_py = (tmp_path / "experiments" / "H1" / "run.py").read_text()
    assert "from the first attempt" in run_py
    assert "clobbered" not in run_py


def test_an_unlocalized_failure_still_asks_for_every_section(tmp_path):
    # A logic bug at execution can live anywhere, so the whole program is fair
    # game — the behaviour every fix had before localization existed.
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    fix_prompt = model.prompts_by_kind["fix"][0]
    placeholders = dict(prompts.EXPERIMENT_SECTION_PLACEHOLDERS)
    for section in ("load_data_function", "run_experiment_function", "readme"):
        assert placeholders[section] in fix_prompt
    assert "do NOT return the others" not in fix_prompt
    assert result["experiments"][0]["fix_history"][0]["regenerated_sections"] == []


def test_a_missing_section_asks_only_for_what_was_missing(tmp_path):
    # Doubly right here: the other sections did arrive intact, and a shorter
    # answer is less likely to be truncated — truncation being the usual reason
    # a section goes missing at all.
    incomplete = {**GOOD_SECTIONS, "evaluate_function": ""}
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response(incomplete)], fix=[_codegen_response(GOOD_SECTIONS)]
    )
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    assert result["experiments"][0]["status"] == "completed"
    placeholders = dict(prompts.EXPERIMENT_SECTION_PLACEHOLDERS)
    assert placeholders["evaluate_function"] in model.prompts_by_kind["fix"][0]
    assert placeholders["build_model_function"] not in model.prompts_by_kind["fix"][0]


def test_assemble_generation_without_a_previous_generation_is_unchanged():
    # The merging path must not alter what a first generation assembles to.
    sections = {**GOOD_SECTIONS, "readme": "r", "requirements_txt": "", "assumptions_made": ""}
    assembled = CoderAgent._assemble_generation(sections)
    assert assembled["run_py_sections"] == GOOD_SECTIONS
    assert assembled["assumptions_made"] == []
    assert assembled["needs_gpu"] is False


def test_assemble_generation_merges_a_subset_onto_the_previous_one():
    previous = CoderAgent._assemble_generation(
        {**GOOD_SECTIONS, "readme": "the original readme", "needs_gpu": "true"}
    )
    merged = CoderAgent._assemble_generation(
        {"evaluate_function": "def evaluate(o):\n    return {}\n"}, previous=previous
    )
    assert merged["run_py_sections"]["evaluate_function"] == "def evaluate(o):\n    return {}\n"
    assert merged["run_py_sections"]["load_data_function"] == GOOD_SECTIONS["load_data_function"]
    assert merged["readme"] == "the original readme"
    assert merged["needs_gpu"] is True


# ── Keeping the best attempt, not merely the last ─────────────────────────────


def test_the_furthest_attempt_is_restored_when_the_last_one_regressed(tmp_path):
    # Attempt 1 reaches a real execution failure; the fix regresses to code that
    # no longer compiles. The budget then runs out. What a human is handed
    # should be the version that got further, not the one that happens to be
    # last.
    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS)],
        fix=[_codegen_response(BROKEN_SYNTAX_SECTIONS)],
    )
    result = _agent(tmp_path, model, max_fix_attempts=1).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert exp["reported_attempt"] == 1
    assert "Reporting attempt 1" in exp["reason"]
    assert "regressed to compile_check" in exp["reason"]
    run_py = (tmp_path / "experiments" / "H1" / "run.py").read_text()
    assert "raise RuntimeError('boom')" in run_py
    assert "def evaluate(experiment_output:" not in run_py


def test_nothing_is_restored_when_the_final_attempt_is_the_best(tmp_path):
    # The common case: the run never regressed, so the newest code stays.
    # The syntax error goes in run_experiment_function so that the targeted
    # regeneration it triggers is the section carrying RAISING_SECTIONS' raising
    # body — targeting evaluate_function would keep the working one and the run
    # would simply succeed.
    broken_run_experiment = {
        **GOOD_SECTIONS,
        "run_experiment_function": "def run_experiment(data, model:\n    pass\n",
    }
    model = ScriptedChatModel(
        codegen=[_codegen_response(broken_run_experiment)],
        fix=[_codegen_response(RAISING_SECTIONS)],
    )
    result = _agent(tmp_path, model, max_fix_attempts=1).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "code_generated_not_run"
    assert exp["reported_attempt"] == 2
    assert "Reporting attempt" not in exp["reason"]
    assert "raise RuntimeError('boom')" in (tmp_path / "experiments" / "H1" / "run.py").read_text()


def test_a_restored_attempt_reports_its_own_assumptions(tmp_path):
    model = ScriptedChatModel(
        codegen=[_codegen_response(RAISING_SECTIONS, assumptions=["the first attempt's"])],
        fix=[_codegen_response(BROKEN_SYNTAX_SECTIONS, assumptions=["the last attempt's"])],
    )
    result = _agent(tmp_path, model, max_fix_attempts=1).run(
        _planner_output([_plan("H1", complexity="low")])
    )

    assert result["experiments"][0]["assumptions_made"] == ["the first attempt's"]


def test_best_candidate_ignores_an_attempt_with_no_code_on_disk(tmp_path):
    # An unparseable response never got as far as writing a run.py, so there is
    # nothing to restore even though its snapshot directory exists.
    missing = tmp_path / "fix_attempts" / "attempt_1" / "run.py"
    history = [{"attempt": 1, "error_source": "results_json", "code_path": str(missing)}]
    assert _best_candidate(history, {"error_source": "compile_check"}) is None


# ── The removed-API repair ────────────────────────────────────────────────────
# The third no-model-call repair, beside installing a package and shrinking a
# run. Ported from barkla-wip/coder-format-retries (0c799c8).


def test_patch_removed_pandas_fillna_rewrites_the_non_inplace_form():
    """Job 10416110's actual failing line — a pure syntax swap since the whole
    chain's result is assigned, never mutated in place."""
    code = "vol = pd.Series(returns).rolling(window=10).std().fillna(method='bfill').values\n"

    patched, changes = repair.patch_removed_pandas_fillna(code)

    assert patched == "vol = pd.Series(returns).rolling(window=10).std().bfill().values\n"
    assert len(changes) == 1


def test_patch_removed_pandas_fillna_rewrites_the_inplace_form_as_reassignment():
    """Job 10411325's actual failing line. A plain syntax swap here would drop
    the mutation and turn a working line into a silent no-op, so this shape
    gets reassignment instead of a bare call."""
    code = "    df.fillna(method='bfill', inplace=True)\n"

    patched, changes = repair.patch_removed_pandas_fillna(code)

    assert patched == "    df = df.bfill()\n"
    assert len(changes) == 1


def test_patch_removed_pandas_fillna_refuses_an_unsafe_inplace_receiver():
    """`get_df()` cannot be reassigned back to — only a plain dotted-name
    receiver is provably safe to rewrite, so anything else is left for the
    model rather than guessed at."""
    code = "get_df().fillna(method='bfill', inplace=True)\n"

    patched, changes = repair.patch_removed_pandas_fillna(code)

    assert patched == code
    assert changes == []


def test_patch_removed_pandas_fillna_leaves_a_constant_fillna_alone():
    """`.fillna(0)` is unaffected by the pandas 3 removal and must not match."""
    code = "df = df.fillna(0)\n"

    patched, changes = repair.patch_removed_pandas_fillna(code)

    assert patched == code
    assert changes == []


def test_patch_removed_pandas_fillna_keeps_the_following_line():
    """The inplace regex ends `[ \\t]*$`, not `\\s*$`: under MULTILINE the `$`
    sits before the newline, and `\\s*` would consume that newline too, silently
    joining the next line onto the rewritten one."""
    code = "df.fillna(method='ffill', inplace=True)\nreturn df\n"

    patched, _ = repair.patch_removed_pandas_fillna(code)

    assert patched == "df = df.ffill()\nreturn df\n"


def test_a_removed_api_is_diagnosed_as_obsolete_and_carries_the_replacement():
    diagnosis = diagnose.classify_execution_failure(
        "TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'"
    )
    assert diagnosis.error_source == "obsolete_dependency"
    assert diagnosis.route == diagnose.ROUTE_REGENERATE
    assert ".ffill()" in diagnosis.guidance
    # An install cannot fix it, so it must never be routed to the installer.
    assert diagnosis.needs_install is False


def test_a_removed_pandas_call_is_patched_without_spending_a_fix_attempt(tmp_path, monkeypatch):
    """The exact shape of Barkla job 10416110: the model reproduced
    byte-identical code across two fix attempts despite the fix prompt naming
    the exact replacement. The deterministic patch must resolve it on the first
    execution retry, spending zero fix attempts and zero model calls."""
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response()})
    agent = _agent(
        tmp_path, fake_model, network_check=lambda: True, huggingface_lookup_fn=lambda *a, **k: None
    )

    executed_sources = []
    attempts = {"n": 0}

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        attempts["n"] += 1
        executed_sources.append(Path(run_script).read_text())
        if attempts["n"] == 1:
            # Written directly rather than through codegen: the point under test
            # is the patch-and-rerun loop, not getting the model to produce this
            # exact line.
            Path(run_script).write_text(
                "vol = pd.Series([1.0]).fillna(method='bfill')\n"
                "print('unreachable if the pandas call above still raises')\n"
            )
            return False, (
                "run.py exited with code 1: Traceback (most recent call last):\n"
                "TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'"
            )
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)

    result = agent.run(_planner_output([_plan("H1", complexity="low")]))
    exp = result["experiments"][0]

    assert exp["status"] == "completed", exp.get("reason")
    assert attempts["n"] == 2, "one failed run, one re-run of the patched file"
    assert "fillna(method=" not in executed_sources[1]
    assert ".bfill()" in executed_sources[1]
    assert exp["fix_attempts"] == 0, "a deterministic patch is not a fix attempt"
    # What must never happen is a *fix* call — the opening line of
    # EXPERIMENT_CODEGEN_FIX_PROMPT — since the patch resolved the failure
    # before the loop asked to regenerate anything.
    assert not any(
        "The code you generated for hypothesis" in messages[-1][1] for messages in fake_model.calls
    )


def test_the_smoke_run_does_not_pre_empt_the_removed_api_patch(tmp_path, monkeypatch):
    """The reconciliation between the two features. A removed-API failure raises
    TypeError, which diagnose.is_scale_independent treats as a real defect — so
    without the explicit fall-through in _smoke_failure, the smoke run reports
    it, the fix loop takes over, and the deterministic patch below is never
    reached in the one case it exists for."""
    sections = {
        **GOOD_SECTIONS,
        "configuration": "epochs = 40\n",  # gives the smoke run something to shrink
        "helpers": "def _fill(series):\n    return series.fillna(method='bfill')\n",
    }
    fake_model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response(sections)})
    agent = _agent(
        tmp_path, fake_model, network_check=lambda: True, huggingface_lookup_fn=lambda *a, **k: None
    )

    ran = []

    def fake_run_experiment(python_executable, run_script, cwd, timeout_seconds):
        ran.append(Path(run_script).name)
        if "fillna(method=" in Path(run_script).read_text():
            return False, (
                "run.py exited with code 1: Traceback (most recent call last):\n"
                "TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'"
            )
        (Path(cwd) / "results.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9}, "meets_success_criteria": True})
        )
        return True, ""

    monkeypatch.setattr(sandbox, "run_experiment", fake_run_experiment)

    result = agent.run(_planner_output([_plan("H1", complexity="low")]))
    exp = result["experiments"][0]

    # The smoke run really did happen and really did see the failure — it just
    # declined to answer for it.
    assert ran == ["run_smoke.py", "run.py", "run.py"]
    assert exp["status"] == "completed", exp.get("reason")
    assert exp["fix_attempts"] == 0, "the patch resolved it; no fix attempt was spent"
    assert ".bfill()" in (tmp_path / "experiments" / "H1" / "run.py").read_text()
    assert not any(
        "The code you generated for hypothesis" in messages[-1][1] for messages in fake_model.calls
    )


# ── Provenance: what the code actually read ───────────────────────────────────
# Ported from barkla-wip/coder-format-retries (0c799c8). verify_downloads_used
# tightens (fewer inputs credited as real), supersede_unresolved loosens (fewer
# phantom surrogates); both are deterministic and both run after the experiment.


def _download(name="World Bank", uri="https://api.worldbank.org/v2", **kw):
    return provenance.DataSource(
        name=name, kind=provenance.KIND_REAL_DOWNLOAD, uri=uri, reason="openly fetchable", **kw
    )


def test_a_declared_download_the_code_never_fetches_is_downgraded():
    # The over-claiming direction: without this the experiment earns the
    # "findings are interpretable as evidence" stamp for data it invented.
    sources = [_download()]

    verified = provenance.verify_downloads_used(sources, "def load_data():\n    return [1, 2]\n")

    assert verified[0].kind == provenance.KIND_SURROGATE
    assert "never fetches api.worldbank.org" in verified[0].reason
    assert provenance.all_real(verified) is False


def test_a_download_the_code_does_fetch_is_left_alone():
    code = "requests.get('https://api.worldbank.org/v2/country')\n"
    assert provenance.verify_downloads_used([_download()], code)[0].kind == (
        provenance.KIND_REAL_DOWNLOAD
    )


def test_one_obtained_input_vouches_for_the_experiment_as_a_whole():
    # Asked of the experiment, not of each requirement: a Hub dataset routinely
    # answers a requirement the code then reads from a local copy, and
    # per-requirement matching would turn that into a phantom surrogate.
    sources = [_download(name="Hub", uri="https://x.co", usage_verified=True), _download()]

    verified = provenance.verify_downloads_used(sources, "read('data.jsonl')")

    assert [s.kind for s in verified] == [provenance.KIND_REAL_DOWNLOAD] * 2


def test_a_local_file_is_real_whatever_the_code_looks_like():
    local = provenance.DataSource(
        name="staged", kind=provenance.KIND_REAL_LOCAL, local_path="/data/x.csv", reason="staged"
    )
    assert provenance.verify_downloads_used([local], "print('hi')")[0].kind == (
        provenance.KIND_REAL_LOCAL
    )


def _unresolved(name="stock prices"):
    return provenance.DataSource(
        name=name,
        kind=provenance.KIND_SURROGATE,
        reason="no open source identified",
        unresolved=True,
    )


def test_a_phantom_surrogate_is_superseded_by_data_the_code_really_reads():
    sources = [_download(name="Hub", usage_verified=True), _unresolved()]

    kept = provenance.supersede_unresolved(sources, "df = read_jsonl('data.jsonl')\n")

    assert [s.name for s in kept] == ["Hub"]
    assert provenance.all_real(kept) is True


def test_a_synthesize_generator_blocks_superseding():
    # The name prompt_block instructs, so its presence is the trace that
    # something really was invented — no verdict is credited past it.
    sources = [_download(name="Hub", usage_verified=True), _unresolved()]
    code = "def synthesize_prices(n):\n    return [1.0] * n\n"

    assert provenance.supersede_unresolved(sources, code) == sources


def test_a_restricted_source_is_never_superseded():
    # Naming real data that specifically was not obtained is not a phantom, and
    # no amount of other data answers it.
    restricted = provenance.DataSource(
        name="UK Biobank", kind=provenance.KIND_SURROGATE, reason="restricted access"
    )
    sources = [_download(name="Hub", usage_verified=True), restricted]

    assert provenance.supersede_unresolved(sources, "read('data.jsonl')") == sources


def test_reads_dataset_matches_the_percent_encoded_form():
    # The prompt hands over a rows URL, which encodes the namespace slash — a
    # plain `id in code` test misses exactly the form the model is given.
    assert CoderAgent._reads_dataset("acme/sleep-survey", "url = '...dataset=acme%2Fsleep-survey'")
    assert CoderAgent._reads_dataset("acme/sleep-survey", "load('acme/sleep-survey')")
    assert not CoderAgent._reads_dataset("acme/sleep-survey", "load('other/thing')")


def test_an_offered_dataset_is_described_to_the_model_as_real_not_as_a_surrogate(tmp_path):
    # The contradiction this fixes: hf_dataset_block introduced the dataset as
    # real while provenance_block, in the same prompt, called it a surrogate and
    # ordered a `synthesize_` generator — and the model does as it is told.
    model = RecordingScriptedChatModel(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    lookup, _ = _recording_lookup(HF_DATASET_MATCH)
    _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    # Asserted against the provenance block specifically, not the whole prompt:
    # the dataset was always named by hf_dataset_block. What was missing is its
    # appearing *here*, as a resolved real input.
    prompt = model.prompts_by_kind["codegen"][0]
    block = prompt[prompt.index("RESOLVED DATA INPUTS") :]
    assert "Hugging Face dataset acme/sleep-survey\n   REAL" in block
    # The plan's own "synthetic" requirement is still listed as a surrogate —
    # this fix credits the offered dataset, it does not silence the rest.
    assert "SURROGATE" in block


def test_require_real_data_skips_a_synthetic_only_plan_before_any_codegen(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, coder_require_real_data=True)
    model = FakeChatModel({})  # no responses configured — any model call fails the test
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "skipped"
    assert exp["code_path"] is None
    assert "CODER_REQUIRE_REAL_DATA" in exp["reason"]
    assert exp["data_provenance"]["all_inputs_real"] is False
    assert model.calls == [], "skipped before a single codegen call"
    assert not (tmp_path / "experiments" / "H1" / "run.py").exists()


def test_require_real_data_off_generates_exactly_as_before(tmp_path):
    # The default. The fixture plan's data_requirements are synthetic, and the
    # experiment is still generated, run and reported "inconclusive".
    model = FakeChatModel({'"hypothesis_id":"H1"': _codegen_response()})
    result = _agent(tmp_path, model).run(_planner_output([_plan("H1", complexity="low")]))

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert exp["results"]["meets_success_criteria"] == "unknown"


def test_a_dataset_the_code_reads_is_counted_as_a_real_input(tmp_path):
    """Regression for a key that never matched. _provenance_for read
    hf_dataset["dataset"], but huggingface_client returns "dataset_id" — so this
    branch never once fired, and an experiment that genuinely fetched a real Hub
    dataset still had its verdict withheld as though it had invented the data.
    That is the one path by which a real verdict is reachable at all."""
    agent = _agent(tmp_path, FakeChatModel({}))
    code = (
        "resp = requests.get('https://datasets-server.huggingface.co/rows"
        "?dataset=acme%2Fsleep-survey&config=default&split=train&offset=0&length=100')\n"
    )

    sources = agent._provenance_for(
        _plan("H1"), network_available=True, hf_dataset=HF_DATASET_MATCH, run_py=code
    )

    dataset = next(s for s in sources if "acme/sleep-survey" in s.name)
    assert dataset.is_real
    # The uri must carry the real host, or verify_downloads_used stops vouching
    # for a dataset the code did fetch.
    assert "datasets-server.huggingface.co" in dataset.uri
    # The plan's own "synthetic" requirement is a phantom here — the dataset
    # answered it under another name, and no synthesize_ generator was written.
    assert provenance.all_real(sources)


def test_a_dataset_the_code_ignores_still_withholds_the_verdict(tmp_path):
    # The other direction, and the one that must not regress: an offered-but-
    # unread dataset is not evidence.
    agent = _agent(tmp_path, FakeChatModel({}))

    sources = agent._provenance_for(
        _plan("H1"),
        network_available=True,
        hf_dataset=HF_DATASET_MATCH,
        run_py="def load_data():\n    return synthesize_rows(100)\n",
    )

    assert not any("acme/sleep-survey" in s.name for s in sources)
    assert provenance.all_real(sources) is False


# ---------------------------------------------------------------------------
# Data acquisition (agents/coder/acquire.py) — wired into the graph.
#
# acquire.py's own behaviour (the URL safety gate, paging, parsing, the cache)
# is covered in tests/test_coder_acquire.py. These are about the wiring: that
# the node runs at the right point, that what it fetched reaches the prompt and
# the provenance document, and that every way it can fail leaves the run exactly
# as it was before the module existed.
# ---------------------------------------------------------------------------


def _hf_rows_response(rows=2):
    """The Dataset Viewer payload acquire.py would fetch for HF_DATASET_MATCH."""

    class _Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        is_redirect = False
        is_permanent_redirect = False

        def iter_content(self, chunk_size=65536):
            payload = {
                "rows": [
                    {"row_idx": i, "row": {"hours_slept": 7.0 + i, "score": 80 + i}}
                    for i in range(rows)
                ],
                "num_rows_total": rows,
            }
            yield json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Response()


@pytest.fixture
def acquisition_on(monkeypatch, tmp_path):
    """CODER_ENABLE_DATA_ACQUISITION on, a cache under tmp_path, and a faked
    network — no test here touches the real one."""
    from research_pipeline.agents.coder import acquire

    _patch_settings(
        monkeypatch,
        coder_enable_data_acquisition=True,
        coder_data_cache_dir=str(tmp_path / "data_cache"),
    )
    monkeypatch.setattr(
        acquire.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _hf_rows_response()

    monkeypatch.setattr(acquire.requests, "get", fake_get)
    return calls


def test_a_fetched_dataset_reaches_the_model_as_a_local_file(tmp_path, acquisition_on):
    model = RecordingScriptedChatModel(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    lookup, _ = _recording_lookup(HF_DATASET_MATCH)
    _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    prompt = model.prompts_by_kind["codegen"][0]
    assert "already downloaded for you" in prompt
    assert "Read that file from disk" in prompt
    assert str(tmp_path / "data_cache") in prompt
    # And crucially *not* the instruction to build the REST call, which is the
    # step that used to go wrong and burn fix attempts.
    assert "Read it over HTTP with `requests`" not in prompt


def test_a_fetched_dataset_is_a_real_local_input_in_the_provenance_document(
    tmp_path, acquisition_on
):
    model = RecordingScriptedChatModel(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    lookup, _ = _recording_lookup(HF_DATASET_MATCH)
    result = _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    inputs = result["experiments"][0]["data_provenance"]["inputs"]
    fetched = [entry for entry in inputs if entry.get("acquired")]
    assert len(fetched) == 1
    assert fetched[0]["kind"] == "real_local"
    assert fetched[0]["acquired"]["row_count"] == 2
    assert len(fetched[0]["acquired"]["sha256"]) == 64
    # The origin URL is kept, so the record still says where the bytes came from.
    assert fetched[0]["uri"].startswith("https://datasets-server.huggingface.co/rows")


def test_the_fetch_happens_once_per_plan_not_once_per_fix_attempt(tmp_path, acquisition_on):
    # Same reasoning as the dataset lookup being threaded through state: a fix
    # attempt must reuse what the first generation read rather than re-download.
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response({**GOOD_SECTIONS_WITH_HF_DATASET, "imports": "import ("})],
        fix=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)],
    )
    lookup, _ = _recording_lookup(HF_DATASET_MATCH)
    _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    assert len(model.prompts_by_kind["fix"]) == 1, "the fix loop ran"
    # One page of rows, fetched for the first generation and reused for the fix.
    assert len(acquisition_on) == 1
    assert "already downloaded for you" in model.prompts_by_kind["fix"][0]


def test_acquisition_is_skipped_without_the_setting(tmp_path, monkeypatch):
    """The default. Nothing is fetched and the prompt keeps handing over a URL."""
    from research_pipeline.agents.coder import acquire

    def fail(*args, **kwargs):
        raise AssertionError("no request may be made with acquisition off")

    monkeypatch.setattr(acquire.requests, "get", fail)
    model = RecordingScriptedChatModel(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    lookup, _ = _recording_lookup(HF_DATASET_MATCH)
    result = _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert "Read it over HTTP with `requests`" in model.prompts_by_kind["codegen"][0]
    assert not any(
        entry.get("acquired") for entry in result["experiments"][0]["data_provenance"]["inputs"]
    )


def test_acquisition_is_skipped_without_a_network(tmp_path, monkeypatch):
    from research_pipeline.agents.coder import acquire

    _patch_settings(monkeypatch, coder_enable_data_acquisition=True)

    def fail(*args, **kwargs):
        raise AssertionError("no request may be made when the network probe failed")

    monkeypatch.setattr(acquire.requests, "get", fail)
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, network_check=lambda: False).run(
        _planner_output([_plan("H1")])
    )
    assert result["experiments"][0]["status"] == "completed"


def test_a_failing_fetch_generates_exactly_as_before(tmp_path, monkeypatch):
    """The degradation contract, at the agent level: a download that breaks
    leaves the input a real_download and the code fetching it as it always did.
    An experiment must never go ungenerated because a fetch failed."""
    from research_pipeline.agents.coder import acquire

    _patch_settings(
        monkeypatch,
        coder_enable_data_acquisition=True,
        coder_data_cache_dir=str(tmp_path / "data_cache"),
    )
    monkeypatch.setattr(
        acquire.requests,
        "get",
        lambda url, **kwargs: (_ for _ in ()).throw(acquire.requests.RequestException("down")),
    )
    model = RecordingScriptedChatModel(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    lookup, _ = _recording_lookup(HF_DATASET_MATCH)
    result = _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    assert result["experiments"][0]["status"] == "completed"
    assert "Read it over HTTP with `requests`" in model.prompts_by_kind["codegen"][0]


def test_require_real_data_sees_the_fetch_before_deciding_to_skip(
    tmp_path, monkeypatch, acquisition_on
):
    """The routing order that matters: acquire_data runs *before* the skip
    decision, so an input the pipeline managed to fetch counts as real when that
    decision is made rather than after it."""
    _patch_settings(
        monkeypatch,
        coder_enable_data_acquisition=True,
        coder_data_cache_dir=str(tmp_path / "data_cache"),
        coder_require_real_data=True,
    )
    model = RecordingScriptedChatModel(codegen=[_codegen_response(GOOD_SECTIONS_WITH_HF_DATASET)])
    lookup, _ = _recording_lookup(HF_DATASET_MATCH)
    result = _agent(tmp_path, model, network_check=lambda: True, huggingface_lookup_fn=lookup).run(
        _planner_output([_plan("H1")])
    )

    assert acquisition_on, "the fetch ran before the skip decision, not after it"
    # This fixture plan also names a requirement nothing can resolve, so the run
    # is still skipped — but the skip's own provenance record shows the fetched
    # dataset as a real_local input, which is only possible if acquire_data had
    # already run when _route_after_data_lookup resolved the sources.
    inputs = result["experiments"][0]["data_provenance"]["inputs"]
    assert [entry["kind"] for entry in inputs if entry.get("acquired")] == ["real_local"]


# ---------------------------------------------------------------------------
# Source discovery (agents/coder/discover.py) — wired into the same node.
#
# discover.py's own behaviour (connectors, the relevance gate, the probe cap)
# is covered in tests/test_coder_discover.py. These are about the wiring.
# ---------------------------------------------------------------------------


def _plan_needing_data(hid="H1"):
    """A plan whose data requirement names no source — the case that becomes an
    invented input today, and the only case discovery is allowed to answer."""
    plan = _plan(hid)
    plan["data_requirements"] = {
        "source": "bicycle collision casualty records",
        "description": "reported cycling casualties by year",
        "preprocessing_steps": [],
    }
    return plan


@pytest.fixture
def discovery_on(monkeypatch, tmp_path, acquisition_on):
    """Discovery enabled, with the catalogues faked to offer one relevant CSV."""
    from research_pipeline.agents.coder import discover

    _patch_settings(
        monkeypatch,
        coder_enable_data_acquisition=True,
        coder_enable_source_discovery=True,
        coder_data_cache_dir=str(tmp_path / "data_cache"),
    )
    searched = []

    def fake_search(requirement):
        searched.append(requirement)
        return [
            discover.Candidate(
                name=requirement,
                url="https://data.example/collisions.csv",
                connector="ckan:data.gov.uk",
                title="Reported road collisions and casualties",
                description="Bicycle collision casualty records by year",
                landing_page="https://www.data.gov.uk/dataset/collisions",
            )
        ]

    monkeypatch.setattr(discover, "CONNECTORS", [("fake", fake_search)])
    return searched


def _collisions_csv_response():
    class _Response:
        status_code = 200
        headers = {"content-type": "text/csv"}
        is_redirect = False
        is_permanent_redirect = False

        def iter_content(self, chunk_size=65536):
            yield b"year,casualties\n2020,15\n2021,18\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Response()


def test_a_discovered_source_turns_an_invented_input_into_a_real_one(
    tmp_path, monkeypatch, discovery_on
):
    from research_pipeline.agents.coder import acquire

    monkeypatch.setattr(acquire.requests, "get", lambda url, **kw: _collisions_csv_response())
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, network_check=lambda: True).run(
        _planner_output([_plan_needing_data()])
    )

    assert discovery_on == ["bicycle collision casualty records"]
    inputs = result["experiments"][0]["data_provenance"]["inputs"]
    assert [entry["kind"] for entry in inputs] == ["real_local"]
    assert result["experiments"][0]["data_provenance"]["all_inputs_real"] is True
    # The audit trail: what was searched, by which connector, and which record.
    assert inputs[0]["discovered"]["connector"] == "ckan:data.gov.uk"
    assert inputs[0]["discovered"]["landing_page"] == ("https://www.data.gov.uk/dataset/collisions")
    assert inputs[0]["acquired"]["columns"] == ["year", "casualties"]

    # Real data, and still not a verdict: nobody named this dataset, so the
    # experiment runs on it and reports its metrics while the hypothesis stays
    # inconclusive until a human checks the landing page. A live sweep found two
    # of five discovered datasets were real, plausible and wrong — see
    # provenance.needs_confirmation.
    exp = result["experiments"][0]
    assert exp["results"]["meets_success_criteria"] == "unknown"
    assert exp["data_provenance"]["unconfirmed_discovered_inputs"] == [
        "bicycle collision casualty records"
    ]
    assert "found by keyword search" in exp["results"]["verdict_withheld_because"]


def test_the_model_is_told_the_source_was_discovered_not_named(tmp_path, monkeypatch, discovery_on):
    from research_pipeline.agents.coder import acquire

    monkeypatch.setattr(acquire.requests, "get", lambda url, **kw: _collisions_csv_response())
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    _agent(tmp_path, model, network_check=lambda: True).run(_planner_output([_plan_needing_data()]))

    prompt = model.prompts_by_kind["codegen"][0]
    assert "no source was named for this input" in prompt
    assert "Columns: year, casualties" in prompt
    assert "say so in assumptions_made rather than synthesizing" in prompt
    # And it is no longer ordered to write a generator for this input.
    block = prompt[prompt.index("RESOLVED DATA INPUTS") :]
    assert "SURROGATE" not in block


def test_discovery_is_skipped_without_the_setting(tmp_path, monkeypatch, acquisition_on):
    """The default. No catalogue is searched and the input stays a surrogate."""
    from research_pipeline.agents.coder import discover

    monkeypatch.setattr(
        discover,
        "CONNECTORS",
        [("fake", lambda r: pytest.fail("no catalogue may be searched with discovery off"))],
    )
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, network_check=lambda: True).run(
        _planner_output([_plan_needing_data()])
    )

    inputs = result["experiments"][0]["data_provenance"]["inputs"]
    assert [entry["kind"] for entry in inputs] == ["synthetic_surrogate"]
    assert result["experiments"][0]["results"]["meets_success_criteria"] == "unknown"


def test_discovery_runs_once_per_plan_not_once_per_fix_attempt(tmp_path, monkeypatch, discovery_on):
    from research_pipeline.agents.coder import acquire

    monkeypatch.setattr(acquire.requests, "get", lambda url, **kw: _collisions_csv_response())
    model = RecordingScriptedChatModel(
        codegen=[_codegen_response({**GOOD_SECTIONS, "imports": "import ("})],
        fix=[_codegen_response()],
    )
    _agent(tmp_path, model, network_check=lambda: True).run(_planner_output([_plan_needing_data()]))

    assert len(model.prompts_by_kind["fix"]) == 1, "the fix loop ran"
    assert len(discovery_on) == 1, "four catalogue searches are not repeated per attempt"
    assert "no source was named for this input" in model.prompts_by_kind["fix"][0]


def test_a_failing_discovery_leaves_the_input_a_surrogate(tmp_path, monkeypatch, acquisition_on):
    """Degradation: nothing relevant found means the experiment is generated
    exactly as it was, on a documented surrogate, with the verdict withheld."""
    from research_pipeline.agents.coder import discover

    _patch_settings(
        monkeypatch,
        coder_enable_data_acquisition=True,
        coder_enable_source_discovery=True,
        coder_data_cache_dir=str(tmp_path / "data_cache"),
    )
    monkeypatch.setattr(
        discover, "CONNECTORS", [("dead", lambda r: (_ for _ in ()).throw(RuntimeError("down")))]
    )
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, network_check=lambda: True).run(
        _planner_output([_plan_needing_data()])
    )

    exp = result["experiments"][0]
    assert exp["status"] == "completed"
    assert [entry["kind"] for entry in exp["data_provenance"]["inputs"]] == ["synthetic_surrogate"]


# ---------------------------------------------------------------------------
# Model-assisted data sourcing (Phase 3). The model nominates; Python rules.
#
# discover.py's side of this (rank_with's validation, pooling, the probe budget)
# is covered in tests/test_coder_discover.py. These cover the two model calls
# themselves: what they send, what they accept back, and that every failure mode
# degrades to the keyword behaviour rather than losing the requirement.
# ---------------------------------------------------------------------------


def _sourcing_agent(tmp_path, model):
    return _agent(tmp_path, model, network_check=lambda: True)


def _cands():
    from research_pipeline.agents.coder import discover

    good = discover.Candidate(
        name="r",
        url="https://x.example/hourly.csv",
        connector="ckan:data.gov.uk",
        title="Air quality",
        description="Monitoring",
        resource="Hourly PM2.5 readings",
    )
    bad = discover.Candidate(
        name="r",
        url="https://x.example/ref.csv",
        connector="ckan:data.gov.uk",
        title="Air quality",
        description="Monitoring",
        resource="Geographic reference table",
    )
    return [good, bad]


def test_the_chooser_prompt_shows_the_resource_name_of_each_candidate(tmp_path):
    model = FakeChatModel({"RESOURCE:": '{"ranked": [0], "why": "hourly readings"}'})
    chosen = _sourcing_agent(tmp_path, model)._choose_data_source("PM2.5 hourly", _cands())

    assert chosen == [0]
    prompt = model.calls[0][-1][1]
    # The resource line is the whole point: a dataset with the right title
    # routinely contains a station list rather than the data.
    assert "RESOURCE: Hourly PM2.5 readings" in prompt
    assert "RESOURCE: Geographic reference table" in prompt
    assert "0." in prompt and "1." in prompt


def test_the_chooser_passes_through_an_empty_rejection(tmp_path):
    """[] is a real answer — "I looked and none of these fit" — and must reach
    discover.rank_with, which honours it by leaving a labelled surrogate."""
    model = FakeChatModel({"RESOURCE:": '{"ranked": [], "why": "none hold the measurements"}'})
    assert _sourcing_agent(tmp_path, model)._choose_data_source("PM2.5", _cands()) == []


def test_the_chooser_returns_none_when_the_model_fails(tmp_path):
    """None, not [] — no answer was obtained, so the keyword ordering stands
    rather than the requirement being rejected outright."""
    model = FakeChatModel({"RESOURCE:": "not json at all, and not on the retry either"})
    assert _sourcing_agent(tmp_path, model)._choose_data_source("PM2.5", _cands()) is None


def test_the_chooser_returns_none_for_a_malformed_shape(tmp_path):
    model = FakeChatModel({"RESOURCE:": '{"ranked": "the first one"}'})
    assert _sourcing_agent(tmp_path, model)._choose_data_source("PM2.5", _cands()) is None


def test_the_chooser_drops_non_integer_entries(tmp_path):
    model = FakeChatModel({"RESOURCE:": '{"ranked": [0, "1", null, 1], "why": "w"}'})
    assert _sourcing_agent(tmp_path, model)._choose_data_source("PM2.5", _cands()) == [0, 1]


def test_proposed_sources_become_candidates(tmp_path):
    model = FakeChatModel(
        {
            "Name up to": (
                '{"sources": [{"url": "https://api.example/v1/pm25.csv", "name": "PM2.5 hourly",'
                ' "format": "csv"}], "why": "public API"}'
            )
        }
    )
    proposed = _sourcing_agent(tmp_path, model)._propose_data_sources("PM2.5 hourly readings")

    assert [c.url for c in proposed] == ["https://api.example/v1/pm25.csv"]
    assert proposed[0].connector == "model"
    assert proposed[0].resource == "PM2.5 hourly"


def test_proposed_sources_are_capped(tmp_path):
    from research_pipeline.agents.coder.coder_agent import _MAX_PROPOSED_SOURCES

    sources = ", ".join(
        f'{{"url": "https://api.example/{i}.csv", "name": "n{i}"}}' for i in range(12)
    )
    model = FakeChatModel({"Name up to": f'{{"sources": [{sources}], "why": "w"}}'})
    proposed = _sourcing_agent(tmp_path, model)._propose_data_sources("anything")
    assert len(proposed) == _MAX_PROPOSED_SOURCES


def test_a_proposal_with_no_usable_urls_yields_nothing(tmp_path):
    for payload in (
        '{"sources": [], "why": "I do not know a real URL"}',
        '{"sources": [{"name": "no url here"}], "why": "w"}',
        '{"sources": "not a list"}',
        "not json at all, and not on the retry either",
    ):
        model = FakeChatModel({"Name up to": payload})
        assert _sourcing_agent(tmp_path, model)._propose_data_sources("anything") == []


def test_the_model_connector_is_last_and_only_when_enabled(tmp_path, monkeypatch):
    """A catalogue hit is a file someone published and described; a proposed URL
    is the model's recollection of a URL shape. Try the first one first."""
    agent = _agent(tmp_path, FakeChatModel({}))
    assert [name for name, _ in agent._data_connectors()] == ["direct", "ckan", "zenodo"]

    _patch_settings(monkeypatch, coder_enable_model_data_sourcing=True)
    assert [name for name, _ in agent._data_connectors()][-1] == "model"


def test_a_model_proposed_url_still_has_to_fetch_and_parse(tmp_path, monkeypatch):
    """The reason the model is allowed to guess at all: an invented URL costs a
    download and can never become a source."""
    from research_pipeline.agents.coder import acquire, discover

    _patch_settings(monkeypatch, coder_enable_model_data_sourcing=True)
    monkeypatch.setattr(
        acquire.requests,
        "get",
        lambda url, **kw: _hallucinated_404(),
    )
    model = FakeChatModel(
        {
            "Name up to": (
                '{"sources": [{"url": "https://invented.example/does-not-exist.csv",'
                ' "name": "n"}], "why": "w"}'
            )
        }
    )
    agent = _agent(tmp_path, model, network_check=lambda: True)
    found = discover.find_source(
        "some requirement nothing has",
        cache_dir=tmp_path / "cache",
        connectors=[("model", agent._propose_data_sources)],
    )
    assert found is None


def _hallucinated_404():
    class _Response:
        status_code = 404
        headers = {}
        is_redirect = False
        is_permanent_redirect = False

        def iter_content(self, chunk_size=65536):
            yield b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Response()


def test_model_sourcing_does_not_change_the_verdict_rule(tmp_path, monkeypatch, discovery_on):
    """A model picking the dataset improves the hit rate, not the epistemic
    status: whether real data answers a research question is not something
    Python can verify, so a discovered input stays inconclusive whoever chose
    it."""
    from research_pipeline.agents.coder import acquire

    _patch_settings(
        monkeypatch,
        coder_enable_data_acquisition=True,
        coder_enable_source_discovery=True,
        coder_enable_model_data_sourcing=True,
        coder_data_cache_dir=str(tmp_path / "data_cache"),
    )
    monkeypatch.setattr(acquire.requests, "get", lambda url, **kw: _collisions_csv_response())
    model = RecordingScriptedChatModel(codegen=[_codegen_response()])
    result = _agent(tmp_path, model, network_check=lambda: True).run(
        _planner_output([_plan_needing_data()])
    )

    exp = result["experiments"][0]
    assert exp["data_provenance"]["all_inputs_real"] is True
    assert exp["results"]["meets_success_criteria"] == "unknown"
    assert exp["data_provenance"]["unconfirmed_discovered_inputs"] == [
        "bicycle collision casualty records"
    ]


# -- static_safety_check must not flag attribute calls -------------------------
#
# Regression for Barkla job 10423680: `\beval\s*\(` matched `model.eval()`,
# because `.` is a word boundary. The plan spent all three fix attempts
# regenerating correct code and was reported code_generated_not_run. A false
# positive here is strictly worse than a missing pattern — the model cannot fix
# a finding about code that has no defect.


@pytest.mark.parametrize(
    "code",
    [
        "baseline_model.eval()",
        "enhanced_model.eval()\n    with torch.no_grad():\n        pass",
        "self.model.eval()",
        "cursor.exec(query)",
        "session.exec(statement)",
        "results = df.eval('a + b')",  # pandas.DataFrame.eval, also legitimate
    ],
)
def test_static_safety_check_allows_attribute_calls(code):
    assert sandbox.static_safety_check(code) == []


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("result = eval(user_input)", "eval"),
        ("eval(payload)", "eval"),
        ("x = 1; exec(code)", "exec"),
        ("mod = __import__(name)", "__import__"),
        ("value = eval  ( expr )", "eval"),
    ],
)
def test_static_safety_check_still_flags_the_bare_builtins(code, expected):
    findings = sandbox.static_safety_check(code)
    assert findings, f"{code!r} should still be flagged"
    assert expected in findings[0]


def test_the_pytorch_inference_idiom_is_clean_end_to_end():
    """The exact shape job 10423680 generated and was punished for."""
    code = (
        "def run_experiment(data, model):\n"
        "    baseline_model, enhanced_model = model\n"
        "    baseline_model.eval()\n"
        "    enhanced_model.eval()\n"
        "    with torch.no_grad():\n"
        "        return {'preds': enhanced_model(data)}\n"
    )
    assert sandbox.static_safety_check(code) == []


# -- an acquired dataset must be recorded as the input it is --------------------
#
# Regression for Barkla job 10424136: load_data read the acquired JSONL and
# computed real metrics from it, but because acquire._safe_label sanitises the
# namespace slash the cached filename matches neither the raw dataset id nor its
# percent-encoded form, so _reads_dataset said no and the dataset was left out of
# data_provenance.json — which named a staged CSV the code never opened. Cosmetic
# there only because that staged file independently made the run real; with
# nothing staged, the experiment's one real input vanishes and the verdict is
# withheld from a run that earned it.


def test_reads_dataset_accepts_the_acquired_local_path():
    dataset_id = "chuyin0321/timeseries-daily-stocks"
    # The real cached path from job 10424136 — note the sanitised slash.
    path = (
        "/mnt/fastscratch/users/sgyshere/coder-data-cache/37f0c4d9d9e16f7b/"
        "hugging_face_dataset_chuyin0321_timeseries-daily-stocks.jsonl"
    )
    code = f'df_hf = pd.read_json("{path}", lines=True)'

    assert dataset_id not in code, "the sanitised filename must not contain the raw id"
    assert not CoderAgent._reads_dataset(dataset_id, code)
    assert CoderAgent._reads_dataset(dataset_id, code, path)


def test_reads_dataset_still_matches_the_id_and_url_forms():
    dataset_id = "acme/sleep-survey"
    assert CoderAgent._reads_dataset(dataset_id, "load('acme/sleep-survey')", "/tmp/x.jsonl")
    assert CoderAgent._reads_dataset(dataset_id, "url='...dataset=acme%2Fsleep-survey'")
    assert not CoderAgent._reads_dataset(dataset_id, "load('other/thing')", "/tmp/x.jsonl")


def test_an_acquired_dataset_the_code_reads_is_a_real_input(tmp_path, monkeypatch):
    """End to end through _provenance_for: the acquired file is recorded, so a
    run with nothing staged still reaches a real verdict."""
    _patch_settings(monkeypatch, coder_data_dir="")  # nothing staged
    agent = _agent(tmp_path, FakeChatModel({}))
    rows_url = CoderAgent._rows_url(HF_DATASET_MATCH)
    path = str(tmp_path / "cache" / "hugging_face_dataset_acme_sleep-survey.jsonl")
    acquisitions = {
        rows_url: {
            "url": rows_url,
            "path": path,
            "sha256": "d" * 64,
            "byte_count": 10,
            "data_format": "jsonl",
            "columns": ["hours_slept", "score"],
            "row_count": 100,
        }
    }
    run_py = f'df = pd.read_json("{path}", lines=True)'

    sources = agent._provenance_for(
        _plan("H1"),
        network_available=True,
        hf_dataset=HF_DATASET_MATCH,
        run_py=run_py,
        acquisitions=acquisitions,
    )
    hub = [s for s in sources if "acme/sleep-survey" in s.name]
    assert hub, "the acquired dataset must appear as an input"
    assert hub[0].kind == provenance.KIND_REAL_LOCAL
    assert hub[0].local_path == path
    assert provenance.verdict(sources) != provenance.VERDICT_SURROGATE


# -- sandbox.check_training_batching -------------------------------------------

_UNBATCHED = (
    "def run_experiment(data, model):\n"
    "    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
    "    for epoch in range(NUM_EPOCHS):\n"
    "        optimizer.zero_grad()\n"
    "        loss = criterion(model(X_train_tensor), y_train_tensor)\n"
    "        loss.backward()\n"
    "        optimizer.step()\n"
    "    return {}\n"
)


def test_training_batching_flags_one_step_per_epoch():
    findings = sandbox.check_training_batching(_UNBATCHED, _UNBATCHED, [])
    assert len(findings) == 1
    assert "mini-batching" in findings[0]


def test_training_batching_accepts_a_dataloader():
    code = _UNBATCHED.replace(
        "    for epoch in range(NUM_EPOCHS):",
        "    loader = DataLoader(TensorDataset(X, y), batch_size=32)\n"
        "    for epoch in range(NUM_EPOCHS):\n        for xb, yb in loader:",
    )
    assert sandbox.check_training_batching(code, code, []) == []


def test_training_batching_accepts_hand_rolled_slicing():
    code = _UNBATCHED.replace(
        "        optimizer.zero_grad()",
        "        for i in range(0, len(X_train_tensor), BATCH_SIZE):\n            optimizer.zero_grad()",
    )
    assert sandbox.check_training_batching(code, code, []) == []


def test_training_batching_ignores_code_with_no_torch_optimizer():
    """An sklearn fit, a closed-form estimator or a statistical test has no
    training loop to be wrong about."""
    code = "def run_experiment(data, model):\n    model.fit(data['X'], data['y'])\n    return {}\n"
    assert sandbox.check_training_batching(code, code, []) == []


def test_training_batching_ignores_a_scheduler_step():
    code = (
        "def run_experiment(data, model):\n"
        "    for epoch in range(10):\n"
        "        scheduler.step()\n"
        "    return {}\n"
    )
    assert sandbox.check_training_batching(code, code, []) == []


def test_training_batching_accepts_a_declared_full_batch_choice():
    """Same sanctioned escape as check_hf_dataset_usage: full-batch is right for
    a small dataset or a second-order optimizer. The point is that the choice be
    deliberate and recorded, not that mini-batching be mandatory."""
    assert (
        sandbox.check_training_batching(
            _UNBATCHED, _UNBATCHED, ["Full-batch training: the dataset is 300 rows"]
        )
        == []
    )


def test_training_batching_survives_a_syntax_error():
    assert sandbox.check_training_batching("def run_experiment(", "", []) == []


def test_neither_check_is_a_retry_on_bad_results():
    """The line this pair must not cross. Both are properties of the program;
    neither reads a metric. A retry keyed on meets_success_criteria would have
    the agent regenerate until the hypothesis came out supported."""
    import ast
    import inspect
    import textwrap

    for fn in (sandbox.check_training_batching,):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        function = tree.body[0]
        assert isinstance(function, ast.FunctionDef)
        # No parameter carries a result into either check...
        parameters = [a.arg for a in function.args.args]
        assert not any(p in parameters for p in ("results", "metrics", "meets_success_criteria"))
        # ...and neither does the executable body, docstring excluded, mention one.
        body = function.body[1:] if ast.get_docstring(function) else function.body
        code = "\n".join(ast.unparse(node) for node in body)
        assert "meets_success_criteria" not in code
        assert "metrics" not in code


# -- the "train it properly" checks --------------------------------------------
#
# From Barkla job 10424865: the run accumulated baseline_losses and
# enhanced_losses every epoch, reported neither, and left an 18x gap that could
# equally have been "this architecture is worse" or "this architecture never
# trained". These two close that, without either of them being able to see which
# arm won.


def _falling(n=40):
    """A curve still descending steeply when the budget ran out."""
    return [10.0 * (0.9**i) for i in range(n)]


def _plateaued(n=40):
    return [10.0 * (0.5**i) if i < 10 else 0.0098 for i in range(n)]


def test_training_diagnostics_requires_a_loss_curve_per_arm():
    findings = sandbox.check_training_diagnostics({"mae": 0.5}, trains_with_optimizer=True)
    assert len(findings) == 1
    assert sandbox.TRAINING_HISTORY_KEY in findings[0]


def test_training_diagnostics_ignores_a_run_with_no_optimizer():
    """An sklearn fit or a statistical test has no curve to report."""
    assert sandbox.check_training_diagnostics({"mae": 0.5}, trains_with_optimizer=False) == []


def test_training_diagnostics_accepts_a_reported_history():
    metrics = {"training_history": {"baseline": _plateaued(), "enhanced": _plateaued()}}
    assert sandbox.check_training_diagnostics(metrics, trains_with_optimizer=True) == []


def test_convergence_flags_a_curve_still_falling():
    metrics = {"training_history": {"enhanced": _falling()}}
    findings = sandbox.check_training_convergence(metrics, trains_with_optimizer=True)
    assert findings and "still improving" in findings[0]
    assert "measures the training budget" in findings[-1]


def test_convergence_accepts_a_plateaued_curve():
    metrics = {"training_history": {"baseline": _plateaued(), "enhanced": _plateaued()}}
    assert sandbox.check_training_convergence(metrics, trains_with_optimizer=True) == []


def test_convergence_flags_a_curve_too_short_to_judge():
    metrics = {"training_history": {"baseline": [5.0, 4.0, 3.0]}}
    findings = sandbox.check_training_convergence(metrics, trains_with_optimizer=True)
    assert findings and "too few to show convergence" in findings[0]


def test_convergence_is_blind_to_which_arm_is_winning():
    """The safety property. The same curve is judged the same way whether it
    belongs to the baseline or the treatment, and whether it is the better or
    the worse arm — so this can fix undertraining but never favour a verdict."""
    as_baseline = sandbox.check_training_convergence(
        {"training_history": {"baseline": _falling(), "enhanced": _plateaued()}}, True
    )
    as_enhanced = sandbox.check_training_convergence(
        {"training_history": {"baseline": _plateaued(), "enhanced": _falling()}}, True
    )
    assert len(as_baseline) == len(as_enhanced) == 2
    assert as_baseline[0].replace("baseline", "X") == as_enhanced[0].replace("enhanced", "X")


def test_convergence_ignores_a_malformed_history():
    for history in ({"a": "not a list"}, {"a": [1, "two"]}, [], "nope", None):
        metrics = {"training_history": history}
        assert sandbox.check_training_convergence(metrics, trains_with_optimizer=True) == []


def test_trains_with_torch_optimizer():
    assert sandbox.trains_with_torch_optimizer("optimizer_baseline.step()")
    assert not sandbox.trains_with_torch_optimizer("scheduler.step()")
    assert not sandbox.trains_with_torch_optimizer("model.fit(X, y)")
    assert not sandbox.trains_with_torch_optimizer("def f(")


def test_the_convergence_loop_never_reads_the_verdict():
    """The line this must not cross, enforced structurally rather than by
    review: a retry keyed on meets_success_criteria would regenerate until the
    hypothesis came out supported."""
    import ast as _ast
    import inspect
    import textwrap

    for fn in (sandbox.check_training_convergence, sandbox.check_training_diagnostics):
        tree = _ast.parse(textwrap.dedent(inspect.getsource(fn)))
        function = tree.body[0]
        assert isinstance(function, _ast.FunctionDef)
        assert "meets_success_criteria" not in [a.arg for a in function.args.args]
        body = function.body[1:] if _ast.get_docstring(function) else function.body
        code = "\n".join(_ast.unparse(node) for node in body)
        assert "meets_success_criteria" not in code
        assert "success" not in code


# -- sandbox.parse_requirements_lines ------------------------------------------
#
# Regression for Barkla job 10424998: the model echoed the prompt's own section
# placeholder, so requirements.txt held the literal text `<empty>`. That was
# merged into .resolved_requirements.txt and made uv reject the whole file,
# taking numpy, pandas, scipy, scikit-learn and torch with it — and because an
# env-provisioning failure is deliberately never retried, one stray line ended
# the plan where a real defect would have got ten attempts.


@pytest.mark.parametrize(
    "text",
    [
        "<empty>",
        "<empty section — no third-party packages needed>",
        "  <empty>  \n",
        "None needed\n",  # starts with a letter, but see the asserts below
    ],
)
def test_parse_requirements_drops_placeholder_prose(text):
    parsed = sandbox.parse_requirements_lines(text)
    assert all(not line.startswith("<") for line in parsed)


def test_parse_requirements_drops_the_exact_line_that_broke_the_run():
    assert sandbox.parse_requirements_lines("<empty>") == []


def test_parse_requirements_keeps_real_packages():
    text = "numpy\npandas>=2.0\nscikit-learn==1.5.0\ntorch\n"
    assert sandbox.parse_requirements_lines(text) == [
        "numpy",
        "pandas>=2.0",
        "scikit-learn==1.5.0",
        "torch",
    ]


def test_parse_requirements_drops_comments_and_blanks():
    text = "# generated\n\nnumpy\n\n  # trailing note\npandas\n"
    assert sandbox.parse_requirements_lines(text) == ["numpy", "pandas"]


def test_parse_requirements_survives_a_placeholder_mixed_with_real_packages():
    """The costly shape: one bad line must not discard the good ones."""
    assert sandbox.parse_requirements_lines("<empty>\nnumpy\ntorch\n") == ["numpy", "torch"]


# -- Hub query construction keeps benchmark names intact -----------------------
#
# Barkla 10426431: the Planner asked for "CoNLL-2003 dataset", the tokenizer
# produced ["conll", "2003", "dataset"], the year was dropped as a bare number,
# and the query became "conll" — for which the Hub returns conll2000, conll2002
# and a non-servable conll2003, spending all three probes. The query "conll2003"
# returns Davlan/conll2003_noMISC, which the viewer serves. The run went to
# synthetic data over one dropped token.


@pytest.mark.parametrize(
    ("description", "expected_first"),
    [
        ("CoNLL-2003 dataset", "conll2003"),
        ("CIFAR-10 images", "cifar10"),
        ("MNIST_10k digits", "mnist10"),
        ("SQuAD 2.0 questions", "squad2"),
    ],
)
def test_keyword_queries_tries_the_benchmark_name_first(description, expected_first):
    from research_pipeline.agents.coder import huggingface_client

    assert huggingface_client._keyword_queries(description)[0] == expected_first


def test_keyword_queries_still_drops_an_unattached_number():
    """A number not joined to a name really does crowd out the words that match."""
    from research_pipeline.agents.coder import huggingface_client

    assert huggingface_client._keyword_queries("500 students survey") == ["students survey"]


def test_keyword_queries_keeps_the_plain_query_as_a_fallback():
    """The compound is speculative — 'survey 2024' merges to something that
    matches nothing — so it is an extra query, never a replacement."""
    from research_pipeline.agents.coder import huggingface_client

    queries = huggingface_client._keyword_queries("survey 2024 responses")
    assert queries[0] == "survey2024"
    assert "survey responses" in queries


def test_keyword_queries_unchanged_for_prose_with_no_benchmark_name():
    from research_pipeline.agents.coder import huggingface_client

    assert huggingface_client._keyword_queries(
        "a survey of 500 undergraduate students measuring sleep quality"
    ) == ["survey undergraduate students measuring", "survey undergraduate"]


def test_a_model_proposed_archive_is_not_probed(tmp_path):
    """Barkla 10426431 spent one of four probes fetching a conll2003.zip that
    acquire.describe could never have read. The catalogue connectors already
    apply this filter to their resources; the model connector did not."""
    model = FakeChatModel(
        {
            "Name up to": json.dumps(
                {
                    "sources": [
                        {"url": "https://x.example/conll2003.zip", "name": "conll"},
                        {"url": "https://x.example/train.csv", "name": "conll csv"},
                    ],
                    "why": "w",
                }
            )
        }
    )
    proposed = _agent(tmp_path, model)._propose_data_sources("CoNLL-2003 dataset")
    assert [c.url for c in proposed] == ["https://x.example/train.csv"]
