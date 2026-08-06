"""Runtime environment probing and isolated execution for generated
experiment code. No LLM calls here — kept separate from coder_agent.py so
the actual execution mechanics are unit-testable without a live model.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import py_compile
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

NETWORK_PROBE_HOST = "pypi.org"
NETWORK_PROBE_PORT = 443
NETWORK_PROBE_TIMEOUT_SECONDS = 3.0

# "high" estimated_complexity is never run synchronously — see coder_agent.py.
TIMEOUT_SECONDS_BY_COMPLEXITY = {"low": 120, "medium": 300}


def _load_template(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text()


def has_network_access(
    host: str = NETWORK_PROBE_HOST, port: int = NETWORK_PROBE_PORT, timeout: float = NETWORK_PROBE_TIMEOUT_SECONDS
) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def missing_packages(requirements: List[str]) -> List[str]:
    """requirements: lines like 'numpy' or 'numpy>=1.20'. Checked via Python's
    import machinery using the requirement name as the module name (good
    enough for the common case; a handful of packages have a distribution
    name that differs from their import name, e.g. scikit-learn -> sklearn —
    generated requirements.txt files are expected to list the import name)."""
    missing = []
    for line in requirements:
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        for sep in ("==", ">=", "<=", "~=", ">", "<"):
            if sep in name:
                name = name.split(sep, 1)[0].strip()
                break
        module_name = name.replace("-", "_")
        if importlib.util.find_spec(module_name) is None:
            missing.append(name)
    return missing


def compile_check(py_files: List[Path]) -> Optional[str]:
    """Byte-compiles each file to catch syntax errors without executing
    anything. Returns None if every file compiles, else a message describing
    the first failure encountered."""
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            return f"{path.name}: {exc.msg}"
    return None


# Regex rather than AST: this runs against model-generated code that the fix
# loop may rewrite several times, and a pattern list is cheap to extend when a
# new footgun shows up. It is a second layer behind the isolated per-experiment
# venv, not the sandboxing boundary itself — but it *is* the only gate on the
# SLURM auto-submit path, where nothing ever runs locally first.
DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    (r"\beval\s*\(", "eval() call"),
    (r"\bexec\s*\(", "exec() call"),
    (r"\b__import__\s*\(", "dynamic __import__() call"),
    (r"subprocess\.\w+\([^)]*shell\s*=\s*True", "subprocess call with shell=True"),
    (r"\bos\.system\s*\(", "os.system() call"),
    (r"\bos\.popen\s*\(", "os.popen() call"),
    (r"\bshutil\.rmtree\s*\(", "shutil.rmtree() call"),
    (r"\bos\.(remove|unlink)\s*\(", "file deletion via os.remove()/os.unlink()"),
    (r"\bos\.chmod\s*\(", "os.chmod() call"),
    (r"\bsocket\.(socket|create_connection)\s*\(", "raw socket usage"),
    (r"\bpickle\.loads?\s*\(", "pickle load (arbitrary code execution on untrusted data)"),
    (r"\bctypes\b", "ctypes usage"),
    (r"os\.environ(?:\.get\(\s*)?\[?[\"'][^\"']*(SECRET|TOKEN|PASSWORD|API_KEY|AWS_)", "credential-like environment variable access"),
]


def static_safety_check(code: str) -> List[str]:
    """Scans generated Python source for patterns that shouldn't appear in an
    experiment script, before it is executed or submitted anywhere. Returns a
    list of human-readable findings; empty means clean."""
    findings = []
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            findings.append(description)
    return findings


def read_results_json_for_diagnosis(experiment_dir: Path) -> Tuple[Optional[dict], Optional[str]]:
    """Reads an experiment's results.json for the fix loop. Returns
    (results, diagnosis): `results` is the parsed payload when the run
    genuinely succeeded, `diagnosis` describes what's wrong otherwise —
    including the traceback run.py writes into results.json["error"], which
    the success-path reader discards."""
    results_path = experiment_dir / "results.json"
    if not results_path.exists():
        return None, "run.py did not write results.json"

    try:
        data = json.loads(results_path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"results.json is not valid JSON: {exc}"

    if not isinstance(data, dict):
        return None, f"results.json should contain an object, got {type(data).__name__}"

    if data.get("error"):
        return None, f"run.py raised an exception:\n{data['error']}"

    missing = [key for key in ("metrics", "meets_success_criteria") if key not in data]
    if missing:
        return None, f"results.json is missing required key(s): {', '.join(missing)}"

    data.setdefault("notes", "")
    return data, None


def ensure_experiment_env(
    experiment_dir: Path,
    requirements_path: Path,
    network_available: bool,
) -> Tuple[Optional[Path], Optional[str]]:
    """Ensures a Python interpreter with the experiment's requirements
    installed. Returns (python_executable, error_message) — exactly one is
    None. Uses the current interpreter directly when nothing extra is needed;
    creates an isolated `uv venv` under the experiment directory only if
    packages are missing, so the shared pipeline environment is never
    modified by generated code."""
    requirements = requirements_path.read_text().splitlines() if requirements_path.exists() else []
    missing = missing_packages(requirements)
    if not missing:
        return Path(sys.executable), None

    if shutil.which("uv") is None:
        return None, f"missing package(s) {missing} and 'uv' is not on PATH to install them"
    if not network_available:
        return None, f"missing package(s) {missing} and no network access to install them"

    venv_dir = experiment_dir / ".venv"
    venv_python = venv_dir / "bin" / "python"
    try:
        subprocess.run(["uv", "venv", str(venv_dir)], check=True, capture_output=True, text=True, timeout=120)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(venv_python), "-r", str(requirements_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc))[-500:]
        return None, f"failed to provision an isolated environment for {missing}: {detail}"
    except subprocess.TimeoutExpired:
        return None, f"provisioning an isolated environment for {missing} timed out"

    return venv_python, None


def run_experiment(python_executable: Path, run_script: Path, cwd: Path, timeout_seconds: int) -> Tuple[bool, str]:
    """Runs run_script with python_executable, cwd set to the experiment
    directory (so relative paths like ./results.json resolve correctly).
    Returns (succeeded, message) — message is empty on success, or a
    diagnosable tail of stdout/stderr (or a timeout note) on failure."""
    try:
        proc = subprocess.run(
            [str(python_executable), str(run_script)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"execution timed out after {timeout_seconds}s"

    if proc.returncode != 0:
        tail = (proc.stdout[-1000:] + "\n" + proc.stderr[-1000:]).strip()
        return False, f"run.py exited with code {proc.returncode}: {tail[-1500:]}"
    return True, ""


def render_sbatch_template(hypothesis_id: str, has_requirements: bool) -> str:
    venv_activation = (
        "# TODO: activate/create a venv and `pip install -r requirements.txt` here if needed\n" if has_requirements else ""
    )
    return _load_template("run.sbatch.template").format(hypothesis_id=hypothesis_id, venv_activation=venv_activation)


def render_experiment_template(
    *,
    hypothesis_id: str,
    objective: str,
    design: str,
    data_description: str,
    baseline: str,
    success_criteria: str,
    agent_imports: str,
    agent_configuration: str,
    load_data_function: str,
    build_model_function: str,
    run_experiment_function: str,
    evaluate_function: str,
    agent_helpers: str,
) -> str:
    """Renders the complete run.py from templates/run.py.template: the
    EXPERIMENT METADATA block and ORCHESTRATION are fixed/deterministic (not
    model-generated); the four functions plus imports/configuration/helpers
    are the model-generated pieces, spliced in as complete text blocks.

    Uses plain string replacement on unique __TOKEN__ markers rather than
    str.format() — the spliced-in content is arbitrary LLM-generated Python
    source, which routinely contains literal '{' / '}' (dict literals,
    f-strings) that would break format()'s placeholder syntax.
    """
    template = _load_template("run.py.template")
    substitutions = {
        "__EXPERIMENT_ID__": repr(hypothesis_id),
        "__EXPERIMENT_NAME__": repr(f"Experiment {hypothesis_id}"),
        "__HYPOTHESIS_TEXT__": repr(objective),
        "__DESCRIPTION_TEXT__": repr(f"{design} Data: {data_description}"),
        "__BASELINE_TEXT__": repr(baseline),
        "__SUCCESS_CRITERIA_TEXT__": repr(success_criteria),
        "__AGENT_IMPORTS__": agent_imports.strip() or "# (no extra imports needed)",
        "__AGENT_CONFIGURATION__": agent_configuration.strip() or "# (no extra configuration needed)",
        "__AGENT_LOAD_DATA_FUNCTION__": load_data_function.strip(),
        "__AGENT_BUILD_MODEL_FUNCTION__": build_model_function.strip(),
        "__AGENT_RUN_EXPERIMENT_FUNCTION__": run_experiment_function.strip(),
        "__AGENT_EVALUATE_FUNCTION__": evaluate_function.strip(),
        "__AGENT_HELPERS__": agent_helpers.strip() or "# (no helper functions needed)",
    }
    for token, value in substitutions.items():
        template = template.replace(token, value)
    return template
