"""Runtime environment probing and isolated execution for generated
experiment code. No LLM calls here — kept separate from coder_agent.py so
the actual execution mechanics are unit-testable without a live model.
"""

from __future__ import annotations

import ast
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

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

NETWORK_PROBE_HOST = "pypi.org"
NETWORK_PROBE_PORT = 443
NETWORK_PROBE_TIMEOUT_SECONDS = 3.0


def _load_template(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text()


def has_network_access(
    host: str = NETWORK_PROBE_HOST,
    port: int = NETWORK_PROBE_PORT,
    timeout: float = NETWORK_PROBE_TIMEOUT_SECONDS,
) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def missing_packages(requirements: list[str]) -> list[str]:
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


def compile_check(py_files: list[Path]) -> str | None:
    """Byte-compiles each file to catch syntax errors without executing
    anything. Returns None if every file compiles, else a message describing
    the first failure encountered."""
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            return f"{path.name}: {exc.msg}"
    return None


# Small quantized models writing a multi-line call/dict/list habitually
# append a trailing '\' to every line regardless of whether Python needs one
# there — it already continues implicitly inside an open (), [], or {}. That
# habit isn't just harmless noise: once the previous statement's brackets
# have already closed, a stray trailing backslash still force-continues onto
# the next (unrelated) line, merging two statements into one invalid logical
# line. Only a bare backslash immediately before the newline (no trailing
# whitespace after it) is ever valid line-continuation syntax, so that's the
# only shape this strips.
_TRAILING_BACKSLASH_RE = re.compile(r"[ \t]*\\\n")


def _strip_redundant_line_continuations(source: str) -> str:
    return _TRAILING_BACKSLASH_RE.sub("\n", source)


# A newline the model intended inside a generated code section occasionally
# arrives as the two literal characters '\' + 'n' rather than a real line
# break: a small quantized model habitually double-escapes when it re-quotes
# its own previous output during a repair turn, which decodes to that literal
# 2-char sequence instead of a newline. Python's tokenizer then treats the
# lone backslash as an (invalid) explicit line continuation: "unexpected
# character after line continuation character". Unlike
# _strip_redundant_line_continuations, this can't be a blanket find/replace
# across the whole file — a *valid* string literal legitimately containing
# "\n" (e.g. print("a\nb")) is indistinguishable from the corrupted case by
# regex alone, and rewriting every occurrence would corrupt those too. So
# this only touches the exact line and column py_compile's SyntaxError
# reports, which is precise because the corruption itself is what caused the
# error there.
_LINE_CONTINUATION_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}


def _repair_literal_line_continuation(source: str, exc: SyntaxError) -> str | None:
    if exc.msg != "unexpected character after line continuation character":
        return None
    if not exc.lineno or not exc.offset:
        return None
    lines = source.splitlines(keepends=True)
    line_idx = exc.lineno - 1
    if line_idx >= len(lines):
        return None
    line = lines[line_idx]
    # exc.offset (1-indexed) points at the offending character itself — the
    # one right after the backslash — not at the backslash.
    char_idx = exc.offset - 1
    if char_idx < 1 or line[char_idx - 1] != "\\":
        return None
    replacement = _LINE_CONTINUATION_ESCAPES.get(line[char_idx])
    if replacement is None:
        return None
    lines[line_idx] = line[: char_idx - 1] + replacement + line[char_idx + 1 :]
    return "".join(lines)


def lenient_compile_check(
    source: str, filename: str, max_repair_attempts: int = 5
) -> tuple[str, str | None]:
    """Byte-compiles `source` (a string, not yet written to disk). On a
    SyntaxError, retries after applying whichever repair applies —
    _repair_literal_line_continuation for a backslash+letter sequence that
    should have been a real newline/tab/carriage-return, else
    _strip_redundant_line_continuations for a harmless trailing backslash —
    bounded by max_repair_attempts since a single generation can contain more
    than one corrupted spot. Returns (source_to_use, error_message):
    error_message is None on success, and source_to_use is the repaired
    version only if that's what actually compiled — never a transformation
    applied without verifying it fixed something, exactly like
    llm_json._fix_invalid_escapes/_loads_lenient apply their own repair only
    when the strict parse already failed."""
    current = source
    last_error: SyntaxError | None = None
    for _ in range(max_repair_attempts):
        try:
            compile(current, filename, "exec")
            return current, None
        except SyntaxError as exc:
            last_error = exc
            repaired = _repair_literal_line_continuation(current, exc)
            if repaired is None:
                repaired = _strip_redundant_line_continuations(current)
            if repaired == current:
                break
            current = repaired
    assert last_error is not None
    return source, f"{filename}: {last_error.msg} (line {last_error.lineno})"


# Regex rather than AST: this runs against model-generated code that the fix
# loop may rewrite several times, and a pattern list is cheap to extend when a
# new footgun shows up. It is a second layer behind the isolated per-experiment
# venv, not the sandboxing boundary itself — but it *is* the only gate on the
# SLURM auto-submit path, where nothing ever runs locally first.
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
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
    (
        r"os\.environ(?:\.get\(\s*)?\[?[\"'][^\"']*(SECRET|TOKEN|PASSWORD|API_KEY|AWS_)",
        "credential-like environment variable access",
    ),
]


def static_safety_check(code: str) -> list[str]:
    """Scans generated Python source for patterns that shouldn't appear in an
    experiment script, before it is executed or submitted anywhere. Returns a
    list of human-readable findings; empty means clean."""
    findings = []
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            findings.append(description)
    return findings


# AST rather than regex, unlike static_safety_check above — and for a reason that
# isn't stylistic: the question here is "is this call inside a try/except?",
# which is structural. A regex can find `pd.read_csv(` but cannot tell a guarded
# read from an unguarded one, and flagging every read would fire on correct code.
#
# Attribute names, matched regardless of the module alias, so pd/pandas/whatever
# all work. `load`/`loadtxt`/`genfromtxt` are numpy's file readers.
_LOCAL_READ_ATTRS = {
    "read_csv",
    "read_excel",
    "read_parquet",
    "read_json",
    "read_table",
    "read_feather",
    "read_stata",
    "read_pickle",
    "read_hdf",
    "load",
    "loadtxt",
    "genfromtxt",
}
_LOCAL_READ_NAMES = {"open"}
# A read whose first argument comes from one of these is parsing bytes already in
# memory — a fetched response body — not opening a file that has to exist on
# disk. `read_csv(StringIO(response.text))` and `read_json(response.text)` are
# the two shapes the Dataset Viewer path produces.
_IN_MEMORY_CALLS = {"StringIO", "BytesIO", "json", "text", "decode"}
_IN_MEMORY_ATTRS = {"text", "content", "body", "stdout"}
# The Dataset Viewer host the codegen prompt points the model at. A load_data
# that fetches from it is on the sanctioned remote-data path, where
# `pd.read_csv(StringIO(response.text))`-style parsing is normal — see
# check_data_fallback's docstring for why that exempts the whole function.
_DATASET_VIEWER_HOST = "datasets-server.huggingface.co"


def _called_name(node: ast.Call) -> str:
    """The called function's name as written: "open", "read_csv", "get"."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _reads_in_memory_bytes(first_arg: ast.expr | None) -> bool:
    if isinstance(first_arg, ast.Call):
        return _called_name(first_arg) in _IN_MEMORY_CALLS
    if isinstance(first_arg, ast.Attribute):
        return first_arg.attr in _IN_MEMORY_ATTRS
    return False


def _reads_a_local_file(node: ast.Call) -> bool:
    name = _called_name(node)
    if isinstance(node.func, ast.Name):
        if name not in _LOCAL_READ_NAMES:
            return False
    elif isinstance(node.func, ast.Attribute):
        if name not in _LOCAL_READ_ATTRS:
            return False
    else:
        return False
    return not _reads_in_memory_bytes(node.args[0] if node.args else None)


def _fetches_from_dataset_viewer(tree: ast.AST) -> bool:
    """Whether this function names the Dataset Viewer host anywhere — an inline
    URL, one assigned to a local first, or an f-string built around it. Every
    string constant in the function is scanned rather than only those inside the
    request call, because the URL is usually built a line or two before it's
    used."""
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _DATASET_VIEWER_HOST in node.value
        for node in ast.walk(tree)
    )


def _unguarded_read_calls(node: ast.AST, inside_try: bool) -> list[ast.Call]:
    """Every local-file read reachable from `node` that is not inside a
    try-block's body. Recursive rather than ast.walk because "am I inside a try?"
    is exactly the context ast.walk discards.

    Only `Try.body` counts as guarded — not its handlers or `finally`. A read in
    an `except:` branch is the fallback path, and if *it* assumes a file exists
    the experiment still dies; it needs its own guard.
    """
    found: list[ast.Call] = []
    if isinstance(node, ast.Call) and not inside_try and _reads_a_local_file(node):
        found.append(node)
    for field, value in ast.iter_fields(node):
        children = value if isinstance(value, list) else [value]
        for child in children:
            if not isinstance(child, ast.AST):
                continue
            child_inside_try = inside_try or (isinstance(node, ast.Try) and field == "body")
            found.extend(_unguarded_read_calls(child, child_inside_try))
    return found


def check_data_fallback(load_data_function_source: str) -> list[str]:
    """Checks that a generated `load_data` doesn't simply assume its data is
    there. Returns human-readable findings; empty means clean.

    This enforces an instruction the prompt has always given and nothing ever
    verified: read the data defensively and fall back to a synthesized stand-in
    if it isn't available. A real production run failed exactly here — the model
    recorded "assumes survey_data.csv is present" in assumptions_made for a plan
    that required collecting *new* data, and the experiment died on a
    FileNotFoundError that no fix attempt could diagnose as a design problem.
    Same principle as everywhere else in this package: if compliance is
    checkable, Python checks it rather than the model being trusted.

    Flagged: `open`/`pandas.read_*`/`numpy.load`-style reads that aren't inside a
    try-block's body. Not flagged: a read of an in-memory buffer
    (`read_csv(StringIO(body))`), and nothing at all if the function fetches from
    the Dataset Viewer host the prompt offers it — that path *is* the sanctioned
    real-data route, and parsing its response body with `read_csv`/`read_json` is
    the normal way to consume it.

    Scoped to load_data's own source, so a read hidden in a `helpers` function
    isn't seen. That's deliberate: `helpers` is parsed in isolation, so a read
    there that load_data already wraps in try/except would look unguarded and
    produce a false failure — and a false failure burns the fix budget on code
    that was already correct.
    """
    try:
        tree = ast.parse(load_data_function_source)
    except SyntaxError:
        # Not this check's job to report: _attempt_once runs the compile check on
        # the whole rendered run.py first, and it reports syntax errors with
        # proper line numbers.
        return []

    if _fetches_from_dataset_viewer(tree):
        return []

    findings = []
    for call in _unguarded_read_calls(tree, inside_try=False):
        name = _called_name(call)
        findings.append(
            f"line {call.lineno}: {name}(...) reads a local file with no try/except around it, so "
            "the experiment dies if that file doesn't exist. Wrap the read, log the failure, and "
            "fall back to a small synthesized stand-in dataset (recording that in assumptions_made "
            "and the README) — or fetch the data over HTTP instead"
        )
    return findings


# templates/run.py.template's orchestration calls exactly these four functions,
# by these exact bare global names — the wiring is fixed template text, not
# model output, so a section that defines something differently named is simply
# not callable.
REQUIRED_FUNCTION_NAMES: dict[str, str] = {
    "load_data_function": "load_data",
    "build_model_function": "build_model",
    "run_experiment_function": "run_experiment",
    "evaluate_function": "evaluate",
}


def check_required_function_names(sections: dict[str, str]) -> list[str]:
    """Checks each generated code section actually defines the function the
    template will call. Returns human-readable findings; empty means clean.

    The model is asked for four sections and trusted to put a correctly named
    function in each. A section that instead defines `def load_the_dataset():`
    compiles fine, is safe, and guards its reads — so it clears every other
    check, gets a full `uv venv` provisioned (up to ~600s, the single most
    expensive step in the loop), and only then dies on a NameError at
    execution. The defect is cheaply detectable before any of that, so it is
    detected here instead: same "check it rather than trust it" rule the rest of
    this package follows.

    Only *top-level* definitions in the section count, deliberately — not
    ast.walk. A function nested inside a class or another function isn't
    reachable as a bare global name from the template's orchestration, so
    finding one there would be a false pass.

    A section that doesn't parse is skipped silently rather than flagged:
    reporting syntax errors is compile_check's job, on the whole rendered run.py
    where the line numbers are meaningful, and a broken section should not also
    collect a spurious "wrong function name" finding for a function it may well
    define correctly.
    """
    findings = []
    for section_name, expected_fn in REQUIRED_FUNCTION_NAMES.items():
        source = sections.get(section_name, "")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if expected_fn not in defined:
            findings.append(
                f"{section_name} does not define a top-level `def {expected_fn}(...)` "
                f"(found: {sorted(defined) or 'no top-level function'}) — run.py's fixed "
                f"orchestration calls {expected_fn}() by that exact name, so the experiment "
                "would fail with a NameError"
            )
    return findings


def check_shared_infra_files(files: dict[str, str]) -> tuple[dict[str, str], str]:
    """Runs the same checks a single experiment's run.py gets — lenient
    compile check, then static safety check — against every shared
    infrastructure file.

    This exists because every experiment that imports shared infrastructure
    trusts it unconditionally: a bug there is invisible to (and permanently
    unfixable by) the per-experiment fix loop, which only ever regenerates
    that one experiment's own run_py_sections and never touches
    experiments/_shared/ — see coder_agent.py's module docstring. Checking it
    here, once, right after it's generated, is what makes a shared-infra bug
    something the fix loop can actually see and react to.

    Returns (files with any backslash-continuation repairs applied, problem
    description) — problem is "" when every .py file is clean."""
    repaired = dict(files)
    for name, content in files.items():
        if not name.endswith(".py"):
            continue
        fixed_content, error = lenient_compile_check(content, name)
        repaired[name] = fixed_content
        if error:
            return repaired, f"syntax error: {error}"
    for name, content in repaired.items():
        if not name.endswith(".py"):
            continue
        findings = static_safety_check(content)
        if findings:
            return repaired, f"{name} was flagged by the static safety check: {'; '.join(findings)}"
    return repaired, ""


def read_results_json_for_diagnosis(experiment_dir: Path) -> tuple[dict | None, str | None]:
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
) -> tuple[Path | None, str | None]:
    """Ensures a Python interpreter with the experiment's requirements
    installed. Returns (python_executable, error_message) — exactly one is
    None. Uses the current interpreter directly when nothing extra is needed;
    creates an isolated venv under the experiment directory only if packages
    are missing, so the shared pipeline environment is never modified by
    generated code. Prefers `uv` (faster) when it's on PATH, and falls back
    to the stdlib `venv` + `pip` otherwise, so provisioning works regardless
    of which environment launched the pipeline (e.g. an Apptainer container
    that was only ever set up with plain pip)."""
    requirements = requirements_path.read_text().splitlines() if requirements_path.exists() else []
    missing = missing_packages(requirements)
    if not missing:
        return Path(sys.executable), None

    if not network_available:
        return None, f"missing package(s) {missing} and no network access to install them"

    venv_dir = experiment_dir / ".venv"
    venv_python = venv_dir / "bin" / "python"
    use_uv = shutil.which("uv") is not None
    tool = "uv" if use_uv else "pip"
    # A prior fix attempt can get partway through provisioning (venv created,
    # then the pip/uv install step fails or times out) and leave venv_dir
    # behind. Both `uv venv` and the stdlib `venv` module refuse to
    # (re)populate an existing directory, so without this every subsequent
    # fix attempt would fail on venv creation itself with the same
    # "already exists" error, regardless of what changed in the generated
    # code — the fix loop could never actually recover.
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    try:
        if use_uv:
            subprocess.run(
                ["uv", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(venv_python),
                    "-r",
                    str(requirements_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc))[-500:]
        return (
            None,
            f"failed to provision an isolated environment for {missing} via {tool}: {detail}",
        )
    except subprocess.TimeoutExpired:
        return None, f"provisioning an isolated environment for {missing} via {tool} timed out"

    # A zero-exit-code `uv venv`/`pip install` isn't proof the interpreter is
    # actually there: a 2026-08-12 production run on an Apptainer container
    # with fastscratch (a network filesystem) as CODER_EXPERIMENTS_DIR saw
    # both subprocess calls report success with no venv_python at the end —
    # both `uv venv` and the stdlib venv module create it via a symlink (or a
    # copy on some platforms) that network filesystems and container overlay
    # layers don't always honor the same way a local disk does. Without this
    # check, the caller trusts the returned path on faith and
    # run_experiment's subprocess.run raises a bare, uncaught
    # FileNotFoundError that crashes the whole orchestrator run instead of
    # degrading to the already-handled "couldn't provision an environment"
    # result every other failure in this function produces.
    if not venv_python.exists():
        return (
            None,
            f"{tool} reported success provisioning {missing}, but no interpreter exists at "
            f"{venv_python} afterward — the venv directory may not be usable on this filesystem",
        )

    return venv_python, None


def run_experiment(
    python_executable: Path, run_script: Path, cwd: Path, timeout_seconds: int
) -> tuple[bool, str]:
    """Runs run_script with python_executable, cwd set to the experiment
    directory (so relative paths like ./results.json resolve correctly).
    Returns (succeeded, message) — message is empty on success, or a
    diagnosable tail of stdout/stderr (or a timeout note) on failure."""
    # run_script and python_executable are both resolved to absolute paths
    # before being handed to the subprocess. experiments_dir
    # (CODER_EXPERIMENTS_DIR) defaults to a relative "experiments", so a
    # caller passing experiment_dir / "run.py" (or a venv python under it)
    # hands us relative paths here too; since cwd below is that same
    # relative directory, a relative path would get re-resolved against the
    # subprocess's cwd instead of the launching process's, doubling the
    # directory (experiments/H1/experiments/H1/run.py) — or, for
    # python_executable, raising a FileNotFoundError for a venv interpreter
    # that does exist, just not under that doubled path.
    try:
        proc = subprocess.run(
            [str(python_executable.resolve()), str(run_script.resolve())],
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
        "# TODO: activate/create a venv and `pip install -r requirements.txt` here if needed\n"
        if has_requirements
        else ""
    )
    return _load_template("run.sbatch.template").format(
        hypothesis_id=hypothesis_id, venv_activation=venv_activation
    )


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
        "__AGENT_CONFIGURATION__": agent_configuration.strip()
        or "# (no extra configuration needed)",
        "__AGENT_LOAD_DATA_FUNCTION__": load_data_function.strip(),
        "__AGENT_BUILD_MODEL_FUNCTION__": build_model_function.strip(),
        "__AGENT_RUN_EXPERIMENT_FUNCTION__": run_experiment_function.strip(),
        "__AGENT_EVALUATE_FUNCTION__": evaluate_function.strip(),
        "__AGENT_HELPERS__": agent_helpers.strip() or "# (no helper functions needed)",
    }
    for token, value in substitutions.items():
        template = template.replace(token, value)
    return template
