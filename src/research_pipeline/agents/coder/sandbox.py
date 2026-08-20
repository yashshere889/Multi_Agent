"""Runtime environment probing and isolated execution for generated
experiment code. No LLM calls here — kept separate from coder_agent.py so
the actual execution mechanics are unit-testable without a live model.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import math
import py_compile
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

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


# Import name -> the distribution that actually provides it, for the cases where
# they differ. extract_third_party_imports returns *import* names by design, and
# those get installed verbatim — which is fine for numpy and pandas and fatal for
# sklearn: the `sklearn` distribution on PyPI is a deprecation shim that fails
# the install on purpose and tells you to use scikit-learn. Barkla job 10279165
# died exactly there, on ['numpy', 'pandas', 'sklearn'], with uv printing the
# answer in its own hint.
#
# Applied to the model's declared requirements too, not only to extracted
# imports: a model writing `sklearn` in requirements.txt produces the identical
# failure, and every entry here is a name that is never correct to install as-is.
IMPORT_TO_DISTRIBUTION: dict[str, str] = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "Crypto": "pycryptodome",
    "pkg_resources": "setuptools",
    "mpl_toolkits": "matplotlib",
    "OpenSSL": "pyOpenSSL",
    "serial": "pyserial",
}


def installable_name(requirement: str) -> str:
    """The name to hand a package installer for a requirement line or import name.

    Preserves any version specifier — only the name part is rewritten — so
    `sklearn>=1.3` becomes `scikit-learn>=1.3` rather than losing the pin.
    """
    stripped = requirement.strip()
    if not stripped or stripped.startswith("#"):
        return requirement
    name = stripped
    suffix = ""
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        if sep in name:
            index = name.index(sep)
            name, suffix = name[:index].strip(), name[index:]
            break
    mapped = IMPORT_TO_DISTRIBUTION.get(name)
    return f"{mapped}{suffix}" if mapped else requirement


def _normalize_requirements(requirements: list[str]) -> list[tuple[str, str]]:
    """Strips comments/version specifiers from requirement lines, returning
    (requirement_name, importable_module_name) pairs. Shared by
    missing_packages and _verify_bare_interpreter_imports so both agree on
    what a requirement line actually names."""
    pairs = []
    for line in requirements:
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        for sep in ("==", ">=", "<=", "~=", ">", "<"):
            if sep in name:
                name = name.split(sep, 1)[0].strip()
                break
        pairs.append((name, name.replace("-", "_")))
    return pairs


def missing_packages(requirements: list[str]) -> list[str]:
    """requirements: lines like 'numpy' or 'numpy>=1.20'. Checked via Python's
    import machinery using the requirement name as the module name (good
    enough for the common case; a handful of packages have a distribution
    name that differs from their import name, e.g. scikit-learn -> sklearn —
    generated requirements.txt files are expected to list the import name)."""
    return [
        name
        for name, module_name in _normalize_requirements(requirements)
        if importlib.util.find_spec(module_name) is None
    ]


def _verify_bare_interpreter_imports(requirements: list[str], cwd: Path) -> list[str]:
    """find_spec() (missing_packages, above) only proves a module is
    importable in *this* process — not in a subprocess launched with the same
    interpreter from the experiment directory, which is what run_experiment
    actually does. A 2026-08-19 production run (job 10271093) hit exactly
    that gap: pandas wasn't "missing" by find_spec's reckoning (likely
    visible via an HPC module-loaded site-packages path this process
    inherits), so ensure_experiment_env hands back the bare interpreter on
    faith — but the subprocess run_experiment launches from experiments/H1/
    couldn't import it, and 3 fix attempts regenerated code against a
    failure that was never about the code at all. Runs one real
    `python -c "import ..."` from the same cwd run_experiment will use, so
    what's trusted is what will actually run — same reasoning as this
    function's neighbour, the venv_python.exists() check below. Returns the
    requirement names (not module names) that failed to import, or [] if
    every one imported cleanly (including when there's nothing to check)."""
    pairs = _normalize_requirements(requirements)
    if not pairs:
        return []
    import_stmt = "; ".join(f"import {module_name}" for _, module_name in pairs)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", import_stmt],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return [name for name, _ in pairs]
    return [] if proc.returncode == 0 else [name for name, _ in pairs]


# experiments/_shared/ is generated once per run and freely imports whatever
# it needs (e.g. pandas) — but each experiment's own requirements_txt is
# generated from a prompt that only asks the model to list what *that
# experiment's own sections* import, so a package the shared module alone
# depends on can silently never make it into requirements.txt. A 2026-08-14
# production run (job 10229968) reproduced exactly this: run.py's only fault
# was `from experiments._shared import data_utils`, which imported pandas;
# requirements.txt was empty, so ensure_experiment_env saw nothing missing
# and ran with the bare interpreter, which didn't have pandas either — 3 fix
# attempts regenerated run.py without ever touching the actual gap. This
# extracts the shared module's own imports so ensure_experiment_env can
# provision for them regardless of what the model wrote in requirements.txt.
def extract_third_party_imports(source: str) -> set[str]:
    """AST-parses `source` for its top-level import statements and returns
    the distinct top-level module names it imports, excluding stdlib modules
    and the pipeline's own `experiments` package (shared-infra files import
    each other via `from experiments._shared import ...`, which is never a
    pip-installable requirement). A source that fails to parse returns an
    empty set — syntax errors are compile_check's job, not this one's."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    return names - sys.stdlib_module_names - {"experiments"}


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
# Builtins whose *call* is the problem. Deliberately not in DANGEROUS_PATTERNS:
# a regex cannot tell `eval(user_input)` from `model.eval()`, and `\beval\s*\(`
# matches both — the `.` before `eval` is a word boundary. That false positive
# blocked every PyTorch experiment this pipeline generates, since switching a
# model to inference mode is spelled `model.eval()`, and it was unfixable by
# regeneration: the code was already correct, so all three fix attempts were
# spent being told to remove something that had to stay. See _builtin_call_findings.
DANGEROUS_BUILTIN_CALLS: dict[str, str] = {
    "eval": "eval() call",
    "exec": "exec() call",
    "__import__": "dynamic __import__() call",
}

DANGEROUS_PATTERNS: list[tuple[str, str]] = [
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


class _BuiltinCallVisitor(ast.NodeVisitor):
    """Finds calls to the builtins in DANGEROUS_BUILTIN_CALLS.

    Only a bare name (`eval(...)`) or an explicit `builtins.eval(...)` counts.
    An attribute call on some object — `model.eval()`, `cursor.exec()` — is a
    method that happens to share the name and is not the builtin at all.
    """

    def __init__(self) -> None:
        self.findings: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "builtins":
                name = func.attr
        if name in DANGEROUS_BUILTIN_CALLS:
            self.findings.append((node.lineno, DANGEROUS_BUILTIN_CALLS[name]))
        self.generic_visit(node)


def _builtin_call_findings(code: str) -> list[str]:
    """Dangerous builtin calls, with the line each was found on.

    Both callers run this only after a successful compile check, so the parse
    below all but always succeeds. The regex fallback covers the case where it
    doesn't — a caller reaching for this function directly on unparsed source.
    It is the imprecise path on purpose: better a false positive than silently
    passing an `eval` nobody looked at.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [
            description
            for name, description in DANGEROUS_BUILTIN_CALLS.items()
            if re.search(rf"\b{re.escape(name)}\s*\(", code)
        ]

    visitor = _BuiltinCallVisitor()
    visitor.visit(tree)
    lines = code.splitlines()
    findings = []
    for lineno, description in visitor.findings:
        # The line itself, because "eval() call" alone gave the fix loop nothing
        # to act on — it could not tell the model *where*, and a run that hit
        # this spent every attempt guessing.
        source = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
        findings.append(f"{description} on line {lineno}: {source}" if source else description)
    return findings


def static_safety_check(code: str) -> list[str]:
    """Scans generated Python source for patterns that shouldn't appear in an
    experiment script, before it is executed or submitted anywhere. Returns a
    list of human-readable findings; empty means clean."""
    findings = _builtin_call_findings(code)
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


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _raises_not_implemented(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc
    call_func = exc.func if isinstance(exc, ast.Call) else exc
    name = call_func.id if isinstance(call_func, ast.Name) else getattr(call_func, "attr", None)
    return name == "NotImplementedError"


def check_nontrivial_function_bodies(sections: dict[str, str]) -> list[str]:
    """Checks each required function actually does something, not just names
    itself correctly — see check_required_function_names for the "right name,
    wrong content" half of this problem.

    A real production run reported `status: "completed", fix_attempts: 0` for
    an experiment whose four required functions were each just a comment
    followed by a bare `pass`: valid Python, safe, clears compile_check and
    static_safety_check (nothing dangerous happens) and check_data_fallback
    (no read call to guard), so nothing before execution ever looked at
    whether the body did anything. The model's comments plausibly echoed its
    own instructions back, so this deliberately ignores comments/docstrings
    entirely and looks only at the executable statements.

    A body counts as trivial (after dropping a leading docstring) if it is
    empty, only `pass`, only `...`, or only a bare `raise NotImplementedError`
    — the exact anti-patterns prompts.py's SYSTEM_PROMPT already asks the
    model not to write. This is deliberately narrower than "did it use its
    arguments": a function that does real work with a fixed, hardcoded input
    is legitimate (e.g. build_model() takes no arguments at all), and a
    broader heuristic risks flagging real short implementations as hollow.
    """
    findings = []
    for section_name, expected_fn in REQUIRED_FUNCTION_NAMES.items():
        source = sections.get(section_name, "")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == expected_fn
            ):
                continue
            body = node.body
            if body and _is_docstring_expr(body[0]):
                body = body[1:]
            is_trivial = not body or all(
                isinstance(stmt, ast.Pass)
                or (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is Ellipsis
                )
                or _raises_not_implemented(stmt)
                for stmt in body
            )
            if is_trivial:
                findings.append(
                    f"{section_name}'s `{expected_fn}` has no real implementation — its body "
                    "(ignoring comments and any docstring) is empty, `pass`, `...`, or just "
                    "raises NotImplementedError, so it would run without error but do nothing"
                )
    return findings


def check_hf_dataset_usage(
    configuration_source: str,
    load_data_source: str,
    assumptions_made: list[str],
    hf_dataset: dict,
) -> list[str]:
    """When the dataset appraisal accepted a real, scored dataset and offered
    it to the model (see _hf_dataset_block), checks that load_data() shows some
    sign of actually using it. Returns human-readable
    findings; empty means clean, including whenever no dataset was offered at
    all (`hf_dataset` has no `dataset_id`).

    This is not "the dataset id must appear in the code or the attempt
    fails": prompts.py's HF_DATASET_USAGE_NOTE explicitly sanctions ignoring
    the offered dataset when it doesn't genuinely fit the plan, provided the
    model says so in assumptions_made — a real Hub search match can still be
    the wrong shape for a given plan. So this checks for the dataset id in
    either of the two sanctioned places (used in the generated code, or named
    as declined in assumptions_made) and only flags the silent third option:
    neither — load_data() quietly using something else (typically synthesized
    data) with no trace the offer was ever engaged with, which is exactly
    what a model that echoes past the offer without reading it produces.

    Checked as a plain substring rather than an AST walk for a specific call
    shape, since these strings are the one fixed trace consistent with how the
    prompt hands the dataset over — the model can build the actual read in more
    shapes than are worth enumerating. Four spellings count, because there are
    two ways the dataset can be offered: the repo id raw or percent-encoded
    (the REST path, where the id is in the URL), and the local JSONL's full path
    or bare filename (the download path, where the id never appears in the code
    at all — insisting on it there would fail correct code).
    """
    dataset_id = hf_dataset.get("dataset_id")
    if not dataset_id:
        return []
    dataset_id = str(dataset_id)
    local_path = str(hf_dataset.get("local_path") or "")
    code = f"{configuration_source}\n{load_data_source}"
    spellings = [dataset_id, quote(dataset_id, safe="")]
    if local_path:
        spellings.extend([local_path, Path(local_path).name])
    if any(spelling and spelling in code for spelling in spellings):
        return []
    if any(dataset_id in note for note in assumptions_made):
        return []
    return [
        f"a real dataset ({dataset_id}) was found and offered for this experiment's "
        "load_data(), but neither the generated code nor assumptions_made shows any sign "
        "of it — load_data() appears to have silently used something else (e.g. "
        "synthesized data) instead, without saying so. Either read this dataset as "
        "instructed, or explicitly say in assumptions_made that it doesn't fit and why."
    ]


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


# Case-insensitive placeholder strings a model sometimes leaves in place of a
# real computed metric — not code (compile_check/static_safety_check already
# ran), just a value that made it all the way to results.json.
_PLACEHOLDER_METRIC_VALUES = {
    "n/a",
    "na",
    "tbd",
    "todo",
    "unknown",
    "placeholder",
    "none",
    "pending",
}


def _nonfinite_within(value: object, _depth: int = 0) -> list[str]:
    """The NaN/Infinity values nested anywhere inside a metric, as strings.

    Depth-bounded rather than fully recursive: metrics are small structures, and
    a self-referencing one would otherwise hang the check that is meant to
    protect the run.
    """
    if _depth > 4:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        return [bad for item in value for bad in _nonfinite_within(item, _depth + 1)]
    if isinstance(value, dict):
        return [bad for item in value.values() for bad in _nonfinite_within(item, _depth + 1)]
    return []


def check_results_plausibility(metrics: dict) -> list[str]:
    """Sanity-checks a completed experiment's own reported metrics before
    read_results_json_for_diagnosis's success is trusted as a real result.
    Returns human-readable findings; empty means clean.

    Deliberately narrow: this catches the sloppy tail of hollow completions —
    evaluate() returning no metrics at all, NaN/Infinity (almost always a
    division by zero or a computation that never touched real data), or an
    obvious placeholder string ("N/A", "TBD") standing in for a number — not
    a plausible-looking fabricated one. A hardcoded `return {"accuracy":
    0.85}` that ignores its input entirely produces a result indistinguishable
    from a real one at this remove; nothing this cheap can tell the two
    apart. check_nontrivial_function_bodies already covers the bodies simple
    enough to catch structurally (bare pass/.../NotImplementedError); this is
    the equivalent check for the *output* of a body that runs without error.

    `bool` values are excluded from the numeric checks even though Python
    treats bool as an int subclass — a metric that is a genuine boolean flag
    (e.g. "converged": false) is not a "metric is exactly zero" finding.
    """
    if not metrics:
        return [
            "evaluate() returned no metrics at all — results.json's metrics object is "
            "empty, so there is nothing to check success criteria against"
        ]

    findings = []
    numeric_values = {
        name: value
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for name, value in numeric_values.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            findings.append(
                f"metric '{name}' is {value} — usually a division by zero or a computation "
                "that never actually ran on real data"
            )

    # A metric is often a container rather than a scalar — a credible interval
    # is [low, high], a per-group result is a dict — and a NaN inside one is the
    # same broken computation as a NaN at the top level, just out of reach of
    # the isinstance check above. A Barkla run reported a posterior mean beside
    # its own [low, high] interval; had the interval been the part that failed,
    # nothing here would have noticed.
    for name, value in metrics.items():
        if isinstance(value, (list, tuple, dict)):
            bad = _nonfinite_within(value)
            if bad:
                findings.append(
                    f"metric '{name}' contains {', '.join(bad)} — a value inside it is not a "
                    "real number, so whatever produced it failed rather than returned"
                )

    if numeric_values and not findings and all(value == 0 for value in numeric_values.values()):
        findings.append(
            f"every numeric metric is exactly 0 ({', '.join(sorted(numeric_values))}) — check "
            "whether evaluate() is actually computing anything from its input, or just "
            "returning a default"
        )

    for name, value in metrics.items():
        if isinstance(value, str) and value.strip().lower() in _PLACEHOLDER_METRIC_VALUES:
            findings.append(
                f"metric '{name}' is the placeholder value {value!r}, not a real number"
            )

    return findings


def ensure_experiment_env(
    experiment_dir: Path,
    requirements_path: Path,
    network_available: bool,
    extra_requirements: list[str] | None = None,
    venv_root: Path | None = None,
) -> tuple[Path | None, str | None]:
    """Ensures a Python interpreter with the experiment's requirements
    installed. Returns (python_executable, error_message) — exactly one is
    None. Uses the current interpreter directly when nothing extra is needed;
    creates an isolated venv under the experiment directory only if packages
    are missing, so the shared pipeline environment is never modified by
    generated code. Prefers `uv` (faster) when it's on PATH, and falls back
    to the stdlib `venv` + `pip` otherwise, so provisioning works regardless
    of which environment launched the pipeline (e.g. an Apptainer container
    that was only ever set up with plain pip).

    `venv_root` places the venv somewhere other than beside the results. On a
    cluster that matters: a venv is thousands of small files, Barkla's scratch
    and fastscratch carry inode quotas (300k / 500k files), and localscratch has
    none and is node-local NVMe besides. The venv is disposable and the results
    are not, so they need not share a filesystem. Defaults to the experiment
    directory, which is right on a laptop.

    `extra_requirements` is for packages the model never had a reason to put
    in requirements.txt — namely what experiments/_shared/ itself imports
    (see extract_third_party_imports). They're checked the same way as
    requirements.txt entries (skipped if already importable) and, when
    actually missing, installed the same way; they're just never written to
    requirements.txt on disk, since that file documents what *this
    experiment's own code* declared, not what shared infra happens to need."""
    requirements = requirements_path.read_text().splitlines() if requirements_path.exists() else []
    combined_requirements = requirements + list(extra_requirements or [])
    missing = missing_packages(combined_requirements)
    if not missing:
        # find_spec() says nothing's missing, but that alone isn't proof the
        # bare interpreter can actually import these from a subprocess — see
        # _verify_bare_interpreter_imports's docstring for the incident (job
        # 10271093) this closes. Falls through to real provisioning below,
        # exactly like anything missing_packages itself flagged, if it can't.
        missing = _verify_bare_interpreter_imports(combined_requirements, experiment_dir)
        if not missing:
            return Path(sys.executable), None

    if not network_available:
        return None, f"missing package(s) {missing} and no network access to install them"

    # `pip`/`uv pip install -r` reads whatever file is named below, not the
    # `missing` list above — so when extra_requirements added something not
    # already in requirements.txt on disk, install from a merged copy instead
    # of the original, or that package would be "detected" as needed but
    # never actually installed. The original requirements.txt on disk is left
    # untouched either way — it documents what the model itself declared.
    # Names are mapped to what a package installer can actually resolve at this
    # point, not earlier: missing_packages above checks *importability*, which is
    # keyed on the import name, so rewriting sooner would ask whether
    # `scikit-learn` is importable and always conclude it is missing.
    merged = [
        installable_name(line)
        for line in dict.fromkeys(requirements + list(extra_requirements or []))
    ]
    if merged != list(dict.fromkeys(requirements)) or extra_requirements:
        install_requirements_path = experiment_dir / ".resolved_requirements.txt"
        install_requirements_path.write_text("\n".join(merged) + "\n")
    else:
        install_requirements_path = requirements_path

    venv_dir = (
        (venv_root / experiment_dir.name / ".venv") if venv_root else experiment_dir / ".venv"
    )
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
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
                    str(install_requirements_path),
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
                [str(venv_python), "-m", "pip", "install", "-r", str(install_requirements_path)],
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


def module_importable(python_executable: Path, module: str, cwd: Path) -> bool:
    """Can *this* interpreter import this module, from the directory run.py runs in?

    A subprocess, not find_spec, and the target interpreter rather than
    sys.executable — same reasoning as _verify_bare_interpreter_imports, whose
    docstring records the run where the two answers differed. What is trusted
    here has to be what will actually run.

    Exists because an installer's exit code is not proof of repair: `uv pip
    install pandas` returns 0 when it believes pandas is already present for
    that interpreter, and on Barkla job 10279290 that happened six times in a
    row while the experiment kept failing to import it. Believing the exit code
    turned one unfixable environment into six wasted installs per attempt.
    """
    if not module:
        return False
    try:
        proc = subprocess.run(
            [str(python_executable), "-c", f"import {module.split('.')[0]}"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def install_into_env(python_executable: Path, packages: list[str]) -> tuple[bool, str]:
    """Install packages into an already-provisioned interpreter. Returns (ok, detail).

    ensure_experiment_env provisions once, up front, from what the model
    declared in requirements.txt. This is the other half: a package the code
    imports but nobody declared only becomes visible when the run fails on it,
    and at that point the repair is to install it and run the *same code*
    again — not to regenerate source that was never wrong. A 2026-08-19 summary
    shows the cost of not having this: three fix attempts against one
    `ModuleNotFoundError: No module named 'pandas'`.

    Same uv-preferred/pip-fallback choice as ensure_experiment_env, for the same
    reason: uv is faster where present, and an Apptainer container set up with
    plain pip must still work.

    Never raises — the caller needs the detail to decide whether to try a
    different distribution name or stop, and an exception here would lose the
    difference between "no such package" and "no network".
    """
    wanted = [p for p in dict.fromkeys(packages) if p]
    if not wanted:
        return False, "nothing to install"

    use_uv = shutil.which("uv") is not None
    if use_uv:
        command = ["uv", "pip", "install", "--python", str(python_executable), *wanted]
    else:
        command = [str(python_executable), "-m", "pip", "install", *wanted]

    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, f"installing {wanted} via {'uv' if use_uv else 'pip'} timed out after 600s"
    except OSError as exc:
        return False, f"could not run {'uv' if use_uv else 'pip'}: {exc}"

    if proc.returncode == 0:
        return True, (proc.stdout or "").strip()[-500:]
    return False, ((proc.stderr or "") + (proc.stdout or "")).strip()[-500:]


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
