"""Classify an execution failure by *kind*, so the repair goes somewhere useful.

Every runtime failure used to be recorded as `run_experiment` — the stage that
broke, never what broke — and every `error_source` routes to the same place:
regenerate the code and try again. That is right for a logic bug and useless
for a missing package, and the summary from a 2026-08-19 run shows the cost:
three fix attempts, three regenerations, and `ModuleNotFoundError: No module
named 'pandas'` returned verbatim each time. Rewriting the source could not
have installed anything.

So this module answers one question — what kind of failure is this? — and hands
back a route. `repair.py` decides what to do about it; nothing here executes,
installs or calls a model, which is what makes it testable against a corpus of
real tracebacks.

The classes map onto `schema.VALID_ERROR_SOURCES` rather than replacing it:
`run_experiment` remains the fallback for a genuine logic bug, and the four new
members split off the cases that need a different repair. `_ERROR_STAGE_ORDER`
in coder_agent.py must list them in the same order — see AGENTS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import sandbox

# Routes. `repair.route_execution_failure` dispatches on these.
ROUTE_ENV = "env_repair"
ROUTE_DOWNSCALE = "downscale"
ROUTE_REGENERATE = "regenerate"
ROUTE_TERMINAL = "terminal"

# Import name -> the distribution that provides it, for the cases where they
# differ AND installing that distribution genuinely satisfies that import.
# sandbox._normalize_requirements already maps the common direction; this is the
# repair direction, used only when a traceback names a module we don't have.
# The table itself lives in sandbox.IMPORT_TO_DISTRIBUTION, beside the install
# path that also needs it — one table, so a name fixed for the repair path is
# fixed for provisioning too.

# Imports that NO installable distribution provides, mapped to the replacement
# and the API note the fix prompt should carry.
#
# The distinction from IMPORT_TO_PACKAGE above is the entire point, and
# collapsing the two is a specific, observed failure: installing `pymc` in
# response to `import pymc3` succeeds, changes nothing, and the identical
# ImportError comes back on the re-run — an install that reports success while
# the loop makes no progress. `sklearn` -> `scikit-learn` is an alias and the
# install works; `pymc3` -> `pymc` is a successor and it cannot.
DEAD_IMPORTS: dict[str, tuple[str, str]] = {
    "pymc3": (
        "pymc",
        "PyMC 3 is end-of-life and no package provides `pymc3`. Use PyMC 5: `import pymc as pm`. "
        "`pm.Model()`, `pm.sample()` and the distributions carry over; `return_inferencedata=True` "
        "is now the default, and theano/aesara references become pytensor.",
    ),
    "pystan": (
        "cmdstanpy",
        "PyStan 2.x is unmaintained and will not build on a cluster. Use CmdStanPy: write the Stan "
        "program to a .stan file, then `CmdStanModel(stan_file=...)` and `.sample()`.",
    ),
    "theano": ("pytensor", "Theano is dead. PyMC 5 uses PyTensor: `import pytensor.tensor as pt`."),
    "aesara": ("pytensor", "Aesara was renamed. Use `import pytensor.tensor as pt`."),
}

# APIs removed by a major release of a package the experiments routinely use,
# matched on the traceback line and mapped to the replacement to write instead.
#
# Same reasoning as DEAD_IMPORTS, one level down: the failure is not that
# something is missing from the environment but that the model wrote a call
# that no longer exists, and the error text names what broke without naming
# what to write instead. requirements_txt pins nothing (deliberately — see
# prompts.py), so experiments resolve the newest major while the model writes
# the idioms it was trained on, and the fix loop rediscovers the same
# TypeError every attempt. Barkla job 10411325 spent all three that way on one
# `fillna(method=...)` call, the identical error each time.
REMOVED_APIS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"fillna\(\) got an unexpected keyword argument ['\"]method['\"]"),
        "pandas 3 removed the `method=` argument of .fillna(). Use the dedicated methods: "
        "`df.ffill()` / `df.bfill()` (and `df.ffill(axis=1)` for the axis form). "
        "`.fillna(value)` for a constant is unchanged. Note `inplace=True` is also being "
        "phased out — assign the result instead: `df = df.bfill()`.",
    ),
    (
        re.compile(r"'DataFrame' object has no attribute 'append'"),
        "pandas 2 removed DataFrame.append(). Use `pd.concat([df, other], ignore_index=True)`.",
    ),
    (
        re.compile(
            r"module ['\"]numpy['\"] has no attribute ['\"](?:float|int|bool|object|str)['\"]"
        ),
        "numpy removed the `np.float`/`np.int`/`np.bool`/`np.object`/`np.str` aliases. Use the "
        "Python builtins (`float`, `int`, `bool`, `object`, `str`) or the sized dtypes "
        "(`np.float64`, `np.int64`).",
    ),
    (
        re.compile(r"get_feature_names\(\)"),
        "scikit-learn removed .get_feature_names(). Use `.get_feature_names_out()`.",
    ),
]


def removed_api(text: str) -> str | None:
    """Replacement guidance for a call a major release deleted, else None."""
    for pattern, guidance in REMOVED_APIS:
        if pattern.search(text):
            return guidance
    return None


def api_currency_note() -> str:
    """The same removals, told to the model *before* it writes the code.

    REMOVED_APIS repairs one of these after it has already cost an attempt, and
    the table is the only place that knows the replacements — so the codegen
    prompt is rendered from it rather than restating them, the same "one table"
    rule `resolve_package` follows for import names. Prevention is worth more
    than the repair here: requirements_txt pins nothing, so every experiment
    resolves the newest major of packages the model learned an older major of,
    and this is a standing mismatch rather than an occasional accident.
    """
    return "\n".join(f"- {guidance}" for _, guidance in REMOVED_APIS)


_MODULE_RE = re.compile(r"ModuleNotFoundError:\s*No module named ['\"]([\w.]+)['\"]")
_IMPORT_NO_MODULE_RE = re.compile(r"ImportError:\s*No module named ['\"]?([\w.]+)")
_SHARED_LIB_RE = re.compile(r"(lib[\w.+-]*\.so[\w.]*)[^\n]*cannot open shared object file")
_CUDA_OOM_RE = re.compile(r"CUDA out of memory|CUDA error: out of memory|OutOfMemoryError")
_TIMEOUT_RE = re.compile(r"execution timed out after \d+s")
_HOST_OOM_RE = re.compile(r"\bMemoryError\b|Killed\b|exited with code -9\b")


@dataclass(frozen=True)
class ExecutionDiagnosis:
    """What kind of failure this was, in the terms the repair router needs."""

    error_source: str
    route: str
    summary: str
    module: str | None = None
    package: str | None = None
    guidance: str = ""

    @property
    def needs_install(self) -> bool:
        return self.route == ROUTE_ENV


def resolve_package(module: str) -> str:
    """The distribution to install for a missing import. Unknown names pass through.

    Falling through to the import's own name is right far more often than it is
    wrong (numpy, pandas, scipy, arviz, geopandas all match), and a wrong guess
    costs one failed install, which repair.py handles by retrying with the raw
    import name.
    """
    if module in sandbox.IMPORT_TO_DISTRIBUTION:
        return sandbox.IMPORT_TO_DISTRIBUTION[module]
    root = module.split(".")[0]
    return sandbox.IMPORT_TO_DISTRIBUTION.get(root, root)


def dead_import(module: str) -> tuple[str, str] | None:
    """Replacement and API guidance for an import nothing can satisfy, else None."""
    if module in DEAD_IMPORTS:
        return DEAD_IMPORTS[module]
    return DEAD_IMPORTS.get(module.split(".")[0])


def classify_execution_failure(message: str) -> ExecutionDiagnosis:
    """Diagnose the message `sandbox.run_experiment` returns on failure.

    Order matters: a ModuleNotFoundError also looks like a generic non-zero
    exit, so the specific cases are tried before the fallback.
    """
    text = message or ""

    # --- a call a major release deleted -------------------------------------
    # Regeneration, not an env repair: the package is installed and correct, it
    # is the call that is out of date, so installing anything changes nothing.
    # Carries the replacement, because the traceback says what broke but not
    # what to write instead — and without it the model rewrites the same call.
    guidance = removed_api(text)
    if guidance:
        return ExecutionDiagnosis(
            error_source="obsolete_dependency",
            route=ROUTE_REGENERATE,
            summary=f"The experiment calls an API that has been removed. {guidance}",
            guidance=guidance,
        )

    # --- an import nothing can install --------------------------------------
    # Checked before the installable case, or `pymc3` would be sent to pip.
    match = _MODULE_RE.search(text) or _IMPORT_NO_MODULE_RE.search(text)
    if match:
        module = match.group(1)
        replacement = dead_import(module)
        if replacement:
            package, guidance = replacement
            return ExecutionDiagnosis(
                error_source="obsolete_dependency",
                route=ROUTE_REGENERATE,
                summary=(
                    f"`import {module}` cannot be satisfied — no installable package provides it. "
                    f"{guidance}"
                ),
                module=module,
                package=package,
                guidance=guidance,
            )
        return ExecutionDiagnosis(
            error_source="missing_dependency",
            route=ROUTE_ENV,
            summary=f"The experiment imports {module!r}, which is not installed in its environment.",
            module=module,
            package=resolve_package(module),
        )

    # --- a native library pip cannot supply ----------------------------------
    shared = _SHARED_LIB_RE.search(text)
    if shared:
        return ExecutionDiagnosis(
            error_source="missing_system_library",
            route=ROUTE_TERMINAL,
            summary=(
                f"A native library ({shared.group(1)}) is missing. pip cannot supply this — it "
                "needs a cluster module or a rebuilt base environment."
            ),
            module=shared.group(1),
        )

    # --- too big or too slow --------------------------------------------------
    if _TIMEOUT_RE.search(text):
        return ExecutionDiagnosis(
            error_source="resource_limit",
            route=ROUTE_DOWNSCALE,
            summary="The experiment exceeded its wall-clock budget and was killed.",
        )
    if _CUDA_OOM_RE.search(text):
        return ExecutionDiagnosis(
            error_source="resource_limit",
            route=ROUTE_DOWNSCALE,
            summary="The experiment ran out of GPU memory.",
        )
    if _HOST_OOM_RE.search(text):
        return ExecutionDiagnosis(
            error_source="resource_limit",
            route=ROUTE_DOWNSCALE,
            summary="The experiment ran out of host memory.",
        )

    # --- anything else is a code defect, exactly as before -------------------
    return ExecutionDiagnosis(
        error_source="run_experiment",
        route=ROUTE_REGENERATE,
        summary=f"Execution failed: {text}",
    )
