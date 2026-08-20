"""Repairs that are not "ask the model for new code".

Two of them, and what they have in common is that neither needs a model call:
installing a package the code correctly imported, and making an experiment
smaller when it ran out of time or memory. Both re-run source that was never
wrong, which is why neither should consume the fix-attempt budget —
`max_fix_attempts` exists to bound bad code.

No settings are read here and nothing is logged, same rule as sandbox.py, so
this stays unit-testable without a cluster or a model. `coder_agent` supplies
the budgets and does the dispatching.
"""

from __future__ import annotations

import re

from . import diagnose, sandbox

# Knobs that make a run smaller without changing what it measures. Halving
# `draws` costs posterior precision; it does not turn the experiment into a
# different experiment — which is what makes this a legitimate automatic repair
# where editing, say, the model formula would not be.
#
# Each has a floor: `chains` halved to 0 would not sample at all.
DOWNSCALE_KNOBS: dict[str, int] = {
    "draws": 250,
    "tune": 250,
    "chains": 2,
    "iter_sampling": 250,
    "iter_warmup": 250,
    "n_samples": 500,
    "num_samples": 500,
    "n_iter": 100,
    "max_iter": 100,
    "epochs": 1,
    "batch_size": 8,
    "n_estimators": 50,
    "n_rows": 1000,
    "sample_size": 1000,
    "max_rows": 1000,
    "n_boot": 100,
    "bootstrap_samples": 100,
    "n_permutations": 100,
}


def downscale(code: str) -> tuple[str, list[str]]:
    """Halve the known cost knobs. Returns (new_code, human-readable changes).

    Only assignments whose *name* is a known knob are touched, so a literal
    `2020` that happens to be a year is left alone. An empty change list means
    there was nothing to shrink — the caller should ask for a different approach
    rather than re-running the same thing.
    """
    changes: list[str] = []
    result = code

    for knob, floor in DOWNSCALE_KNOBS.items():
        pattern = re.compile(rf"(?P<prefix>\b{re.escape(knob)}\b\s*=\s*)(?P<value>\d+)")

        def shrink(match: re.Match[str], knob: str = knob, floor: int = floor) -> str:
            current = int(match.group("value"))
            reduced = max(floor, current // 2)
            if reduced >= current:
                return match.group(0)
            changes.append(f"{knob}: {current} -> {reduced}")
            return f"{match.group('prefix')}{reduced}"

        result = pattern.sub(shrink, result)

    return result, changes


def install_for(python_executable, diagnosis: diagnose.ExecutionDiagnosis) -> tuple[bool, str]:
    """Install what a `missing_dependency` diagnosis says is missing.

    Tries the mapped distribution first, then the raw import name: the mapping
    is a lookup table with gaps, and a package whose import and distribution
    names happen to match is the common case, so one failed install is a cheap
    way to cover both without needing the table to be complete.
    """
    module = diagnosis.module or ""
    package = diagnosis.package or diagnose.resolve_package(module)

    ok, detail = sandbox.install_into_env(python_executable, [package])
    if not ok and package != module and module:
        ok, fallback_detail = sandbox.install_into_env(python_executable, [module])
        if ok:
            return True, f"installed {module!r} (mapped name {package!r} was not on the index)"
        detail = f"{detail}\n--- retry as {module!r} ---\n{fallback_detail}"

    if ok:
        return True, f"installed {package!r} to satisfy `import {module}`"
    return False, f"could not install {package!r} for `import {module}`: {detail[-800:]}"
