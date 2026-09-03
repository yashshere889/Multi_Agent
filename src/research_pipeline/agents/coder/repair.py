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


# pandas 3 removed .fillna(method=...). Two Barkla jobs hit this and the model
# reproduced byte-identical code across every fix attempt despite the fix
# prompt naming the exact replacement (see diagnose.REMOVED_APIS) — the
# guidance is correct but the model does not reliably apply it, the same
# reason `downscale` below does not leave a cost knob to the model either.
# `.bfill()`/`.ffill()` compute exactly what `.fillna(method=...)` did, so this
# is a pure syntax swap, not a rewrite decision.
_FILLNA_METHOD_TO_CALL = {"ffill": "ffill", "pad": "ffill", "bfill": "bfill", "backfill": "bfill"}

# Non-inplace form: `.fillna(method='bfill')` sitting inside a larger
# expression (assigned, chained, returned). The closing paren must follow the
# method value directly — an `inplace=True` kwarg between them means this
# regex does not match, which is what keeps the two forms below from
# overlapping.
_FILLNA_METHOD_EXPR_RE = re.compile(
    r"\.fillna\(\s*method\s*=\s*['\"](?P<method>ffill|pad|bfill|backfill)['\"]\s*\)"
)

# Inplace form as its own statement: `df.fillna(method='bfill', inplace=True)`.
# Narrower on purpose. A plain syntax swap here would drop the mutation and
# turn a working line into a silent no-op — worse than the TypeError it
# replaces, and in a pipeline whose whole point is not reporting numbers that
# were never really computed, a silent no-op is the one failure mode worth
# refusing to guess at. Only rewritten when the entire line is exactly this
# shape: a bare `<receiver>.fillna(...)` call with nothing else on it, so
# reassigning `<receiver>` is provably equivalent to what `inplace=True` did.
# A receiver that is itself a call (`get_df().fillna(..., inplace=True)`)
# cannot be reassigned back to safely and is deliberately excluded by the
# character class, not merely unmatched by accident.
_FILLNA_METHOD_INPLACE_STMT_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<receiver>[A-Za-z_][\w.\[\]'\"]*)\.fillna\(\s*"
    r"method\s*=\s*['\"](?P<method>ffill|pad|bfill|backfill)['\"]\s*,\s*"
    r"inplace\s*=\s*True\s*\)[ \t]*$",  # [ \t]*, not \s*: MULTILINE's $ sits
    # before the newline, and \s* would still consume that newline itself
    # (it's whitespace too), silently deleting the following line's break.
    re.MULTILINE,
)


def patch_removed_pandas_fillna(code: str) -> tuple[str, list[str]]:
    """Rewrite pandas 3's removed `.fillna(method=...)` to `.ffill()`/`.bfill()`.

    Two shapes, two safety bars — see the regexes above for why. Returns
    (new_code, human-readable changes); an empty change list means neither
    shape was found, the same "nothing to shrink" contract `downscale` uses,
    so the caller can tell "already safe" apart from "did nothing".

    Covers exactly the two forms observed on Barkla (10411325's inplace form,
    10416110's expression form) — this is not a general Python rewriter, and
    every other REMOVED_APIS entry (DataFrame.append, numpy's scalar aliases,
    get_feature_names) still goes to the model, which is where a rewrite that
    needs judgment belongs.
    """
    changes: list[str] = []

    def _expr_sub(match: re.Match[str]) -> str:
        method = _FILLNA_METHOD_TO_CALL[match.group("method")]
        changes.append(f".fillna(method={match.group('method')!r}) -> .{method}()")
        return f".{method}()"

    result = _FILLNA_METHOD_EXPR_RE.sub(_expr_sub, code)

    def _stmt_sub(match: re.Match[str]) -> str:
        method = _FILLNA_METHOD_TO_CALL[match.group("method")]
        receiver = match.group("receiver")
        changes.append(
            f"{receiver}.fillna(method={match.group('method')!r}, inplace=True) -> "
            f"{receiver} = {receiver}.{method}()"
        )
        return f"{match.group('indent')}{receiver} = {receiver}.{method}()"

    result = _FILLNA_METHOD_INPLACE_STMT_RE.sub(_stmt_sub, result)
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
