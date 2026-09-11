"""Withhold the hypothesis verdict when the metrics could not have come out
any other way.

The third question asked of a completed experiment, after "were its inputs
real?" (`provenance.py`) and "did it run at the size it was designed for?"
(`compute_provenance.py`). Both of those can be answered yes and the numbers
still say nothing, because the measurement itself had nowhere to go.

Two shapes, each observed rather than supposed:

  * **A label derived from an input.** Batch 10460809's Experiment Planner
    designed q026 to split documents at the median entropy and then predict
    which half a document fell in, using entropy as a feature. The Coder built
    exactly that plan, and accuracy, F1 and R² all came back 1.0. The same
    design scores 0.99 on random numbers: the model learns a cutoff on its own
    input. q034 is the same failure in correlations — Spearman 1.0 for both
    arms, p = 1.0 — and was published as a *refutation*.
  * **A comparison with no headroom.** Benchmark case 12 reported precision@5
    of 1.0 for raw counts and 1.0 for TF-IDF in runs 10460812 and 10460813. A
    difference of zero between two arms pinned at the ceiling is not evidence
    that one fails to beat the other, and `writer_agent` maps that `False` to
    **"refuted"**.

Measured over every completed experiment this project recorded — 184 distinct
runs across the benchmarks and pipeline batches — the gate fires on 17, and 9
of those had published a verdict: eight refutations and one "supported". The
recurring cases are benchmark case 03 (every accuracy and F1 at 1.0 in five
runs) and case 12 (both retrieval arms at 1.0 in four).

The response is the one the other two gates give: `meets_success_criteria`
becomes the string "unknown", which the Writer reads as "inconclusive"; the
metrics are reported untouched; the model's own claim is kept under
`model_reported_meets_success_criteria`. It is deliberately *not* a fix-loop
route. `sandbox.check_results_plausibility` sends all-zero metrics back to the
model, which is right for code that computed nothing, but a saturated result
usually comes from a sound implementation of an unsound plan, and regeneration
never touches the plan — q027 spent seven consecutive attempts failing the
plausibility check without escaping it.

Conservative in the direction every gate here is. A metric counts only when
its name identifies it as bounded; a difference, spread or interval end is
never one, because a delta of 0 is what two saturated arms *produce* rather
than an instance of saturation; and the gate fires only when at least two such
metrics exist and every one sits at its perfect value. A genuinely easy task
scoring 1.0 twice is withheld too — withholding wrongly costs an
"inconclusive", and not withholding costs a refutation nobody earned.

Reads no settings and calls no model, same rule as `sandbox.py`, `repair.py`
and `compute_provenance.py`.
"""

from __future__ import annotations

import math
import re

VERDICT_SATURATED = (
    "every bounded metric sits exactly at its perfect value — the measurement could not have come "
    "out differently, so the metrics are NOT interpretable as evidence for or against the hypothesis"
)

WITHHELD_BECAUSE = (
    "Every bounded metric this experiment reported sits exactly at its perfect value ({names}). "
    "That is the signature of a label derived from one of the model's own inputs, or of a "
    "comparison whose arms are both at the ceiling and so cannot differ; either way the numbers "
    "could not have come out otherwise, so the verdict is withheld."
)

# One perfect metric is too weak to judge on — an easy task can score 1.0.
# Two independent bounded measures both perfect is the signature.
MIN_BOUNDED_METRICS = 2

# Bounded above by a perfect 1.0.
_SCORE_TOKENS = frozenset(
    {
        "accuracy",
        "acc",
        "ari",
        "auc",
        "dice",
        "f1",
        "hit",
        "hits",
        "iou",
        "jaccard",
        "map",
        "mrr",
        "ndcg",
        "nmi",
        "precision",
        "r2",
        "recall",
        "roc",
        "sensitivity",
        "specificity",
    }
)
# Bounded by ±1 and perfect at either end: a correlation of -1 is exactly as
# saturated as one of +1.
_CORRELATION_TOKENS = frozenset(
    {"corr", "correlation", "kendall", "pearson", "rho", "spearman", "tau"}
)
# Bounded below by a perfect 0.
_ERROR_TOKENS = frozenset({"brier", "ece"})
# A difference between arms, a spread, or an interval end. Never counted: an
# improvement of 0 is the consequence of saturation, not a case of it.
_EXCLUDED_TOKENS = frozenset(
    {
        "change",
        "ci",
        "delta",
        "diff",
        "difference",
        "gain",
        "improvement",
        "lift",
        "lower",
        "margin",
        "p",
        "sd",
        "se",
        "sem",
        "std",
        "stdev",
        "upper",
        "var",
        "variance",
    }
)


def _perfect_value(name: str) -> tuple[float, bool] | None:
    """(perfect value, sign-insensitive) for a recognisably bounded metric, else None."""
    lowered = name.lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", lowered) if token}
    compact = re.sub(r"[^a-z0-9]", "", lowered)
    if tokens & _EXCLUDED_TOKENS or "pvalue" in compact:
        return None
    if tokens & _ERROR_TOKENS:
        return 0.0, False
    if tokens & _CORRELATION_TOKENS:
        return 1.0, True
    if tokens & _SCORE_TOKENS or "rsquared" in compact:
        return 1.0, False
    return None


def findings(metrics: dict) -> list[str]:
    """The bounded metrics pinned at their perfect value — empty unless every one is.

    Top-level scalars only, and `bool` excluded even though Python counts it as
    an int, for the same reason `sandbox.check_results_plausibility` excludes
    it: `"converged": True` is a flag, not an accuracy of 1.
    """
    if not isinstance(metrics, dict):
        return []
    bounded: dict[str, tuple[float, tuple[float, bool]]] = {}
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        perfect = _perfect_value(str(name))
        if perfect is not None:
            bounded[str(name)] = (float(value), perfect)
    if len(bounded) < MIN_BOUNDED_METRICS:
        return []
    pinned = [
        name
        for name, (value, (target, sign_insensitive)) in bounded.items()
        if math.isclose(abs(value) if sign_insensitive else value, target, rel_tol=0, abs_tol=1e-9)
    ]
    return sorted(pinned) if len(pinned) == len(bounded) else []


def saturated(metrics: dict) -> bool:
    return bool(findings(metrics))


def apply_to_results(results: dict) -> dict:
    """Withhold the verdict when every bounded metric sits at its perfect value.

    Returns `results` itself, untouched, otherwise — so a run this has nothing to
    say about is identical to one that predates it.
    """
    pinned = findings(results.get("metrics") or {})
    if not pinned:
        return results

    stamped = dict(results)
    # setdefault, not assignment, for the reason compute_provenance gives: an
    # earlier gate may already have withheld this verdict and recorded the
    # model's real claim here, and assigning would overwrite it with "unknown".
    stamped.setdefault(
        "model_reported_meets_success_criteria", results.get("meets_success_criteria")
    )
    stamped["meets_success_criteria"] = "unknown"
    stamped["measurement_validity"] = VERDICT_SATURATED
    stamped["saturated_metrics"] = pinned
    reason = WITHHELD_BECAUSE.format(names=", ".join(pinned))
    existing = stamped.get("verdict_withheld_because")
    stamped["verdict_withheld_because"] = f"{existing} {reason}" if existing else reason
    return stamped
