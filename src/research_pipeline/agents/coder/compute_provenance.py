"""Say what an experiment's numbers cost in compute, and withhold the
hypothesis verdict when a deterministic repair shrank the run itself.

This is `provenance.py`'s failure arriving down a different road. There, an
experiment invents its inputs and still produces metrics that look exactly like
results. Here, an experiment that exceeded its time or memory budget has its own
cost knobs halved by `repair.downscale` and is re-run — and the metrics that
come back look exactly like results too, because nothing on them records that
the model saw half the epochs it was written to see.

Which knob was halved is the whole question:

    PRECISION_KNOBS     draws, chains, bootstrap resamples. A smaller value
                        estimates the same quantity less tightly. Reported as
                        normal; the verdict stands.
    MEASUREMENT_KNOBS   epochs, n_estimators, max_iter, dataset size. A smaller
                        value fits a different, worse model. An undertrained
                        model loses to its baseline for a reason that has
                        nothing to do with the hypothesis, and `writer_agent`
                        maps that losing `False` to **"refuted"** — so a
                        timeout would publish a refutation the run never
                        earned.

So a downscale that touched a measurement knob turns `meets_success_criteria`
into the string `"unknown"`, which the Writer already reads as "inconclusive",
exactly as a synthetic input does. The metrics themselves are untouched and
still reported — they describe what the pipeline did, which is worth reading —
and the model's own claim is kept under
`model_reported_meets_success_criteria`. Only the verdict is withheld.

Downscaling remains the right repair: a truncated run beats no run at all, and
this does not stop one happening. It only stops the truncation being invisible
in the thing the Writer reads.

The knob classification itself lives in `repair.py`, next to the tables that
define it, so there is one answer to "does shrinking this change what we
measure". Reads no settings and calls no model, same rule as `sandbox.py` and
`repair.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import repair

VERDICT_FULL = "ran at the size it was generated for — metrics describe the experiment as written"
VERDICT_PRECISION = (
    "resampling was reduced to fit the compute budget — the estimates are less precise but "
    "measure the same quantity, so findings remain interpretable as evidence"
)
VERDICT_TRUNCATED = (
    "the run was truncated to fit the compute budget — the metrics describe a smaller or "
    "undertrained model rather than the experiment as designed, and are NOT interpretable as "
    "evidence for or against the hypothesis"
)

WITHHELD_BECAUSE = (
    "A deterministic downscale shrank this run to fit its compute budget, changing what the "
    "experiment measures rather than only how precisely it measures it. These metrics describe "
    "a smaller or undertrained model, so they say nothing about the hypothesis. See "
    "compute_provenance.json."
)


def truncated(changes: list[str]) -> bool:
    """Whether any downscale changed what this experiment measures."""
    return bool(repair.measurement_changes(changes))


def verdict(changes: list[str]) -> str:
    """The compute-validity stamp — computed, never asked of the model."""
    if not changes:
        return VERDICT_FULL
    return VERDICT_TRUNCATED if truncated(changes) else VERDICT_PRECISION


def as_document(changes: list[str], timeout_seconds: int | None = None) -> dict[str, Any]:
    measurement = repair.measurement_changes(changes)
    precision = [change for change in changes if change not in measurement]
    return {
        "downscaled": bool(changes),
        "changes": list(changes),
        # Split rather than merely labelled, because these are the two entries
        # a reader actually acts on: one explains a wider credible interval,
        # the other explains why there is no verdict at all.
        "measurement_changes": measurement,
        "precision_changes": precision,
        "timeout_seconds": timeout_seconds,
        "compute_validity": verdict(changes),
        "ran_at_full_size": not changes,
    }


def write(changes: list[str], path: Path, timeout_seconds: int | None = None) -> dict[str, Any]:
    document = as_document(changes, timeout_seconds)
    path.write_text(json.dumps(document, indent=2))
    return document


def apply_to_results(results: dict, changes: list[str]) -> dict:
    """Withhold the hypothesis verdict when the run was truncated to fit its
    budget.

    A precision-only downscale returns `results` untouched: fewer posterior
    draws is a wider interval, not a different experiment.
    """
    if not truncated(changes):
        return results

    stamped = dict(results)
    # setdefault, not assignment. `provenance.apply_to_results` may already have
    # withheld this verdict for synthetic inputs, and when it did it recorded
    # the model's real claim here first. Assigning would overwrite that claim
    # with the "unknown" that replaced it, losing the only copy of what the
    # experiment actually reported.
    stamped.setdefault(
        "model_reported_meets_success_criteria", results.get("meets_success_criteria")
    )
    stamped["meets_success_criteria"] = "unknown"
    stamped["compute_validity"] = verdict(changes)
    # Both withholding reasons can be true at once, and a reader who sees only
    # the data one would go looking for a provenance problem that isn't the
    # whole story.
    existing = stamped.get("verdict_withheld_because")
    stamped["verdict_withheld_because"] = (
        f"{existing} {WITHHELD_BECAUSE}" if existing else WITHHELD_BECAUSE
    )
    return stamped
