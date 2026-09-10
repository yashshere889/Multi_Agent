#!/usr/bin/env python3
"""One-screen summary of a pipeline run directory.

    python scripts/summarize_run.py <run-dir> [<run-dir> ...]

Written for the single-question debug loop, where the question after every run
is the same three things and reading four JSON files by hand to answer them is
most of the wall clock: what did the Coder Agent actually run on, did the
verdict survive, and is the Writer/Reviewer loop converging.

Reads only the files a run already writes, so it works equally on a debug run, a
batch's output directory, or a run someone else produced days ago.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ISSUE_CATEGORIES = (
    "hallucinations",
    "citation_issues",
    "results_accuracy_issues",
    "hypothesis_coverage_issues",
)


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _question_dirs(root: Path) -> list[Path]:
    """A run dir, or a batch whose outputs hold one dir per question."""
    outputs = root / "outputs" if (root / "outputs").is_dir() else root
    nested = sorted(p for p in outputs.glob("q*") if p.is_dir())
    return nested or [outputs]


def summarize_experiments(directory: Path) -> None:
    for summary_path in sorted(directory.glob("coder_agent_summary_*.json")):
        for experiment in _load(summary_path).get("experiments") or []:
            results = experiment.get("results") or {}
            provenance = experiment.get("data_provenance") or {}
            verdict = results.get("meets_success_criteria")
            print(
                f"  [{experiment.get('hypothesis_id')}] {experiment.get('status')}"
                f"  verdict={verdict!r}  fixes={experiment.get('fix_attempts')}"
            )
            if verdict == "unknown":
                print(f"      withheld: {str(results.get('verdict_withheld_because'))[:96]}")
            for entry in provenance.get("inputs") or []:
                # `discovered` is the whole point of the line: it is the
                # difference between data a human named and data a keyword
                # search found, and it is what needs_confirmation reads.
                mark = "FOUND-BY-SEARCH" if entry.get("discovered") else "named"
                print(f"      {entry.get('kind'):<20} {mark:<16} {str(entry.get('name'))[:52]}")
            unconfirmed = provenance.get("unconfirmed_discovered_inputs") or []
            if unconfirmed:
                print(f"      unconfirmed: {unconfirmed}")


def summarize_reviews(directory: Path) -> None:
    reviews = sorted(
        path
        for path in directory.glob("review_*.json")
        if path.name != "review_log.json"
    )
    if not reviews:
        return
    for path in reviews:
        review = _load(path)
        counts = {c: len(review.get(c) or []) for c in ISSUE_CATEGORIES}
        located = collections.Counter(
            entry.get("location", "?") for entry in (review.get("hallucinations") or [])
        )
        worst = ", ".join(f"{k}:{v}" for k, v in located.most_common(3))
        print(
            f"  iteration {review.get('iteration')}: "
            f"{sum(counts.values()):>4} issues  pass={review.get('overall_pass')}  "
            f"{counts}"
        )
        if worst:
            print(f"      hallucinations by section: {worst}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for target in argv:
        root = Path(target)
        if not root.exists():
            print(f"{target}: no such directory")
            continue
        for directory in _question_dirs(root):
            print(f"\n=== {directory.name} ===")
            summarize_experiments(directory)
            summarize_reviews(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
