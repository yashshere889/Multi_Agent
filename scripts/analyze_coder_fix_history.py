"""Aggregates error_source frequency across every coder_agent_summary_*.json
this project has ever produced, so "which failure category actually dominates"
is a command to run instead of a memory note written by hand after each batch
(see e.g. the 2026-08-11 batch post-mortem — five recurring bug patterns found
by manually cross-referencing fix_history across ten runs).

    uv run python scripts/analyze_coder_fix_history.py
    uv run python scripts/analyze_coder_fix_history.py outputs/ runs/ --json report.json

Deliberately outside `research_pipeline`: this reads whatever
`coder_agent_summary_*.json` files it's pointed at (default: everything under
`outputs/` and `runs/`, recursively — the two places this repo writes them,
per config.py's `coder_output_dir` and the webapp's `runs/<uuid>/` layout) and
never imports the package, so it works against JSON produced by any version of
schema.py, past or future, without needing to import that version's code.

Every field this reads is treated as optional except `experiments` and each
entry's `error_source` — old summaries predate `resolved`, `same_error_streak`
and `starter_used`; this script's whole reason to exist is comparing runs
across time, so it must not choke on the shape a prior version of the pipeline
wrote. A summary file that doesn't parse as JSON, or has no top-level
`experiments` list, is skipped with a warning rather than aborting the whole
scan — one corrupt file from an interrupted run shouldn't hide every other
file's data.

Three counts per error_source, beyond a raw count:
  resolved       - the regeneration that followed got past this check
                    (FixAttempt.resolved — see coder_agent._cleared_previous_error).
  stuck          - this attempt was already the model's 2nd+ consecutive
                    failure on the same error_source (same_error_streak >= 2 —
                    see coder_agent._consecutive_error_streak). Only
                    meaningful for summaries written after that field existed;
                    older ones simply score 0 here, not "not stuck".
  unresolved_at_end - this was the *last* fix_history entry for its experiment
                    and it was never resolved: the failure category still
                    standing when the fix loop stopped, whatever the reason
                    (budget exhausted, or the last regeneration's own output
                    was itself unparseable). The strongest signal for "which
                    check is actually costing finished experiments" — a
                    resolved last entry (e.g. a plan whose only fix attempt
                    succeeded before it was separately deferred to sbatch for
                    an unrelated reason) does not count here even though the
                    experiment's own status may still read
                    "code_generated_not_run".
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ROOTS = ["outputs", "runs"]
SUMMARY_GLOB = "**/coder_agent_summary_*.json"


@dataclass
class ErrorSourceStats:
    occurrences: int = 0
    resolved: int = 0
    stuck: int = 0
    unresolved_at_end: int = 0

    @property
    def resolved_rate(self) -> float:
        return self.resolved / self.occurrences if self.occurrences else 0.0


@dataclass
class Report:
    files_scanned: int = 0
    files_skipped: int = 0
    experiments_scanned: int = 0
    fix_attempts_scanned: int = 0
    by_error_source: dict[str, ErrorSourceStats] = field(default_factory=dict)

    def stats_for(self, error_source: str) -> ErrorSourceStats:
        return self.by_error_source.setdefault(error_source, ErrorSourceStats())


def find_summary_files(roots: list[Path]) -> list[Path]:
    """Every coder_agent_summary_*.json under any of `roots`, recursively. A
    root that's itself a summary file is accepted as-is, so a single-file
    invocation doesn't need a glob."""
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.glob(SUMMARY_GLOB)))
    return files


def _load(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping %s — could not parse: %s", path, exc)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("experiments"), list):
        logger.warning("Skipping %s — no top-level 'experiments' list", path)
        return None
    return data


def aggregate(paths: list[Path]) -> Report:
    report = Report()
    for path in paths:
        data = _load(path)
        if data is None:
            report.files_skipped += 1
            continue
        report.files_scanned += 1
        for exp in data["experiments"]:
            if not isinstance(exp, dict):
                continue
            history = exp.get("fix_history")
            if not isinstance(history, list) or not history:
                continue
            report.experiments_scanned += 1
            last_index = len(history) - 1
            for i, entry in enumerate(history):
                if not isinstance(entry, dict):
                    continue
                source = entry.get("error_source")
                if not source:
                    continue
                report.fix_attempts_scanned += 1
                stats = report.stats_for(source)
                stats.occurrences += 1
                resolved = bool(entry.get("resolved"))
                if resolved:
                    stats.resolved += 1
                streak = entry.get("same_error_streak")
                if isinstance(streak, int) and streak >= 2:
                    stats.stuck += 1
                if i == last_index and not resolved:
                    stats.unresolved_at_end += 1
    return report


def render(report: Report) -> str:
    lines = [
        f"Scanned {report.files_scanned} summary file(s) "
        f"({report.files_skipped} skipped/unparseable), "
        f"{report.experiments_scanned} experiment(s) with at least one fix attempt, "
        f"{report.fix_attempts_scanned} fix attempt(s) total.",
        "",
    ]
    if not report.by_error_source:
        lines.append("No fix_history entries found.")
        return "\n".join(lines)

    header = (
        f"{'error_source':<28}{'count':>7}{'resolved':>10}{'resolved%':>11}"
        f"{'stuck':>7}{'unresolved@end':>16}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    ranked = sorted(report.by_error_source.items(), key=lambda kv: kv[1].occurrences, reverse=True)
    for source, stats in ranked:
        lines.append(
            f"{source:<28}{stats.occurrences:>7}{stats.resolved:>10}"
            f"{stats.resolved_rate * 100:>10.0f}%{stats.stuck:>7}{stats.unresolved_at_end:>16}"
        )
    return "\n".join(lines)


def _report_to_json(report: Report) -> dict:
    return {
        "files_scanned": report.files_scanned,
        "files_skipped": report.files_skipped,
        "experiments_scanned": report.experiments_scanned,
        "fix_attempts_scanned": report.fix_attempts_scanned,
        "by_error_source": {
            source: {
                "occurrences": stats.occurrences,
                "resolved": stats.resolved,
                "resolved_rate": stats.resolved_rate,
                "stuck": stats.stuck,
                "unresolved_at_end": stats.unresolved_at_end,
            }
            for source, stats in report.by_error_source.items()
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help=f"Directories (searched recursively) or summary files to scan. Default: {DEFAULT_ROOTS}",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the aggregated counts as JSON to this path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args(argv)
    paths = find_summary_files([Path(r) for r in args.roots])
    report = aggregate(paths)
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(_report_to_json(report), indent=2) + "\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
