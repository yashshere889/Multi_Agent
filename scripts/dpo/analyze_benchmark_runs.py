"""Score Coder Agent benchmark runs, correcting for the verdict-serialisation defect.

    uv run python scripts/dpo/analyze_benchmark_runs.py \
        dissertation/data/coder-benchmark-runs/10431703 \
        dissertation/data/coder-benchmark-runs/10431840 --latex

`benchmark.py` counts a case as interpretable only when it completed *and*
`meets_success_criteria` is a real bool — deliberately strict, since a truthy
non-verdict must never read as success. Generated `evaluate()` code, however,
routinely computes that field from a comparison over numpy values, yielding
numpy.bool_; the template's `json.dump(..., default=str)` then records the
string "True". The strict test rejects it, and a correct verdict is silently
discarded.

This script reports both numbers: `interpretable` exactly as the harness scored
it, and `interpretable (corrected)` after parsing verdicts that survived only as
strings. The gap between them is the measurement error, not a change in the
system's behaviour, and the dissertation reports it as such. Runs produced after
the template fix should show no gap at all.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

STATUS_INCOMPLETE = ("skipped", "code_generated_not_run", "submitted_to_slurm", "slurm_job_failed")


def parse_verdict(value: object) -> object:
    """Recover a verdict that survived serialisation only as a string.

    Mirrors the template's own `_normalize_verdict`: parse, never cast, because
    bool("False") is True. Anything unrecognised stays unrecognised.
    """
    if value is True or value is False:
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return value


def score(run_dir: Path) -> dict:
    cases = []
    for case_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        summaries = glob.glob(str(case_dir / "coder_agent_summary_*.json"))
        if not summaries:
            continue
        try:
            summary = json.loads(Path(summaries[0]).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for exp in summary.get("experiments", []) or []:
            results = exp.get("results") or {}
            raw = results.get("meets_success_criteria")
            fixed = parse_verdict(raw)
            status = exp.get("status")
            completed = status == "completed"
            inputs = (exp.get("data_provenance") or {}).get("inputs") or []
            cases.append(
                {
                    "case": case_dir.name,
                    "status": status,
                    "raw": raw,
                    "corrected": fixed,
                    "interpretable_as_scored": completed and (raw is True or raw is False),
                    "interpretable_corrected": completed and (fixed is True or fixed is False),
                    "stringified": completed and raw is not fixed,
                    "kinds": sorted({i.get("kind") for i in inputs if i.get("kind")}),
                    "fixes": len(exp.get("fix_history") or []),
                    "sources": [h.get("error_source") for h in (exp.get("fix_history") or [])],
                }
            )
    return {"run": run_dir.name, "cases": cases}


def summarize(scored: dict) -> dict:
    cases = scored["cases"]
    total = len(cases)
    return {
        "run": scored["run"],
        "total": total,
        "completed": sum(c["status"] == "completed" for c in cases),
        "skipped": sum(c["status"] in STATUS_INCOMPLETE for c in cases),
        "as_scored": sum(c["interpretable_as_scored"] for c in cases),
        "corrected": sum(c["interpretable_corrected"] for c in cases),
        "stringified": sum(c["stringified"] for c in cases),
        "real_data": sum(bool(c["kinds"]) and "synthetic_surrogate" not in c["kinds"] for c in cases),
        "fixes": sum(c["fixes"] for c in cases),
    }


def aggregate(summaries: list[dict]) -> dict:
    """Mean and observed range for one arm.

    Range rather than a standard deviation: at four or five runs an sd invites a
    confidence interval this sample size does not support, whereas min-max states
    exactly what was seen and nothing more.
    """
    out = {"n": len(summaries)}
    for key in ("completed", "as_scored", "corrected", "real_data", "fixes"):
        values = [s[key] for s in summaries]
        out[key] = {
            "mean": sum(values) / len(values) if values else float("nan"),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "values": values,
        }
    return out


LABELS = (
    ("Completed", "completed"),
    ("Interpretable (as scored)", "as_scored"),
    ("Interpretable (corrected)", "corrected"),
    ("Real data only", "real_data"),
    ("Fix attempts", "fixes"),
)


def report_arms(arms: dict) -> None:
    aggregates = {name: aggregate(runs) for name, runs in arms.items()}
    width = max(len(label) for label, _ in LABELS) + 2

    header = "metric".ljust(width)
    for name in arms:
        header += ("%s (n=%d)" % (name, aggregates[name]["n"])).rjust(26)
    print(header)
    print("-" * len(header))
    for label, key in LABELS:
        row = label.ljust(width)
        for name in arms:
            stat = aggregates[name][key]
            row += ("%.1f  [%d-%d]" % (stat["mean"], stat["min"], stat["max"])).rjust(26)
        print(row)

    print("\nper-run values:")
    for name in arms:
        print("  %s:" % name)
        for label, key in LABELS:
            print("      %-26s %s" % (label, aggregates[name][key]["values"]))

    names = list(arms)
    if len(names) == 2:
        first, second = (aggregates[n]["corrected"] for n in names)
        print(
            "\ninterpretable (corrected): %s mean %.2f [%d-%d]  vs  %s mean %.2f [%d-%d]"
            % (
                names[0], first["mean"], first["min"], first["max"],
                names[1], second["mean"], second["min"], second["max"],
            )
        )
        if first["max"] >= second["min"] and second["max"] >= first["min"]:
            print(
                "  RANGES OVERLAP - this sample does not separate the arms. Report the "
                "overlap; do not report a difference."
            )
        else:
            print("  ranges are disjoint across every run in each arm.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--latex", action="store_true", help="also emit a LaTeX table")
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=DIR[,DIR...]",
        help="group runs into an arm, e.g. --arm 'discovery on=a,b,c'. Repeatable.",
    )
    args = parser.parse_args()

    if args.arm:
        arms = {}
        for spec in args.arm:
            name, _, dirs = spec.partition("=")
            arms[name.strip()] = [
                summarize(score(Path(d.strip()))) for d in dirs.split(",") if d.strip()
            ]
        report_arms(arms)
        return

    scored = [score(run) for run in args.runs]
    summaries = [summarize(s) for s in scored]

    header = f"{'metric':<28}" + "".join(f"{s['run']:>14}" for s in summaries)
    print(header)
    print("-" * len(header))
    for label, key in (
        ("cases", "total"),
        ("completed", "completed"),
        ("skipped / not run", "skipped"),
        ("interpretable (as scored)", "as_scored"),
        ("interpretable (corrected)", "corrected"),
        ("  of which stringified", "stringified"),
        ("real data only", "real_data"),
        ("fix attempts", "fixes"),
    ):
        print(f"{label:<28}" + "".join(f"{s[key]:>14}" for s in summaries))

    for s in scored:
        affected = [c for c in s["cases"] if c["stringified"]]
        if affected:
            print(f"\n{s['run']}: verdict lost to serialisation in {len(affected)} case(s):")
            for c in affected:
                print(f"    {c['case']:<40} recorded {c['raw']!r} -> {c['corrected']!r}")

    if args.latex:
        print("\n% ---- for the dissertation ----")
        print("\\begin{tabular}{l" + "r" * len(summaries) + "}")
        print("\\toprule")
        print("\\textbf{Metric} & " + " & ".join(f"\\textbf{{{s['run']}}}" for s in summaries) + " \\\\")
        print("\\midrule")
        for label, key in (
            ("Cases", "total"),
            ("Completed", "completed"),
            ("Interpretable (as scored)", "as_scored"),
            ("Interpretable (corrected)", "corrected"),
            ("Real data only", "real_data"),
            ("Fix attempts", "fixes"),
        ):
            print(f"{label} & " + " & ".join(str(s[key]) for s in summaries) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
