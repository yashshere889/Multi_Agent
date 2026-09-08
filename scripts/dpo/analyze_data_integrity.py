"""Find verdicts that rest on data the provenance system did not question.

    uv run python scripts/dpo/analyze_data_integrity.py dissertation/data --latex

`provenance.py` withholds a hypothesis verdict when a declared input could not be
resolved and was synthesised: an input labelled `synthetic_surrogate` forces
`meets_success_criteria` to "unknown". Two cases slip past that rule, and both
*assert* a verdict rather than withholding one:

  * **wrong real data** — a real dataset was fetched, but not the one the plan
    called for, so the input is `real_download` and nothing downstream objects;
  * **synthetic content in a real file** — a file genuinely present on disk is
    classed `real_local` whatever it contains.

In every observed instance the model *declared* the problem in
`assumptions_made` and nothing read the declaration. This script reads it.

It is a heuristic over model-written prose, and is therefore deliberately not
wired into the pipeline as a gate by this script alone: it is a measurement
instrument for reporting how often the gap is exercised. Findings are split into
two tiers because the distinction matters and text alone resolves it only
roughly:

  * **asserted** — the text says a substitute *was used* ("Used X as a proxy
    for Y", "was used as the primary data source"). Serious: the verdict rests
    on it.
  * **conditional** — the text describes a guarded fallback that may never have
    fired ("falls back to synthetic data if the fetch fails"). This is the
    behaviour `check_data_fallback` *requires*, so it is reported separately and
    is not by itself a defect.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

SUBSTITUTION = re.compile(
    r"\bproxy\b|\bstand[- ]in\b|\bsubstitut|\binstead of\b|\bin place of\b|\bas a replacement\b",
    re.I,
)
FABRICATED_LABELS = re.compile(
    r"keyword[- ]based (?:sentiment )?label|true labels? (?:are|aren'?t|is|isn'?t) "
    r"(?:not )?(?:provided|available)|heuristic label|pseudo[- ]label|synthetic label|"
    r"derived the labels|labell?ed .{0,20}heuristic",
    re.I,
)
SYNTHETIC_USE = re.compile(
    r"synthetic (?:data|dataset|corpus)|simulated (?:data|dataset)|generated (?:the )?data",
    re.I,
)
# Markers that make a statement hypothetical rather than a report of what happened.
CONDITIONAL = re.compile(r"\bif\b|\bwhen\b|\bunless\b|\bshould\b|\bin case\b|\bfall(?:s|ing)? back\b", re.I)

CATEGORIES = (
    ("substituted_dataset", SUBSTITUTION),
    ("fabricated_labels", FABRICATED_LABELS),
    ("synthetic_content", SYNTHETIC_USE),
)


def classify(assumption: str) -> list[tuple[str, str]]:
    """Return (category, tier) for each risk the assumption text carries."""
    tier = "conditional" if CONDITIONAL.search(assumption) else "asserted"
    return [(name, tier) for name, pattern in CATEGORIES if pattern.search(assumption)]


def parse_verdict(value: object) -> object:
    if value is True or value is False:
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return value


def scan(roots: list[Path]) -> list[dict]:
    findings = []
    for root in roots:
        for path in sorted(root.glob("**/coder_agent_summary_*.json")):
            try:
                summary = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for exp in summary.get("experiments", []) or []:
                verdict = parse_verdict((exp.get("results") or {}).get("meets_success_criteria"))
                if verdict not in (True, False):
                    continue  # no verdict is being claimed; nothing at risk
                kinds = sorted(
                    {i.get("kind") for i in ((exp.get("data_provenance") or {}).get("inputs") or [])}
                )
                if "synthetic_surrogate" in kinds:
                    continue  # provenance already withheld; this is the working path
                risks = []
                for assumption in exp.get("assumptions_made") or []:
                    for category, tier in classify(assumption):
                        risks.append({"category": category, "tier": tier, "text": assumption})
                if risks:
                    findings.append(
                        {
                            "run": path.parent.parent.name,
                            "case": path.parent.name,
                            "hypothesis_id": exp.get("hypothesis_id"),
                            "verdict": verdict,
                            "kinds": kinds,
                            "risks": risks,
                        }
                    )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    findings = scan(args.roots)
    asserted = [f for f in findings if any(r["tier"] == "asserted" for r in f["risks"])]
    conditional = [f for f in findings if f not in asserted]

    print(f"verdict-carrying experiments with a declared data risk: {len(findings)}")
    print(f"  asserted    {len(asserted):3d}  (a substitute was reported as used)")
    print(f"  conditional {len(conditional):3d}  (guarded fallback; may never have fired)")

    for label, group in (("ASSERTED", asserted), ("CONDITIONAL", conditional)):
        if not group:
            continue
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
        for finding in group:
            print(
                f"\n  {finding['run']}/{finding['case']}  "
                f"verdict={finding['verdict']}  data={','.join(finding['kinds'])}"
            )
            for risk in finding["risks"]:
                if label == "ASSERTED" and risk["tier"] != "asserted":
                    continue
                print(f"      [{risk['category']}] {risk['text'][:150]}")

    if args.latex and asserted:
        print("\n% ---- for the dissertation ----")
        print("\\begin{tabular}{llll}")
        print("\\toprule")
        print("\\textbf{Run} & \\textbf{Case} & \\textbf{Verdict} & \\textbf{Declared risk} \\\\")
        print("\\midrule")
        for finding in asserted:
            categories = ", ".join(
                sorted({r["category"].replace("_", " ") for r in finding["risks"] if r["tier"] == "asserted"})
            )
            case = finding["case"].replace("_", "\\_")[:34]
            print(f"{finding['run']} & {case} & {finding['verdict']} & {categories} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
