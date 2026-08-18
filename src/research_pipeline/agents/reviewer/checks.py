"""Deterministic (non-LLM) checks for the Reviewer Agent: citation validity,
results-accuracy, and a coarse hypothesis-coverage pre-check. These run
regex/dict-diff logic against the ground-truth upstream data rather than
asking the model to judge things that can be verified exactly — reserve LLM
judgment (see prompts.py / reviewer_agent.py) for what actually needs it:
hallucination nuance, tone, flow, and honesty of framing.

All of these are heuristics over PDF-extracted text (agents.writer.pdf_reader)
rather than the model's own structured intermediate representation — the
Reviewer deliberately checks what actually got rendered, not what the Writer
Agent *says* it did (its own JSON summary is useful context, not ground
truth about the printed paper). That means false negatives are possible
(a genuine issue phrased in a way a regex doesn't catch) but false positives
are kept low by design — every check here is documented with what it can and
can't catch, so ReviewerAgent's LLM-based checks are a deliberate complement,
not a redundant re-check of the same ground.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from research_pipeline.agents.writer.citations import (
    _NARRATIVE_CITE_RE,
    _PARENTHETICAL_CITE_RE,
    IndexedPaper,
    build_surname_year_lookup,
)

_NOT_COMPLETED_PHRASES = (
    "not executed", "not run", "no results", "not completed", "not implemented",
    "no empirical", "not been run", "were not executed", "not carried out", "was skipped",
)
_COMPLETED_CLAIM_PHRASES = ("results show", "we found", "successfully", "demonstrates that", "confirms that", "achieved")

_SUPPORTED_KEYWORDS = ("support", "confirm", "consistent with", "validated", "corroborat")
_REFUTED_KEYWORDS = ("refute", "contradict", "not support", "disconfirm", "inconsistent with", "fail to support")
_INCONCLUSIVE_KEYWORDS = ("inconclusive", "not run", "not executed", "no results", "unclear", "not completed", "undetermined")

_NUMBER_RE = re.compile(r"-?\d+\.?\d*%?")

# Any literal "[[cite:" / "[[citet:" that survives into the rendered text is
# always a defect — a well-formed marker is fully consumed by
# CitationRegistry.resolve(), so this can only match a marker malformed badly
# enough to escape it. Captures a short trailing snippet (not the whole
# marker, which can run to an unclosed end-of-line) so the issue text stays
# readable — [^\[]{0,60} rather than .{0,60}: excluding "[" stops the snippet
# before it can swallow a *second* leaked marker's own opening "[[", which
# would otherwise hide it from a later finditer() match entirely. Capped per
# section so a single degenerate generation — the same repetition-loop
# failure mode citations.py's _MAX_UNRESOLVED_NOTES_PER_KEY guards against —
# can't flood citation_issues with near-duplicates either.
_LEAKED_MARKER_RE = re.compile(r"\[\[(?:cite|citet):[^\[]{0,60}")
_MAX_LEAKED_MARKER_ISSUES_PER_SECTION = 5


def check_citations(
    section_texts: Dict[str, str], paper_index: Dict[str, IndexedPaper], citations_used: List[str]
) -> List[dict]:
    """Three checks, all deterministic:
    1. Every id the Writer's own summary claims was cited (`citations_used`)
       actually exists in the Literature Agent's paper index — should always
       hold given how the Writer resolves citations, but re-checked here
       independently rather than trusted.
    2. Every "(Surname, YYYY)" / "Surname (YYYY)" -shaped citation actually
       found in the rendered text corresponds to *some* paper in the
       Literature Agent's output — catches a citation that's real-looking but
       doesn't match any known paper (fabricated, or a surname/year typo).
       Regex-based and NeurIPS-author-year-specific: it can miss citations
       written in a different shape, and can't (on its own) tell a citation
       is to the *wrong* paper if that paper happens to share a surname/year
       with a real one — this is a coarse net, not a full disambiguator.
    3. No literal `[[cite:`/`[[citet:` marker syntax survives into the
       rendered text at all. A well-formed marker is always fully consumed by
       CitationRegistry.resolve() (replaced with a real citation or with ""),
       so any leftover match here means a marker was malformed badly enough
       to escape the Writer's own resolution — a 2026-08-17 production run
       (job 10247173) shipped a "reviewed" paper with raw
       "[[cite:ID1],[ID2]]"-shaped garbage printed directly in the text, and
       neither this function nor the LLM hallucination pass had a check aimed
       at that specific defect — the LLM pass quoted the leaked text back as
       an ordinary "needs more grounding" hallucination instead, so the
       feedback that reached the Writer never named the real problem. This
       check exists so that class of defect is always a citation_issue, named
       correctly, rather than depending on the LLM noticing and phrasing it
       usefully.
    """
    issues: List[dict] = []

    for paper_id in citations_used:
        if paper_id not in paper_index:
            issues.append(
                {
                    "location": "References",
                    "issue": (
                        f"citations_used lists paper id '{paper_id}', which is not present in the Literature "
                        "Agent's output — the paper this cites cannot be verified."
                    ),
                }
            )

    lookup = build_surname_year_lookup(paper_index)
    for section, text in section_texts.items():
        seen: set[tuple[str, str]] = set()
        for pattern in (_PARENTHETICAL_CITE_RE, _NARRATIVE_CITE_RE):
            for match in pattern.finditer(text):
                surname_group, year = match.group(1), match.group(2)
                surname = re.split(r"\s+(?:and|et al\.)", surname_group)[0].strip()
                key = (surname.lower(), year)
                if key in seen:
                    continue
                seen.add(key)
                if key not in lookup:
                    issues.append(
                        {
                            "location": section,
                            "issue": (
                                f"citation-like text \"({surname_group}, {year})\" doesn't match any paper in the "
                                "Literature Agent's output by first-author surname and year — possibly fabricated "
                                "or mismatched; verify against the source list."
                            ),
                        }
                    )

    for section, text in section_texts.items():
        for match in list(_LEAKED_MARKER_RE.finditer(text))[:_MAX_LEAKED_MARKER_ISSUES_PER_SECTION]:
            issues.append(
                {
                    "location": section,
                    "issue": (
                        f'Found literal unresolved citation marker text "{match.group(0)}..." printed in the '
                        "paper — a [[cite:ID]]/[[citet:ID]] marker survived into the rendered text instead of "
                        "being resolved to an author-year citation. This means the marker's syntax was malformed "
                        "(commonly: an extra bracket around one of the ids, e.g. [[cite:[ID1],[ID2]]] instead of "
                        "[[cite:ID1,ID2]]) — rewrite this citation using the exact [[cite:ID1,ID2]] / "
                        "[[citet:ID]] format, with no brackets around individual ids."
                    ),
                }
            )

    return issues


def _plausible_number_reprs(value: object) -> set[str]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {str(value)}
    reprs = {str(value), f"{value:.1f}", f"{value:.2f}", f"{value:.3f}"}
    if 0 <= value <= 1:
        pct = value * 100
        reprs |= {f"{pct:.0f}", f"{pct:.1f}", f"{pct:.0f}%", f"{pct:.1f}%", f"{pct:.2f}%"}
    return reprs


def check_results_accuracy(results_subsections: Dict[str, str], coder_output: dict) -> List[dict]:
    """Per experiment:
    - if status != "completed": the hypothesis's Results text must read like
      it wasn't run (contain a not-run/not-executed-style phrase) and must not
      contain a completion-claiming phrase — independent re-check of the same
      invariant WriterAgent enforces at draft time, in case a revision dropped it.
    - if status == "completed": every metric's value must appear, in some
      plausible textual form (raw, rounded, or as a percentage), somewhere in
      that hypothesis's Results text — a metric with no matching number
      anywhere is flagged, surfacing whatever numbers *are* present as the
      likely altered/mismatched figure.
    """
    issues: List[dict] = []
    for experiment in coder_output.get("experiments", []):
        hid = experiment["hypothesis_id"]
        text = results_subsections.get(hid, "")
        location = f"Results > {hid}"

        if not text:
            issues.append({"location": location, "claimed": "(no Results subsection found for this hypothesis)", "actual": f"status={experiment['status']}"})
            continue

        lowered = text.lower()
        if experiment["status"] != "completed":
            has_disclaimer = any(phrase in lowered for phrase in _NOT_COMPLETED_PHRASES)
            overstates = any(phrase in lowered for phrase in _COMPLETED_CLAIM_PHRASES)
            if not has_disclaimer or overstates:
                issues.append(
                    {
                        "location": location,
                        "claimed": "text reads as if this experiment produced results",
                        "actual": f"status='{experiment['status']}' ({experiment['reason']}) — no results exist",
                    }
                )
            continue

        metrics = (experiment.get("results") or {}).get("metrics", {})
        numbers_in_text = set(_NUMBER_RE.findall(text))
        for metric_name, metric_value in metrics.items():
            expected = _plausible_number_reprs(metric_value)
            if expected & numbers_in_text:
                continue
            claimed = ", ".join(sorted(numbers_in_text)) if numbers_in_text else "(no number found in this text at all)"
            issues.append(
                {
                    "location": location,
                    "claimed": f"figure(s) actually present in the text: {claimed}",
                    "actual": f"Coder Agent reports {metric_name}={metric_value!r}",
                }
            )
    return issues


def _keyword_verdict_guess(text: str) -> Optional[str]:
    lowered = text.lower()
    is_inconclusive = any(kw in lowered for kw in _INCONCLUSIVE_KEYWORDS)
    is_supported = any(kw in lowered for kw in _SUPPORTED_KEYWORDS)
    is_refuted = any(kw in lowered for kw in _REFUTED_KEYWORDS)
    if is_inconclusive:
        return "inconclusive"
    if is_supported and not is_refuted:
        return "supported"
    if is_refuted and not is_supported:
        return "refuted"
    return None  # ambiguous or no signal — not flagged, left to the LLM pass


def check_hypothesis_coverage(
    results_subsections: Dict[str, str], discussion_subsections: Dict[str, str], hypothesis_ids: List[str], verdicts: Dict[str, dict]
) -> List[dict]:
    """Two deterministic checks per hypothesis:
    1. It has a dedicated Results *and* Discussion subsection at all (the
       pdf_reader splitter found its "N.k <id>" marker) — a missing
       subsection is a structural gap, not a judgment call.
    2. A coarse keyword scan of its Discussion text against the independently
       computed true verdict (agents.writer.writer_agent.compute_hypothesis_verdict)
       — flags only the unambiguous case where the text's keywords
       contradict the true verdict outright (e.g. true verdict is
       "inconclusive" but the text uses "supported"/"confirms" language with
       no inconclusive/not-run hedge). Ambiguous or hedged text is left alone
       here and is exactly what the LLM-based framing check is for — this is
       a first-pass net for the clear-cut cases, not a full honesty judge.
    """
    issues: List[dict] = []
    for hid in hypothesis_ids:
        if hid not in results_subsections:
            issues.append({"hypothesis_id": hid, "location": "Results", "issue": f"{hid} has no dedicated Results subsection in the rendered paper."})
        if hid not in discussion_subsections:
            issues.append({"hypothesis_id": hid, "location": "Discussion", "issue": f"{hid} has no dedicated Discussion subsection in the rendered paper."})
            continue

        true_verdict = verdicts[hid]["verdict"]
        guessed = _keyword_verdict_guess(discussion_subsections[hid])
        if guessed is not None and guessed != true_verdict:
            issues.append(
                {
                    "hypothesis_id": hid,
                    "location": f"Discussion > {hid}",
                    "issue": (
                        f"Discussion text reads as '{guessed}' but the actual outcome (from the Coder Agent's data) "
                        f"is '{true_verdict}': {verdicts[hid]['reason']} — correct the framing."
                    ),
                }
            )
    return issues
