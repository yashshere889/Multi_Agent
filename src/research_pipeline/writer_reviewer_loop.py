"""Writer/Reviewer feedback loop.

A small orchestrator, not an agent itself: runs the Writer Agent and Reviewer
Agent in a loop — draft, review, revise, re-review — until the Reviewer finds
zero hallucinations/citation/results-accuracy/hypothesis-coverage issues and
every quality score is at or above `quality_threshold`, or `max_iterations` is
reached, whichever comes first. Both agents stay independent of this file and
of each other: the Writer knows nothing about the Reviewer's schema, the
Reviewer knows nothing about the Writer's revision API. This module is the
only place that translates one agent's output into the other's input.

Stop conditions
-----------------
- Converged: `review["overall_pass"]` is True (see agents.reviewer.reviewer_agent
  — zero issues across every category, every quality score >= quality_threshold).
- Exhausted: `max_iterations` reached without converging. The final draft is
  still returned — never silently accepted as done — alongside
  `unresolved_issues`, a flattened list of everything still outstanding.

Traceability
-------------
Every iteration's paper is kept (`<output_dir>/v1.pdf`, `v2.pdf`, ...), not
just the final one, and every iteration's review is appended to
`<output_dir>/review_log.json` alongside which paper version it reviewed —
the full history survives the run, not just the final verdict.

Entry point
------------
    from research_pipeline.writer_reviewer_loop import run_writer_reviewer_loop
    result = run_writer_reviewer_loop(literature_output, hypothesis_output, planner_output, coder_output)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from research_pipeline.agents.reviewer import ReviewerAgent
from research_pipeline.agents.writer import WriterAgent
from research_pipeline.config import settings

logger = logging.getLogger(__name__)

# Headings WriterAgent's per-section drafting calls actually accept revision
# feedback for (see writer_agent.py:_revision_note) — "References" is
# mechanically built, not drafted, so it's never a valid routing target here.
_REVISABLE_HEADINGS = (
    "Title", "Abstract", "Introduction", "Related Work", "Hypotheses",
    "Methods", "Results", "Discussion", "Limitations", "Future Work",
)


def route_feedback_to_sections(review: dict, quality_threshold: int) -> Dict[str, List[str]]:
    """Turns a ReviewerAgent review's structured issue lists into
    {heading: [actionable bullet, ...]} for WriterAgent.revise()'s
    feedback_by_section — an issue whose `location` starts with a known
    heading (e.g. "Results > H2") routes there; anything else (e.g. a
    References-located citation issue, or a paper-wide quality score) falls
    into "__general__", which every section's revision prompt also sees."""
    routed: Dict[str, List[str]] = {}

    def _add(heading: str, bullet: str) -> None:
        routed.setdefault(heading, []).append(bullet)

    def _heading_for(location: str) -> str:
        top = location.split(">")[0].strip()
        return top if top in _REVISABLE_HEADINGS else "__general__"

    for h in review.get("hallucinations", []):
        _add(_heading_for(h["location"]), f'Hallucination: "{h["claim"]}" — {h["issue"]}')
    for c in review.get("citation_issues", []):
        _add(_heading_for(c["location"]), f'Citation issue: {c["issue"]}')
    for r in review.get("results_accuracy_issues", []):
        _add(_heading_for(r["location"]), f'Results accuracy: claimed — {r["claimed"]}; actual — {r["actual"]}. Fix this.')
    for hc in review.get("hypothesis_coverage_issues", []):
        _add(_heading_for(hc["location"]), f'{hc.get("hypothesis_id", "")}: {hc["issue"]}')

    for dimension, score in review.get("quality_scores", {}).items():
        if score < quality_threshold:
            _add("__general__", f"Quality — {dimension}: scored {score}/5, below the {quality_threshold}/5 threshold.")

    return routed


def _consolidate_unresolved(review: dict, quality_threshold: int) -> List[str]:
    items: List[str] = []
    items += [f'Hallucination ({h["location"]}): {h["issue"]}' for h in review["hallucinations"]]
    items += [f'Citation ({c["location"]}): {c["issue"]}' for c in review["citation_issues"]]
    items += [
        f'Results accuracy ({r["location"]}): claimed — {r["claimed"]}; actual — {r["actual"]}'
        for r in review["results_accuracy_issues"]
    ]
    items += [f'{hc["location"]}: {hc["issue"]}' for hc in review["hypothesis_coverage_issues"]]
    items += [
        f"Quality — {dim}: {score}/5 (below {quality_threshold}/5 threshold)"
        for dim, score in review["quality_scores"].items()
        if score < quality_threshold
    ]
    return items


def run_writer_reviewer_loop(
    literature_output: object,
    hypothesis_output: dict,
    planner_output: dict,
    coder_output: dict,
    max_iterations: Optional[int] = None,
    quality_threshold: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    writer: Optional[WriterAgent] = None,
    reviewer: Optional[ReviewerAgent] = None,
) -> dict:
    """Runs the Writer -> Reviewer -> revise loop to convergence or
    `max_iterations` (default from WRITER_REVIEWER_MAX_ITERATIONS, 3).
    `quality_threshold` (default from WRITER_REVIEWER_QUALITY_THRESHOLD, 4) is
    the minimum acceptable 1-5 score on every ReviewerAgent quality dimension.

    Pass `writer`/`reviewer` to reuse already-configured agents (e.g. a shared
    chat_model); otherwise default-configured ones are built, both pointed at
    `output_dir`.

    Returns:
        {
          "final_paper_path": str,
          "iterations_run": int,
          "converged": bool,
          "unresolved_issues": [str, ...],   # empty if converged
          "review_history_path": str,        # outputs/paper/review_log.json
        }
    """
    max_iterations = max_iterations if max_iterations is not None else settings.writer_reviewer_max_iterations
    quality_threshold = quality_threshold if quality_threshold is not None else settings.writer_reviewer_quality_threshold
    resolved_output_dir = Path(output_dir or settings.writer_reviewer_loop_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    writer = writer or WriterAgent(output_dir=resolved_output_dir)
    reviewer = reviewer or ReviewerAgent(output_dir=resolved_output_dir)

    history: List[dict] = []
    paper_summary: dict = {}
    review: dict = {}
    converged = False

    for iteration in range(1, max_iterations + 1):
        paper_filename = f"v{iteration}.pdf"
        summary_filename = f"v{iteration}_summary.json"

        if iteration == 1:
            logger.info("Writer/Reviewer loop iteration %d: drafting", iteration)
            paper_summary = writer.run(
                literature_output, hypothesis_output, planner_output, coder_output,
                paper_filename=paper_filename, summary_filename=summary_filename,
            )
        else:
            logger.info("Writer/Reviewer loop iteration %d: revising against iteration %d's feedback", iteration, iteration - 1)
            feedback_by_section = route_feedback_to_sections(review, quality_threshold)
            paper_summary = writer.revise(
                literature_output, hypothesis_output, planner_output, coder_output,
                previous_paper_path=history[-1]["paper_path"],
                previous_summary=history[-1]["paper_summary"],
                feedback_by_section=feedback_by_section,
                paper_filename=paper_filename, summary_filename=summary_filename,
            )

        review = reviewer.run(
            paper_summary["paper_path"], paper_summary, literature_output, hypothesis_output, planner_output, coder_output,
            iteration=iteration, quality_threshold=quality_threshold,
        )

        history.append({"iteration": iteration, "paper_path": paper_summary["paper_path"], "paper_summary": paper_summary, "review": review})
        logger.info("Iteration %d: overall_pass=%s", iteration, review["overall_pass"])

        if review["overall_pass"]:
            converged = True
            break

    unresolved_issues = [] if converged else _consolidate_unresolved(review, quality_threshold)

    review_log_path = resolved_output_dir / "review_log.json"
    review_log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))

    result = {
        "final_paper_path": history[-1]["paper_path"],
        "iterations_run": len(history),
        "converged": converged,
        "unresolved_issues": unresolved_issues,
        "review_history_path": str(review_log_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "Writer/Reviewer loop finished after %d iteration(s): converged=%s, unresolved=%d",
        len(history), converged, len(unresolved_issues),
    )
    return result
