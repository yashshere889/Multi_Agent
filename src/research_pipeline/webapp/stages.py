"""Turning the orchestrator's state deltas into something a page can render.

`graph.stream(..., stream_mode="updates")` yields `{node_name: delta}` after
every node returns, where `delta` is exactly what that node returned. So the
graph already emits a stage-completion event at precisely the right boundary,
and this module is only the translation layer: pick the few display fields out
of each delta, and lay the completed events out against the known node order.

Nothing here re-derives pipeline facts. Every number below is read straight out
of an agent's own validated output — the hypothesis verdicts, the quality
scores, whether the review passed. The two pieces of control flow it needs to
know are both answered by asking the orchestrator rather than restating its
conditions: "does another Writer/Reviewer iteration come next" by calling
`should_continue_revising`, and "where does this run actually stop" by calling
`_resolved_end_stage` (private to that module, but the alternative is a second
copy of the end_stage/include_interdisciplinary interaction here, which could
disagree with the routing the graph will actually take).
"""

from __future__ import annotations

from typing import Optional

from research_pipeline.orchestrator.graph import (
    NODE_FOR_STAGE,
    _resolved_end_stage,
    should_continue_revising,
)

LITERATURE = "literature"
INTERDISCIPLINARY = "interdisciplinary_literature"
HYPOTHESIS = "hypothesis"
EXPERIMENT_PLANNER = "experiment_planner"
CODER = "coder"
DRAFT_OR_REVISE = "draft_or_revise"
REVIEW = "review"
FINALIZE = "finalize"

# The node order in orchestrator/graph.py. Kept here rather than introspected
# off the compiled graph because the conditional edge means the writer/reviewer
# pair repeats, which a topological walk can't express.
UPSTREAM_STAGES = (LITERATURE, INTERDISCIPLINARY, HYPOTHESIS, EXPERIMENT_PLANNER, CODER)
STAGE_ORDER = UPSTREAM_STAGES + (DRAFT_OR_REVISE, REVIEW, FINALIZE)

# The bring-your-own-papers entry point is a different *node* but the same
# *stage*: it returns the same literature_output delta run_literature_node does,
# and nothing downstream can tell which one ran (see orchestrator/graph.py's
# _entry_router), so it is displayed as the Literature stage rather than as a
# stage of its own — which also keeps it from being dropped as an unknown node.
SEED_LITERATURE = "seed_literature"
NODE_ALIASES = {SEED_LITERATURE: LITERATURE}

LABELS = {
    LITERATURE: "Literature",
    INTERDISCIPLINARY: "Interdisciplinary Literature",
    HYPOTHESIS: "Hypothesis",
    EXPERIMENT_PLANNER: "Experiment Planner",
    CODER: "Coder",
    DRAFT_OR_REVISE: "Writer",
    REVIEW: "Reviewer",
    FINALIZE: "Finalize",
}

DONE = "done"
RUNNING = "running"
PENDING = "pending"


def stage_for_node(node: str) -> str:
    """The stage a stream update's node name is displayed as. Only the seeded
    literature entry point differs from its node name today."""
    return NODE_ALIASES.get(node, node)


def label_for(stage: str, iteration: Optional[int] = None) -> str:
    base = LABELS.get(stage, stage)
    if stage == DRAFT_OR_REVISE and iteration:
        return f"{base} — draft" if iteration == 1 else f"{base} — revision {iteration - 1}"
    if stage == REVIEW and iteration:
        return f"{base} — iteration {iteration}"
    return base


def summarize(stage: str, delta: dict) -> dict:
    """Reduce one node's returned delta to the handful of fields the UI shows.
    Defensive about missing keys: a summary is a display nicety, and failing to
    build one must never be what ends a run that otherwise succeeded."""
    if stage == LITERATURE:
        papers = (delta.get("literature_output") or {}).get("merged_papers") or []
        return {
            "papers_found": len(papers),
            "papers_downloaded": sum(1 for p in papers if p.get("local_path")),
            "queries": (delta.get("literature_output") or {}).get("search_queries") or [],
            "titles": [p.get("title") for p in papers[:5]],
        }

    if stage == INTERDISCIPLINARY:
        output = delta.get("interdisciplinary_output") or {}
        return {
            "fields": [
                {"field": f.get("field"), "rationale": f.get("rationale")} for f in output.get("fields_explored") or []
            ],
            "cross_field_papers": len(output.get("cross_field_papers") or []),
            "papers_total": len(output.get("papers") or []),
            "insights": [
                {"insight": i.get("insight"), "source_field": i.get("source_field")}
                for i in output.get("bridge_insights") or []
            ],
        }

    if stage == HYPOTHESIS:
        output = delta.get("hypothesis_output") or {}
        ranks = {r.get("hypothesis_id"): r.get("rank") for r in output.get("ranking") or []}
        return {
            "hypotheses": [
                {"id": h.get("id"), "statement": h.get("statement"), "rank": ranks.get(h.get("id"))}
                for h in output.get("hypotheses") or []
            ],
            "selected": output.get("selected_hypothesis_id"),
            "gaps": len(output.get("gaps") or []),
            "papers_used": len(output.get("source_paper_ids") or []),
        }

    if stage == EXPERIMENT_PLANNER:
        plans = (delta.get("planner_output") or {}).get("experiment_plans") or []
        return {
            "plans": [
                {
                    "hypothesis_id": p.get("hypothesis_id"),
                    "feasible": p.get("feasible"),
                    "complexity": p.get("estimated_complexity"),
                    "objective": p.get("objective"),
                }
                for p in plans
            ]
        }

    if stage == CODER:
        experiments = (delta.get("coder_output") or {}).get("experiments") or []
        return {
            "experiments": [
                {
                    "hypothesis_id": e.get("hypothesis_id"),
                    "status": e.get("status"),
                    "fix_attempts": e.get("fix_attempts", 0),
                    # "unknown" is a legitimate value here, not a missing one —
                    # the Coder Agent uses it when the run produced metrics but
                    # the plan's success criteria couldn't be evaluated.
                    "meets_success_criteria": (e.get("results") or {}).get("meets_success_criteria"),
                    "reason": e.get("reason"),
                }
                for e in experiments
            ]
        }

    if stage == DRAFT_OR_REVISE:
        paper_summary = delta.get("paper_summary") or {}
        return {
            "iteration": delta.get("iteration"),
            "paper_path": paper_summary.get("paper_path"),
            "supported": len(paper_summary.get("hypotheses_supported") or []),
            "refuted": len(paper_summary.get("hypotheses_refuted") or []),
            "inconclusive": len(paper_summary.get("hypotheses_inconclusive") or []),
            "citations": len(paper_summary.get("citations_used") or []),
            "sections": len(paper_summary.get("sections_generated") or []),
        }

    if stage == REVIEW:
        review = delta.get("review") or {}
        issues = {
            "hallucinations": len(review.get("hallucinations") or []),
            "citations": len(review.get("citation_issues") or []),
            "results_accuracy": len(review.get("results_accuracy_issues") or []),
            "hypothesis_coverage": len(review.get("hypothesis_coverage_issues") or []),
        }
        return {
            "iteration": review.get("iteration"),
            "overall_pass": review.get("overall_pass"),
            "converged": delta.get("converged"),
            "quality_scores": review.get("quality_scores") or {},
            "issues": issues,
            "total_issues": sum(issues.values()),
            "feedback": review.get("feedback_for_writer"),
        }

    if stage == FINALIZE:
        return dict(delta.get("final_result") or {})

    return {}


def _end_node(end_stage: Optional[str], include_interdisciplinary: bool) -> str:
    """The last *node* this run will reach before finalize, asked of the
    orchestrator's own resolver so a disabled cross-field stage moves the end
    here exactly as it does there. `_resolved_end_stage` reads only these two
    keys off the mapping it is given, so an ad-hoc dict is enough."""
    resolved = _resolved_end_stage(
        {"end_stage": end_stage, "include_interdisciplinary": include_interdisciplinary}  # type: ignore[arg-type]
    )
    return NODE_FOR_STAGE[resolved]


def _next_stage(
    stage: str,
    summary: dict,
    max_iterations: Optional[int],
    end_stage: Optional[str] = None,
    include_interdisciplinary: bool = True,
) -> Optional[str]:
    """Which node the graph will run next, given the one that just finished."""
    if stage in UPSTREAM_STAGES:
        # An upstream stage that *is* where this run stops goes straight to
        # finalize. The writer/reviewer end is not decided here — that cycle
        # ends on should_continue_revising below, as it always has.
        if stage == _end_node(end_stage, include_interdisciplinary):
            return FINALIZE
        for candidate in STAGE_ORDER[STAGE_ORDER.index(stage) + 1 :]:
            if candidate == INTERDISCIPLINARY and not include_interdisciplinary:
                continue
            return candidate
        return FINALIZE
    if stage == DRAFT_OR_REVISE:
        return REVIEW
    if stage == REVIEW:
        # The orchestrator's own router, so "one more revision or finalize?" is
        # answered by the same condition the graph will apply, not a copy of it.
        route = should_continue_revising(
            {
                "converged": bool(summary.get("converged")),
                "iteration": summary.get("iteration") or 0,
                "max_iterations": max_iterations,
            }
        )
        return DRAFT_OR_REVISE if route == "revise" else FINALIZE
    return None


def build_progress(
    stage_events: list[dict],
    is_active: bool,
    max_iterations: Optional[int] = None,
    end_stage: Optional[str] = None,
    include_interdisciplinary: bool = True,
) -> list[dict]:
    """The rows the progress fragment renders, in the order they happened.

    Completed stages come straight from the events. If the run is still going,
    one `running` row is appended for whatever comes next, and the remaining
    upstream stages are listed as `pending` so the shape of the run is visible
    from the first second rather than materializing a row at a time.

    `end_stage` / `include_interdisciplinary` are the run's own params (absent
    for a default full run). They shape only what is *predicted*: a run that
    stops after the Hypothesis Agent must not advertise a Coder stage it will
    never reach, and a run with the cross-field stage off must not list it.
    """
    rows = [
        {
            "key": event.get("stage"),
            "label": label_for(event.get("stage"), (event.get("summary") or {}).get("iteration")),
            "status": DONE,
            "summary": event.get("summary") or {},
            "ts": event.get("ts"),
        }
        for event in stage_events
        if event.get("stage") in STAGE_ORDER
    ]

    if not is_active:
        return rows

    if rows:
        last = stage_events[-1]
        upcoming = _next_stage(
            last.get("stage"),
            last.get("summary") or {},
            max_iterations,
            end_stage=end_stage,
            include_interdisciplinary=include_interdisciplinary,
        )
    else:
        upcoming = LITERATURE

    if upcoming is None:
        return rows

    iteration = None
    if upcoming in (DRAFT_OR_REVISE, REVIEW):
        done_drafts = sum(1 for r in rows if r["key"] == DRAFT_OR_REVISE)
        iteration = done_drafts if upcoming == REVIEW else done_drafts + 1

    rows.append({"key": upcoming, "label": label_for(upcoming, iteration), "status": RUNNING, "summary": {}, "ts": None})

    # Placeholders for the rest of the run, so its shape is visible from the
    # first second rather than materializing a row at a time. Only while the
    # upstream stages are still going: once drafting starts, how many
    # Writer/Reviewer iterations remain isn't knowable, and inventing rows for
    # them would be guessing at the run rather than reporting it.
    if upcoming in UPSTREAM_STAGES:
        end_node = _end_node(end_stage, include_interdisciplinary)
        end_index = STAGE_ORDER.index(end_node)
        pending = []
        for stage in UPSTREAM_STAGES[UPSTREAM_STAGES.index(upcoming) + 1 :]:
            if STAGE_ORDER.index(stage) > end_index:
                break  # this run stops before here; it is not upcoming, it is never
            if stage == INTERDISCIPLINARY and not include_interdisciplinary:
                continue
            pending.append(stage)
        rows += [
            {"key": stage, "label": label_for(stage), "status": PENDING, "summary": {}, "ts": None}
            for stage in pending
        ]
        if end_node == DRAFT_OR_REVISE:
            rows.append(
                {"key": DRAFT_OR_REVISE, "label": "Writer ⇄ Reviewer", "status": PENDING, "summary": {}, "ts": None}
            )
        rows.append({"key": FINALIZE, "label": label_for(FINALIZE), "status": PENDING, "summary": {}, "ts": None})

    return rows
