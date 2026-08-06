"""LangGraph node functions for the top-level pipeline orchestrator.

Every node is a thin wrapper around an agent's existing entry point — no agent
logic is reimplemented here, and no node re-validates an upstream output, since
each run_<name>_agent() already validates its input against the upstream
agent's own validate_output() before doing anything else.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig

from research_pipeline.agents.coder import run_coder_agent
from research_pipeline.agents.experiment_planner import run_experiment_planner_agent
from research_pipeline.agents.hypothesis import run_hypothesis_agent
from research_pipeline.agents.literature.graph import build_literature_graph
from research_pipeline.agents.reviewer import ReviewerAgent
from research_pipeline.agents.writer import WriterAgent
from research_pipeline.config import settings
from research_pipeline.orchestrator.state import PipelineState
from research_pipeline.writer_reviewer_loop import _consolidate_unresolved, route_feedback_to_sections

logger = logging.getLogger(__name__)


def _quality_threshold(state: PipelineState) -> int:
    value = state.get("quality_threshold")
    return value if value is not None else settings.writer_reviewer_quality_threshold


def _paper_output_dir(state: PipelineState) -> Path:
    return Path(state.get("output_dir") or settings.writer_reviewer_loop_output_dir)


def _injected(config: Optional[RunnableConfig], key: str):
    """Agents come from the run config, not from state: state is checkpointed
    (and so must stay serializable), while an already-configured WriterAgent /
    ReviewerAgent — sharing a chat_model, or a fake one in tests — is not."""
    return (config or {}).get("configurable", {}).get(key)


def run_literature_node(state: PipelineState) -> dict:
    download_dir = state.get("download_dir", "papers")
    literature_graph = build_literature_graph()
    result = literature_graph.invoke(
        {
            "research_question": state["research_question"],
            "max_results_per_query": state.get("max_results_per_query", settings.default_max_results_per_query),
            "download_dir": download_dir,
            "metadata_path": state.get("metadata_path", f"{download_dir}/metadata.json"),
        },
        # The literature subgraph keeps its own checkpointer/thread; the
        # orchestrator's thread_id is unrelated to it.
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    logger.info("Literature stage found %d unique papers", len(result["merged_papers"]))
    return {"literature_output": result}


def run_hypothesis_node(state: PipelineState) -> dict:
    literature_output = state["literature_output"]
    result = run_hypothesis_agent(
        literature_output["merged_papers"],
        research_question=literature_output.get("research_question"),
        output_dir=state.get("output_dir"),
    )
    return {"hypothesis_output": result}


def run_planner_node(state: PipelineState) -> dict:
    return {"planner_output": run_experiment_planner_agent(state["hypothesis_output"], output_dir=state.get("output_dir"))}


def run_coder_node(state: PipelineState) -> dict:
    return {"coder_output": run_coder_agent(state["planner_output"], output_dir=state.get("output_dir"))}


def draft_or_revise_node(state: PipelineState, config: Optional[RunnableConfig] = None) -> dict:
    """Iteration 1 drafts from scratch; every later iteration revises the
    previous draft against the previous review — the same split as
    writer_reviewer_loop.run_writer_reviewer_loop's loop body."""
    iteration = state.get("iteration", 0) + 1
    output_dir = _paper_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = _injected(config, "writer") or WriterAgent(output_dir=output_dir)
    paper_filename = f"v{iteration}.pdf"
    summary_filename = f"v{iteration}_summary.json"

    upstream = (state["literature_output"], state["hypothesis_output"], state["planner_output"], state["coder_output"])

    if iteration == 1:
        logger.info("Writer/Reviewer iteration %d: drafting", iteration)
        paper_summary = writer.run(*upstream, paper_filename=paper_filename, summary_filename=summary_filename)
    else:
        logger.info("Writer/Reviewer iteration %d: revising against iteration %d's feedback", iteration, iteration - 1)
        previous = state["review_history"][-1]
        paper_summary = writer.revise(
            *upstream,
            previous_paper_path=previous["paper_path"],
            previous_summary=previous["paper_summary"],
            feedback_by_section=route_feedback_to_sections(state["review"], _quality_threshold(state)),
            paper_filename=paper_filename,
            summary_filename=summary_filename,
        )

    return {"paper_summary": paper_summary, "iteration": iteration}


def review_node(state: PipelineState, config: Optional[RunnableConfig] = None) -> dict:
    output_dir = _paper_output_dir(state)
    reviewer = _injected(config, "reviewer") or ReviewerAgent(output_dir=output_dir)
    paper_summary = state["paper_summary"]

    review = reviewer.run(
        paper_summary["paper_path"],
        paper_summary,
        state["literature_output"],
        state["hypothesis_output"],
        state["planner_output"],
        state["coder_output"],
        iteration=state["iteration"],
        quality_threshold=_quality_threshold(state),
    )
    logger.info("Iteration %d: overall_pass=%s", state["iteration"], review["overall_pass"])

    history = state.get("review_history", []) + [
        {
            "iteration": state["iteration"],
            "paper_path": paper_summary["paper_path"],
            "paper_summary": paper_summary,
            "review": review,
        }
    ]
    return {"review": review, "review_history": history, "converged": bool(review["overall_pass"])}


def finalize_node(state: PipelineState) -> dict:
    """Writes review_log.json and assembles the run's result — the same shape
    and same persistence as run_writer_reviewer_loop, plus every upstream
    stage's own output files, which each agent already wrote itself."""
    history = state["review_history"]
    converged = state["converged"]
    unresolved_issues = [] if converged else _consolidate_unresolved(state["review"], _quality_threshold(state))

    review_log_path = _paper_output_dir(state) / "review_log.json"
    review_log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))

    logger.info(
        "Pipeline finished after %d Writer/Reviewer iteration(s): converged=%s, unresolved=%d",
        len(history), converged, len(unresolved_issues),
    )
    return {
        "final_result": {
            "final_paper_path": history[-1]["paper_path"],
            "iterations_run": len(history),
            "converged": converged,
            "unresolved_issues": unresolved_issues,
            "review_history_path": str(review_log_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    }
