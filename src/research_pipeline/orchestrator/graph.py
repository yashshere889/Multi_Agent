"""Builds the top-level pipeline orchestrator LangGraph.

Ties all seven agents into one graph — Literature -> Interdisciplinary
Literature -> Hypothesis -> Experiment Planner -> Coder -> Writer <-> Reviewer
-> finalize — so a full run is one invoke() instead of the hand-chained CLI
calls in README's "Chaining agents". Every individual agent subcommand and
writer_reviewer_loop.py still work standalone for partial or disk-decoupled runs.

Only the hypothesis the Hypothesis Agent ranked first is planned and executed
(see run_planner_node); the Writer and Reviewer still receive the full
hypothesis output, so the paper can say which hypotheses were considered and
why one was taken forward.

The Writer/Reviewer cycle is a real conditional edge here rather than a Python
loop: review routes back to draft_or_revise until the paper passes or
max_iterations is spent. The routing decision reads review["overall_pass"], a
value the Reviewer Agent already derives deterministically — the orchestrator
adds no model judgment of its own.

    from research_pipeline.orchestrator.graph import build_pipeline_graph
    graph = build_pipeline_graph()
    result = graph.invoke({"research_question": "..."}, config={"configurable": {"thread_id": "..."}})
    result["final_result"]  # {final_paper_path, iterations_run, converged, ...}

Pass already-configured agents (e.g. sharing one chat_model) as "writer" /
"reviewer" alongside thread_id in config["configurable"]; otherwise the nodes
build default-configured ones.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from research_pipeline.checkpointer import get_checkpointer
from research_pipeline.config import settings
from research_pipeline.orchestrator.nodes import (
    draft_or_revise_node,
    finalize_node,
    review_node,
    run_coder_node,
    run_hypothesis_node,
    run_interdisciplinary_literature_node,
    run_literature_node,
    run_planner_node,
)
from research_pipeline.orchestrator.state import PipelineState


def should_continue_revising(state: PipelineState) -> str:
    """Same stop conditions as run_writer_reviewer_loop: converged, or out of
    iterations. An exhausted budget still finalizes — the last draft is
    returned with its unresolved issues, never silently accepted as done."""
    max_iterations = state.get("max_iterations")
    if max_iterations is None:
        max_iterations = settings.writer_reviewer_max_iterations
    if state["converged"] or state["iteration"] >= max_iterations:
        return "finalize"
    return "revise"


def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("literature", run_literature_node)
    graph.add_node("interdisciplinary_literature", run_interdisciplinary_literature_node)
    graph.add_node("hypothesis", run_hypothesis_node)
    graph.add_node("experiment_planner", run_planner_node)
    graph.add_node("coder", run_coder_node)
    graph.add_node("draft_or_revise", draft_or_revise_node)
    graph.add_node("review", review_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("literature")
    # Unconditional, like every other upstream edge: the cross-field search is
    # part of what this pipeline does, not an opt-in mode. An agent that finds
    # no adjacent fields simply passes the in-domain papers through.
    graph.add_edge("literature", "interdisciplinary_literature")
    graph.add_edge("interdisciplinary_literature", "hypothesis")
    graph.add_edge("hypothesis", "experiment_planner")
    graph.add_edge("experiment_planner", "coder")
    graph.add_edge("coder", "draft_or_revise")
    graph.add_edge("draft_or_revise", "review")

    graph.add_conditional_edges(
        "review",
        should_continue_revising,
        {"revise": "draft_or_revise", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    # Checkpointing, as in the literature subgraph: a crash after the Coder
    # stage can resume from the same thread_id instead of re-running the
    # searches, the LLM synthesis, and the generated experiments. In-memory by
    # default; CHECKPOINTER_BACKEND makes that survive process restarts too,
    # which matters most here — this is the graph a pre-empted SLURM job or a
    # restarted Kaggle kernel is usually halfway through.
    #
    # No RetryPolicy on any node here on purpose: each one wraps an entire
    # sub-agent invocation, so a graph-level retry would re-run a whole agent
    # (including non-idempotent work like the Coder's file writes and
    # subprocess executions) instead of one call. Granular retry belongs inside
    # each sub-agent's own graph, and that's where it lives.
    return graph.compile(checkpointer=get_checkpointer())
