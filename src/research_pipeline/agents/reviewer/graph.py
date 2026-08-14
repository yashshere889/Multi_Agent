"""Builds the Reviewer Agent's LangGraph.

Shaped like agents/literature/graph.py rather than agents/hypothesis/graph.py:
the Reviewer's parallel work is a *fixed* set of three deterministic checks
(agents.reviewer.checks), not a per-item branch over a variable-length list, so
the fan-out is three plain `add_edge` calls out of one node — the same pattern
as literature's arxiv/semantic_scholar pair — and not a conditional edge
returning `Send`s. Fan-in is likewise plain edges converging on the next node,
which reads all three checks' keys off the state once every branch has finished
(exactly how literature's merge_and_dedupe reads both searches' results).

After that the graph is a straight chain: the LLM passes have to run in order
because `check_discussion` extends the hallucination and coverage lists that
`check_hallucinations` and `check_hypothesis_coverage` produced, and the final
verdict depends on all of them.

The graph is built per-agent-instance because every node is a bound method on a
configured ReviewerAgent (its chat model and output dir) — nodes stay thin
adapters over the agent's existing private helpers, so the LLM/IO logic lives in
exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

from research_pipeline.agents.reviewer.state import ReviewerState
from research_pipeline.checkpointer import get_checkpointer

# Outer safety net on the three LLM-calling nodes only; the deterministic checks
# (citations, results accuracy, hypothesis coverage) touch no network and would
# fail identically on a retry, so they're left alone.
_RETRY = RetryPolicy(max_attempts=2)

if TYPE_CHECKING:  # avoids a circular import — reviewer_agent imports this module
    from research_pipeline.agents.reviewer.reviewer_agent import ReviewerAgent


def build_reviewer_graph(agent: "ReviewerAgent"):
    graph = StateGraph(ReviewerState)

    graph.add_node("prepare_context", agent._node_prepare_context)
    graph.add_node("check_citations", agent._node_check_citations)
    graph.add_node("check_results_accuracy", agent._node_check_results_accuracy)
    graph.add_node("check_hypothesis_coverage", agent._node_check_hypothesis_coverage)
    graph.add_node("check_hallucinations", agent._node_check_hallucinations, retry_policy=_RETRY)
    graph.add_node("check_discussion", agent._node_check_discussion, retry_policy=_RETRY)
    graph.add_node("score_quality", agent._node_score_quality, retry_policy=_RETRY)
    graph.add_node("compute_overall_pass", agent._node_compute_overall_pass)
    graph.add_node("build_feedback", agent._node_build_feedback)
    graph.add_node("assemble_and_validate", agent._node_assemble_and_validate)

    graph.set_entry_point("prepare_context")

    # Fan out: all three deterministic checks start as soon as the ground truth
    # is indexed and the PDF is split. They touch no LLM and write disjoint
    # state keys, so they're safe to run concurrently.
    graph.add_edge("prepare_context", "check_citations")
    graph.add_edge("prepare_context", "check_results_accuracy")
    graph.add_edge("prepare_context", "check_hypothesis_coverage")

    # Fan in: the first LLM pass waits for all three branches to complete — no
    # separate merge node is needed, since a node with several incoming edges
    # runs once, after every predecessor, and just reads their keys off state.
    graph.add_edge("check_citations", "check_hallucinations")
    graph.add_edge("check_results_accuracy", "check_hallucinations")
    graph.add_edge("check_hypothesis_coverage", "check_hallucinations")

    graph.add_edge("check_hallucinations", "check_discussion")
    graph.add_edge("check_discussion", "score_quality")
    graph.add_edge("score_quality", "compute_overall_pass")
    graph.add_edge("compute_overall_pass", "build_feedback")
    graph.add_edge("build_feedback", "assemble_and_validate")
    graph.add_edge("assemble_and_validate", END)

    # Checkpointing, matching the literature/hypothesis/planner graphs: a crash
    # partway through can resume from the last completed node (via the same
    # thread_id) rather than re-running every LLM call. In-memory by default;
    # CHECKPOINTER_BACKEND makes that survive process restarts too.
    return graph.compile(checkpointer=get_checkpointer())
