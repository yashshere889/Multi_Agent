"""Builds the Hypothesis Agent's LangGraph.

Same shape as agents/literature/graph.py, with one difference: the fan-out
width isn't known until runtime (it's however many batches `chunk_papers`
produced), so the branch is a conditional edge returning a list of `Send`s
rather than a fixed pair of `add_edge` calls.

The graph is built per-agent-instance because every node is a bound method on
a configured HypothesisAgent (its chat model, batch budget, and output dir) —
nodes stay thin adapters over the agent's existing private helpers, so the
LLM/IO logic lives in exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from research_pipeline.agents.hypothesis.state import HypothesisState

if TYPE_CHECKING:  # avoids a circular import — hypothesis_agent imports this module
    from research_pipeline.agents.hypothesis.hypothesis_agent import HypothesisAgent


def fan_out_batches(state: HypothesisState) -> List[Send]:
    """One `analyze_batch` invocation per batch, each seeded with the batch it
    owns. LangGraph runs them concurrently and each returns `{"partials": [...]}`,
    which the state's operator.add reducer accumulates."""
    return [Send("analyze_batch", {**state, "current_batch": batch}) for batch in state["batches"]]


def build_hypothesis_graph(agent: "HypothesisAgent"):
    graph = StateGraph(HypothesisState)

    graph.add_node("normalize_and_chunk", agent._node_normalize_and_chunk)
    graph.add_node("analyze_batch", agent._node_analyze_batch)
    graph.add_node("synthesize", agent._node_synthesize)
    graph.add_node("generate_hypotheses", agent._node_generate_hypotheses)
    graph.add_node("rank_hypotheses", agent._node_rank_hypotheses)
    graph.add_node("assemble_and_validate", agent._node_assemble_and_validate)

    graph.set_entry_point("normalize_and_chunk")

    # Fan out: one analyze_batch branch per batch of papers.
    graph.add_conditional_edges("normalize_and_chunk", fan_out_batches, ["analyze_batch"])

    # Fan in: a plain edge out of a Send-spawned node only fires once every
    # branch has finished, so synthesize sees the complete `partials` list.
    graph.add_edge("analyze_batch", "synthesize")

    graph.add_edge("synthesize", "generate_hypotheses")
    # Ranking is its own node rather than part of hypothesis generation: it's a
    # separate LLM call with a separate contract, and keeping it separate means
    # the generated hypotheses are already fixed before anything scores them.
    graph.add_edge("generate_hypotheses", "rank_hypotheses")
    graph.add_edge("rank_hypotheses", "assemble_and_validate")
    graph.add_edge("assemble_and_validate", END)

    # In-memory checkpointing, matching the literature graph: a crash partway
    # through can resume from the last completed node (via the same thread_id)
    # rather than re-running every LLM call. Swap for a SqliteSaver if that
    # should survive process restarts too.
    return graph.compile(checkpointer=MemorySaver())
