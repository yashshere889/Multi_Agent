"""Builds the Writer Agent's LangGraph.

The simplest shape of the four converted agents: a straight chain, no fan-out at
all. A paper is written in a fixed order — the body first, then the Abstract and
Title once the body's content is settled, then citation resolution over
everything at once, then the honesty pass, the PDF, and the summary — and each
of those steps genuinely depends on the previous one's output, so there is
nothing here to branch on (contrast agents/literature/graph.py's fixed two-way
fan-out, or the `Send` fan-out in agents/hypothesis/graph.py).

The two places the Writer *is* concurrent — drafting Related Work batch by batch,
and drafting Methods/Results/Discussion per hypothesis — keep the
ThreadPoolExecutors they already had inside their single node, rather than
becoming graph branches. Splitting them into `Send` sub-graphs would be a real
change in shape (the per-hypothesis bundle writes three keys at once, and the
related-work batches feed a synthesis call), so it's deliberately left alone
here; making each *section* a node is what buys the traceability.

The graph is built per-agent-instance because every node is a bound method on a
configured WriterAgent (its chat model and output dir) — nodes stay thin adapters
over the agent's existing private helpers, so the LLM/IO logic lives in exactly
one place. Both entry points share this one graph: `revise()` differs only in
seeding `revision` into the initial state, which each drafting node reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from research_pipeline.agents.writer.state import WriterState

if TYPE_CHECKING:  # avoids a circular import — writer_agent imports this module
    from research_pipeline.agents.writer.writer_agent import WriterAgent


def build_writer_graph(agent: "WriterAgent"):
    graph = StateGraph(WriterState)

    graph.add_node("prepare_context", agent._node_prepare_context)
    graph.add_node("draft_introduction", agent._node_draft_introduction)
    graph.add_node("draft_related_work", agent._node_draft_related_work)
    graph.add_node("draft_hypotheses_section", agent._node_draft_hypotheses_section)
    graph.add_node("draft_per_hypothesis_sections", agent._node_draft_per_hypothesis_sections)
    graph.add_node("draft_limitations", agent._node_draft_limitations)
    graph.add_node("draft_future_work", agent._node_draft_future_work)
    graph.add_node("draft_abstract", agent._node_draft_abstract)
    graph.add_node("draft_title", agent._node_draft_title)
    graph.add_node("resolve_citations", agent._node_resolve_citations)
    graph.add_node("validate_paper_honesty", agent._node_validate_paper_honesty)
    graph.add_node("render_pdf", agent._node_render_pdf)
    graph.add_node("assemble_and_validate_summary", agent._node_assemble_and_validate_summary)

    graph.set_entry_point("prepare_context")

    graph.add_edge("prepare_context", "draft_introduction")
    graph.add_edge("draft_introduction", "draft_related_work")
    graph.add_edge("draft_related_work", "draft_hypotheses_section")
    graph.add_edge("draft_hypotheses_section", "draft_per_hypothesis_sections")
    graph.add_edge("draft_per_hypothesis_sections", "draft_limitations")
    graph.add_edge("draft_limitations", "draft_future_work")
    # Abstract and Title come last on purpose: both summarize a body that has to
    # exist first (see writer_agent.py's module docstring).
    graph.add_edge("draft_future_work", "draft_abstract")
    graph.add_edge("draft_abstract", "draft_title")

    graph.add_edge("draft_title", "resolve_citations")
    graph.add_edge("resolve_citations", "validate_paper_honesty")
    graph.add_edge("validate_paper_honesty", "render_pdf")
    graph.add_edge("render_pdf", "assemble_and_validate_summary")
    graph.add_edge("assemble_and_validate_summary", END)

    # In-memory checkpointing, matching the literature/hypothesis/planner/reviewer
    # graphs: a crash partway through can resume from the last completed node (via
    # the same thread_id) rather than re-running every LLM call — worth more here
    # than anywhere else in the pipeline, since a full draft is a dozen-plus calls.
    # Swap for a SqliteSaver if that should survive process restarts too.
    return graph.compile(checkpointer=MemorySaver())
