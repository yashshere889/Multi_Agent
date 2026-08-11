"""Builds the Interdisciplinary Literature Agent's LangGraph.

Same shape as agents/hypothesis/graph.py: the fan-out width isn't known until
runtime (it's however many adjacent fields the model identified), so the branch
is a conditional edge returning a list of `Send`s rather than a fixed set of
`add_edge` calls. One extra wrinkle over the hypothesis graph — that list can
legitimately be *empty* (no adjacent field was identified), and an empty Send
list would strand the run with no path to the output, so the same conditional
edge routes straight to the merge node in that case.

The graph is built per-agent-instance because every node is a bound method on a
configured InterdisciplinaryLiteratureAgent (its chat model, field budget,
search clients, and output dir) — nodes stay thin adapters over the agent's
existing private helpers, so the LLM/IO logic lives in exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from research_pipeline.agents.interdisciplinary_literature.state import InterdisciplinaryState

if TYPE_CHECKING:  # avoids a circular import — the agent module imports this one
    from research_pipeline.agents.interdisciplinary_literature.interdisciplinary_literature_agent import (
        InterdisciplinaryLiteratureAgent,
    )


def fan_out_fields(state: InterdisciplinaryState) -> Union[List[Send], str]:
    """One `search_field` invocation per adjacent field, each seeded with the
    field it owns. LangGraph runs them concurrently and each returns
    `{"field_results": [...]}`, which the state's operator.add reducer
    accumulates.

    With no fields to explore there is nothing to fan out to, so the run skips
    ahead to the merge node — which then produces a pool identical to the
    in-domain papers it was given, rather than failing a run the Literature
    Agent's own output was fine for."""
    fields = state.get("fields_explored") or []
    if not fields:
        return "merge_cross_field"
    return [Send("search_field", {**state, "current_field": field}) for field in fields]


def build_interdisciplinary_literature_graph(agent: "InterdisciplinaryLiteratureAgent"):
    graph = StateGraph(InterdisciplinaryState)

    graph.add_node("identify_adjacent_fields", agent._node_identify_adjacent_fields)
    graph.add_node("search_field", agent._node_search_field)
    graph.add_node("merge_cross_field", agent._node_merge_cross_field)
    graph.add_node("synthesize_bridges", agent._node_synthesize_bridges)
    graph.add_node("assemble_and_validate", agent._node_assemble_and_validate)

    graph.set_entry_point("identify_adjacent_fields")

    # Fan out: one search_field branch per adjacent field (or straight to the
    # merge when there are none).
    graph.add_conditional_edges("identify_adjacent_fields", fan_out_fields, ["search_field", "merge_cross_field"])

    # Fan in: a plain edge out of a Send-spawned node only fires once every
    # branch has finished, so merge_cross_field sees the complete
    # `field_results` list.
    graph.add_edge("search_field", "merge_cross_field")

    graph.add_edge("merge_cross_field", "synthesize_bridges")
    graph.add_edge("synthesize_bridges", "assemble_and_validate")
    graph.add_edge("assemble_and_validate", END)

    # In-memory checkpointing, matching every other agent's graph: a crash
    # partway through can resume from the last completed node (via the same
    # thread_id) rather than re-running the searches and LLM calls. Swap for a
    # SqliteSaver if that should survive process restarts too.
    return graph.compile(checkpointer=MemorySaver())
