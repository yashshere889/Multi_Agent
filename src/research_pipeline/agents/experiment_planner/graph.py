"""Builds the Experiment Planner Agent's LangGraph.

Same shape as agents/hypothesis/graph.py: the fan-out width isn't known until
runtime (it's however many hypotheses the input carried), so the branch is a
conditional edge returning a list of `Send`s rather than a fixed set of
`add_edge` calls.

The graph is built per-agent-instance because every node is a bound method on a
configured ExperimentPlannerAgent (its chat model and output dir) — nodes stay
thin adapters over the agent's existing private helpers, so the LLM/IO logic
lives in exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy, Send

from research_pipeline.agents.experiment_planner.state import ExperimentPlannerState
from research_pipeline.checkpointer import get_checkpointer

# Outer safety net on the two LLM-calling nodes, default retry_on. See
# agents/literature/graph.py for the full reasoning.
_RETRY = RetryPolicy(max_attempts=2)

if TYPE_CHECKING:  # avoids a circular import — experiment_planner_agent imports this module
    from research_pipeline.agents.experiment_planner.experiment_planner_agent import ExperimentPlannerAgent


def fan_out_hypotheses(state: ExperimentPlannerState) -> List[Send]:
    """One `plan_one` invocation per hypothesis, each seeded with the hypothesis
    it owns. LangGraph runs them concurrently — replacing the ThreadPoolExecutor
    the sequential implementation used — and each returns `{"plans": [...]}`,
    which the state's operator.add reducer accumulates."""
    return [Send("plan_one", {**state, "current_hypothesis": hypothesis}) for hypothesis in state["hypotheses"]]


def build_experiment_planner_graph(agent: "ExperimentPlannerAgent"):
    graph = StateGraph(ExperimentPlannerState)

    graph.add_node("validate_input", agent._node_validate_input)
    graph.add_node("plan_one", agent._node_plan_one, retry_policy=_RETRY)
    graph.add_node("reorder_plans", agent._node_reorder_plans)
    graph.add_node("plan_cross_cutting", agent._node_plan_cross_cutting, retry_policy=_RETRY)
    graph.add_node("assemble_and_validate", agent._node_assemble_and_validate)

    graph.set_entry_point("validate_input")

    # Fan out: one plan_one branch per hypothesis.
    graph.add_conditional_edges("validate_input", fan_out_hypotheses, ["plan_one"])

    # Fan in: a plain edge out of a Send-spawned node only fires once every
    # branch has finished, so reorder_plans sees the complete `plans` list.
    graph.add_edge("plan_one", "reorder_plans")

    graph.add_edge("reorder_plans", "plan_cross_cutting")
    graph.add_edge("plan_cross_cutting", "assemble_and_validate")
    graph.add_edge("assemble_and_validate", END)

    # Checkpointing, matching the hypothesis/literature graphs: a crash partway
    # through can resume from the last completed node (via the same thread_id)
    # rather than re-running every LLM call. In-memory by default;
    # CHECKPOINTER_BACKEND makes that survive process restarts too.
    return graph.compile(checkpointer=get_checkpointer())
