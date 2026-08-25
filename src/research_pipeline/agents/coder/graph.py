"""Builds the Coder Agent's LangGraph.

The only one of the six graphs with real cycles, and the only one that is
deliberately **not** parallel. Two loops are modelled as graph cycles rather
than Python `for` loops — the same move the top-level orchestrator already makes
for the Writer/Reviewer cycle (see orchestrator/graph.py's
`should_continue_revising`):

- the **per-plan loop**: `process_current_plan` -> ... -> `finalize_current_plan`
  (or `give_up_current_plan`) -> back to `process_current_plan` with
  `plan_index` advanced, until every plan in `ordered_plans` is done.
- the **fix loop**, nested inside it: `attempt` -> `snapshot_and_regenerate` ->
  `attempt`, until a check passes, a terminal result is produced, or
  `max_fix_attempts` is spent.

Why not `Send` fan-out over the plans, the way agents/hypothesis/graph.py and
agents/experiment_planner/graph.py fan out their per-item work? Because plans
are not independent here. `CODER_MAX_SLURM_JOBS_PER_RUN` is enforced against a
counter incremented on each successful submission, and each plan must see the
true up-to-date count before deciding whether it may auto-submit. Running the
plans concurrently would let two plans read the same pre-submission count and
both submit — a silent change to real submission-gating behaviour, not just to
tracing. The sequencing is the contract; the graph exists to make each step
observable, not to make it concurrent. (The per-plan generate/run work is I/O
and LLM bound, so this costs nothing that was ever being saved.)

Routing:
- `route_plan_loop` — "process" while `plan_index` is in range, else "done".
  Used from `start_plan_loop` and from both per-plan exits, so all three share
  one definition of "is there another plan?".
- `route_after_process` — infeasible plans never reach the fix loop: that node
  has already appended their "skipped" entry and advanced the cursor, so it
  falls straight back through `route_plan_loop`.
- `CoderAgent._route_after_attempt` — a bound method rather than a module-level
  function because it compares against `agent.max_fix_attempts`, which is
  per-instance (constructor argument, falling back to settings).

The graph is built per-agent-instance because every node is a bound method on a
configured CoderAgent (its chat model, experiments/output dirs, network/GPU
probes, fix budget) — nodes stay thin adapters over the agent's existing private
helpers, so the codegen/execution/submission logic lives in exactly one place.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

from research_pipeline.agents.coder.state import CoderState
from research_pipeline.checkpointer import get_checkpointer

if TYPE_CHECKING:  # avoids a circular import — coder_agent imports this module
    from research_pipeline.agents.coder.coder_agent import CoderAgent

# Deliberately narrow coverage: only the two nodes whose work is a retryable
# call with no side effects worth repeating. `attempt` and
# `snapshot_and_regenerate` are the fix loop, and retrying them at the graph
# level would re-execute generated code or re-provision a venv — this module's
# whole point is that the loop is sequential and each step's effects are real
# (and CLAUDE.md is explicit that env-provisioning failures are not retried).
# `probe_environment` and the five dataset-selection nodes are best-effort:
# every one of them absorbs its own failures in-node and degrades to "no
# dataset", so there is nothing left for a graph-level retry to catch.
_RETRY = RetryPolicy(max_attempts=2)


def route_plan_loop(state: CoderState) -> str:
    """Is there another plan to process? The single place the outer loop's
    termination condition is written down."""
    return "process" if state["plan_index"] < len(state["ordered_plans"]) else "done"


def route_after_process(state: CoderState) -> str:
    """Feasible plans go on to dataset selection and then their first
    generation; infeasible ones were already recorded as "skipped" (and the
    cursor advanced) by process_current_plan, so they re-enter the outer loop
    directly without a single LLM call or HTTP call."""
    if state["current_plan"]["feasible"]:
        return "specify_data_requirements"
    return route_plan_loop(state)


def build_coder_graph(agent: CoderAgent):
    graph = StateGraph(CoderState)

    graph.add_node("validate_input", agent._node_validate_input)
    graph.add_node("probe_environment", agent._node_probe_environment)
    graph.add_node(
        "setup_shared_infrastructure",
        agent._node_setup_shared_infrastructure,
        retry_policy=_RETRY,
    )
    graph.add_node("start_plan_loop", agent._node_start_plan_loop)
    graph.add_node("process_current_plan", agent._node_process_current_plan)
    graph.add_node("specify_data_requirements", agent._node_specify_data_requirements)
    graph.add_node("shortlist_datasets", agent._node_shortlist_datasets)
    graph.add_node("appraise_datasets", agent._node_appraise_datasets)
    graph.add_node("critique_leading_dataset", agent._node_critique_leading_dataset)
    graph.add_node("acquire_dataset", agent._node_acquire_dataset)
    graph.add_node(
        "generate_experiment_code",
        agent._node_generate_experiment_code,
        retry_policy=_RETRY,
    )
    graph.add_node("attempt", agent._node_attempt)
    graph.add_node("snapshot_and_regenerate", agent._node_snapshot_and_regenerate)
    graph.add_node("finalize_current_plan", agent._node_finalize_current_plan)
    graph.add_node("give_up_current_plan", agent._node_give_up_current_plan)
    graph.add_node("assemble_and_validate", agent._node_assemble_and_validate)

    graph.set_entry_point("validate_input")
    graph.add_edge("validate_input", "probe_environment")
    graph.add_edge("probe_environment", "setup_shared_infrastructure")
    graph.add_edge("setup_shared_infrastructure", "start_plan_loop")

    # Hashable keys, not str: add_conditional_edges' path_map parameter is typed
    # dict[Hashable, str] and dict is invariant in its key type, so a plain
    # dict[str, str] doesn't type-check against it.
    _plan_loop_targets: dict[Hashable, str] = {
        "process": "process_current_plan",
        "done": "assemble_and_validate",
    }

    graph.add_conditional_edges("start_plan_loop", route_plan_loop, _plan_loop_targets)

    # process_current_plan is both the loop body's entry and (for infeasible
    # plans) a self-loop: it can record a "skipped" plan and immediately move to
    # the next one without going near the fix loop.
    graph.add_conditional_edges(
        "process_current_plan",
        route_after_process,
        {"specify_data_requirements": "specify_data_requirements", **_plan_loop_targets},
    )

    # Dataset selection: five steps, ahead of the first generation, because what
    # they settle on goes *into* the codegen prompt — and because each is a real
    # decision with an observable outcome that belongs in a trace rather than
    # buried inside the generation call. They are separate nodes rather than one
    # because they answer separate questions, and a run whose experiments quietly
    # stopped getting real data should be diagnosable to the step that stopped:
    #
    #   specify_data_requirements  (LLM)   what does this plan actually need?
    #   shortlist_datasets         (HTTP)  who is worth appraising?
    #   appraise_datasets          (LLM)   what does the evidence say, and what
    #                                      does Python score it at?
    #   critique_leading_dataset   (LLM)   can the leader be rejected?
    #   acquire_dataset            (disk)  download, normalize, record
    #
    # None of them branches: every failure is a smaller answer (an empty spec, no
    # candidates, nothing accepted, no local copy), and generation proceeds.
    graph.add_edge("specify_data_requirements", "shortlist_datasets")
    graph.add_edge("shortlist_datasets", "appraise_datasets")
    graph.add_edge("appraise_datasets", "critique_leading_dataset")
    graph.add_edge("critique_leading_dataset", "acquire_dataset")
    graph.add_edge("acquire_dataset", "generate_experiment_code")
    graph.add_edge("generate_experiment_code", "attempt")

    # The fix loop. `attempt` is the only node that runs generated code; every
    # exit from it is a real check's verdict, never model judgment.
    graph.add_conditional_edges(
        "attempt",
        agent._route_after_attempt,
        {
            "finalize": "finalize_current_plan",
            "give_up": "give_up_current_plan",
            "regenerate": "snapshot_and_regenerate",
        },
    )
    graph.add_edge("snapshot_and_regenerate", "attempt")

    # Both per-plan exits advance plan_index and re-enter the outer loop.
    graph.add_conditional_edges("finalize_current_plan", route_plan_loop, _plan_loop_targets)
    graph.add_conditional_edges("give_up_current_plan", route_plan_loop, _plan_loop_targets)

    graph.add_edge("assemble_and_validate", END)

    # Checkpointing, matching the other five graphs: a crash partway through can
    # resume from the last completed node (via the same thread_id) rather than
    # re-running every LLM call and every generated experiment. In-memory by
    # default; CHECKPOINTER_BACKEND makes that survive process restarts too.
    return graph.compile(checkpointer=get_checkpointer())
