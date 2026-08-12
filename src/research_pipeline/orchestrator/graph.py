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

A run can also cover a *contiguous slice* of the pipeline rather than all of it:
`start_stage` picks between the Literature Agent's search and papers the caller
already has (`seed_papers`, see paper_seed.py), `end_stage` picks where the run
stops, and `include_interdisciplinary` toggles the cross-field stage. All three
are optional and their defaults reproduce the original fixed linear run.
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
    seed_literature_node,
    run_planner_node,
)
from research_pipeline.orchestrator.state import PipelineState

# The logical stage order. Stages can only be skipped as a *contiguous range* —
# pick where the run starts, pick where it stops — because the chain is a data
# dependency, not a menu: the Coder needs planner_output, which only the
# Experiment Planner produces, so "skip the planner" isn't a meaningful request.
# The only interior stage that can drop out is interdisciplinary_literature,
# which passes papers straight through in the same shape it received them.
STAGE_SEQUENCE = (
    "literature",
    "interdisciplinary_literature",
    "hypothesis",
    "experiment_planner",
    "coder",
    "writer_reviewer",
)

# "writer_reviewer" is the logical name for the Writer <-> Reviewer cycle; the
# node that actually starts it is draft_or_revise. Public because the webapp's
# progress view needs the same stage-name -> node-name answer this module routes
# on, and a second copy of it there could drift out of step with this one.
NODE_FOR_STAGE = {stage: stage for stage in STAGE_SEQUENCE}
NODE_FOR_STAGE["writer_reviewer"] = "draft_or_revise"

DEFAULT_END_STAGE = "writer_reviewer"


def _resolved_end_stage(state: PipelineState) -> str:
    """The last stage this run will execute, after applying
    include_interdisciplinary — a run told to stop after the cross-field stage
    while that stage is disabled stops after literature instead, rather than
    routing to a node that will never run."""
    end_stage = state.get("end_stage") or DEFAULT_END_STAGE
    if end_stage == "interdisciplinary_literature" and not state.get("include_interdisciplinary", True):
        return "literature"
    return end_stage


def _after(stage_key: str):
    """Router *factory* for the edge leaving `stage_key`.

    Returns a closure over the stage name because LangGraph hands a conditional
    edge's router only the state — the "which stage am I leaving?" half has to
    be bound at graph-build time.

    With no start_stage/end_stage/include_interdisciplinary set at all, this
    reduces to exactly the original fixed linear chain.
    """

    def router(state: PipelineState) -> str:
        end_stage = _resolved_end_stage(state)
        if stage_key == end_stage:
            return "finalize"

        include_interdisciplinary = state.get("include_interdisciplinary", True)
        for stage in STAGE_SEQUENCE[STAGE_SEQUENCE.index(stage_key) + 1 :]:
            if stage == "interdisciplinary_literature" and not include_interdisciplinary:
                continue
            return NODE_FOR_STAGE[stage]
        return "finalize"

    return router


def _entry_router(state: PipelineState) -> str:
    """Start from the Literature Agent's search, or from papers the caller
    already has. Both paths leave `literature_output` populated in the same
    shape, so everything downstream is identical (see paper_seed.py)."""
    return "seed_literature" if state.get("start_stage") == "own_papers" else "literature"


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
    graph.add_node("seed_literature", seed_literature_node)
    graph.add_node("interdisciplinary_literature", run_interdisciplinary_literature_node)
    graph.add_node("hypothesis", run_hypothesis_node)
    graph.add_node("experiment_planner", run_planner_node)
    graph.add_node("coder", run_coder_node)
    graph.add_node("draft_or_revise", draft_or_revise_node)
    graph.add_node("review", review_node)
    graph.add_node("finalize", finalize_node)

    graph.set_conditional_entry_point(
        _entry_router,
        {"literature": "literature", "seed_literature": "seed_literature"},
    )

    # Every upstream edge is conditional on where this run is told to stop and
    # whether the cross-field stage is enabled. Defaults reproduce the original
    # unconditional chain exactly: literature -> interdisciplinary_literature ->
    # hypothesis -> experiment_planner -> coder -> draft_or_revise.
    _stage_destinations = {
        "literature": "literature",
        "interdisciplinary_literature": "interdisciplinary_literature",
        "hypothesis": "hypothesis",
        "experiment_planner": "experiment_planner",
        "coder": "coder",
        "draft_or_revise": "draft_or_revise",
        "finalize": "finalize",
    }
    # Both literature paths take the same decision — each leaves
    # literature_output populated, and nothing downstream can tell which ran.
    graph.add_conditional_edges("literature", _after("literature"), _stage_destinations)
    graph.add_conditional_edges("seed_literature", _after("literature"), _stage_destinations)
    for stage in ("interdisciplinary_literature", "hypothesis", "experiment_planner", "coder"):
        graph.add_conditional_edges(stage, _after(stage), _stage_destinations)

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
