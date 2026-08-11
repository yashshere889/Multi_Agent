"""Shared state schema for the Experiment Planner Agent's LangGraph.

Mirrors agents/hypothesis/state.py: a single TypedDict threaded through every
node, each node returning only the keys it produces. `total=False` because keys
appear progressively as the graph advances — nothing but `hypothesis_output`
exists at entry.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict


class ExperimentPlannerState(TypedDict, total=False):
    # Inputs — the Hypothesis Agent's output dict, exactly as handed to run(),
    # plus the optional subset of hypothesis ids to actually plan. None (the
    # default) plans every hypothesis in the input, as this agent always has.
    hypothesis_output: dict
    hypothesis_ids: Optional[List[str]]

    # validate_input — `hypotheses`/`expected_ids` are the *filtered* set the
    # rest of the graph plans over; the full input is still validated in full.
    hypotheses: List[dict]
    expected_ids: List[str]

    # plan_one — one instance per hypothesis, fanned out via Send. Each Send
    # carries the hypothesis that branch should plan as `current_hypothesis`
    # (node input only; never written back to the graph's state), and `plans`
    # accumulates across concurrent branches rather than being overwritten by
    # whichever writer finishes last. Every accumulated plan already has its
    # `hypothesis_id` force-set from the source hypothesis, never the model's echo.
    current_hypothesis: dict
    plans: Annotated[List[dict], operator.add]

    # reorder_plans — the same plans back in the input's hypothesis order.
    # Send branches finish in whatever order they finish in, so the ordering is
    # re-derived deterministically from `expected_ids` rather than inherited
    # from completion order.
    experiment_plans: List[dict]

    # plan_cross_cutting
    shared_infrastructure: list
    priority_order: list

    # assemble_and_validate — the schema-valid dict run() returns
    result: dict
