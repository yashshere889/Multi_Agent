"""Shared state schema for the Interdisciplinary Literature Agent's LangGraph.

Mirrors agents/hypothesis/state.py: a single TypedDict threaded through every
node, each node returning only the keys it produces. `total=False` because keys
appear progressively as the graph advances — nothing but `papers` (and
optionally `research_question`) exists at entry.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict


class InterdisciplinaryState(TypedDict, total=False):
    # Inputs — `papers` is the Literature Agent's paper list, exactly the shape
    # the Hypothesis Agent already accepts.
    papers: List[dict]
    research_question: Optional[str]

    # identify_adjacent_fields
    core_papers: List[dict]
    core_paper_ids: List[str]
    fields_explored: list

    # search_field — one instance per adjacent field, fanned out via Send. Each
    # Send carries the field that branch should search as `current_field` (node
    # input only; never written back to the graph's state), and `field_results`
    # accumulates across concurrent branches rather than being overwritten by
    # whichever writer finishes last.
    current_field: dict
    field_results: Annotated[List[dict], operator.add]

    # merge_cross_field — the deduped merged pool handed downstream, plus the
    # subset of it this agent actually added.
    merged_papers: List[dict]
    cross_field_papers: List[dict]

    # synthesize_bridges
    bridge_insights: list

    # assemble_and_validate — the schema-valid dict run() returns
    result: dict
