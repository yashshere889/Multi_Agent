"""Shared state schema for the Hypothesis Agent's LangGraph.

Mirrors agents/literature/state.py: a single TypedDict threaded through every
node, each node returning only the keys it produces. `total=False` because
keys appear progressively as the graph advances — nothing but `papers` (and
optionally `research_question`) exists at entry.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict

from research_pipeline.agents.hypothesis.papers import NormalizedPaper


class HypothesisState(TypedDict, total=False):
    # Inputs. `interdisciplinary_context` is the Interdisciplinary Literature
    # Agent's `bridge_insights` when one ran upstream, and None/[] otherwise —
    # the agent works exactly as before without it.
    papers: List[dict]
    research_question: Optional[str]
    interdisciplinary_context: Optional[list]

    # normalize_and_chunk
    normalized: List[NormalizedPaper]
    all_paper_ids: List[str]
    batches: List[List[NormalizedPaper]]

    # analyze_batch — one instance per batch, fanned out via Send. Each Send
    # carries the batch that branch should analyze as `current_batch` (node
    # input only; never written back to the graph's state), and `partials`
    # accumulates across concurrent branches rather than being overwritten by
    # whichever writer finishes last.
    current_batch: List[NormalizedPaper]
    partials: Annotated[List[dict], operator.add]

    # synthesize
    literature_summary: str
    methods_overview: list
    gaps: list

    # generate_hypotheses
    hypotheses: list

    # rank_hypotheses — the model's scored ordering, plus the winner, which is
    # derived in Python from whichever entry holds rank 1 rather than being a
    # second thing the model gets to assert.
    ranking: list
    selected_hypothesis_id: str

    # assemble_and_validate — the schema-valid dict run() returns
    result: dict
