"""Shared state schema for the Reviewer Agent's LangGraph.

Mirrors agents/hypothesis/state.py and agents/experiment_planner/state.py: a
single TypedDict threaded through every node, each node returning only the keys
it produces. `total=False` because keys appear progressively as the graph
advances — only the eight `run()` arguments exist at entry.

Unlike those two agents there is no `Annotated[..., operator.add]` reducer here,
because the Reviewer has no dynamic-count fan-out: its three deterministic
checks are a fixed set fanned out with plain edges (as in
agents/literature/graph.py), and each writes its own distinct key, so nothing is
ever written concurrently by more than one node. The two keys that *are* written
twice — `hallucinations` and `hypothesis_coverage_issues`, appended to by
`check_discussion` — are written in strictly separate super-steps, so the plain
last-value channel is exactly the right semantics: that node reads the current
list and returns the extended one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from research_pipeline.agents.writer.citations import IndexedPaper


class ReviewerState(TypedDict, total=False):
    # Inputs — exactly the arguments run() was called with (`quality_threshold`
    # already resolved against settings, so no node has to re-derive it).
    paper_path: str | Path
    paper_summary: dict
    literature_output: object
    hypothesis_output: dict
    planner_output: dict
    coder_output: dict
    iteration: int
    quality_threshold: int

    # prepare_context — upstream outputs validated, ground truth indexed, and
    # the rendered PDF read back and split up.
    hypotheses: List[dict]
    expected_ids: List[str]
    raw_papers: List[dict]
    paper_index: Dict[str, IndexedPaper]
    plan_by_id: Dict[str, dict]
    experiment_by_id: Dict[str, dict]
    verdicts: Dict[str, dict]
    full_text: str
    sections: Dict[str, str]
    results_subsections: Dict[str, str]
    discussion_subsections: Dict[str, str]

    # The three deterministic checks (agents.reviewer.checks), fanned out from
    # prepare_context with plain edges and fanned back in on check_hallucinations.
    citation_issues: List[dict]
    results_accuracy_issues: List[dict]
    hypothesis_coverage_issues: List[dict]  # check_discussion appends framing issues

    # check_hallucinations / check_discussion (LLM)
    hallucinations: List[dict]
    check_errors: List[dict]  # checks whose response couldn't be parsed

    # score_quality (LLM)
    quality_scores: dict
    quality_notes: dict

    # compute_overall_pass / build_feedback (deterministic)
    overall_pass: bool
    feedback_for_writer: str

    # assemble_and_validate — the schema-valid dict run() returns
    result: dict
