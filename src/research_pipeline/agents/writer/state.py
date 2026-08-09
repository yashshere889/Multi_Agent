"""Shared state schema for the Writer Agent's LangGraph.

Mirrors agents/reviewer/state.py: a single TypedDict threaded through every
node, each node returning only the keys it produces. `total=False` because keys
appear progressively as the graph advances — only the `run()`/`revise()`
arguments exist at entry.

Like the Reviewer's there is no `Annotated[..., operator.add]` reducer, and
unlike the Hypothesis/Experiment Planner graphs there is no `Send` fan-out: the
Writer's graph is a straight chain (each section's draft is its own node), so no
key is ever written by two nodes in the same super-step. The per-batch Related
Work drafting and the per-hypothesis Methods/Results/Discussion drafting are
still concurrent, but that concurrency stays where it already was — inside those
two nodes' existing ThreadPoolExecutors — rather than becoming graph branches.

`revision` is the one optional *input*: absent for a fresh draft (`run()`),
present for a revision pass (`revise()`), and every drafting node reads it to
decide whether to append its section's revision note.

One thing deliberately **not** in this state: the `CitationRegistry`. Every
value written to state is serialized by the checkpointer, and a plain Python
object like the registry is not msgpack-serializable (unlike `IndexedPaper`,
which is a TypedDict, i.e. a plain dict) — putting it in state fails the run
outright. So the registry is rebuilt, deterministically, from `paper_index` plus
the raw drafted text already in state, by `build_scanned_registry` in
writer_agent.py; only its *outputs* (resolved text, references, citation order,
unresolved markers) are threaded through the graph.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypedDict

from research_pipeline.agents.writer.citations import IndexedPaper


class WriterState(TypedDict, total=False):
    # Inputs — exactly the arguments run() (or revise(), which adds `revision`)
    # was called with.
    literature_output: object
    hypothesis_output: dict
    planner_output: dict
    coder_output: dict
    paper_filename: Optional[str]
    summary_filename: Optional[str]
    # {"previous_sections": {heading: text}, "previous_subsections": {heading:
    # {hypothesis_id: text}}, "feedback_by_section": {heading: [bullet, ...]}} —
    # only present on a revision pass; see WriterAgent.revise().
    revision: dict

    # prepare_context — upstream outputs validated and the grounding every
    # drafting node writes against indexed.
    hypotheses: List[dict]
    expected_ids: List[str]
    raw_papers: List[dict]
    paper_index: Dict[str, IndexedPaper]
    valid_paper_ids: List[str]
    plan_by_id: Dict[str, dict]
    experiment_by_id: Dict[str, dict]
    verdicts: Dict[str, dict]

    # Section drafting (one node each, in the order the paper is written) —
    # still carrying `[[cite:ID]]` markers at this point.
    introduction: str
    related_work: str
    hypotheses_section: str
    methods_by_id: Dict[str, str]
    results_by_id: Dict[str, str]
    discussion_by_id: Dict[str, str]
    limitations: str
    future_work: str
    raw_abstract: str
    raw_title: str

    # resolve_citations — the drafted sections assembled in printing order, then
    # the same texts with every marker replaced by its final author-year form.
    raw_sections: List[Tuple[str, str]]
    resolved_sections: List[Tuple[str, str]]
    abstract: str
    title: str
    references: List[str]
    citations_used: List[str]

    # validate_paper_honesty (deterministic — dropped citation markers surface
    # here as review notes) / render_pdf
    notes_for_review: List[str]
    paper_path: str

    # assemble_and_validate_summary — the schema-valid dict run()/revise() return
    result: dict
