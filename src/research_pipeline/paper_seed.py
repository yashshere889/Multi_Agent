"""Builds a Literature-Agent-shaped output from papers the user already has.

Lets a run start from a curated reading list instead of the Literature Agent's
arXiv/Semantic Scholar search. The result is a plain dict shaped exactly like
`LiteratureState` (see agents/literature/state.py), so every downstream
consumer — the Interdisciplinary Literature Agent, the Hypothesis Agent, the
Writer's citation registry — cannot tell it apart from a real search run. That
is the point: seeding is a different *source* of papers, not a different
contract, so no agent needs a "was this seeded?" branch.

    from research_pipeline.paper_seed import seed_literature_output
    literature_output = seed_literature_output(my_papers, "does RAG help?")

Dedupe follows the Literature Agent's own rule rather than a new one — same key,
same tie-break — because "are these the same paper?" must only ever have one
answer in this pipeline.
"""

from __future__ import annotations

import logging
from typing import List

# The Literature Agent's own dedupe key, imported rather than re-implemented for
# the same reason the Interdisciplinary Literature Agent imports it: copying it
# here would be exactly the kind of duplicate that drifts.
from research_pipeline.agents.literature.nodes import _normalize_title

logger = logging.getLogger(__name__)


def seed_literature_output(papers: List[dict], research_question: str) -> dict:
    """Turn a list of user-supplied paper dicts into a LiteratureState-shaped dict.

    Papers with no title are dropped (the Literature Agent's own rule in
    merge_and_dedupe_node — a title-less entry can't be deduped or cited).
    Duplicates collapse on `doi or normalized title`, preferring whichever entry
    carries a `pdf_url`, again matching merge_and_dedupe_node exactly.

    Ids are left exactly as given: `agents/hypothesis/papers.py:normalize_paper`
    already falls back to `arxiv_id or paper_id or doi or f"paper_{index}"`
    downstream, so inventing an id scheme here would only create a second one.
    """
    merged: dict[str, dict] = {}
    dropped = 0
    for paper in papers:
        if not paper.get("title"):
            dropped += 1
            continue
        paper = dict(paper)
        # Don't overwrite a source the caller deliberately set (e.g. re-seeding
        # a previous run's arXiv results).
        paper.setdefault("source", "user_provided")
        key = paper.get("doi") or _normalize_title(paper["title"])
        if key in merged:
            if not merged[key].get("pdf_url") and paper.get("pdf_url"):
                merged[key] = paper
        else:
            merged[key] = paper

    merged_list = list(merged.values())
    logger.info(
        "Seeded literature stage with %d user-provided paper(s) (%d dropped for missing title, %d duplicate(s) collapsed)",
        len(merged_list),
        dropped,
        len(papers) - dropped - len(merged_list),
    )
    return {
        "research_question": research_question,
        "merged_papers": merged_list,
        "search_queries": [],
        "arxiv_papers": [],
        "semantic_scholar_papers": [],
    }
