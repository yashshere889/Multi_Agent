"""Normalizes raw paper dicts (from the Literature Agent, or any future source
using the same shape) and chunks them for batched LLM calls.

Pure, deterministic, no LLM/network calls — kept separate from
hypothesis_agent.py so it's trivially unit-testable.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, TypedDict

logger = logging.getLogger(__name__)


class NormalizedPaper(TypedDict):
    id: str
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: str
    source: str
    full_text: str  # empty unless the source paper dict provides one


def normalize_paper(raw: dict, index: int) -> Optional[NormalizedPaper]:
    """Best-effort normalization of a single paper dict. Returns None if there's
    nothing usable (no title, abstract, or full text) rather than raising —
    the Literature Agent's output can be sparse (e.g. missing abstracts)."""
    if not isinstance(raw, dict):
        logger.warning("Skipping paper at index %d: expected a dict, got %s", index, type(raw).__name__)
        return None

    title = (raw.get("title") or "").strip()
    abstract = (raw.get("abstract") or "").strip()

    # Today's Literature Agent only produces title/abstract-level metadata (see
    # agents/literature/state.py — no full_text/sections field exists yet). This
    # picks one up automatically if a future version adds it, preferring it over
    # the abstract, without requiring any change here.
    full_text = raw.get("full_text") or raw.get("sections") or ""
    if isinstance(full_text, (dict, list)):
        full_text = json.dumps(full_text)
    full_text = str(full_text).strip()

    if not title and not abstract and not full_text:
        logger.warning("Skipping paper at index %d: no title, abstract, or full text", index)
        return None

    paper_id = raw.get("arxiv_id") or raw.get("paper_id") or raw.get("doi") or f"paper_{index}"
    authors = raw.get("authors")
    if not isinstance(authors, list):
        authors = []
    year = raw.get("year")
    if not isinstance(year, int):
        year = None

    return {
        "id": str(paper_id),
        "title": title or "(untitled)",
        "authors": [str(a) for a in authors],
        "year": year,
        "abstract": abstract,
        "source": raw.get("source") or "unknown",
        "full_text": full_text,
    }


def normalize_papers(raw_papers: List[dict]) -> List[NormalizedPaper]:
    normalized = [p for p in (normalize_paper(raw, i) for i, raw in enumerate(raw_papers or [])) if p is not None]
    dropped = len(raw_papers or []) - len(normalized)
    if dropped:
        logger.warning("Dropped %d/%d papers with no usable content", dropped, len(raw_papers or []))
    return normalized


def paper_to_text(paper: NormalizedPaper) -> str:
    authors = ", ".join(paper["authors"]) or "unknown authors"
    year = paper["year"] if paper["year"] is not None else "n.d."
    body = paper["full_text"] or paper["abstract"] or "(no abstract available)"
    return f"[{paper['id']}] {paper['title']} ({authors}, {year}) [source: {paper['source']}]\n{body}"


def chunk_papers(papers: List[NormalizedPaper], max_chars_per_batch: int = 12000) -> List[List[NormalizedPaper]]:
    """Greedily groups papers into batches so each batch's rendered text stays
    under max_chars_per_batch — a character-count proxy for a token budget, so
    the model is never asked to hold the entire paper set (abstracts, or full
    text if present) in one call. A single paper longer than the budget still
    gets its own batch rather than being dropped or split mid-paper."""
    batches: List[List[NormalizedPaper]] = []
    current: List[NormalizedPaper] = []
    current_chars = 0

    for paper in papers:
        paper_chars = len(paper_to_text(paper))
        if current and current_chars + paper_chars > max_chars_per_batch:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(paper)
        current_chars += paper_chars

    if current:
        batches.append(current)

    return batches
