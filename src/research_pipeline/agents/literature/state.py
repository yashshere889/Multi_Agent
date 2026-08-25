"""Shared state schema for the literature-search agent's LangGraph."""

from __future__ import annotations

from typing import List, Optional, TypedDict


class Paper(TypedDict, total=False):
    source: str
    arxiv_id: str
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int]
    pdf_url: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    local_path: Optional[str]

    # Bibliometric / provenance signal. Every search client fills these in with
    # whatever its own API actually knows and leaves the rest at None/[], so
    # the shape is uniform across sources even though the coverage isn't:
    # citation counts and `tldr` are Semantic Scholar-only, while
    # `fields_of_study` is populated from arXiv's categories and CORE's
    # fieldOfStudy too. Nothing downstream may assume a value is present.
    citation_count: Optional[int]
    influential_citation_count: Optional[int]
    venue: Optional[str]
    fields_of_study: List[str]
    tldr: Optional[str]


class LiteratureState(TypedDict, total=False):
    research_question: str
    max_results_per_query: int
    search_queries: List[str]
    arxiv_papers: List[Paper]
    semantic_scholar_papers: List[Paper]
    core_papers: List[Paper]
    merged_papers: List[Paper]
    download_dir: str
    metadata_path: str
