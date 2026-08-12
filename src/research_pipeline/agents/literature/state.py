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
