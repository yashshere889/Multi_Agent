"""Shared state schema for the top-level pipeline orchestrator LangGraph.

Each stage's key holds that stage's *entire* output dict, matching the shape
every run_<name>_agent() already expects from its upstream agent (see
README's "Chaining agents individually") — no adapter/remapping between nodes.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    # --- inputs ---
    research_question: str
    max_results_per_query: int
    download_dir: str
    metadata_path: str
    output_dir: Optional[str]
    max_iterations: Optional[int]
    quality_threshold: Optional[int]

    # --- stage outputs, each the exact dict shape run_<name>_agent returns ---
    literature_output: dict
    hypothesis_output: dict
    planner_output: dict
    coder_output: dict

    # --- writer/reviewer cycle state ---
    paper_summary: dict
    review: dict
    iteration: int
    review_history: List[dict]
    converged: bool

    # --- final result ---
    final_result: dict
