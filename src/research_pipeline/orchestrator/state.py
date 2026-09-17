"""Shared state schema for the top-level pipeline orchestrator LangGraph.

Each stage's key holds that stage's *entire* output dict, matching the shape
every run_<name>_agent() already expects from its upstream agent (see
README's "Chaining agents individually") — no adapter/remapping between nodes.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

StartStage = Literal["literature", "own_papers"]
EndStage = Literal[
    "literature",
    "interdisciplinary_literature",
    "hypothesis",
    "experiment_planner",
    "coder",
    "writer_reviewer",
]


class PipelineState(TypedDict, total=False):
    # --- inputs ---
    research_question: str
    max_results_per_query: int
    download_dir: str
    metadata_path: str
    output_dir: Optional[str]
    max_iterations: Optional[int]
    quality_threshold: Optional[int]

    # --- run shape: which contiguous slice of the pipeline this run covers ---
    # All optional, and every default reproduces the original fixed linear run,
    # so a caller that sets none of them gets exactly today's behavior.
    start_stage: StartStage  # default "literature" (run the search)
    end_stage: EndStage  # default "writer_reviewer" (run everything)
    include_interdisciplinary: bool  # default True
    # Set only when picking a stopped run back up: the last stage that already
    # ran, whose *_output keys below are seeded straight into this run's state.
    # The run then enters at whatever comes *after* it (see graph._entry_router),
    # so this is the one key that changes where the graph starts without a
    # matching start_stage value of its own.
    resume_from: EndStage
    # Which hypothesis (or hypotheses) the Experiment Planner takes forward.
    # Absent by default, and absent means "whichever the ranking put first" —
    # exactly the behaviour before this key existed. Set it to steer the run
    # after reading the three hypotheses the Hypothesis Agent generated, rather
    # than accepting its ranking. See orchestrator/nodes.run_planner_node.
    planned_hypothesis_ids: List[str]
    # Required iff start_stage == "own_papers": raw paper dicts to seed the
    # literature stage with, in place of running a search. See paper_seed.py.
    seed_papers: List[dict]

    # --- stage outputs, each the exact dict shape run_<name>_agent returns ---
    literature_output: dict
    interdisciplinary_output: dict
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
