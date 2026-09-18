"""Offline evaluation for the search agents.

The one thing this package is for: making "did that change help?" answerable
about the Literature and Interdisciplinary Literature agents, which until it
existed could only be judged by reading a few titles and forming an impression.

    research-pipeline eval-bootstrap --survey arXiv:<id> --question "..."
    research-pipeline eval-run --gold evals/gold --name baseline
    research-pipeline eval-score --run evals/runs/baseline.json --judge
    research-pipeline eval-compare evals/runs/a.json evals/runs/b.json

See harness.py for why a run searches once and scores many times, gold.py for
where ground truth comes from (real survey bibliographies, never a model), and
judge.py for why the LLM judge must not reuse the relevance screen's rubric.
"""

from research_pipeline.eval.gold import GoldSetError, bootstrap_from_survey, load_gold_set
from research_pipeline.eval.harness import compare, read_run, run_gold_set, score_run, write_run

__all__ = [
    "GoldSetError",
    "bootstrap_from_survey",
    "compare",
    "load_gold_set",
    "read_run",
    "run_gold_set",
    "score_run",
    "write_run",
]
