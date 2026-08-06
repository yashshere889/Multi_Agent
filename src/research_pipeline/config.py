"""Central settings, loaded from environment variables / .env."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    semantic_scholar_api_key: str
    default_max_results_per_query: int
    hypothesis_output_dir: str
    hypothesis_batch_max_chars: int
    experiment_planner_output_dir: str
    coder_experiments_dir: str
    coder_output_dir: str
    writer_output_dir: str
    writer_related_work_batch_max_chars: int
    writer_paper_authors: str
    writer_paper_affiliation: str
    reviewer_output_dir: str
    writer_reviewer_loop_output_dir: str
    writer_reviewer_max_iterations: int
    writer_reviewer_quality_threshold: int


def load_settings() -> Settings:
    semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if not semantic_scholar_api_key:
        logger.warning(
            "SEMANTIC_SCHOLAR_API_KEY is not set — Semantic Scholar search will be skipped "
            "(unauthenticated requests are aggressively rate-limited / rejected)."
        )
    return Settings(
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY", "not-needed"),
        llm_model=os.environ.get("LLM_MODEL", "gemma-4-12b-it"),
        llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        semantic_scholar_api_key=semantic_scholar_api_key,
        default_max_results_per_query=int(os.environ.get("MAX_RESULTS_PER_QUERY", "5")),
        hypothesis_output_dir=os.environ.get("HYPOTHESIS_OUTPUT_DIR", "outputs"),
        hypothesis_batch_max_chars=int(os.environ.get("HYPOTHESIS_BATCH_MAX_CHARS", "12000")),
        experiment_planner_output_dir=os.environ.get("EXPERIMENT_PLANNER_OUTPUT_DIR", "outputs"),
        coder_experiments_dir=os.environ.get("CODER_EXPERIMENTS_DIR", "experiments"),
        coder_output_dir=os.environ.get("CODER_OUTPUT_DIR", "outputs"),
        writer_output_dir=os.environ.get("WRITER_OUTPUT_DIR", "outputs"),
        writer_related_work_batch_max_chars=int(os.environ.get("WRITER_RELATED_WORK_BATCH_MAX_CHARS", "12000")),
        # No real author identity flows through the pipeline, so this defaults to
        # NeurIPS's own anonymized-submission placeholder text rather than fabricating one.
        writer_paper_authors=os.environ.get("WRITER_PAPER_AUTHORS", "Anonymous Author(s)"),
        writer_paper_affiliation=os.environ.get("WRITER_PAPER_AFFILIATION", "Anonymous Institution"),
        reviewer_output_dir=os.environ.get("REVIEWER_OUTPUT_DIR", "outputs"),
        writer_reviewer_loop_output_dir=os.environ.get("WRITER_REVIEWER_LOOP_OUTPUT_DIR", "outputs/paper"),
        writer_reviewer_max_iterations=int(os.environ.get("WRITER_REVIEWER_MAX_ITERATIONS", "3")),
        writer_reviewer_quality_threshold=int(os.environ.get("WRITER_REVIEWER_QUALITY_THRESHOLD", "4")),
    )


settings = load_settings()
