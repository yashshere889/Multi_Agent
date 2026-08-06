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
    llm_top_p: float
    llm_max_tokens: int
    llm_enable_thinking: bool
    llm_reasoning_budget: int | None
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


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
        llm_model=os.environ.get("LLM_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
        llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        llm_top_p=float(os.environ.get("LLM_TOP_P", "0.95")),
        # Every agent parses structured JSON or full paper sections out of the
        # response, so a truncated completion is a hard failure — leave enough
        # room for the longest Writer section (and for a reasoning trace, if
        # LLM_ENABLE_THINKING is turned back on).
        llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "8192")),
        # Nemotron 3 Nano is a reasoning model that emits <think>...</think>
        # before its answer by default. Off by default here: nothing in this
        # pipeline reads reasoning traces, and the codebase already prefers
        # determinism over model deliberation wherever an answer is verifiable.
        # Traces are stripped defensively regardless (see llm_json.strip_reasoning).
        llm_enable_thinking=_env_bool("LLM_ENABLE_THINKING", False),
        llm_reasoning_budget=_env_optional_int("LLM_REASONING_BUDGET"),
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
