"""Central settings, loaded from environment variables / .env."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


VALID_LLM_BACKENDS = ("openai", "huggingface")


@dataclass(frozen=True)
class Settings:
    llm_backend: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    llm_top_p: float
    llm_max_tokens: int
    llm_enable_thinking: bool
    llm_reasoning_budget: int | None
    llm_hf_device_map: str
    llm_hf_dtype: str
    semantic_scholar_api_key: str
    default_max_results_per_query: int
    hypothesis_output_dir: str
    hypothesis_batch_max_chars: int
    experiment_planner_output_dir: str
    coder_experiments_dir: str
    coder_output_dir: str
    coder_max_fix_attempts: int
    coder_auto_submit_slurm: bool
    coder_max_concurrent_slurm_jobs: int
    coder_max_slurm_jobs_per_run: int
    writer_output_dir: str
    writer_related_work_batch_max_chars: int
    writer_paper_authors: str
    writer_paper_affiliation: str
    reviewer_output_dir: str
    writer_reviewer_loop_output_dir: str
    writer_reviewer_max_iterations: int
    writer_reviewer_quality_threshold: int
    batch_output_root: str
    batch_max_consecutive_failures: int
    webapp_runs_dir: str
    webapp_host: str
    webapp_port: int
    webapp_max_concurrent_runs: int


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


def _env_backend() -> str:
    """Validated at load time rather than at first model construction: a typo'd
    backend would otherwise surface dozens of LLM calls into a run, on whichever
    agent happened to build a model first."""
    backend = os.environ.get("LLM_BACKEND", "openai").strip().lower() or "openai"
    if backend not in VALID_LLM_BACKENDS:
        raise ValueError(f"LLM_BACKEND must be one of {VALID_LLM_BACKENDS}, got {backend!r}")
    return backend


def load_settings() -> Settings:
    semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if not semantic_scholar_api_key:
        logger.warning(
            "SEMANTIC_SCHOLAR_API_KEY is not set — Semantic Scholar search will be skipped "
            "(unauthenticated requests are aggressively rate-limited / rejected)."
        )
    return Settings(
        # "openai": talk to any OpenAI-compatible server (vLLM on a Barkla GPU
        # node, LM Studio, llama-server) over LLM_BASE_URL.
        # "huggingface": load the model in-process with transformers, no server
        # at all — see llm.py for the tradeoffs and the settings that stop
        # applying under it.
        llm_backend=_env_backend(),
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
        # huggingface backend only. "auto" lets accelerate shard the model
        # across whatever GPUs SLURM allocated, and lets transformers take the
        # dtype from the checkpoint (BF16 for the models here) — override the
        # dtype to "float16" on pre-Ampere cards, which have no bfloat16.
        llm_hf_device_map=os.environ.get("LLM_HF_DEVICE_MAP", "auto"),
        llm_hf_dtype=os.environ.get("LLM_HF_DTYPE", "auto"),
        semantic_scholar_api_key=semantic_scholar_api_key,
        default_max_results_per_query=int(os.environ.get("MAX_RESULTS_PER_QUERY", "5")),
        hypothesis_output_dir=os.environ.get("HYPOTHESIS_OUTPUT_DIR", "outputs"),
        hypothesis_batch_max_chars=int(os.environ.get("HYPOTHESIS_BATCH_MAX_CHARS", "12000")),
        experiment_planner_output_dir=os.environ.get("EXPERIMENT_PLANNER_OUTPUT_DIR", "outputs"),
        coder_experiments_dir=os.environ.get("CODER_EXPERIMENTS_DIR", "experiments"),
        coder_output_dir=os.environ.get("CODER_OUTPUT_DIR", "outputs"),
        coder_max_fix_attempts=int(os.environ.get("CODER_MAX_FIX_ATTEMPTS", "3")),
        # Off by default: run.sbatch is generated from code nothing has ever
        # executed, and submitting it spends GPU allocation on a cluster other
        # people are queueing for. Turning this on is a deliberate choice for
        # unattended batch runs, and is still gated by the two caps below plus
        # a clean static safety check.
        coder_auto_submit_slurm=_env_bool("CODER_AUTO_SUBMIT_SLURM", False),
        # Checked against squeue, so it holds across every process in a batch.
        coder_max_concurrent_slurm_jobs=int(os.environ.get("CODER_MAX_CONCURRENT_SLURM_JOBS", "4")),
        # Per-question ceiling, so one runaway plan set can't flood the queue.
        coder_max_slurm_jobs_per_run=int(os.environ.get("CODER_MAX_SLURM_JOBS_PER_RUN", "10")),
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
        batch_output_root=os.environ.get("BATCH_OUTPUT_ROOT", "outputs/batch"),
        # Stops a long batch early when something systemic is wrong (the model
        # server is down, the API key expired) instead of burning the rest of
        # the question list against the same failure.
        batch_max_consecutive_failures=int(os.environ.get("BATCH_MAX_CONSECUTIVE_FAILURES", "5")),
        # One directory per run, holding that run's events, logs and every
        # artifact it produced — so a run is self-contained and rsync-able off a
        # compute node, and no two runs share an output directory.
        webapp_runs_dir=os.environ.get("WEBAPP_RUNS_DIR", "runs"),
        # Loopback by default: the web app has no authentication and can start
        # jobs and read files, so binding it to a routable address on a shared
        # cluster hands those abilities to everyone else on the node. Reach it
        # from elsewhere with an SSH tunnel instead (see README).
        webapp_host=os.environ.get("WEBAPP_HOST", "127.0.0.1"),
        webapp_port=int(os.environ.get("WEBAPP_PORT", "8000")),
        # The pipeline points at a single LLM endpoint, so a second concurrent
        # run mostly just contends with the first for it.
        webapp_max_concurrent_runs=int(os.environ.get("WEBAPP_MAX_CONCURRENT_RUNS", "1")),
    )


settings = load_settings()
