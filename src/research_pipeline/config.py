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
    llm_context_window: int
    llm_enable_thinking: bool
    llm_reasoning_budget: int | None
    semantic_scholar_api_key: str
    core_api_key: str
    huggingface_api_token: str
    default_max_results_per_query: int
    checkpointer_backend: str
    checkpointer_sqlite_path: str
    checkpointer_postgres_uri: str
    enable_paper_search_cache: bool
    paper_search_cache_ttl_seconds: int
    interdisciplinary_output_dir: str
    interdisciplinary_max_fields: int
    hypothesis_output_dir: str
    hypothesis_batch_max_chars: int
    experiment_planner_output_dir: str
    coder_experiments_dir: str
    coder_output_dir: str
    coder_max_fix_attempts: int
    coder_max_env_repairs: int
    coder_max_structural_retries: int
    coder_data_dir: str
    coder_venv_root: str
    coder_share_venvs: bool
    coder_dataset_cache_dir: str
    coder_dataset_max_rows: int
    coder_enable_hf_dataset_search: bool
    coder_require_real_data: bool
    coder_enable_fix_pattern_store: bool
    coder_fix_store_backend: str
    coder_fix_store_sqlite_path: str
    coder_fix_store_postgres_uri: str
    coder_run_high_complexity_when_gpu_available: bool
    coder_high_complexity_timeout_seconds: int
    coder_low_complexity_timeout_seconds: int
    coder_medium_complexity_timeout_seconds: int
    coder_enable_smoke_run: bool
    coder_smoke_timeout_seconds: int
    coder_auto_submit_slurm: bool
    coder_interactive_slurm_review: bool
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


def load_settings() -> Settings:
    semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if not semantic_scholar_api_key:
        logger.warning(
            "SEMANTIC_SCHOLAR_API_KEY is not set — Semantic Scholar search will be skipped "
            "(unauthenticated requests are aggressively rate-limited / rejected)."
        )
    core_api_key = os.environ.get("CORE_API_KEY", "")
    if not core_api_key:
        logger.warning(
            "CORE_API_KEY is not set — CORE search will be skipped "
            "(sign up for a free key at https://core.ac.uk/services/api)."
        )
    return Settings(
        # Any OpenAI-compatible server (vLLM on a Barkla GPU node, LM Studio,
        # llama-server) reached over LLM_BASE_URL.
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY", "not-needed"),
        llm_model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        llm_top_p=float(os.environ.get("LLM_TOP_P", "0.95")),
        # Every agent parses structured JSON or full paper sections out of the
        # response, so a truncated completion is a hard failure — leave enough
        # room for the longest Writer section (and for a reasoning trace, if
        # LLM_ENABLE_THINKING is turned back on). 16384 wasn't enough: on
        # Barkla job 10271093 (2026-08-19) the Reviewer's hallucination check
        # on a long Discussion section produced a ~75K-char hallucinations
        # list that got cut off mid-string at exactly that cap, and the
        # repair retry — bound by the same max_tokens — hit the identical
        # truncation point again and crashed the whole pipeline run.
        llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "32768")),
        # The model's *total* prompt+completion budget, read client-side only to
        # bound how many completion tokens a request may ask for — distinct from
        # llm_max_tokens above, which caps the completion length alone. A long
        # prompt (the Coder Agent's fix prompts carry previous code sections,
        # errors, plan JSON and shared-infra context; the Reviewer's grounding
        # blocks carry the whole merged paper pool) plus a fixed max_tokens can
        # exceed this and get a 400 from the server instead of a completion — see
        # llm.bounded_max_tokens, which every agent goes through.
        #
        # This MUST match whatever --max-model-len the vLLM server was actually
        # started with (see scripts/slurm/_vllm_serve.sh). 131072 is measured,
        # not assumed: on Barkla job 10274103, Qwen3-Coder-30B-A3B served at
        # TP=1 on one 80GB card with --gpu-memory-utilization 0.90 reported
        # "GPU KV cache size: 150,272 tokens" — above the 131072 requested, so
        # the window is real rather than silently clipped.
        #
        # TP=1 (rather than sharding across both GPUs) leaves the job's second
        # GPU free for generated experiments to use. Serving at TP=2 would allow
        # this model's full 262144, at the cost of that card; if you change one,
        # change both here and in _vllm_serve.sh.
        llm_context_window=int(os.environ.get("LLM_CONTEXT_WINDOW", "131072")),
        # Qwen3-Coder-30B-A3B-Instruct is NOT a reasoning model — it emits no
        # <think> trace and its chat template has no enable_thinking kwarg, so
        # this stays off and llm._extra_body sends nothing when it is. The
        # setting is kept rather than deleted because the pipeline should still
        # be able to serve a reasoning model (Nemotron 3 Nano, which this
        # replaced) by changing LLM_MODEL alone. Traces are stripped defensively
        # regardless (see llm_json.strip_reasoning).
        llm_enable_thinking=_env_bool("LLM_ENABLE_THINKING", False),
        llm_reasoning_budget=_env_optional_int("LLM_REASONING_BUDGET"),
        semantic_scholar_api_key=semantic_scholar_api_key,
        core_api_key=core_api_key,
        # Optional, and deliberately not warned about when unset (unlike the two
        # keys above): the Coder Agent's Hugging Face dataset lookup works fine
        # unauthenticated against public datasets — a token only buys higher rate
        # limits, which matters for a long batch sweep from one IP.
        huggingface_api_token=os.environ.get("HUGGINGFACE_API_TOKEN", ""),
        default_max_results_per_query=int(os.environ.get("MAX_RESULTS_PER_QUERY", "5")),
        # Where every graph's LangGraph checkpoints go — one of "memory",
        # "sqlite" or "postgres" (see checkpointer.get_checkpointer). "memory"
        # is the default and reproduces the behaviour every graph.py hardcoded
        # before this setting existed: checkpoints live in the process and die
        # with it. The other two make a crashed run's progress survive the
        # process, which is the whole point on a pre-empted SLURM job or a
        # restarted Kaggle kernel; both need an optional dependency
        # (uv sync --extra checkpoint-sqlite / --extra checkpoint-postgres),
        # which is why neither is the default.
        checkpointer_backend=os.environ.get("CHECKPOINTER_BACKEND", "memory"),
        # One file for every graph in the process. Safe to share: checkpoints
        # are keyed by (thread_id, checkpoint_ns) and every call site already
        # mints its own thread_id (a fresh uuid4 per agent call, the run_id for
        # the orchestrator), so there is nothing to collide.
        checkpointer_sqlite_path=os.environ.get(
            "CHECKPOINTER_SQLITE_PATH", "checkpoints/pipeline.db"
        ),
        # Blank = unset, matching SEMANTIC_SCHOLAR_API_KEY's style. Only read
        # when CHECKPOINTER_BACKEND=postgres.
        checkpointer_postgres_uri=os.environ.get("CHECKPOINTER_POSTGRES_URI", ""),
        # Caches the paper-search nodes (arXiv / Semantic Scholar / CORE, and
        # the interdisciplinary per-field search) on their inputs, so two runs
        # that generate the same queries hit the APIs once. LangGraph's only
        # cache backend is in-memory, so this helps a *long-lived* process —
        # the web app across several runs, an orchestrate-batch sweep across
        # questions — and does nothing at all for a one-shot CLI invocation,
        # which starts with an empty cache every time. On by default because
        # re-querying the same public APIs is pure waste; turn it off to force
        # a genuinely fresh search on every node execution.
        enable_paper_search_cache=_env_bool("ENABLE_PAPER_SEARCH_CACHE", True),
        # How long a cached search stays usable. An hour is short enough that a
        # long sweep still picks up newly published work, long enough to cover
        # a batch of related questions.
        paper_search_cache_ttl_seconds=int(
            os.environ.get("PAPER_SEARCH_CACHE_TTL_SECONDS", "3600")
        ),
        interdisciplinary_output_dir=os.environ.get("INTERDISCIPLINARY_OUTPUT_DIR", "outputs"),
        # How many adjacent fields the agent is allowed to explore. Each field
        # costs one arXiv + one Semantic Scholar + one CORE search per generated
        # query, so this is the knob that bounds the cross-field search fan-out;
        # the per-query result count reuses MAX_RESULTS_PER_QUERY rather than
        # adding a second, near-identical knob.
        interdisciplinary_max_fields=int(os.environ.get("INTERDISCIPLINARY_MAX_FIELDS", "3")),
        hypothesis_output_dir=os.environ.get("HYPOTHESIS_OUTPUT_DIR", "outputs"),
        hypothesis_batch_max_chars=int(os.environ.get("HYPOTHESIS_BATCH_MAX_CHARS", "12000")),
        experiment_planner_output_dir=os.environ.get("EXPERIMENT_PLANNER_OUTPUT_DIR", "outputs"),
        coder_experiments_dir=os.environ.get("CODER_EXPERIMENTS_DIR", "experiments"),
        coder_output_dir=os.environ.get("CODER_OUTPUT_DIR", "outputs"),
        coder_max_fix_attempts=int(os.environ.get("CODER_MAX_FIX_ATTEMPTS", "3")),
        # Bounded separately from the fix attempts above, because the two fail
        # for unrelated reasons and one should not drain the other. A fix
        # attempt is spent asking the model for different code; an env repair
        # installs a package and re-runs the code unchanged, which is not the
        # model's fault and not the model's problem. Sharing one budget is how a
        # 2026-08-19 run spent all three fix attempts on a missing pandas.
        coder_max_env_repairs=int(os.environ.get("CODER_MAX_ENV_REPAIRS", "6")),
        # Regenerations spent on a response that was malformed or hollow rather
        # than wrong — see coder_agent._STRUCTURAL_ERROR_SOURCES. Separate from
        # the fix budget for the same reason installs are: nothing was executed,
        # so nothing was learned about the experiment, and the budget for
        # debugging code should not be spent on the model failing to answer in
        # the requested shape. Small on purpose — a model that cannot produce
        # the format twice running will not produce it on the fifth try, and
        # the identical-failure stop still applies on top of this.
        coder_max_structural_retries=int(os.environ.get("CODER_MAX_STRUCTURAL_RETRIES", "2")),
        # A directory of data files staged by hand — CMS extracts obtained under
        # a DUA, a licensed cohort, anything the agent cannot fetch for itself.
        # Inputs matched here resolve as real, which is what lets a run report a
        # hypothesis verdict at all; see agents/coder/provenance.py.
        coder_data_dir=os.environ.get("CODER_DATA_DIR", ""),
        # Where each experiment's throwaway venv is created. Empty means "beside
        # the results", which is right on a laptop. On Barkla it should point at
        # localscratch (/tmp/users/$USER): a venv is thousands of small files,
        # scratch and fastscratch have inode quotas that a run per experiment
        # eats into, and localscratch has none, is node-local NVMe, and is
        # cleared automatically. The venv is rebuilt per job either way, so
        # nothing is lost by keeping it off the shared filesystem.
        coder_venv_root=os.environ.get("CODER_VENV_ROOT", ""),
        # On by default. Experiment venvs are keyed by the requirements they
        # hold rather than by which experiment asked for them, so a sweep of
        # twenty plans that all want numpy/pandas/scikit-learn provisions one
        # venv instead of twenty. That is a large amount of wall clock and, on
        # a quota'd cluster home, an enormous number of inodes — a single torch
        # install is thousands of files, and re-doing it per experiment is what
        # makes a torch-dependent sweep impossible rather than merely slow.
        # The switch exists for a run that wants a guaranteed-pristine
        # environment per experiment, not because sharing is risky: the key is
        # the resolved requirement set, so a shared venv is always a superset of
        # what the experiment asked for.
        coder_share_venvs=_env_bool("CODER_SHARE_VENVS", True),
        # Empty (the default) keeps the long-standing behaviour: a matched Hub
        # dataset is handed to the model as a Dataset Viewer REST URL and read
        # ~100 rows at a time. Set it to a directory and the agent instead
        # downloads the dataset once, to one JSONL file there, and hands the
        # generated code that local path — which is the difference between an
        # experiment that samples a dataset and one that can train on it.
        # One file per dataset, deliberately: a Hugging Face cache directory is
        # thousands of small files, and Barkla's scratch inode quotas are the
        # binding constraint long before disk space is. Shared across every
        # experiment and every run pointed at the same directory.
        coder_dataset_cache_dir=os.environ.get("CODER_DATASET_CACHE_DIR", ""),
        # The cap on that download, in rows. Only read when the cache directory
        # above is set. Bounded rather than unbounded because the rows endpoint
        # pages 100 at a time, so this is also a request count: the default is
        # 500 requests, a few minutes once, then free for every later run.
        coder_dataset_max_rows=int(os.environ.get("CODER_DATASET_MAX_ROWS", "50000")),
        # On by default: looking up a real Hugging Face dataset for a plan's data
        # requirements is what stops a generated experiment inventing numbers or
        # assuming some CSV is already on disk. It's still only ever attempted
        # when the runtime network probe succeeds, and any failure degrades to
        # generating exactly as before — so the switch is here for reproducible
        # offline runs (and to opt a whole batch out of the extra HTTP calls),
        # not because the lookup is risky.
        coder_enable_hf_dataset_search=_env_bool("CODER_ENABLE_HF_DATASET_SEARCH", True),
        # Off by default, and a policy choice rather than a repair: when set, a
        # plan whose every data input resolves to a surrogate is skipped before
        # a single codegen call, instead of being generated, run, and reported
        # "inconclusive" by the provenance gate. Right for a sweep collecting
        # only interpretable results; wrong whenever the generated code itself
        # is the artefact you want, which is why it is not the default.
        coder_require_real_data=_env_bool("CODER_REQUIRE_REAL_DATA", False),
        # On by default, and — unlike CHECKPOINTER_BACKEND — defaulting to
        # "sqlite" rather than "memory": a checkpointer's in-memory default is
        # fine because most runs are one-shot processes anyway, but this
        # store's entire value is accumulating across many separate process
        # invocations, so an in-memory default would silently do nothing for
        # the pipeline's most common usage pattern. See
        # agents/coder/fix_pattern_store.py's module docstring.
        coder_enable_fix_pattern_store=_env_bool("CODER_ENABLE_FIX_PATTERN_STORE", True),
        coder_fix_store_backend=os.environ.get("CODER_FIX_STORE_BACKEND", "sqlite"),
        coder_fix_store_sqlite_path=os.environ.get(
            "CODER_FIX_STORE_SQLITE_PATH", "coder_fix_patterns.db"
        ),
        # Only read when CODER_FIX_STORE_BACKEND=postgres.
        coder_fix_store_postgres_uri=os.environ.get("CODER_FIX_STORE_POSTGRES_URI", ""),
        # Off by default: "high" complexity always defers to run.sbatch,
        # regardless of GPU availability, because the SLURM path is written
        # for a *shared* cluster where nothing should run unreviewed. On a
        # single-tenant GPU already attached to this process (a Kaggle
        # notebook, a Barkla node reached via run_pipeline.sbatch), that
        # concern doesn't apply — there's no queue to jump and no one else's
        # allocation to spend. Turning this on lets `high` complexity plans
        # run synchronously exactly like low/medium, but only when gpu_check()
        # confirms a GPU is actually present; needs_gpu-without-a-GPU still
        # always defers, since there's nothing to run it on either way.
        coder_run_high_complexity_when_gpu_available=_env_bool(
            "CODER_RUN_HIGH_COMPLEXITY_WHEN_GPU_AVAILABLE", False
        ),
        # High-complexity work (e.g. fine-tuning) legitimately runs longer
        # than low/medium's 120s/300s; only consulted when the flag above is on.
        coder_high_complexity_timeout_seconds=int(
            os.environ.get("CODER_HIGH_COMPLEXITY_TIMEOUT_SECONDS", "1800")
        ),
        # The low/medium execution timeouts, previously a hardcoded dict in
        # sandbox.py. Same pattern (and same read site in coder_agent.py) as the
        # high-complexity timeout above — sandbox.py deliberately reads no
        # settings at all, so all three are resolved by coder_agent.py. Defaults
        # are exactly the values that dict held; "high" is not one of these
        # because it only ever runs synchronously under the opt-in flag above.
        coder_low_complexity_timeout_seconds=int(
            os.environ.get("CODER_LOW_COMPLEXITY_TIMEOUT_SECONDS", "120")
        ),
        coder_medium_complexity_timeout_seconds=int(
            os.environ.get("CODER_MEDIUM_COMPLEXITY_TIMEOUT_SECONDS", "300")
        ),
        # A deliberately shrunken first execution (repair.smoke_variant pins
        # every cost knob to its floor), so a defect that would surface anywhere
        # in the program is found in seconds instead of after the full timeout
        # above — and each round of the fix loop costs seconds too. It can only
        # ever fail an experiment early, never pass one: a smoke run that
        # succeeds, times out, or fails for a reason the shrinking could have
        # caused is followed by the real run regardless. See
        # CoderAgent._smoke_failure.
        coder_enable_smoke_run=_env_bool("CODER_ENABLE_SMOKE_RUN", True),
        # Capped against the real timeout at the call site: a smoke run must
        # never be given longer than the run it is meant to be cheaper than.
        coder_smoke_timeout_seconds=int(os.environ.get("CODER_SMOKE_TIMEOUT_SECONDS", "60")),
        # Off by default: run.sbatch is generated from code nothing has ever
        # executed, and submitting it spends GPU allocation on a cluster other
        # people are queueing for. Turning this on is a deliberate choice for
        # unattended batch runs, and is still gated by the two caps below plus
        # a clean static safety check.
        coder_auto_submit_slurm=_env_bool("CODER_AUTO_SUBMIT_SLURM", False),
        # Off by default, and meant only for a direct, attended `research-pipeline
        # coder ...` CLI call — never set this for orchestrate/orchestrate-batch/the
        # webapp, which run unattended and would hang waiting on a prompt no one is
        # there to answer. When on, a plan that can't run here (no GPU, or high
        # complexity without CODER_RUN_HIGH_COMPLEXITY_WHEN_GPU_AVAILABLE) pauses the
        # graph (LangGraph interrupt()) instead of unconditionally handing the
        # decision to a human outside the pipeline — see
        # CoderAgent._handle_unrunnable_locally. The self-review/job-cap gates below
        # still apply even when a human approves; this only adds a gate, it never
        # removes one.
        coder_interactive_slurm_review=_env_bool("CODER_INTERACTIVE_SLURM_REVIEW", False),
        # Checked against squeue, so it holds across every process in a batch.
        coder_max_concurrent_slurm_jobs=int(os.environ.get("CODER_MAX_CONCURRENT_SLURM_JOBS", "4")),
        # Per-question ceiling, so one runaway plan set can't flood the queue.
        coder_max_slurm_jobs_per_run=int(os.environ.get("CODER_MAX_SLURM_JOBS_PER_RUN", "10")),
        writer_output_dir=os.environ.get("WRITER_OUTPUT_DIR", "outputs"),
        writer_related_work_batch_max_chars=int(
            os.environ.get("WRITER_RELATED_WORK_BATCH_MAX_CHARS", "12000")
        ),
        # No real author identity flows through the pipeline, so this defaults to
        # NeurIPS's own anonymized-submission placeholder text rather than fabricating one.
        writer_paper_authors=os.environ.get("WRITER_PAPER_AUTHORS", "Anonymous Author(s)"),
        writer_paper_affiliation=os.environ.get(
            "WRITER_PAPER_AFFILIATION", "Anonymous Institution"
        ),
        reviewer_output_dir=os.environ.get("REVIEWER_OUTPUT_DIR", "outputs"),
        writer_reviewer_loop_output_dir=os.environ.get(
            "WRITER_REVIEWER_LOOP_OUTPUT_DIR", "outputs/paper"
        ),
        writer_reviewer_max_iterations=int(os.environ.get("WRITER_REVIEWER_MAX_ITERATIONS", "3")),
        writer_reviewer_quality_threshold=int(
            os.environ.get("WRITER_REVIEWER_QUALITY_THRESHOLD", "4")
        ),
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
