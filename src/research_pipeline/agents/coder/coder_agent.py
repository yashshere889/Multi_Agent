"""Coder Agent.

Sits between the Experiment Planner Agent and the execution/results-analysis
stage: takes the Experiment Planner's output, generates a real runnable
experiment per feasible plan, executes what it safely can in this
environment, and reports structured results — all as JSON a later stage can
consume without re-reading logs.

Each experiment is a single generated `run.py`, rendered from
`templates/run.py.template`: a fixed metadata block and orchestration footer
(timing, exception handling, results.json writing — see
`sandbox.render_experiment_template`) are NOT model-generated; only four
functions are — `load_data`, `build_model`, `run_experiment`, `evaluate` —
plus their imports/configuration/helper sections. This removes an entire
class of failure mode from the previous (4-separate-files) design: there's no
cross-file calling convention for the model to get slightly wrong per
experiment, because the wiring isn't model-generated at all.

Input contract
--------------
`planner_output: dict` — the exact output of
`research_pipeline.agents.experiment_planner.run_experiment_planner_agent`
(or `outputs/experiment_plan_<timestamp>.json` loaded from disk). Validated
against `research_pipeline.agents.experiment_planner.schema.ExperimentPlannerOutput`
on entry — if it doesn't validate, CoderAgentError is raised immediately.

Execution model (confirmed with the pipeline owner, not assumed)
------------------------------------------------------------------
- Plans with `feasible: false` are never given code — logged as "skipped"
  with `feasibility_notes` as the reason.
- An experiment that self-reports `needs_gpu: true` when no GPU is detected
  in this environment: there is nothing to run it on, so code is still
  generated and syntax-checked, but never run synchronously. A SLURM
  `run.sbatch` template is generated instead (matching
  scripts/slurm/run_llm_server.sbatch's style) — status becomes
  "code_generated_not_run". The agent NEVER submits this job itself; it's
  left for a human to review and `sbatch run.sbatch` manually.
- `estimated_complexity == "high"`: same sbatch-only treatment as above by
  default, since run.sbatch targets a *shared* cluster where nothing should
  run unreviewed. If `CODER_RUN_HIGH_COMPLEXITY_WHEN_GPU_AVAILABLE` is set and
  a GPU is actually detected in this process (a Kaggle notebook, a Barkla node
  reached via run_pipeline.sbatch — single-tenant compute already attached to
  this pipeline, not a shared queue), it runs synchronously instead, bounded
  by `settings.coder_high_complexity_timeout_seconds` rather than the
  low/medium timeouts.
- `estimated_complexity in {"low", "medium"}` (and no GPU requirement, or a
  GPU is actually available), or `"high"` under the opt-in above: run
  synchronously in-process with a bounded timeout
  (settings.coder_low_complexity_timeout_seconds /
  settings.coder_medium_complexity_timeout_seconds /
  settings.coder_high_complexity_timeout_seconds — all resolved here, since
  sandbox.py reads no settings), in an isolated `uv venv` if
  the generated requirements.txt needs packages not already importable — the
  shared pipeline environment is never touched. Network access and GPU
  presence are probed at runtime (not hardcoded), so the same code adapts
  whether this runs on a laptop, a Kaggle notebook, or a Barkla compute node.

Output contract
----------------
A dict matching `agents.coder.schema.CoderAgentOutput`:
    {
      "experiments": [ one entry per input plan, always — including skipped
          ones; see schema.py:ExperimentResult for the per-entry shape ],
      "shared_infrastructure_path": str,   # experiments/_shared/, always created
      "source_hypothesis_ids": [str, ...],
      "generated_at": "<UTC ISO 8601>",
      "model": str,
    }
Validated (agents.coder.schema.validate_output) before being returned —
including that every input hypothesis id has a corresponding entry. Also
written to `<output_dir>/coder_agent_summary_<UTC timestamp>.json`. Generated
project code is written to `<experiments_dir>/<hypothesis_id>/` regardless of
validation outcome (code on disk is useful even if the summary JSON needs
fixing); on a summary schema validation failure, the raw (invalid) summary is
still written — suffixed `_invalid` — and CoderAgentError is raised.

Entry points
------------
    from research_pipeline.agents.coder import run_coder_agent
    result = run_coder_agent(planner_output)  # dict from the Experiment Planner Agent

Or, to reuse one configured model/output dir across multiple calls:
    agent = CoderAgent()
    result = agent.run(planner_output)

Internals
---------
`run()` executes a LangGraph StateGraph (see graph.py / state.py) rather than a
pair of nested Python loops, so every step — each generation, each check, each
regeneration, each submission decision — is its own traceable node. Both loops
are real graph cycles: one over the plans, one over the fix attempts within a
plan. Unlike the Hypothesis/Experiment Planner graphs there is no `Send`
fan-out, and that is deliberate rather than incidental: plans must be processed
strictly one at a time so each one sees the true, up-to-date count of jobs this
run has already submitted before deciding whether it may auto-submit
(`CODER_MAX_SLURM_JOBS_PER_RUN`). See graph.py's module docstring. All of this
is purely internal — the entry points above, their signatures, the returned
dict, and the raised CoderAgentError are unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt

from research_pipeline.agents.coder import (
    dataset_inspect,
    dataset_scoring,
    dataset_spec,
    diagnose,
    fix_pattern_store,
    huggingface_client,
    prompts,
    provenance,
    repair,
    sandbox,
    slurm_submit,
    starters,
)
from research_pipeline.agents.coder.schema import (
    ERROR_SUMMARY_MAX_CHARS,
    SchemaValidationError,
    validate_output,
)
from research_pipeline.agents.coder.state import CoderState
from research_pipeline.agents.experiment_planner.schema import (
    SchemaValidationError as PlannerSchemaValidationError,
)
from research_pipeline.agents.experiment_planner.schema import (
    validate_output as validate_planner_output,
)
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import LLMJSONError, invoke_json
from research_pipeline.llm_sections import LLMSectionsError, invoke_sections, render_sections

logger = logging.getLogger(__name__)

# Order the checks run in, used to decide whether a regeneration actually got
# past the failure it was asked to fix. Must stay in sync with schema.py's
# VALID_ERROR_SOURCES (same members; this list additionally encodes order).
_ERROR_STAGE_ORDER = [
    "invalid_format",
    "missing_sections",
    "missing_required_function",
    "empty_body",
    "compile_check",
    "static_lint",
    "missing_data_fallback",
    "ignored_available_dataset",
    "self_review",
    # The execution-failure kinds sit where run_experiment always did: they are
    # all detected at the same point (a non-zero exit), just told apart by what
    # the traceback says. Order within the group is arbitrary but must match
    # schema.VALID_ERROR_SOURCES — _cleared_previous_error reads this list as
    # the definition of "later stage", so a member missing here silently scores
    # every regeneration as having made no progress.
    "missing_dependency",
    "obsolete_dependency",
    "missing_system_library",
    "resource_limit",
    "run_experiment",
    "results_json",
    "implausible_results",
]


# LangGraph's default recursion_limit (25 super-steps) is a limit on the graph's
# *shape*, which is the wrong unit for this graph: both of its loops are real
# cycles, so the step count scales with how many plans there are and how large
# the fix budget is. Derived per run from those two numbers rather than guessed —
# the counts below match the loops wired up in graph.py.
_FIXED_STEPS = 5  # validate_input, probe_environment, setup_shared_infrastructure, start_plan_loop, assemble_and_validate
_STEPS_PER_PLAN = 9  # process_current_plan, the five dataset-selection nodes, generate_experiment_code, the first attempt, finalize/give_up
_STEPS_PER_FIX_ATTEMPT = 2  # snapshot_and_regenerate, then the attempt it feeds


# Inputs to _bounded_max_tokens, which stops a long prompt plus a fixed
# max_tokens from overrunning the model's context window.
#
# 4 chars/token (the standard rough heuristic for English prose) undercounted
# this agent's actual code/JSON-heavy prompts by ~15-20% against vLLM's real
# tokenizer — code and JSON are denser than prose (short identifiers, lots of
# punctuation/whitespace runs), which is exactly the kind of text every prompt
# here carries. That gap crashed 3/3 single-question Barkla runs on
# 2026-08-14 with a 400 from the server (request tokens > context window)
# despite this estimate supposedly bounding for it. 3 chars/token
# deliberately overestimates instead — costing a shorter completion in the
# worst case, which the fix loop already tolerates, rather than a crash. It's
# still an estimate on purpose: the backend is an arbitrary OpenAI-compatible
# HTTP endpoint, so there is no tokenizer on this side to ask for the real
# count, and pulling one in would tie this agent to one specific model.
_CHARS_PER_TOKEN_ESTIMATE = 3
# Below this many tokens a completion would be too truncated to be usable code
# — mid-function at best. Hitting this floor means the *prompt* is the problem,
# so it's raised rather than attempted and silently wasted.
_MIN_GENERATION_TOKENS = 2048
# Headroom for the estimate being wrong and for provider-side rounding (chat
# templates, role tokens, tool preambles) that never appears in the prompt text
# we can measure here. Doubled from 512 alongside the chars/token fix above —
# both were tightened together after the same 2026-08-14 crashes.
_CONTEXT_SAFETY_MARGIN = 1024

# Fix-attempt regeneration runs at temperature 0 — the model's most confident
# completion rather than a fresh sample. The fix prompt asks for every section
# back, "keeping whatever already worked", and at a nonzero temperature that
# full-section rewrite has been observed reintroducing *different* bugs each
# round: in one production trace attempt 1 correctly fixed a ModuleNotFoundError
# while attempts 2 and 3 each introduced a new backslash syntax error at a
# different line. Initial generation keeps the constructor's temperature; only
# the regeneration paths use this.
_FIX_TEMPERATURE = 0.0

# How many times one execution may be shrunk before the model is asked to
# rethink the approach instead. Halving twice is a 4x reduction, which is enough
# to tell "slightly over budget" from "the wrong shape of computation" — past
# that, shrinking further degrades the experiment rather than rescuing it.
_MAX_DOWNSCALES = 2

# Rows shown to the evidence/critic prompts. The full inspection sample is 200+
# rows, which no prompt can carry alongside a plan and a card — the measured
# statistics over all of them go in instead, and these are the concrete examples
# the model reasons about.
_DATASET_PROMPT_SAMPLE_ROWS = 8

_BYTES_PER_GB = 1024**3

# Hits requested per Hub search query. The Hub's own page size for a relevance
# search; asking for fewer is what starved the pool (see _shortlist_datasets).
_DATASET_HITS_PER_QUERY = 10

# How far past the inspection budget the prefilter ranks, so viewer failures can
# be walked past rather than silently costing an inspection slot. Four times is
# enough to absorb the 3-in-5 failure rate a real run hit while still bounding
# the work: describing a candidate is a handful of HTTP calls, and the loop
# stops as soon as the budget is filled.
_DATASET_PREFILTER_WINDOW = 4


def _dataset_bands(hf_dataset: dict) -> dict[str, str]:
    """The per-dimension bands from a scored dataset's state dict, for the
    prompt's one-line score summary. Empty when the dict predates scoring."""
    components = hf_dataset.get("components")
    if not isinstance(components, dict):
        return {}
    return {
        name: str(components.get(name))
        for name in ("task_relevance", "content_relevance", "schema_fit", "license_fit")
        if components.get(name)
    }


def _directory_bytes(path: Path) -> int:
    """Total size of everything under `path`, or 0 if it can't be read. Used to
    charge the run's download budget from what actually landed on disk."""
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE


def _truncate(text: str) -> str:
    return text[:ERROR_SUMMARY_MAX_CHARS]


def _consecutive_error_streak(fix_history: list[dict]) -> int:
    """How many trailing entries in fix_history share the most recent one's
    error_source — 1 if this failure is a new kind, 2+ if the model's last
    fix attempt(s) landed back on the exact same failure category.

    Narrower than _cleared_previous_error's "resolved" flag on purpose:
    "resolved" is False both for a regression to an earlier check *and* for a
    repeat of the same one, which is the right signal for fix_history's own
    bookkeeping. This is specifically "did the model just fail on the same
    thing again" — the case where quoting the model its own most recent
    (unsuccessful) fix is useful context rather than noise."""
    if not fix_history:
        return 0
    current_source = fix_history[-1]["error_source"]
    streak = 0
    for entry in reversed(fix_history):
        if entry["error_source"] != current_source:
            break
        streak += 1
    return streak


_DIGITS_RE = re.compile(r"\b\d+\b")
_PATH_RE = re.compile(r"(/[\w.\-/]+)+")
_HEXADDR_RE = re.compile(r"0x[0-9a-fA-F]+")


def _failure_signature(entry: dict) -> str:
    """A normalized identity for "this same failure again".

    Line numbers, absolute paths and object addresses all move between two runs
    of the same bug, so they are flattened before comparing. What survives is
    the error source plus the shape of the message — the exception type, the
    missing symbol, the assertion text.
    """
    summary = entry.get("error_summary") or ""
    summary = _HEXADDR_RE.sub("0xADDR", summary)
    summary = _PATH_RE.sub(lambda m: m.group(0).rsplit("/", 1)[-1], summary)
    summary = _DIGITS_RE.sub("N", summary)
    return f"{entry.get('error_source')}|{' '.join(summary.split())}"


def _identical_failure_streak(fix_history: list[dict]) -> int:
    """How many trailing attempts failed in the *same* way, not merely at the same stage.

    Deliberately stricter than _consecutive_error_streak, which compares
    error_source alone: three different bugs all surface as `run_experiment`,
    and a model fixing one bug into the next is making progress even though the
    source never changes. What this counts is the model landing on the byte-for-
    byte same failure — which is what a run that cannot recover looks like, and
    what the fix budget should not be spent finishing.
    """
    if not fix_history:
        return 0
    current = _failure_signature(fix_history[-1])
    streak = 0
    for entry in reversed(fix_history):
        if _failure_signature(entry) != current:
            break
        streak += 1
    return streak


# Stop after this many identical failures in a row. Two is a repeat, which the
# prompt already escalates on (see _stuck_block); three is a loop. A production
# summary shows the cost of having no such stop: three attempts, three
# regenerations, and the same ModuleNotFoundError each time, with the budget
# spent and nothing learned after the first.
_NO_PROGRESS_STREAK = 3


# How many lines of run.py to show inline before pointing at the file instead
# — a full generated program can run to hundreds of lines, which would bury
# the actual question ("submit, or leave it?") under a wall of code a
# terminal has to scroll past.
_SLURM_REVIEW_CODE_PREVIEW_LINES = 40


def _default_slurm_review_prompt(payload: dict) -> str:
    """CoderAgent's default slurm_review_prompt — a plain terminal prompt, for
    a direct, attended `research-pipeline coder ...` CLI call (see
    CODER_INTERACTIVE_SLURM_REVIEW). Anything else that wants this decision
    made a different way (a test, a future webapp UI) injects its own
    callable instead of using this one; see the constructor.

    Returns "submit" or "skip" — CoderAgent.run()'s interrupt loop passes
    whatever this returns straight through as Command(resume=...), so any
    other string is treated as "skip" the same as an explicit one (a human's
    stray keystroke shouldn't submit a job to a shared cluster)."""
    lines = payload["run_py"].splitlines()
    preview = "\n".join(lines[:_SLURM_REVIEW_CODE_PREVIEW_LINES])
    if len(lines) > _SLURM_REVIEW_CODE_PREVIEW_LINES:
        preview += f"\n... ({len(lines) - _SLURM_REVIEW_CODE_PREVIEW_LINES} more lines — see {payload['code_path']}/run.py)"
    print(
        f"\n{'=' * 70}\n"
        f"Hypothesis {payload['hypothesis_id']} can't run here: {payload['why_unrunnable']}.\n"
        f"Generated {payload['sbatch_path']} — first "
        f"{min(len(lines), _SLURM_REVIEW_CODE_PREVIEW_LINES)} line(s) of run.py:\n\n"
        f"{preview}\n\n"
        f"Nothing has executed this code. It still has to pass a static safety check and an "
        f"LLM pre-flight review before it can be submitted, and both SLURM job caps still apply.\n"
        f"{'=' * 70}"
    )
    answer = input("Submit to SLURM now? [y/N] ").strip().lower()
    return "submit" if answer in {"y", "yes"} else "skip"


def _compact_json(obj: object) -> str:
    """No pretty-printing whitespace — this is embedded in prompts, not read
    by a human, and indent=2 runs meaningfully more tokens for the same data
    (measured ~20-25% on a representative experiment plan). Every prompt in
    this module uses this instead of json.dumps(..., indent=2); the one
    exception is _write_summary's output file, which a human actually reads."""
    return json.dumps(obj, separators=(",", ":"))


_TRUTHY_TEXT = {"true", "yes", "y", "1"}

# Lines the model writes when it means "nothing to report" — dropped rather than
# recorded as an assumption, since an assumptions_made list containing the word
# "none" reads as a real assumption everywhere downstream (the summary JSON, the
# Writer's framing of what was assumed).
_NO_ASSUMPTIONS_TEXT = {
    "none",
    "n/a",
    "na",
    "nil",
    "no assumptions",
    "no assumptions needed",
    "no assumptions were needed",
    "none needed",
    "none required",
}


def _parse_bool_text(text: str) -> bool:
    """Reads a `needs_gpu`/`needs_network` section, which now arrives as raw
    text ("true"/"false") rather than a JSON boolean.

    Matches on the first word only, so "true — this needs a GPU" is still True.
    Anything unrecognisable is False, which is exactly what
    `bool(generation.get("needs_gpu", False))` did before this transport
    changed: an unparseable needs_gpu means the experiment is attempted locally,
    and a real GPU requirement then surfaces as a normal execution failure the
    fix loop can see — whereas defaulting to True would silently defer a
    perfectly runnable experiment to a SLURM script nobody asked for."""
    stripped = text.strip().lower()
    if not stripped:
        return False
    first_word = re.split(r"[^a-z0-9]+", stripped, maxsplit=1)[0]
    return first_word in _TRUTHY_TEXT


def _parse_assumptions(text: str) -> list[str]:
    """Reads an `assumptions_made` section — one assumption per line, each
    expected to start with "- " — into the list of strings the output schema
    (and every consumer of it) expects. Bullet markers are stripped, blank
    lines and "none"-style placeholders dropped."""
    assumptions = []
    for line in text.splitlines():
        item = line.strip().lstrip("-*•").strip()
        if not item or item.rstrip(".").lower() in _NO_ASSUMPTIONS_TEXT:
            continue
        assumptions.append(item)
    return assumptions


def _plan_count(planner_output: object) -> int:
    """How many plans the graph will loop over, read defensively. The real
    contract check is validate_planner_output in the graph's first node, and a
    malformed input has to reach it (and raise CoderAgentError) rather than
    blowing up here while the recursion limit is being sized."""
    plans = planner_output.get("experiment_plans") if isinstance(planner_output, dict) else None
    return len(plans) if isinstance(plans, list) else 0


def _recursion_limit_for(plan_count: int, max_fix_attempts: int) -> int:
    """Worst case: every plan feasible, every plan exhausting its fix budget."""
    worst_case = _FIXED_STEPS + plan_count * (
        _STEPS_PER_PLAN + _STEPS_PER_FIX_ATTEMPT * max(max_fix_attempts, 0)
    )
    return max(worst_case + 10, 25)


class CoderAgentError(RuntimeError):
    """Raised when the agent can't produce schema-valid output, even after retries."""


class CoderAgent:
    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        experiments_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        network_check: Callable[[], bool] | None = None,
        gpu_check: Callable[[], bool] | None = None,
        max_fix_attempts: int | None = None,
        huggingface_lookup_fn: Callable[[str], dict | None] | None = None,
        dataset_search_fn: Callable[[str, int], list[dict]] | None = None,
        dataset_describe_fn: Callable[[str], dict] | None = None,
        dataset_rows_fn: Callable[[str, str, str, int], list[dict]] | None = None,
        dataset_card_fn: Callable[[str], str] | None = None,
        dataset_download_fn: Callable[[str, str, Path], Path | None] | None = None,
        dataset_normalize_fn: Callable[[Path, Path, int], dict] | None = None,
        fix_store: BaseStore | None = None,
        interactive_slurm_review: bool | None = None,
        slurm_review_prompt: Callable[[dict], str] | None = None,
    ) -> None:
        # Reuses the pipeline's existing LLM client/config, same as every
        # other agent, at a low temperature. streaming=True: this agent's
        # completions are full generated source files (1000s of tokens)
        # requested one experiment at a time, so a slow-but-live stream is
        # safe here in a way it isn't for agents that fan out concurrent
        # calls — see get_chat_model's docstring.
        self.chat_model = chat_model or get_chat_model(temperature=0.1, streaming=True)
        self.experiments_dir = Path(experiments_dir or settings.coder_experiments_dir)
        self.output_dir = Path(output_dir or settings.coder_output_dir)
        self.network_check = network_check or sandbox.has_network_access
        self.gpu_check = gpu_check or sandbox.has_gpu
        # Injectable for exactly the same reason network_check/gpu_check are (and
        # the same reason the Interdisciplinary Literature Agent injects its three
        # search functions): it is this agent's only outbound HTTP call, so a test
        # substitutes one function instead of faking four Hugging Face endpoints,
        # and no test ever reaches the real network.
        self.huggingface_lookup = (
            huggingface_lookup_fn or huggingface_client.find_dataset_for_experiment
        )
        # The dataset-appraisal path's network seams, injected individually for
        # the same reason. Each is one function so a test can fail exactly one
        # step (a search that returns nothing, a card that 404s, a download that
        # can't run because huggingface_hub isn't installed) and assert the
        # pipeline still generates an experiment — which is the property that
        # matters most about all of this, since every one of these is an
        # enhancement to a prompt and none of them is a precondition.
        self.dataset_search = dataset_search_fn or huggingface_client.search_datasets
        self.dataset_describe = dataset_describe_fn or huggingface_client.describe_candidate
        self.dataset_rows = dataset_rows_fn or huggingface_client.fetch_rows
        self.dataset_card = dataset_card_fn or huggingface_client.dataset_card
        self.dataset_download = dataset_download_fn or huggingface_client.download_dataset
        self.dataset_normalize = dataset_normalize_fn or huggingface_client.normalize_to_jsonl
        self.max_fix_attempts = (
            settings.coder_max_fix_attempts if max_fix_attempts is None else max_fix_attempts
        )
        # Injectable for the same reason huggingface_lookup_fn is: a test needs
        # an isolated store (its own patterns, not another test's or a real
        # sqlite file) rather than the process-wide singleton
        # fix_pattern_store.get_store() would otherwise hand back.
        self.fix_store = fix_store or fix_pattern_store.get_store()
        self.interactive_slurm_review = (
            settings.coder_interactive_slurm_review
            if interactive_slurm_review is None
            else interactive_slurm_review
        )
        # Only ever called from run()'s interrupt loop, never from inside a
        # graph node directly — see _handle_unrunnable_locally, which calls
        # interrupt() itself and leaves *asking* a human to whoever resumes
        # the graph. The default reads a real terminal answer; a test (or a
        # future webapp) injects its own callable instead.
        self.slurm_review_prompt = slurm_review_prompt or _default_slurm_review_prompt
        self._slurm_jobs_submitted = 0
        # Run-level dataset budgets. Instance attributes rather than state for
        # exactly the reason _slurm_jobs_submitted is: each plan must see the
        # true up-to-date total before deciding whether it may download, and
        # only the sequential plan loop makes that well-defined. Mirrored into
        # state after each acquisition so a trace shows the running total.
        self._datasets_accepted = 0
        self._dataset_bytes_downloaded = 0

    def run(self, planner_output: dict) -> dict:
        """Runs the agent's graph end to end. Same signature, same returned dict,
        and the same CoderAgentError on failure as the earlier sequential
        implementation — the graph is an internal detail, so callers (the CLI,
        the orchestrator, another agent) are unaffected.

        Node exceptions aren't swallowed by LangGraph, so a CoderAgentError
        raised inside a node propagates out of `.invoke` unchanged.
        """
        # Imported here rather than at module scope: graph.py needs this module's
        # CoderAgent for typing, so a top-level import would be circular.
        from research_pipeline.agents.coder.graph import build_coder_graph

        # The per-run SLURM budget is per run() call, not per agent lifetime —
        # reset before the graph starts, exactly as before. The counter stays a
        # plain instance attribute (and is mirrored into state for tracing, see
        # state.py) because the plan loop is strictly sequential: only one plan
        # is ever in flight, so there is no concurrent writer to reconcile.
        self._slurm_jobs_submitted = 0
        # Same lifetime, same reasoning — see the constructor.
        self._datasets_accepted = 0
        self._dataset_bytes_downloaded = 0

        graph = build_coder_graph(self)
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": _recursion_limit_for(
                _plan_count(planner_output), self.max_fix_attempts
            ),
        }
        final_state = graph.invoke(
            {"planner_output": planner_output, "experiments": [], "slurm_jobs_submitted": 0},
            config=config,
        )
        # A paused graph (self.interactive_slurm_review, see
        # _handle_unrunnable_locally's interrupt() call) reports itself via
        # "__interrupt__" in the returned state rather than raising — asking
        # is this instance's job (slurm_review_prompt), not the graph's; the
        # graph only knows it needs an answer, not how to get one. Resuming
        # replays the paused node from the top up to that same interrupt()
        # call, which then returns the decision instead of pausing again — see
        # https://docs.langchain.com/oss/python/langgraph/interrupts. A single
        # run can hit this more than once (one per plan deferred to sbatch),
        # hence the loop rather than a single check.
        while "__interrupt__" in final_state:
            decision = self.slurm_review_prompt(final_state["__interrupt__"][0].value)
            final_state = graph.invoke(Command(resume=decision), config=config)
        return final_state["result"]

    # -- Graph nodes -----------------------------------------------------------
    # Thin adapters: each takes the graph state, delegates to the private helper
    # below that does the actual work, and returns only the keys it produces.
    # The two loops these nodes form are wired up in graph.py.

    def _node_validate_input(self, state: CoderState) -> dict:
        planner_output = state["planner_output"]
        try:
            validate_planner_output(planner_output)
        except PlannerSchemaValidationError as exc:
            raise CoderAgentError(
                f"Input doesn't match the Experiment Planner's output schema: {exc}"
            ) from exc

        plans = planner_output["experiment_plans"]
        return {
            "expected_ids": [p["hypothesis_id"] for p in plans],
            "ordered_plans": self._order_by_priority(plans, planner_output["priority_order"]),
        }

    def _node_probe_environment(self, state: CoderState) -> dict:
        """Network and GPU are probed at runtime, once per run — never assumed,
        since the same code runs on a laptop, a Kaggle notebook, or a Barkla
        compute node."""
        network_available = self.network_check()
        gpu_available = self.gpu_check()
        logger.info(
            "Processing %d experiment plan(s); network_available=%s, gpu_available=%s",
            len(state["ordered_plans"]),
            network_available,
            gpu_available,
        )
        return {"network_available": network_available, "gpu_available": gpu_available}

    def _node_setup_shared_infrastructure(self, state: CoderState) -> dict:
        """Runs once for the whole run (the helper itself no-ops the LLM call
        when the planner asked for no shared infrastructure)."""
        shared_dir, shared_files, warning = self._setup_shared_infrastructure(
            state["planner_output"], state["ordered_plans"]
        )
        return {
            "shared_dir": str(shared_dir),
            "shared_files": shared_files,
            "shared_infra_warning": warning,
        }

    def _node_start_plan_loop(self, state: CoderState) -> dict:
        return {
            "plan_index": 0,
            "experiments": [],
            "slurm_jobs_submitted": self._slurm_jobs_submitted,
        }

    def _node_process_current_plan(self, state: CoderState) -> dict:
        """The per-plan loop body's entry. Infeasible plans are recorded and the
        cursor advanced right here — they never reach the dataset lookup or the
        fix loop, and cost no LLM call and no HTTP call. Feasible ones get their
        directory and a fresh per-plan working set; the lookup and the first
        generation are the two nodes after this one."""
        plan = state["ordered_plans"][state["plan_index"]]
        hypothesis_id = plan["hypothesis_id"]

        if not plan["feasible"]:
            logger.info("Skipping %s: marked infeasible by the Experiment Planner", hypothesis_id)
            skipped = self._result(
                hypothesis_id,
                status="skipped",
                reason=f"Marked infeasible by the Experiment Planner: {plan['feasibility_notes']}",
                code_path=None,
            )
            return {
                "current_plan": plan,
                "experiments": [*state["experiments"], skipped],
                "plan_index": state["plan_index"] + 1,
            }

        experiment_dir = self.experiments_dir / hypothesis_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic, pure-function selection — no LLM call, no dedicated
        # node needed (contrast the HF dataset lookup below, which is a real
        # network call with its own retry/cache policy).
        selected_starter = starters.select_starter(plan)

        return {
            "current_plan": plan,
            "current_experiment_dir": str(experiment_dir),
            "current_fix_history": [],
            "current_attempt": 0,
            "current_outcome": {},
            "current_hf_dataset": {},
            "current_dataset_spec": {},
            "current_dataset_candidates": [],
            "current_dataset_scores": [],
            "current_dataset_rejections": [],
            "current_starter_id": selected_starter["id"] if selected_starter else "",
        }

    def _node_specify_data_requirements(self, state: CoderState) -> dict:
        """State what this plan actually needs from a dataset, before anything
        is searched for.

        First of the five dataset-selection nodes, and the gate for all of them:
        an empty spec here (no network, the feature switched off) is what every
        later node no-ops on, so the whole sequence costs nothing when it is not
        wanted."""
        return {
            "current_dataset_spec": self._specify_data_requirements(
                state["current_plan"], state["network_available"]
            )
        }

    def _node_shortlist_datasets(self, state: CoderState) -> dict:
        """Search the Hub, then rank on metadata alone. No model involved: this
        node exists to decide who is *worth* a model call, and spending one to
        make that decision would defeat the point."""
        return {
            "current_dataset_candidates": self._shortlist_datasets(
                state.get("current_dataset_spec") or {}, state["current_plan"]
            )
        }

    def _node_appraise_datasets(self, state: CoderState) -> dict:
        """Sample real rows from each shortlisted candidate, measure them, ask
        the model for evidence, and score the result in Python."""
        return {
            "current_dataset_scores": self._appraise_datasets(
                state.get("current_dataset_spec") or {},
                state.get("current_dataset_candidates") or [],
            )
        }

    def _node_critique_leading_dataset(self, state: CoderState) -> dict:
        """Try to reject the leader before accepting it.

        The accepted candidate (if any) lands in `current_hf_dataset`; every
        candidate that was reviewed and rejected is kept in
        `current_dataset_rejections` so the reason is in the trace and in the
        record, rather than a dataset silently not being used."""
        accepted, rejections = self._critique_leading_dataset(
            state.get("current_dataset_spec") or {},
            state.get("current_dataset_scores") or [],
        )
        return {"current_hf_dataset": accepted, "current_dataset_rejections": rejections}

    def _node_acquire_dataset(self, state: CoderState) -> dict:
        """Download the accepted dataset under the run's budgets, normalize it
        to a local JSONL, and write the provenance record.

        The only node here with a side effect on disk, and the only one that
        touches the run-level budget counters — which is why it is last and why
        it is separate from the decision that preceded it."""
        acquired = self._acquire_dataset(
            state.get("current_hf_dataset") or {},
            Path(state["current_experiment_dir"]) if state.get("current_experiment_dir") else None,
        )
        return {
            "current_hf_dataset": acquired,
            "datasets_accepted": self._datasets_accepted,
        }

    def _node_generate_experiment_code(self, state: CoderState) -> dict:
        """The first generation for this plan, given whatever the lookup found."""
        try:
            generation = self._generate_experiment_files(
                state["current_plan"],
                state["shared_files"],
                state["network_available"],
                state.get("shared_infra_warning", ""),
                state.get("current_hf_dataset") or {},
                state.get("current_starter_id", ""),
            )
        except CoderAgentError as exc:
            # A generation whose format can't be parsed is a per-plan failure,
            # not a pipeline-ending one — even after invoke_sections' own repair
            # retry, the model can still return a response with sections
            # missing. Feed it into the same fix loop that handles
            # compile/lint/run failures (via _attempt_once's generation_error
            # check) instead of letting it crash the whole multi-hour run over
            # one bad plan.
            generation = {
                "run_py_sections": {},
                "assumptions_made": [],
                "generation_error": str(exc),
            }
        return {"current_generation": generation}

    def _node_attempt(self, state: CoderState) -> dict:
        """One full pass over the current candidate: validate -> compile -> lint
        -> execute (or defer to sbatch) -> read results. `_attempt_once` is
        called whole, so every branch inside it — including the deferred/SLURM
        path and its job caps — is exactly what it was."""
        outcome = self._attempt_once(
            state["current_plan"],
            state["current_generation"],
            Path(state["current_experiment_dir"]),
            state["network_available"],
            state["gpu_available"],
            state["shared_files"],
            state.get("current_starter_id", ""),
            state.get("current_hf_dataset") or {},
        )

        update: dict = {
            "current_outcome": outcome,
            "slurm_jobs_submitted": self._slurm_jobs_submitted,
        }

        fix_history = state["current_fix_history"]
        if fix_history:
            resolved = self._cleared_previous_error(fix_history[-1]["error_source"], outcome)
            update["current_fix_history"] = [
                *fix_history[:-1],
                {**fix_history[-1], "resolved": resolved},
            ]
            if resolved:
                self._maybe_record_fix_pattern(
                    fix_history[-1]["error_source"],
                    fix_history[-1]["error_summary"],
                    state.get("current_broken_sections") or {},
                    state["current_generation"].get("run_py_sections", {}),
                )
        return update

    def _route_after_attempt(self, state: CoderState) -> str:
        """The fix loop's stop condition, unchanged: a terminal result ends the
        plan, an exhausted budget gives up, anything else is regenerated against
        the concrete error. Bound to the agent because the budget is
        per-instance."""
        if "result" in state["current_outcome"]:
            return "finalize"
        if state["current_attempt"] == self.max_fix_attempts:
            return "give_up"
        # Stop early when the last three attempts failed identically. The budget
        # exists to give the model room to converge; once it has landed on the
        # same failure three times it is not converging, and spending the
        # remainder re-deriving it costs allocation to reach the same place.
        if _identical_failure_streak(state.get("current_fix_history") or []) >= _NO_PROGRESS_STREAK:
            return "give_up"
        return "regenerate"

    def _node_snapshot_and_regenerate(self, state: CoderState) -> dict:
        """Preserves the code that just failed, records it in fix_history, and
        asks the model for a version that fixes that concrete error."""
        plan = state["current_plan"]
        outcome = state["current_outcome"]
        attempt = state["current_attempt"]

        logger.info(
            "Fixing %s after %s failure (attempt %d/%d)",
            plan["hypothesis_id"],
            outcome["error_source"],
            attempt + 1,
            self.max_fix_attempts,
        )
        entry = {
            "attempt": attempt + 1,
            "error_source": outcome["error_source"],
            "error_summary": _truncate(outcome["error_text"]),
            "code_path": str(
                self._snapshot_attempt(Path(state["current_experiment_dir"]), attempt + 1)
            ),
            "resolved": False,
        }
        # Computed against fix_history *with* this entry included, so a streak
        # of 2 means "this failure and the one immediately before it were both
        # this error_source" — i.e. the fix that was just tried for it didn't
        # work. previous_error_summary is that prior attempt's own summary
        # (already in state["current_fix_history"], not the new entry), shown
        # to the model so it doesn't just repeat the same fix. See
        # _consecutive_error_streak's docstring for why this is narrower than
        # "resolved".
        streak = _consecutive_error_streak([*state["current_fix_history"], entry])
        entry["same_error_streak"] = streak
        previous_error_summary = (
            state["current_fix_history"][-1]["error_summary"] if streak >= 2 else ""
        )
        try:
            generation = self._regenerate_with_fix(
                plan,
                state["shared_files"],
                state["current_generation"],
                state["network_available"],
                outcome["error_source"],
                outcome["error_text"],
                state.get("shared_infra_warning", ""),
                state.get("current_hf_dataset") or {},
                state.get("current_starter_id", ""),
                stuck_streak=streak,
                previous_error_summary=previous_error_summary,
            )
        except CoderAgentError as exc:
            # Same as the initial generation call: a regeneration attempt can
            # also come back in an unparseable format. Feed it back through the
            # fix loop rather than crashing the run — _attempt_once's
            # generation_error check turns this into a normal "invalid_format"
            # outcome, so it still counts against max_fix_attempts.
            generation = {
                "run_py_sections": {},
                "assumptions_made": [],
                "generation_error": str(exc),
            }
        return {
            "current_fix_history": [*state["current_fix_history"], entry],
            "current_generation": generation,
            "current_attempt": attempt + 1,
            # The sections that just failed (state["current_generation"], read
            # before this node's own reassignment above) — i.e. what entry's
            # error_source was produced by. Read back once entry's `resolved`
            # is known, to pair "what was broken" with "what fixed it" — see
            # fix_pattern_store.record_fix and state.py's field comment.
            "current_broken_sections": dict(state["current_generation"].get("run_py_sections", {})),
        }

    def _node_finalize_current_plan(self, state: CoderState) -> dict:
        """A check produced a terminal result for this plan — record it and
        advance the cursor."""
        fix_history = state["current_fix_history"]
        experiment = {
            **state["current_outcome"]["result"],
            "fix_attempts": len(fix_history),
            "fix_history": fix_history,
        }
        return {
            "experiments": [*state["experiments"], experiment],
            "plan_index": state["plan_index"] + 1,
        }

    def _node_give_up_current_plan(self, state: CoderState) -> dict:
        """The fix budget is spent and the last error still stands — the code
        stays on disk, reported with what it failed on and how many times."""
        fix_history = state["current_fix_history"]
        attempted = f" after {len(fix_history)} fix attempt(s)" if fix_history else ""
        experiment = {
            **self._result(
                state["current_plan"]["hypothesis_id"],
                status="code_generated_not_run",
                reason=f"{state['current_outcome']['error_text']}{attempted}",
                code_path=state["current_experiment_dir"],
                assumptions_made=state["current_generation"].get("assumptions_made", []),
                starter_used=state.get("current_starter_id", ""),
            ),
            "fix_attempts": len(fix_history),
            "fix_history": fix_history,
        }
        return {
            "experiments": [*state["experiments"], experiment],
            "plan_index": state["plan_index"] + 1,
        }

    def _node_assemble_and_validate(self, state: CoderState) -> dict:
        result: dict = {
            "experiments": state["experiments"],
            "shared_infrastructure_path": state["shared_dir"],
            "source_hypothesis_ids": state["expected_ids"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": settings.llm_model,
        }

        try:
            validate_output(result, expected_hypothesis_ids=state["expected_ids"])
        except SchemaValidationError as exc:
            debug_path = self._write_summary(result, suffix="_invalid")
            raise CoderAgentError(
                f"Assembled output failed schema validation: {exc}. "
                f"Raw (invalid) output written to {debug_path} for inspection. "
                f"Generated experiment code on disk under {self.experiments_dir} is unaffected."
            ) from exc

        summary_path = self._write_summary(result)
        logger.info("Wrote coder agent summary to %s", summary_path)
        return {"result": result}

    # -- Ordering ---------------------------------------------------------------

    @staticmethod
    def _order_by_priority(plans: list[dict], priority_order: list[dict]) -> list[dict]:
        rank_by_id = {entry["hypothesis_id"]: entry["rank"] for entry in priority_order}
        # plans missing from priority_order (shouldn't happen given a validated
        # planner output) sort after every ranked plan, in their original order
        return sorted(plans, key=lambda p: rank_by_id.get(p["hypothesis_id"], len(plans) + 1))

    # -- Shared infrastructure -----------------------------------------------

    def _setup_shared_infrastructure(
        self, planner_output: dict, plans: list[dict]
    ) -> tuple[Path, dict[str, str], str]:
        """Generates shared infrastructure once, then validates it the same
        way a single experiment's run.py is validated — compile check, then
        static safety check — and regenerates against the concrete failure,
        bounded by max_fix_attempts, exactly like the per-experiment fix loop.

        This has to happen here rather than inside that fix loop: every
        experiment that imports shared infrastructure trusts it
        unconditionally, and the fix loop only ever regenerates the one
        experiment currently failing — it has no way to see, let alone fix, a
        bug that actually lives in experiments/_shared/. If it's still broken
        after the budget is spent, every experiment's codegen/fix prompt gets
        an explicit warning instead (see _shared_infra_block) so the model can
        choose not to rely on it, rather than failing at import time with no
        idea why."""
        shared_dir = self.experiments_dir / "_shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        (self.experiments_dir / "__init__.py").touch()
        (shared_dir / "__init__.py").touch()

        shared_items = planner_output.get("shared_infrastructure") or []
        if not shared_items:
            return shared_dir, {}, ""

        prompt = prompts.SHARED_INFRA_PROMPT.format(
            shared_items_block=_compact_json(shared_items),
            plans_block=_compact_json(plans),
        )
        # field_names=None: one section per file, named after the file, so the
        # names come from the response rather than from a fixed list.
        files = self._call_sections(prompt)
        files, problem = sandbox.check_shared_infra_files(files)

        attempt = 0
        while problem and attempt < self.max_fix_attempts:
            attempt += 1
            logger.info(
                "Shared infrastructure failed a check (attempt %d/%d): %s",
                attempt,
                self.max_fix_attempts,
                problem,
            )
            fix_prompt = prompts.SHARED_INFRA_FIX_PROMPT.format(
                shared_items_block=_compact_json(shared_items),
                plans_block=_compact_json(plans),
                previous_files_block=render_sections(files),
                error_text=problem,
            )
            files = self._call_sections(fix_prompt, temperature=_FIX_TEMPERATURE) or files
            files, problem = sandbox.check_shared_infra_files(files)

        warning = ""
        if problem:
            warning = (
                f"WARNING: shared infrastructure still fails a check after {attempt} repair "
                f"attempt(s): {problem}. Do not assume this shared code is safe to import as-is "
                "— if you need this functionality, either avoid importing it or implement the "
                "needed logic standalone in your own experiment file instead."
            )
            logger.warning(
                "Shared infrastructure still broken after %d attempt(s): %s", attempt, problem
            )

        self._write_files(shared_dir, files)
        logger.info("Wrote %d shared infrastructure file(s) to %s", len(files), shared_dir)
        return (
            shared_dir,
            {name: content for name, content in files.items() if name.endswith(".py")},
            warning,
        )

    # -- Per-experiment processing --------------------------------------------
    #
    # The plan loop (process_current_plan -> ... -> finalize/give_up ->
    # process_current_plan) and the fix loop (attempt ->
    # snapshot_and_regenerate -> attempt) that used to live here as
    # `_process_plan`/`_generate_run_and_diagnose` are now real graph cycles —
    # see graph.py. The helpers below (`_attempt_once`,
    # `_generate_experiment_files`, `_regenerate_with_fix`, `_snapshot_attempt`,
    # `_cleared_previous_error`, `_result`) are what the graph's nodes call.

    def _attempt_once(
        self,
        plan: dict,
        generation: dict,
        experiment_dir: Path,
        network_available: bool,
        gpu_available: bool,
        shared_files: dict[str, str],
        starter_id: str = "",
        hf_dataset: dict | None = None,
    ) -> dict:
        """Runs one full pass over a generated candidate. Returns either
        {"result": <terminal experiment dict>} or {"error_source",
        "error_text"} describing a failure the fix loop can regenerate
        against."""
        hypothesis_id = plan["hypothesis_id"]
        generation_error = generation.get("generation_error")
        if generation_error:
            return {
                "error_source": "invalid_format",
                "error_text": f"Model did not return this experiment's code in the required delimited section format: {generation_error}",
            }
        sections = generation.get("run_py_sections", {})
        assumptions_made = generation.get("assumptions_made", [])
        needs_gpu = bool(generation.get("needs_gpu", False))

        required_sections = (
            "load_data_function",
            "build_model_function",
            "run_experiment_function",
            "evaluate_function",
        )
        missing_sections = [
            name for name in required_sections if not (sections.get(name) or "").strip()
        ]
        if missing_sections:
            return {
                "error_source": "missing_sections",
                "error_text": f"Model response was missing required code section(s): {missing_sections} — run.py was not written or executed.",
            }

        # Checked before anything expensive happens: a section defining a
        # differently-named function clears every other check and only fails at
        # execution, after a venv provision — see
        # sandbox.check_required_function_names.
        function_name_findings = sandbox.check_required_function_names(sections)
        if function_name_findings:
            return {
                "error_source": "missing_required_function",
                "error_text": f"Generated code doesn't define the required function name(s): {'; '.join(function_name_findings)}",
            }

        # Checked before anything expensive too, same reasoning as the function-
        # name check above: a body that's just `pass`/`...`/a NotImplementedError
        # clears every check below (nothing dangerous, no unguarded read to
        # flag) and only shows up as a hollow "completed" result after a full
        # venv provision and execution — see
        # sandbox.check_nontrivial_function_bodies.
        empty_body_findings = sandbox.check_nontrivial_function_bodies(sections)
        if empty_body_findings:
            return {
                "error_source": "empty_body",
                "error_text": f"Generated code has no real implementation: {'; '.join(empty_body_findings)}",
            }

        # run.py's metadata block + orchestration are a fixed template, not
        # model output — only the four functions/imports/configuration/helpers
        # are spliced in, so the calling convention is guaranteed correct
        # rather than depending on the model reproducing it exactly every time.
        run_py = sandbox.render_experiment_template(
            hypothesis_id=hypothesis_id,
            objective=plan["objective"],
            design=plan["design"],
            data_description=plan["data_requirements"]["description"],
            baseline=plan["evaluation"]["baseline"],
            success_criteria=plan["evaluation"]["success_criteria"],
            agent_imports=sections.get("imports", ""),
            agent_configuration=sections.get("configuration", ""),
            load_data_function=sections["load_data_function"],
            build_model_function=sections["build_model_function"],
            run_experiment_function=sections["run_experiment_function"],
            evaluate_function=sections["evaluate_function"],
            agent_helpers=sections.get("helpers", ""),
        )

        # lenient_compile_check may silently repair a redundant trailing
        # backslash before writing the file (see its docstring) — run_py is
        # reassigned so the safety check and the file on disk both see
        # whatever version actually compiled.
        run_py, compile_error = sandbox.lenient_compile_check(run_py, "run.py")

        files = {
            "run.py": run_py,
            "README.md": generation.get("readme", ""),
            "requirements.txt": generation.get("requirements_txt", ""),
        }
        self._write_files(experiment_dir, files)

        if compile_error:
            return {
                "error_source": "compile_check",
                "error_text": f"Generated code has a syntax error, not executed: {compile_error}",
            }

        findings = sandbox.static_safety_check(run_py)
        if findings:
            return {
                "error_source": "static_lint",
                "error_text": f"Generated code was flagged by the static safety check: {'; '.join(findings)}",
            }

        # Enforces the "guard the read, fall back to synthesized data" instruction
        # the prompt has always given — checked, not trusted, exactly like every
        # other stop condition in this loop. Runs on load_data's own source rather
        # than the rendered run.py; see sandbox.check_data_fallback.
        fallback_findings = sandbox.check_data_fallback(sections["load_data_function"])
        if fallback_findings:
            return {
                "error_source": "missing_data_fallback",
                "error_text": f"load_data() assumes its data will be present: {'; '.join(fallback_findings)}",
            }

        # A real, pre-verified dataset was offered (see _hf_dataset_block) —
        # checks the offer was actually engaged with rather than silently
        # dropped, the two sanctioned outcomes being "used it" or "declined it
        # in assumptions_made" (HF_DATASET_USAGE_NOTE says either is fine).
        # See sandbox.check_hf_dataset_usage.
        dataset_usage_findings = sandbox.check_hf_dataset_usage(
            sections.get("configuration", ""),
            sections["load_data_function"],
            assumptions_made,
            hf_dataset or {},
        )
        if dataset_usage_findings:
            return {
                "error_source": "ignored_available_dataset",
                "error_text": f"A real dataset was offered but not used: {'; '.join(dataset_usage_findings)}",
            }

        complexity = plan["estimated_complexity"]
        requirements_path = experiment_dir / "requirements.txt"

        run_high_locally = (
            complexity == "high"
            and settings.coder_run_high_complexity_when_gpu_available
            and gpu_available
        )
        if (needs_gpu and not gpu_available) or (complexity == "high" and not run_high_locally):
            return self._handle_unrunnable_locally(
                plan, generation, run_py, experiment_dir, requirements_path, complexity, starter_id
            )

        # Both experiments/_shared/ and this experiment's own run_py can import a
        # package the model never declared in requirements_txt — see
        # sandbox.extract_third_party_imports's docstring for the shared-infra
        # production failure (job 10229968) that originally motivated this. A
        # 2026-08-17 run (job 10247173) reproduced the same gap on the
        # experiment's *own* code: this plan had no shared infrastructure at
        # all, so that guard never even ran, and the model's `imports`/
        # `load_data_function` section imported pandas while its
        # requirements_txt section didn't list it — ensure_experiment_env saw
        # nothing missing and handed back the bare interpreter, which failed
        # identically on all 3 fix attempts since nothing forced the
        # regenerated requirements_txt to actually include it. run_py is
        # parsed here too, not just shared_files, to close that. Passed as
        # extra_requirements rather than folded into requirements.txt on disk,
        # which should keep documenting only what the model itself declared.
        extra_requirements = sorted(
            sandbox.extract_third_party_imports(run_py).union(
                *(sandbox.extract_third_party_imports(src) for src in shared_files.values())
            )
        )
        python_executable, env_error = sandbox.ensure_experiment_env(
            experiment_dir,
            requirements_path,
            network_available,
            extra_requirements,
            venv_root=Path(settings.coder_venv_root) if settings.coder_venv_root else None,
            python_version=settings.coder_experiment_python or None,
        )
        # Checked as `is None` rather than `if env_error` so the interpreter is
        # narrowed to a Path for the run below; ensure_experiment_env's contract
        # is that exactly one of the two is set, so this is the same branch.
        if python_executable is None:
            # Not routed through the fix loop: a missing package or an
            # unreachable index is an environment problem, and regenerating
            # the code can't resolve it.
            return {
                "result": self._result(
                    hypothesis_id,
                    status="code_generated_not_run",
                    reason=env_error or "could not provision an environment for this experiment",
                    code_path=str(experiment_dir),
                    assumptions_made=assumptions_made,
                    starter_used=starter_id,
                )
            }

        # All three timeouts are settings-driven and read here rather than in
        # sandbox.py, which deliberately reads no settings at all.
        if complexity == "high":
            timeout_seconds = settings.coder_high_complexity_timeout_seconds
        elif complexity == "medium":
            timeout_seconds = settings.coder_medium_complexity_timeout_seconds
        else:
            timeout_seconds = settings.coder_low_complexity_timeout_seconds
        # Execution, plus the repairs that don't need the model.
        #
        # A failure is classified by *kind* before anything is decided about it
        # (diagnose.classify_execution_failure). Two kinds are repaired here and
        # the same code re-run: a package the code correctly imports but nobody
        # declared, and a run that was simply too big for its budget. Neither is
        # a defect in the generated source, so neither costs a fix attempt —
        # regenerating in response to a missing package is what made a
        # 2026-08-19 run spend its whole budget re-deriving the same
        # ModuleNotFoundError three times.
        #
        # Everything else returns an error_source as before and goes to the fix
        # loop, which is still the right place for a genuine code defect.
        env_repairs = 0
        downscales = 0
        run_path = experiment_dir / "run.py"

        while True:
            succeeded, message = sandbox.run_experiment(
                python_executable, run_path, experiment_dir, timeout_seconds
            )
            if succeeded:
                break

            failure = diagnose.classify_execution_failure(message)

            if failure.route == diagnose.ROUTE_ENV and env_repairs < settings.coder_max_env_repairs:
                if not network_available:
                    # Say which package and why it can't be had, rather than
                    # letting the fix loop rediscover it three times.
                    return {
                        "error_source": failure.error_source,
                        "error_text": (
                            f"{failure.summary} No network access on this node to install it — "
                            "add it to the environment the pipeline runs in."
                        ),
                    }
                installed, detail = repair.install_for(python_executable, failure)
                env_repairs += 1
                # An installer's exit code is not proof of repair. `uv pip
                # install pandas` returns 0 when it believes pandas is already
                # present for that interpreter, which on Barkla job 10279290 it
                # did six times in a row while the experiment went on failing to
                # import it. Verify against the interpreter that will actually
                # run the code, and treat "installed, still missing" as the
                # environment being broken in a way installing cannot fix —
                # otherwise the budget drains re-running an identical failure,
                # which is the exact behaviour this whole routing layer exists
                # to prevent.
                if installed and sandbox.module_importable(
                    python_executable, failure.module or "", experiment_dir
                ):
                    logger.info("[%s] %s", hypothesis_id, detail)
                    continue  # re-run the *unchanged* code
                if installed:
                    return {
                        "error_source": failure.error_source,
                        "error_text": (
                            f"{failure.summary} The package installed successfully but "
                            f"{failure.module!r} is still not importable by {python_executable} — "
                            "the interpreter running the experiment is not the one being installed "
                            "into, or its site-packages are not on that interpreter's path."
                        ),
                    }
                return {
                    "error_source": failure.error_source,
                    "error_text": f"{failure.summary} {detail}",
                }

            if failure.route == diagnose.ROUTE_DOWNSCALE and downscales < _MAX_DOWNSCALES:
                shrunk, changes = repair.downscale(run_path.read_text())
                if changes:
                    run_path.write_text(shrunk)
                    downscales += 1
                    logger.info(
                        "[%s] %s Reduced deterministically: %s",
                        hypothesis_id,
                        failure.summary,
                        "; ".join(changes),
                    )
                    continue  # re-run the smaller version, still no model call
                # Nothing to shrink: fall through and let the model rethink it.

            return {"error_source": failure.error_source, "error_text": failure.summary}

        results, diagnosis = sandbox.read_results_json_for_diagnosis(experiment_dir)
        if results is None:
            return {
                "error_source": "results_json",
                "error_text": f"run.py exited successfully but did not produce a valid results.json: {diagnosis}",
            }

        # A real result on disk isn't the same as a meaningful one — see
        # sandbox.check_results_plausibility for exactly what this does and
        # doesn't catch.
        plausibility_findings = sandbox.check_results_plausibility(results.get("metrics") or {})
        if plausibility_findings:
            return {
                "error_source": "implausible_results",
                "error_text": f"results.json's metrics look hollow: {'; '.join(plausibility_findings)}",
            }

        # The experiment ran and produced metrics. Whether those metrics are
        # allowed to carry a verdict about the hypothesis depends on where their
        # inputs came from, which is decided here in Python rather than left to
        # the model: any synthetic input turns meets_success_criteria into the
        # string "unknown", which the Writer reads as "inconclusive". Returning
        # False instead would have it publish a *refutation* off generated data.
        sources = self._provenance_for(
            plan, network_available, hf_dataset=hf_dataset, run_py=run_py
        )
        provenance_document = provenance.write(sources, experiment_dir / "data_provenance.json")
        results = provenance.apply_to_results(results, sources)
        if not provenance.all_real(sources):
            logger.info(
                "[%s] verdict withheld — %s",
                hypothesis_id,
                provenance_document["methodological_validity"],
            )

        return {
            "result": self._result(
                hypothesis_id,
                status="completed",
                reason="",
                code_path=str(experiment_dir),
                assumptions_made=assumptions_made,
                results=results,
                starter_used=starter_id,
                data_provenance=provenance_document,
            )
        }

    def _handle_unrunnable_locally(
        self,
        plan: dict,
        generation: dict,
        run_py: str,
        experiment_dir: Path,
        requirements_path: Path,
        complexity: str,
        starter_id: str = "",
    ) -> dict:
        """Plans that can't run here: too heavy, or they need a GPU this
        machine doesn't have. Always writes run.sbatch. Whether it also gets
        submitted is opt-in and capped — by default a human still reviews and
        submits it, since nothing has ever executed this code."""
        hypothesis_id = plan["hypothesis_id"]
        assumptions_made = generation.get("assumptions_made", [])
        why_unrunnable = (
            "estimated_complexity is 'high'"
            if complexity == "high"
            else "experiment needs a GPU, none detected in this environment"
        )

        sbatch_path = experiment_dir / "run.sbatch"
        sbatch_path.write_text(
            sandbox.render_sbatch_template(hypothesis_id, requirements_path.exists())
        )

        def leave_for_review(detail: str) -> dict:
            return {
                "result": self._result(
                    hypothesis_id,
                    status="code_generated_not_run",
                    reason=f"{why_unrunnable} — generated {sbatch_path.name} instead of running synchronously; {detail}",
                    code_path=str(experiment_dir),
                    assumptions_made=assumptions_made,
                    starter_used=starter_id,
                )
            }

        if not settings.coder_auto_submit_slurm:
            if not self.interactive_slurm_review:
                return leave_for_review("review and submit it yourself with `sbatch run.sbatch`.")

            # Pauses the whole graph here — see run()'s interrupt loop, which
            # is what actually asks a human and resumes with the answer. On
            # resume, this node re-executes from the top (LangGraph's
            # documented interrupt contract) and reaches this same call a
            # second time, which is when it returns the decision instead of
            # pausing again — everything above this line (writing run.sbatch,
            # the checks _attempt_once already ran) is either idempotent or
            # already done by the time a real answer comes back, so replaying
            # it is harmless.
            decision = interrupt(
                {
                    "hypothesis_id": hypothesis_id,
                    "why_unrunnable": why_unrunnable,
                    "run_py": run_py,
                    "code_path": str(experiment_dir),
                    "sbatch_path": str(sbatch_path),
                }
            )
            if decision != "submit":
                return leave_for_review(
                    "left for manual review — reviewer chose not to submit it now."
                )
            # decision == "submit": fall through to the exact same
            # safety-check + cap + submit sequence coder_auto_submit_slurm
            # takes. A human saying "submit" is one more gate, not a bypass of
            # the ones below.

        # The static safety check already ran in _attempt_once, before this
        # branch — anything it flags never reaches submission.
        concerns = self._self_review(plan, run_py)
        if concerns:
            # No execution is possible here, so a flagged concern is the only
            # error signal available — feed it back through the same fix loop.
            return {
                "error_source": "self_review",
                "error_text": "Pre-submission review raised: " + "; ".join(concerns),
            }

        if self._slurm_jobs_submitted >= settings.coder_max_slurm_jobs_per_run:
            return leave_for_review(
                f"auto-submission was skipped because this run already submitted {self._slurm_jobs_submitted} job(s) "
                f"(CODER_MAX_SLURM_JOBS_PER_RUN={settings.coder_max_slurm_jobs_per_run})."
            )

        queued = slurm_submit.count_running_jobs()
        if queued >= settings.coder_max_concurrent_slurm_jobs:
            return leave_for_review(
                f"auto-submission was skipped because {queued} job(s) are already queued "
                f"(CODER_MAX_CONCURRENT_SLURM_JOBS={settings.coder_max_concurrent_slurm_jobs})."
            )

        job_id, submit_error = slurm_submit.submit_job(sbatch_path, experiment_dir)
        if submit_error:
            return leave_for_review(
                f"auto-submission failed ({submit_error}) — submit it yourself once resolved."
            )

        self._slurm_jobs_submitted += 1
        logger.info("Submitted %s to SLURM as job %s", hypothesis_id, job_id)
        return {
            "result": self._result(
                hypothesis_id,
                status="submitted_to_slurm",
                reason=f"{why_unrunnable} — auto-submitted as SLURM job {job_id}; results are not available in this run.",
                code_path=str(experiment_dir),
                assumptions_made=assumptions_made,
                slurm_job_id=job_id,
                starter_used=starter_id,
            )
        }

    @staticmethod
    def _cleared_previous_error(previous_source: str, outcome: dict) -> bool:
        """Whether the regenerated code got past the check that failed last
        time — either by finishing, or by failing later in the sequence."""
        if "result" in outcome:
            return True
        order = _ERROR_STAGE_ORDER
        return order.index(outcome["error_source"]) > order.index(previous_source)

    def _maybe_record_fix_pattern(
        self,
        error_source: str,
        error_summary: str,
        broken_sections: dict[str, str],
        fixed_sections: dict[str, str],
    ) -> None:
        """Persists the fix that just resolved error_source, for
        _fix_pattern_block to show a future fix attempt on the same
        error_source. A store outage degrades to "nothing recorded" rather
        than failing the run — same resilience contract as _find_hf_dataset's
        network lookup: this is a prompt enhancement, never a reason a real,
        working experiment result is lost."""
        if not settings.coder_enable_fix_pattern_store:
            return
        try:
            fix_pattern_store.record_fix(
                self.fix_store,
                error_source=error_source,
                error_summary=error_summary,
                broken_sections=broken_sections,
                fixed_sections=fixed_sections,
            )
        except Exception as exc:  # noqa: BLE001 — see the docstring
            logger.warning("Could not persist a fix pattern for %s: %s", error_source, exc)

    @staticmethod
    def _snapshot_attempt(experiment_dir: Path, attempt: int) -> Path:
        """Preserves the code that just failed before it's overwritten, so a
        failed run is still inspectable (and usable as training data) rather
        than only the last attempt surviving on disk."""
        snapshot_dir = experiment_dir / "fix_attempts" / f"attempt_{attempt}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name in ("run.py", "requirements.txt", "results.json"):
            source = experiment_dir / name
            if source.exists():
                (snapshot_dir / name).write_text(source.read_text())
        return snapshot_dir / "run.py"

    @staticmethod
    def _result(
        hypothesis_id: str,
        status: str,
        reason: str,
        code_path: str | None,
        assumptions_made: list[str] | None = None,
        results: dict | None = None,
        slurm_job_id: str | None = None,
        starter_used: str = "",
        data_provenance: dict | None = None,
    ) -> dict:
        return {
            "hypothesis_id": hypothesis_id,
            "status": status,
            "reason": reason,
            "code_path": code_path,
            "assumptions_made": assumptions_made or [],
            "results": results,
            "fix_attempts": 0,
            "fix_history": [],
            "slurm_job_id": slurm_job_id,
            # The starters.STARTERS id this plan's codegen/fix prompts were
            # grounded in, or "" for "general" (no match) — traceability, same
            # instinct as fix_history existing at all.
            "starter_used": starter_used,
            "data_provenance": data_provenance or {},
        }

    # -- LLM calls -----------------------------------------------------------

    def _bounded_max_tokens(self, user_prompt: str) -> int:
        """How many completion tokens this specific prompt may ask for, so the
        request can't exceed the model's context window.

        get_chat_model fixes max_tokens at client construction, which is fine
        until the prompt itself is large: this agent's fix prompts carry the
        previous code sections, the concrete error, the plan JSON and the
        shared-infrastructure block, and `prompt_tokens + max_tokens >
        context_window` is a 400 BadRequestError from the server, not a short
        answer. That crashed 6 of 10 questions in the 2026-08-11 batch run,
        both before and after a fix attempt's prompt was drafted.

        The estimate is deliberately conservative and is never checked against
        the real tokenizer (there isn't one on this side — see
        _CHARS_PER_TOKEN_ESTIMATE). Being conservative costs at most a shorter
        completion, which the fix loop already handles; being wrong the other
        way is exactly the crash this exists to prevent.

        Raises CoderAgentError when the prompt leaves less room than
        _MIN_GENERATION_TOKENS — at that point no completion would be long
        enough to use, so the prompt is the problem. Callers let that propagate:
        the generation nodes already convert a CoderAgentError into a normal
        fix-loop outcome.
        """
        prompt_tokens = _estimate_tokens(prompts.SYSTEM_PROMPT) + _estimate_tokens(user_prompt)
        headroom = settings.llm_context_window - prompt_tokens - _CONTEXT_SAFETY_MARGIN
        if headroom < _MIN_GENERATION_TOKENS:
            raise CoderAgentError(
                f"Prompt is too large for the model's context window: ~{prompt_tokens} estimated "
                f"prompt tokens leave only {headroom} token(s) for the response, below the "
                f"{_MIN_GENERATION_TOKENS}-token minimum a usable completion needs "
                f"(LLM_CONTEXT_WINDOW={settings.llm_context_window})."
            )
        return min(settings.llm_max_tokens, headroom)

    def _call_json(self, user_prompt: str, *, temperature: float | None = None) -> dict:
        """For responses that are short structured fields only — currently just
        the self-review verdict. Anything carrying source code goes through
        _call_sections instead."""
        max_tokens = self._bounded_max_tokens(user_prompt)
        try:
            return invoke_json(
                self.chat_model,
                prompts.SYSTEM_PROMPT,
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMJSONError as exc:
            raise CoderAgentError(str(exc)) from exc

    def _call_sections(
        self,
        user_prompt: str,
        field_names: Sequence[str] | None = None,
        *,
        temperature: float | None = None,
    ) -> dict[str, str]:
        """For every response that carries generated source code.

        Mirrors _call_json's wrapper shape (same system prompt, same
        LLM*Error -> CoderAgentError conversion so callers keep catching one
        exception type), but over llm_sections' delimited transport: code is
        returned as raw text between markers instead of as JSON string values
        the model has to escape by hand. `field_names=None` means "discover the
        section names from the response", which the shared-infrastructure call
        needs since it names its sections after files it hasn't picked yet.

        `temperature` overrides the constructor's for this call only — the fix
        paths pass _FIX_TEMPERATURE; initial generation leaves it None."""
        max_tokens = self._bounded_max_tokens(user_prompt)
        try:
            return invoke_sections(
                self.chat_model,
                prompts.SYSTEM_PROMPT,
                user_prompt,
                field_names,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMSectionsError as exc:
            raise CoderAgentError(str(exc)) from exc

    @staticmethod
    def _assemble_generation(sections: dict[str, str]) -> dict:
        """Turns the flat dict of raw section text `_call_sections` returns into
        the exact `generation` dict shape everything downstream already expects
        (`run_py_sections` nested, `assumptions_made` a list, `needs_gpu` a
        bool) — so `_attempt_once`, `sandbox.render_experiment_template` and
        `schema.py` are untouched by the transport change.

        The text -> bool/list conversions live here rather than in
        llm_sections.py deliberately: that module owns the transport, and what a
        field's text *means* is this agent's domain knowledge, the same split
        the pipeline already makes for `meets_success_criteria`."""
        return {
            "run_py_sections": {
                name: sections.get(name, "") for name in prompts.RUN_PY_SECTION_NAMES
            },
            "readme": sections.get("readme", ""),
            "requirements_txt": sections.get("requirements_txt", ""),
            "assumptions_made": _parse_assumptions(sections.get("assumptions_made", "")),
            "needs_network": _parse_bool_text(sections.get("needs_network", "")),
            "needs_gpu": _parse_bool_text(sections.get("needs_gpu", "")),
        }

    @staticmethod
    def _shared_infra_block(shared_files: dict[str, str], warning: str = "") -> str:
        if not shared_files:
            return "No shared infrastructure applies to this experiment — implement it standalone."
        blocks = "\n\n".join(
            f"--- experiments/_shared/{name} ---\n{content}"
            for name, content in shared_files.items()
        )
        warning_block = f"\n\n{warning}" if warning else ""
        return (
            f"Shared infrastructure already generated for this pipeline:\n\n{blocks}\n\n"
            f"{prompts.SHARED_IMPORT_NOTE}{warning_block}"
        )

    @staticmethod
    def _starter_block(starter_id: str) -> str:
        """Renders the chosen starters.STARTERS entry as a worked reference
        example for the codegen/fix prompt, or "" for "" (no match — the
        "general" fallback), so a plan with no matching starter reads exactly
        as it did before this library existed."""
        starter = starters.STARTERS.get(starter_id) if starter_id else None
        if starter is None:
            return ""
        rendered = "\n\n".join(
            f"--- {name} ---\n{content}"
            for name, content in starter["sections"].items()
            if content.strip()
        )
        return (
            f"A pre-validated reference program for a similar task ({starter['description']}) "
            "is below — real, runnable code that already passes every check this experiment's "
            "own code will be checked against. Adapt its STRUCTURE and patterns (train/test "
            "split, real metric computation, guarded fallbacks, JSON-serializable return "
            "values, the two reserved evaluate() keys) to THIS plan's actual data_requirements/"
            "methods/design — do not reuse its synthetic data or copy it verbatim if the plan "
            f"calls for something different:\n\n{rendered}\n"
        )

    def _fix_pattern_block(self, error_source: str) -> str:
        """Up to 2 real past fixes for this exact error_source, from
        fix_pattern_store — the fix-loop counterpart to _starter_block: same
        "ground the model in a real worked example" idea, populated by this
        pipeline's own run history instead of a hand-authored library. ""
        when disabled, empty, or the store is unreachable, so a fix prompt
        with no recorded history for this error_source reads exactly as it
        did before this store existed."""
        if not settings.coder_enable_fix_pattern_store:
            return ""
        try:
            recalled = fix_pattern_store.recall_fixes(self.fix_store, error_source)
        except Exception as exc:  # noqa: BLE001 — a prompt enhancement, not a dependency
            logger.warning(
                "Fix-pattern lookup failed for %s; continuing without it: %s", error_source, exc
            )
            return ""
        if not recalled:
            return ""

        rendered_patterns = []
        for i, pattern in enumerate(recalled, start=1):
            sections_text = "\n\n".join(
                f"--- {name} (before) ---\n{change['before'] or '<none — new section>'}\n"
                f"--- {name} (after; this is what fixed it) ---\n{change['after']}"
                for name, change in (pattern.get("changed_sections") or {}).items()
            )
            rendered_patterns.append(
                f"Past fix #{i}, for the same failure "
                f"({str(pattern.get('error_summary', ''))[:200]}):\n{sections_text}"
            )
        return (
            "Real fixes that resolved this exact error_source in past runs, most recent "
            "first — this is not this plan's own code, so adapt the PATTERN of what changed "
            "rather than copying it verbatim:\n\n" + "\n\n".join(rendered_patterns) + "\n"
        )

    @staticmethod
    def _stuck_block(streak: int, previous_error_summary: str) -> str:
        """Empty below streak 2 (this failure is a new kind, or the first one) —
        a fix prompt with no repeat failure reads exactly as it did before
        this escalation existed. From streak 2 on, names the repeat explicitly
        and quotes the model's own last (unsuccessful) fix, so it diagnoses a
        different cause instead of resubmitting a variation of the same one."""
        if streak < 2:
            return ""
        return (
            f"STUCK WARNING: this is the same failure category as your last {streak - 1} "
            "fix attempt(s) on this experiment — none of them actually resolved it. Your "
            f"most recent attempt's diagnosis was:\n{previous_error_summary}\n\n"
            "Do not resubmit a small variation of that same fix. Identify a different root "
            "cause, or replace the affected logic with a simpler, more conservative "
            "implementation (e.g. drop the risky pattern or library entirely) rather than "
            "patching around it again.\n\n"
        )

    # ----------------------------------------------------------------------
    # Dataset selection.
    #
    # Five helpers, one per node, in the order the graph calls them. Every one
    # of them degrades to "no dataset" rather than raising: the Coder is about
    # to generate code either way, and a dataset search must never be the reason
    # an experiment doesn't get written. That contract is why none of these
    # nodes carries a RetryPolicy — a failure here is absorbed in-node, so there
    # is nothing for the graph to retry.
    # ----------------------------------------------------------------------

    def _dataset_search_enabled(self, network_available: bool) -> bool:
        return bool(network_available and settings.coder_enable_hf_dataset_search)

    def _specify_data_requirements(self, plan: dict, network_available: bool) -> dict:
        """The plan's data requirement as a structured spec, as a plain dict.

        The model drafts it and `dataset_spec.validate_spec` coerces every field;
        if the call fails or returns junk, `fallback_spec` derives one from the
        plan deterministically. So this returns a usable spec in every case
        except the feature being switched off, and nothing downstream has to
        handle "the model didn't answer" separately from "the model answered
        badly".
        """
        if not self._dataset_search_enabled(network_available):
            return {}
        prompt = prompts.DATASET_SPEC_PROMPT.format(plan_block=_compact_json(plan))
        payload: object = None
        try:
            payload = self._call_json(prompt)
        except Exception as exc:  # noqa: BLE001 — see this section's header
            logger.warning(
                "Dataset requirement spec for %s failed (%s); deriving one from the plan instead",
                plan.get("hypothesis_id"),
                exc,
            )
        spec = dataset_spec.validate_spec(payload, plan)
        logger.info(
            "Dataset spec for %s: task=%r data_types=%s desired_examples=%d",
            plan.get("hypothesis_id"),
            spec.task,
            list(spec.data_types),
            spec.desired_examples,
        )
        return spec.to_dict()

    def _shortlist_datasets(self, spec_dict: dict, plan: dict) -> list[dict]:
        """Candidates worth appraising, best-first.

        Pools the hits from every query the spec produces rather than stopping
        at the first query that returns something — ranking happens over the
        pool, so a later query's better match is not lost to an earlier query's
        merely-adequate one. The deterministic prefilter then cuts the pool to
        `CODER_DATASET_MAX_INSPECTIONS`, and only those survivors pay for the
        viewer round-trips that `describe_candidate` makes.
        """
        if not spec_dict:
            return []
        spec = dataset_spec.from_dict(spec_dict)

        queries = dataset_spec.search_queries(spec, plan)

        # Each query is asked for a full page, and the pages are then
        # interleaved round-robin into the pool. Dividing the pool cap by the
        # query count instead — which this did first — starves every query at
        # once: a live run (Barkla job 10334321) split a 20-candidate cap five
        # ways, took 4 hits per query, and pooled **6** candidates where the
        # same queries at a full page pool 47. Round-robin keeps the equal
        # representation that division was reaching for, without the starvation:
        # no single broad query fills the pool, and no query is capped below
        # what it found.
        pages: list[list[dict]] = []
        for query in queries:
            try:
                hits = self.dataset_search(query, _DATASET_HITS_PER_QUERY)
            except Exception as exc:  # noqa: BLE001 — see this section's header
                logger.warning("Hub search for %r failed: %s", query, exc)
                continue
            pages.append([hit for hit in hits or [] if isinstance(hit, dict)])

        pooled: dict[str, dict] = {}
        for rank in range(_DATASET_HITS_PER_QUERY):
            if len(pooled) >= settings.coder_dataset_max_candidates:
                break
            for page in pages:
                if rank >= len(page):
                    continue
                dataset_id = str(page[rank].get("id") or "")
                if dataset_id and dataset_id not in pooled:
                    pooled[dataset_id] = page[rank]
                if len(pooled) >= settings.coder_dataset_max_candidates:
                    break

        if not pooled:
            logger.info("No Hugging Face candidates for %s", plan.get("hypothesis_id"))
            return []

        # Ranked well past the inspection budget, because the budget bounds how
        # many candidates get *inspected*, not how many get *attempted*. A
        # ranked candidate is dropped whenever the Dataset Viewer cannot serve
        # it — it 500s, it reports viewer:false, it lists no splits — and those
        # failures are common enough to starve the appraisal outright: Barkla
        # job 10334376 pooled 27 candidates, passed the top 5 to describe, lost
        # 3 of them to the viewer, and inspected 2 — while the dataset that
        # actually fitted the plan sat at rank 6 and was never looked at.
        window = settings.coder_dataset_max_inspections * _DATASET_PREFILTER_WINDOW
        ranked = dataset_scoring.prefilter(list(pooled.values()), spec, window)

        described: list[dict] = []
        for candidate in ranked:
            if len(described) >= settings.coder_dataset_max_inspections:
                break
            dataset_id = str(candidate.get("id") or "")
            try:
                details = self.dataset_describe(dataset_id)
            except Exception as exc:  # noqa: BLE001 — see this section's header
                logger.warning("Could not describe %s: %s", dataset_id, exc)
                continue
            if not details:
                logger.info("The Dataset Viewer cannot serve %s; trying the next", dataset_id)
                continue
            details["prefilter_score"] = candidate.get("prefilter_score", 0.0)
            described.append(details)

        logger.info(
            "Shortlisted %d of %d Hugging Face candidates for %s",
            len(described),
            len(pooled),
            plan.get("hypothesis_id"),
        )
        return described

    def _appraise_datasets(self, spec_dict: dict, candidates: list[dict]) -> list[dict]:
        """Score each shortlisted candidate, best-first, as plain state dicts.

        The per-candidate loop lives inside this node rather than becoming a
        graph cycle, the same call the fix loop's internal env-repair loop
        makes: the iterations are independent and bounded by a setting, so a
        cycle would add super-steps and a recursion-limit term without making
        anything more observable than the scores this returns.
        """
        if not spec_dict or not candidates:
            return []
        spec = dataset_spec.from_dict(spec_dict)

        scored: list[dataset_scoring.ScoredDataset] = []
        for candidate in candidates[: settings.coder_dataset_max_appraisals]:
            appraised = self._appraise_one(spec, candidate)
            if appraised is not None:
                scored.append(appraised)

        scored.sort(key=lambda entry: entry.score, reverse=True)
        for entry in scored:
            logger.info(
                "Dataset candidate %s scored %.2f (%s)",
                entry.dataset_id,
                entry.score,
                ", ".join(f"{k}={v}" for k, v in entry.components.labels().items()),
            )
        return [dataset_scoring.to_state(entry) for entry in scored]

    def _appraise_one(
        self, spec: dataset_spec.DatasetSpec, candidate: dict
    ) -> dataset_scoring.ScoredDataset | None:
        """One candidate: sample rows, measure them, gather evidence, score.

        Four of the six dimensions never reach the model — license, quality,
        provenance and (after verification) schema fit are all computed here.
        The model is asked only for the two relevance labels and for a mapping
        it does not get the final word on. See dataset_scoring's docstring.
        """
        dataset_id = str(candidate.get("dataset_id") or "")
        if not dataset_id:
            return None

        config = str(candidate.get("config") or "default")
        split = str(candidate.get("split") or "train")
        columns = [column for column in candidate.get("columns") or [] if isinstance(column, dict)]
        column_names = [str(column.get("name")) for column in columns if column.get("name")]

        try:
            rows = self.dataset_rows(
                dataset_id, config, split, settings.coder_dataset_rows_inspected
            )
        except Exception as exc:  # noqa: BLE001 — see this section's header
            logger.warning("Could not sample rows from %s: %s", dataset_id, exc)
            rows = []
        # The three first-rows already fetched by describe_candidate are a
        # smaller sample, not no sample — better to score on them than to drop a
        # candidate because the paged read failed.
        rows = rows or [row for row in candidate.get("sample_rows") or [] if isinstance(row, dict)]

        report = dataset_inspect.inspect_rows(
            rows,
            columns,
            languages=spec.languages,
            desired_examples=spec.desired_examples,
            num_rows_total=int(candidate.get("num_rows") or 0),
        )

        try:
            card = self.dataset_card(dataset_id)
        except Exception as exc:  # noqa: BLE001 — see this section's header
            logger.warning("Could not read the card for %s: %s", dataset_id, exc)
            card = ""

        appraisal = dataset_scoring.coerce_appraisal(
            self._dataset_llm_call(
                prompts.DATASET_EVIDENCE_PROMPT,
                spec,
                dataset_id,
                column_names,
                rows,
                report,
                card,
            )
        )

        license_id = str(candidate.get("license") or "")
        license_band = dataset_scoring.license_label(license_id, spec)
        provenance_band = dataset_scoring.provenance_label(candidate.get("info"))
        schema_band, mapping = dataset_scoring.schema_fit_label(
            appraisal["column_mapping"], spec, column_names
        )
        quality = dataset_scoring.quality_score(report)

        components = dataset_scoring.ScoreComponents(
            task_relevance=appraisal["task_relevance"],
            content_relevance=appraisal["content_relevance"],
            quality=quality,
            provenance=provenance_band,
            schema_fit=schema_band,
            license_fit=license_band,
            evidence={
                "task_relevance": appraisal["task_relevance_evidence"],
                "content_relevance": appraisal["content_relevance_evidence"],
                "quality": dataset_scoring.quality_evidence(report),
                "provenance": dataset_scoring.provenance_evidence(
                    candidate.get("info"), provenance_band
                ),
                "schema_fit": dataset_scoring.schema_evidence(mapping, spec, schema_band),
                "license_fit": dataset_scoring.license_evidence(license_id, license_band),
            },
        )

        scored = dataset_scoring.ScoredDataset(
            dataset_id=dataset_id,
            components=components,
            report=report,
            revision=str(candidate.get("revision") or ""),
            config=config,
            split=split,
            license=license_id,
            size_bytes=int(candidate.get("num_bytes") or 0),
            rows=int(candidate.get("num_rows") or report.num_rows_total),
            columns=columns,
            sample_rows=[row for row in rows[:3] if isinstance(row, dict)],
            column_mapping=mapping,
            requirements=appraisal["requirements"],
            rows_url=huggingface_client.rows_url_for(dataset_id, config, split),
        )
        return dataset_scoring.rescore(scored)

    def _dataset_llm_call(
        self,
        template: str,
        spec: dataset_spec.DatasetSpec,
        dataset_id: str,
        column_names: list[str],
        rows: list[dict],
        report: dataset_inspect.InspectionReport,
        card: str,
    ) -> dict:
        """The evidence and critic prompts share every input, so they share the
        rendering too. Returns {} on any failure — which `coerce_appraisal` and
        `coerce_findings` both read as "nothing was established", the same
        answer an explicit UNKNOWN gives."""
        prompt = template.format(
            spec_block=_compact_json(spec.to_dict()),
            dataset_id=dataset_id,
            columns_block=", ".join(column_names) or "(unknown)",
            rows_sampled=report.rows_sampled,
            sample_block=_compact_json(rows[:_DATASET_PROMPT_SAMPLE_ROWS]),
            inspection_block=_compact_json(report.to_dict()),
            card_block=card or "(no dataset card published)",
        )
        try:
            return self._call_json(prompt)
        except Exception as exc:  # noqa: BLE001 — see this section's header
            logger.warning("Dataset appraisal call for %s failed: %s", dataset_id, exc)
            return {}

    def _critique_leading_dataset(
        self, spec_dict: dict, scores: list[dict]
    ) -> tuple[dict, list[dict]]:
        """Accept the best candidate that survives an adversarial review.

        Candidates are walked best-first. One that already fails on its score,
        license or schema is rejected without spending a critic call on it —
        there is nothing an adversarial pass could find that would change the
        answer. The rest are critiqued until one is accepted or
        `CODER_DATASET_MAX_CRITIC_REVIEWS` is spent; a veto falls through to the
        next candidate rather than ending the search, which is the whole reason
        several candidates are appraised instead of one.
        """
        if not spec_dict or not scores:
            return {}, []
        spec = dataset_spec.from_dict(spec_dict)
        minimum = settings.coder_dataset_min_score

        rejections: list[dict] = []
        oversized_fallback: dict = {}
        reviews = 0
        for payload in scores:
            scored = dataset_scoring.from_state(payload)
            dataset_scoring.decide(scored, minimum)
            if scored.decision == "reject":
                logger.info(
                    "Dataset %s rejected before review: %s",
                    scored.dataset_id,
                    "; ".join(scored.reasons_for_rejection),
                )
                rejections.append(dataset_scoring.to_state(scored))
                continue

            if reviews >= settings.coder_dataset_max_critic_reviews:
                logger.info(
                    "Critic budget spent; %s was not reviewed and is not used", scored.dataset_id
                )
                break
            reviews += 1

            findings = dataset_scoring.coerce_findings(
                self._dataset_llm_call(
                    prompts.DATASET_CRITIC_PROMPT,
                    spec,
                    scored.dataset_id,
                    [str(column.get("name")) for column in scored.columns if column.get("name")],
                    scored.sample_rows,
                    scored.report,
                    self._dataset_card_quietly(scored.dataset_id),
                )
            )
            dataset_scoring.apply_critic(scored, findings, spec)
            dataset_scoring.decide(scored, minimum)

            if scored.decision == "accept":
                if self._too_large_to_download(scored.size_bytes):
                    # Falls through to the next candidate rather than settling
                    # for a dataset nothing can fetch. Barkla job 10410949
                    # accepted CryptoSpartan/stocks_bars_1m at 0.92, found it
                    # 10.2 GB against a 5 GB cap, offered it as a REST URL the
                    # compute node could not reach, and the experiment ran on
                    # synthesized data with its verdict withheld — while
                    # paperswithbacktest/Stocks-Daily-Price sat one rank below
                    # at 0.88 with quality 1.00 and would have downloaded fine.
                    #
                    # Kept as the fallback, though: if nothing smaller is
                    # acceptable, offering this one over REST still beats
                    # offering nothing, which is what happened before downloads
                    # existed at all. It has been through the critic by this
                    # point, so the fallback is a vetted candidate.
                    logger.info(
                        "Dataset %s accepted at %.2f but is too large to download "
                        "(%.1f GB); trying the next candidate",
                        scored.dataset_id,
                        scored.score,
                        scored.size_bytes / _BYTES_PER_GB,
                    )
                    if not oversized_fallback:
                        oversized_fallback = dataset_scoring.to_state(scored)
                    continue
                logger.info(
                    "Dataset %s accepted at %.2f after review", scored.dataset_id, scored.score
                )
                return dataset_scoring.to_state(scored), rejections

            logger.info(
                "Dataset %s vetoed by review: %s",
                scored.dataset_id,
                "; ".join(scored.reasons_for_rejection),
            )
            rejections.append(dataset_scoring.to_state(scored))

        if oversized_fallback:
            logger.info(
                "No acceptable dataset fits the download budget; falling back to %s over REST",
                oversized_fallback.get("dataset_id"),
            )
        return oversized_fallback, rejections

    def _dataset_card_quietly(self, dataset_id: str) -> str:
        try:
            return self.dataset_card(dataset_id)
        except Exception:  # noqa: BLE001 — see this section's header
            return ""

    def _dataset_cache_dir(self) -> Path:
        """Where downloaded datasets live. Shared across every plan and every
        run on this machine and keyed by (repo id, revision) below, so N
        experiments wanting the same dataset cost one copy — the per-experiment
        copies are exactly what would have justified keeping this pipeline
        download-free on shared scratch."""
        if settings.coder_dataset_cache_dir:
            return Path(settings.coder_dataset_cache_dir)
        if settings.coder_data_dir:
            return Path(settings.coder_data_dir) / "hf_datasets"
        return self.experiments_dir / "_hf_datasets"

    def _download_budget_reason(self, size_bytes: int) -> str:
        """Why this dataset may not be downloaded, or "" if it may.

        A dataset whose size the viewer wouldn't report is allowed through: /size
        fails often enough that blocking on it would disable downloading
        broadly, and the row cap plus the post-download measurement below keep
        the run budget honest anyway.
        """
        if not settings.coder_dataset_download:
            return "CODER_DATASET_DOWNLOAD is off"
        if self._datasets_accepted >= settings.coder_dataset_max_accepted_per_run:
            return (
                f"this run has already downloaded {self._datasets_accepted} datasets "
                f"(CODER_DATASET_MAX_ACCEPTED_PER_RUN)"
            )
        per_dataset = settings.coder_dataset_max_download_gb * _BYTES_PER_GB
        if size_bytes and size_bytes > per_dataset:
            return (
                f"{size_bytes / _BYTES_PER_GB:.1f} GB exceeds the "
                f"{settings.coder_dataset_max_download_gb:.1f} GB per-dataset budget"
            )
        run_budget = settings.coder_dataset_run_download_budget_gb * _BYTES_PER_GB
        if self._dataset_bytes_downloaded + size_bytes > run_budget:
            return (
                f"downloading it would pass this run's "
                f"{settings.coder_dataset_run_download_budget_gb:.1f} GB total budget"
            )
        return ""

    def _too_large_to_download(self, size_bytes: int) -> bool:
        """Whether *this candidate* is too big to fetch, given what's left.

        Deliberately candidate-specific. A global reason — downloads switched
        off, the per-run dataset cap already spent — applies equally to every
        candidate, so it is no reason to prefer a different one and this returns
        False for all of them; `_download_budget_reason` still reports it at
        acquisition time. Size is different: a smaller candidate really can
        succeed where a larger one cannot.
        """
        if not settings.coder_dataset_download:
            return False
        if self._datasets_accepted >= settings.coder_dataset_max_accepted_per_run:
            return False
        if not size_bytes:
            # The viewer wouldn't say. Treat as fetchable and let
            # _download_budget_reason decide — refusing an unmeasured candidate
            # would reject most of the Hub on missing metadata.
            return False
        if size_bytes > settings.coder_dataset_max_download_gb * _BYTES_PER_GB:
            return True
        run_budget = settings.coder_dataset_run_download_budget_gb * _BYTES_PER_GB
        return self._dataset_bytes_downloaded + size_bytes > run_budget

    def _acquire_dataset(self, accepted: dict, experiment_dir: Path | None) -> dict:
        """Fetch the accepted dataset and write its provenance record.

        On success `local_path` names a `data.jsonl` the generated experiment
        reads with the standard library and no network at all. On any failure —
        over budget, huggingface_hub absent, the download or the normalization
        failing — the dataset is still offered, via the Dataset Viewer REST URL
        that was the only path before downloading existed. That fallback is the
        reason none of this needs to be reliable to be useful.
        """
        if not accepted:
            return {}
        scored = dataset_scoring.from_state(accepted)
        scored.downloaded_at = dataset_scoring.now_iso()

        reason = self._download_budget_reason(scored.size_bytes)
        if reason:
            logger.info("Not downloading %s: %s", scored.dataset_id, reason)
        else:
            self._download_and_normalize(scored)

        if experiment_dir is not None:
            self._write_dataset_record(scored, experiment_dir)
        return dataset_scoring.to_state(scored)

    def _download_and_normalize(self, scored: dataset_scoring.ScoredDataset) -> None:
        """Download at a pinned revision, flatten to JSONL, charge the budget.

        The revision is pinned so the provenance record names a commit rather
        than "whatever main was that day" — a dataset that changes under a
        finished experiment is a result nobody can reproduce.
        """
        cache_key = f"{scored.dataset_id.replace('/', '__')}@{scored.revision or 'main'}"
        source_dir = self._dataset_cache_dir() / cache_key
        local_file = source_dir / "data.jsonl"

        if local_file.exists():
            logger.info("Reusing cached %s", local_file)
            scored.local_path = str(local_file)
            return

        try:
            downloaded = self.dataset_download(scored.dataset_id, scored.revision, source_dir)
        except Exception as exc:  # noqa: BLE001 — see this section's header
            logger.warning("Download of %s failed: %s", scored.dataset_id, exc)
            return
        if downloaded is None:
            return

        try:
            normalized = self.dataset_normalize(
                Path(downloaded), local_file, settings.coder_dataset_max_local_rows
            )
        except Exception as exc:  # noqa: BLE001 — see this section's header
            logger.warning("Normalizing %s failed: %s", scored.dataset_id, exc)
            return
        if not normalized:
            return

        scored.local_path = str(local_file)
        scored.rows = int(normalized.get("rows_written") or scored.rows)
        self._datasets_accepted += 1
        # Charged from what actually landed on disk, not from what /size
        # predicted: the pre-check can be fooled by a missing or wrong size, and
        # the run budget is the one that has to hold across a whole sweep.
        self._dataset_bytes_downloaded += _directory_bytes(source_dir)
        logger.info(
            "Downloaded %s@%s -> %s (%d rows, %.2f GB used this run)",
            scored.dataset_id,
            scored.revision or "main",
            local_file,
            scored.rows,
            self._dataset_bytes_downloaded / _BYTES_PER_GB,
        )

    @staticmethod
    def _write_dataset_record(scored: dataset_scoring.ScoredDataset, experiment_dir: Path) -> None:
        """The audit trail: why this dataset, scored how, from which commit.

        Sits beside data_provenance.json for the same reason that file exists —
        a number in a paper should be traceable to the data it came from, and
        "the Coder picked something" is not traceability.
        """
        try:
            experiment_dir.mkdir(parents=True, exist_ok=True)
            (experiment_dir / "dataset_provenance.json").write_text(
                json.dumps(scored.as_record(), indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Could not write dataset_provenance.json: %s", exc)

    def _provenance_for(
        self,
        plan: dict,
        network_available: bool,
        hf_dataset: dict | None = None,
        run_py: str = "",
    ) -> list[provenance.DataSource]:
        """Resolve this plan's data inputs to real-or-surrogate.

        Deterministic and cheap (regex plus, when a staging directory is
        configured, one directory walk), so it is called wherever it is needed
        rather than threaded through graph state like the Hugging Face lookup —
        that one is a real network call with its own cache and retry policy, and
        this is a pure function of the plan.
        """
        requirements = provenance.split_requirements(
            (plan.get("data_requirements") or {}).get("source") or "",
            (plan.get("data_requirements") or {}).get("description") or "",
        )
        staging = Path(settings.coder_data_dir) if settings.coder_data_dir else None
        sources = provenance.resolve(
            requirements, staging_dir=staging, network_available=network_available
        )

        # A dataset the Hugging Face lookup found is a real input — but only if
        # the code actually reads it. It is offered, not imposed: the model may
        # decline it and say why in assumptions_made, and check_hf_dataset_usage
        # accepts that. Counting an offered-but-declined dataset as evidence
        # would be exactly the over-claim this gate exists to prevent, so this
        # requires the rendered run.py to name it.
        dataset_id = str((hf_dataset or {}).get("dataset_id") or "")
        local_path = str((hf_dataset or {}).get("local_path") or "")
        # Two ways the code can name the dataset, and both count: the repo id
        # (the REST path, where the id is in the URL) or the local file the
        # download normalized it into. The local file is checked by its bare
        # name as well as its full path, since generated code routinely builds
        # the path from a constant rather than pasting it.
        named = bool(run_py) and (
            (bool(dataset_id) and dataset_id in run_py)
            or (bool(local_path) and (local_path in run_py or Path(local_path).name in run_py))
        )
        if dataset_id and named:
            sources.insert(
                0,
                provenance.DataSource(
                    name=f"Hugging Face dataset {dataset_id}",
                    # A downloaded, revision-pinned file on disk is real_local:
                    # it is present, it is fixed, and re-running the experiment
                    # reads the same bytes. Without a download it stays
                    # real_download — still real, but fetched at run time.
                    kind=(
                        provenance.KIND_REAL_LOCAL if local_path else provenance.KIND_REAL_DOWNLOAD
                    ),
                    uri=str((hf_dataset or {}).get("rows_url") or ""),
                    local_path=local_path,
                    reason=(
                        "selected by the dataset appraisal (score "
                        f"{(hf_dataset or {}).get('score', 0.0)}) and read by the generated code"
                    ),
                ),
            )
        return sources

    @staticmethod
    def _hf_dataset_block(hf_dataset: dict) -> str:
        """Renders the accepted dataset for the codegen/fix prompt.

        Two variants, and which one is used is a fact about what happened rather
        than a setting: if the dataset was downloaded and normalized there is a
        local `data.jsonl` to point at, and if it wasn't there is the Dataset
        Viewer REST URL that was the only option before downloading existed.
        Empty string when nothing was accepted, so a prompt with no dataset
        reads exactly as it did before any of this existed.

        The score and the evidence behind it are included deliberately. The
        model is allowed to decline the dataset (and `check_hf_dataset_usage`
        accepts a declared decline), so it should see how strong the match
        actually is rather than being handed an unqualified "use this".
        """
        dataset_id = hf_dataset.get("dataset_id")
        if not dataset_id:
            return ""

        columns = ", ".join(
            f"{column.get('name')} ({column.get('type', 'unknown')})"
            for column in hf_dataset.get("columns") or []
        )
        config = str(hf_dataset.get("config") or "default")
        split = str(hf_dataset.get("split") or "train")
        local_path = str(hf_dataset.get("local_path") or "")

        header = (
            "A real, public dataset was selected for this experiment's data requirements. It "
            "was scored against a fixed rubric after its rows were inspected, not just matched "
            "by name:\n"
            f"  dataset: {dataset_id}\n"
            f"  revision: {hf_dataset.get('revision') or '(unpinned)'}\n"
            f"  config: {config}\n"
            f"  split: {split}\n"
            f"  score: {hf_dataset.get('score', 0.0)} "
            f"({', '.join(f'{k}={v}' for k, v in _dataset_bands(hf_dataset).items())})\n"
            f"  columns: {columns or '(unknown)'}\n"
            f"  first rows: {_compact_json(hf_dataset.get('sample_rows') or [])}\n\n"
        )

        if local_path:
            return (
                header
                + "It has already been downloaded and flattened to JSON Lines on this machine. "
                "Read it from this exact path:\n"
                f"  {local_path}\n\n" + prompts.HF_LOCAL_DATASET_USAGE_NOTE
            )

        rows_url = hf_dataset.get("rows_url") or huggingface_client.rows_url_for(
            str(dataset_id), config, split
        )
        return (
            header
            + "It was not downloaded, so read it over HTTP with `requests` (already available — "
            "do NOT add the `datasets` package, and do not download the dataset):\n"
            f"  {rows_url}\n\n" + prompts.HF_DATASET_USAGE_NOTE
        )

    def _generate_experiment_files(
        self,
        plan: dict,
        shared_files: dict[str, str],
        network_available: bool,
        shared_infra_warning: str = "",
        hf_dataset: dict | None = None,
        starter_id: str = "",
    ) -> dict:
        prompt = prompts.EXPERIMENT_CODEGEN_PROMPT.format(
            plan_block=_compact_json(plan),
            shared_infra_block=self._shared_infra_block(shared_files, shared_infra_warning),
            hf_dataset_block=self._hf_dataset_block(hf_dataset or {}),
            provenance_block=provenance.prompt_block(self._provenance_for(plan, network_available)),
            starter_block=self._starter_block(starter_id),
            hypothesis_id=plan["hypothesis_id"],
            objective=plan["objective"],
            network_status="available" if network_available else "NOT available",
            network_note=(
                "Fetch real data as normal."
                if network_available
                else "Do not fetch remote data — generate/synthesize a small stand-in dataset instead, and say so clearly in assumptions_made and the README."
            ),
        )
        return self._assemble_generation(
            self._call_sections(prompt, prompts.EXPERIMENT_FIELD_NAMES)
        )

    def _regenerate_with_fix(
        self,
        plan: dict,
        shared_files: dict[str, str],
        previous_generation: dict,
        network_available: bool,
        error_source: str,
        error_text: str,
        shared_infra_warning: str = "",
        hf_dataset: dict | None = None,
        starter_id: str = "",
        stuck_streak: int = 0,
        previous_error_summary: str = "",
    ) -> dict:
        prompt = prompts.EXPERIMENT_CODEGEN_FIX_PROMPT.format(
            hypothesis_id=plan["hypothesis_id"],
            plan_block=_compact_json(plan),
            # Carried into the fix prompt too, not just the first generation: the
            # dataset is often exactly what a data-loading failure needs to be
            # fixed *with*, and re-running the lookup per attempt would spend
            # three more HTTP calls to learn the same thing. Same reasoning for
            # the starter block — it stays grounded in the same worked example
            # across every fix attempt instead of drifting.
            hf_dataset_block=self._hf_dataset_block(hf_dataset or {}),
            # Same block the codegen prompt was given. A fix attempt that no
            # longer knows which inputs are surrogates is free to "fix" a
            # failure by quietly inventing data, which is the outcome the
            # provenance gate exists to prevent.
            provenance_block=provenance.prompt_block(self._provenance_for(plan, network_available)),
            starter_block=self._starter_block(starter_id),
            # Looked up fresh every fix call, unlike the starter/dataset blocks
            # above — it depends on error_source, which can (and often does)
            # change attempt to attempt, so caching it per-plan the way those
            # two are would show the wrong error's patterns after a shift.
            fix_pattern_block=self._fix_pattern_block(error_source),
            shared_infra_block=self._shared_infra_block(shared_files, shared_infra_warning),
            # Shown back in the same delimited format it's being asked to answer
            # in — quoting the previous attempt as escaped JSON would invite the
            # model to answer in kind, which is the failure mode this transport
            # exists to remove.
            previous_sections_block=render_sections(previous_generation.get("run_py_sections", {})),
            stuck_block=self._stuck_block(stuck_streak, previous_error_summary),
            error_source=error_source,
            error_text=error_text,
            network_status="available" if network_available else "NOT available",
            network_note=(
                ""
                if network_available
                else " Do not fetch remote data — synthesize a small stand-in dataset instead, and say so in assumptions_made and the README."
            ),
        )
        return self._assemble_generation(
            self._call_sections(
                prompt, prompts.EXPERIMENT_FIELD_NAMES, temperature=_FIX_TEMPERATURE
            )
        )

    def _self_review(self, plan: dict, run_py: str) -> list[str]:
        """Reads the code back critically before it goes to a shared cluster.
        This is the one place model judgment is used, and only because the
        code cannot be executed here first — everywhere else a real check
        decides."""
        prompt = prompts.EXPERIMENT_SELF_REVIEW_PROMPT.format(
            hypothesis_id=plan["hypothesis_id"],
            plan_block=_compact_json(plan),
            code_block=run_py,
        )
        response = self._call_json(prompt)
        if response.get("looks_correct"):
            return []
        return [str(concern) for concern in response.get("concerns", [])]

    # -- File / summary persistence -------------------------------------------

    @staticmethod
    def _write_files(directory: Path, files: dict[str, str]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (directory / name).write_text(content)

    def _write_summary(self, result: dict, suffix: str = "") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"coder_agent_summary_{timestamp}{suffix}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return path


def run_coder_agent(
    planner_output: dict,
    output_dir: str | Path | None = None,
    *,
    interactive_slurm_review: bool | None = None,
) -> dict:
    """Module-level entry point — the stable call the pipeline (or a manual
    trigger) should use. Builds a default-configured CoderAgent and runs it
    once; use the CoderAgent class directly to reuse one model/config across
    multiple calls.

    `interactive_slurm_review` defaults to None (CODER_INTERACTIVE_SLURM_REVIEW,
    itself defaulting to off) rather than being a plain bool default — every
    caller that doesn't pass it explicitly gets exactly today's behavior. The
    orchestrator and batch.py deliberately never pass it: they run unattended,
    and this setting exists specifically for a direct, attended CLI call (see
    cli.py's --interactive-slurm-review)."""
    return CoderAgent(output_dir=output_dir, interactive_slurm_review=interactive_slurm_review).run(
        planner_output
    )
