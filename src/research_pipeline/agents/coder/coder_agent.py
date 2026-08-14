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
from urllib.parse import quote

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.coder import huggingface_client, prompts, sandbox, slurm_submit
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
    "compile_check",
    "static_lint",
    "missing_data_fallback",
    "self_review",
    "run_experiment",
    "results_json",
]


# LangGraph's default recursion_limit (25 super-steps) is a limit on the graph's
# *shape*, which is the wrong unit for this graph: both of its loops are real
# cycles, so the step count scales with how many plans there are and how large
# the fix budget is. Derived per run from those two numbers rather than guessed —
# the counts below match the loops wired up in graph.py.
_FIXED_STEPS = 5  # validate_input, probe_environment, setup_shared_infrastructure, start_plan_loop, assemble_and_validate
_STEPS_PER_PLAN = 5  # process_current_plan, search_hf_dataset, generate_experiment_code, the first attempt, finalize/give_up
_STEPS_PER_FIX_ATTEMPT = 2  # snapshot_and_regenerate, then the attempt it feeds


# Inputs to _bounded_max_tokens, which stops a long prompt plus a fixed
# max_tokens from overrunning the model's context window.
#
# 4 characters per token is the standard rough heuristic for English + code.
# It's an estimate on purpose: the backend is an arbitrary OpenAI-compatible
# HTTP endpoint, so there is no tokenizer on this side to ask for the real
# count, and pulling one in would tie this agent to one specific model.
_CHARS_PER_TOKEN_ESTIMATE = 4
# Below this many tokens a completion would be too truncated to be usable code
# — mid-function at best. Hitting this floor means the *prompt* is the problem,
# so it's raised rather than attempted and silently wasted.
_MIN_GENERATION_TOKENS = 2048
# Headroom for the estimate being wrong and for provider-side rounding (chat
# templates, role tokens, tool preambles) that never appears in the prompt text
# we can measure here.
_CONTEXT_SAFETY_MARGIN = 512

# Fix-attempt regeneration runs at temperature 0 — the model's most confident
# completion rather than a fresh sample. The fix prompt asks for every section
# back, "keeping whatever already worked", and at a nonzero temperature that
# full-section rewrite has been observed reintroducing *different* bugs each
# round: in one production trace attempt 1 correctly fixed a ModuleNotFoundError
# while attempts 2 and 3 each introduced a new backslash syntax error at a
# different line. Initial generation keeps the constructor's temperature; only
# the regeneration paths use this.
_FIX_TEMPERATURE = 0.0


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE


def _truncate(text: str) -> str:
    return text[:ERROR_SUMMARY_MAX_CHARS]


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
        self.max_fix_attempts = (
            settings.coder_max_fix_attempts if max_fix_attempts is None else max_fix_attempts
        )
        self._slurm_jobs_submitted = 0

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

        graph = build_coder_graph(self)
        final_state = graph.invoke(
            {"planner_output": planner_output, "experiments": [], "slurm_jobs_submitted": 0},
            config={
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": _recursion_limit_for(
                    _plan_count(planner_output), self.max_fix_attempts
                ),
            },
        )
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

        return {
            "current_plan": plan,
            "current_experiment_dir": str(experiment_dir),
            "current_fix_history": [],
            "current_attempt": 0,
            "current_outcome": {},
            "current_hf_dataset": {},
        }

    def _node_search_hf_dataset(self, state: CoderState) -> dict:
        """Looks up one real Hugging Face dataset for this plan's data
        requirements, before any code is generated.

        Its own node rather than a line inside the generation call, for the same
        reason every other step here is a node: the lookup is a real decision
        point with an observable outcome (matched / didn't), and a run where
        experiments quietly stopped using real data should be diagnosable from
        the trace rather than from log archaeology."""
        return {
            "current_hf_dataset": self._find_hf_dataset(
                state["current_plan"], state["network_available"]
            )
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

        complexity = plan["estimated_complexity"]
        requirements_path = experiment_dir / "requirements.txt"

        run_high_locally = (
            complexity == "high"
            and settings.coder_run_high_complexity_when_gpu_available
            and gpu_available
        )
        if (needs_gpu and not gpu_available) or (complexity == "high" and not run_high_locally):
            return self._handle_unrunnable_locally(
                plan, generation, run_py, experiment_dir, requirements_path, complexity
            )

        python_executable, env_error = sandbox.ensure_experiment_env(
            experiment_dir, requirements_path, network_available
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
        succeeded, message = sandbox.run_experiment(
            python_executable, experiment_dir / "run.py", experiment_dir, timeout_seconds
        )
        if not succeeded:
            return {"error_source": "run_experiment", "error_text": f"Execution failed: {message}"}

        results, diagnosis = sandbox.read_results_json_for_diagnosis(experiment_dir)
        if results is None:
            return {
                "error_source": "results_json",
                "error_text": f"run.py exited successfully but did not produce a valid results.json: {diagnosis}",
            }

        return {
            "result": self._result(
                hypothesis_id,
                status="completed",
                reason="",
                code_path=str(experiment_dir),
                assumptions_made=assumptions_made,
                results=results,
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
                )
            }

        if not settings.coder_auto_submit_slurm:
            return leave_for_review("review and submit it yourself with `sbatch run.sbatch`.")

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

    def _find_hf_dataset(self, plan: dict, network_available: bool) -> dict:
        """One real dataset for this plan, or {} — never an exception.

        Skipped without a request when the network probe already failed or
        CODER_ENABLE_HF_DATASET_SEARCH is off. The lookup client already degrades
        every failure to None, so the try/except here is only for an injected
        lookup function that raises: this is an enhancement to a prompt, and a
        broken dataset search must never be the reason an experiment doesn't get
        generated at all.
        """
        if not network_available or not settings.coder_enable_hf_dataset_search:
            return {}
        description = (plan.get("data_requirements") or {}).get("description") or plan["objective"]
        try:
            dataset = self.huggingface_lookup(str(description))
        except Exception as exc:  # noqa: BLE001 — see the docstring
            logger.warning(
                "Hugging Face dataset lookup raised for %s; generating without it: %s",
                plan["hypothesis_id"],
                exc,
            )
            return {}
        return dataset or {}

    @staticmethod
    def _hf_dataset_block(hf_dataset: dict) -> str:
        """Renders a matched dataset for the codegen/fix prompt: what it is, its
        real column names and a few real rows, and the exact REST URL to read it
        from. Empty string when nothing matched, so a prompt with no dataset
        reads exactly as it did before this lookup existed."""
        dataset_id = hf_dataset.get("dataset_id")
        if not dataset_id:
            return ""
        columns = ", ".join(
            f"{column.get('name')} ({column.get('type', 'unknown')})"
            for column in hf_dataset.get("columns") or []
        )
        config = str(hf_dataset.get("config") or "default")
        split = str(hf_dataset.get("split") or "train")
        # safe="" so the namespace slash in "owner/name" is percent-encoded:
        # these are query-parameter *values*, and a bare slash there is what makes
        # a hand-built dataset-viewer URL 404.
        rows_url = (
            f"{huggingface_client.DATASET_VIEWER_ROWS_URL}"
            f"?dataset={quote(str(dataset_id), safe='')}"
            f"&config={quote(config, safe='')}&split={quote(split, safe='')}"
            "&offset=0&length=100"
        )
        return (
            "A real, public dataset matching this experiment's data requirements was found and "
            "verified as servable by the Hugging Face Dataset Viewer:\n"
            f"  dataset: {dataset_id}\n"
            f"  config: {config}\n"
            f"  split: {split}\n"
            f"  columns: {columns or '(unknown)'}\n"
            f"  first rows: {_compact_json(hf_dataset.get('sample_rows') or [])}\n\n"
            "Read it over HTTP with `requests` (already available — do NOT add the `datasets` "
            "package, and do not download the dataset):\n"
            f"  {rows_url}\n\n" + prompts.HF_DATASET_USAGE_NOTE
        )

    def _generate_experiment_files(
        self,
        plan: dict,
        shared_files: dict[str, str],
        network_available: bool,
        shared_infra_warning: str = "",
        hf_dataset: dict | None = None,
    ) -> dict:
        prompt = prompts.EXPERIMENT_CODEGEN_PROMPT.format(
            plan_block=_compact_json(plan),
            shared_infra_block=self._shared_infra_block(shared_files, shared_infra_warning),
            hf_dataset_block=self._hf_dataset_block(hf_dataset or {}),
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
    ) -> dict:
        prompt = prompts.EXPERIMENT_CODEGEN_FIX_PROMPT.format(
            hypothesis_id=plan["hypothesis_id"],
            plan_block=_compact_json(plan),
            # Carried into the fix prompt too, not just the first generation: the
            # dataset is often exactly what a data-loading failure needs to be
            # fixed *with*, and re-running the lookup per attempt would spend
            # three more HTTP calls to learn the same thing.
            hf_dataset_block=self._hf_dataset_block(hf_dataset or {}),
            shared_infra_block=self._shared_infra_block(shared_files, shared_infra_warning),
            # Shown back in the same delimited format it's being asked to answer
            # in — quoting the previous attempt as escaped JSON would invite the
            # model to answer in kind, which is the failure mode this transport
            # exists to remove.
            previous_sections_block=render_sections(previous_generation.get("run_py_sections", {})),
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


def run_coder_agent(planner_output: dict, output_dir: str | Path | None = None) -> dict:
    """Module-level entry point — the stable call the pipeline (or a manual
    trigger) should use. Builds a default-configured CoderAgent and runs it
    once; use the CoderAgent class directly to reuse one model/config across
    multiple calls."""
    return CoderAgent(output_dir=output_dir).run(planner_output)
