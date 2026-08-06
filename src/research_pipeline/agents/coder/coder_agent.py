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
- `estimated_complexity == "high"`, or an experiment that self-reports
  `needs_gpu: true` when no GPU is detected in this environment: code is
  still generated and syntax-checked, but never run synchronously. A SLURM
  `run.sbatch` template is generated instead (matching
  scripts/slurm/run_llm_server.sbatch's style) — status becomes
  "code_generated_not_run". The agent NEVER submits this job itself; it's
  left for a human to review and `sbatch run.sbatch` manually.
- `estimated_complexity in {"low", "medium"}` (and no GPU requirement, or a
  GPU is actually available): run synchronously in-process with a bounded
  timeout (research_pipeline.agents.coder.sandbox.TIMEOUT_SECONDS_BY_COMPLEXITY),
  in an isolated `uv venv` if the generated requirements.txt needs packages
  not already importable — the shared pipeline environment is never touched.
  Network access and GPU presence are probed at runtime (not hardcoded), so
  the same code adapts whether this runs on a laptop or a Barkla compute node.

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
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.coder import prompts, sandbox, slurm_submit
from research_pipeline.agents.coder.schema import ERROR_SUMMARY_MAX_CHARS, SchemaValidationError, validate_output
from research_pipeline.agents.experiment_planner.schema import SchemaValidationError as PlannerSchemaValidationError
from research_pipeline.agents.experiment_planner.schema import validate_output as validate_planner_output
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import LLMJSONError, invoke_json

logger = logging.getLogger(__name__)

# Order the checks run in, used to decide whether a regeneration actually got
# past the failure it was asked to fix.
_ERROR_STAGE_ORDER = [
    "missing_sections",
    "compile_check",
    "static_lint",
    "self_review",
    "run_experiment",
    "results_json",
]


def _truncate(text: str) -> str:
    return text[:ERROR_SUMMARY_MAX_CHARS]


class CoderAgentError(RuntimeError):
    """Raised when the agent can't produce schema-valid output, even after retries."""


class CoderAgent:
    def __init__(
        self,
        chat_model: Optional[BaseChatModel] = None,
        experiments_dir: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        network_check: Optional[Callable[[], bool]] = None,
        gpu_check: Optional[Callable[[], bool]] = None,
        max_fix_attempts: Optional[int] = None,
    ) -> None:
        # Reuses the pipeline's existing LLM client/config, same as the
        # Hypothesis and Experiment Planner agents, at a low temperature.
        self.chat_model = chat_model or get_chat_model(temperature=0.1)
        self.experiments_dir = Path(experiments_dir or settings.coder_experiments_dir)
        self.output_dir = Path(output_dir or settings.coder_output_dir)
        self.network_check = network_check or sandbox.has_network_access
        self.gpu_check = gpu_check or sandbox.has_gpu
        self.max_fix_attempts = settings.coder_max_fix_attempts if max_fix_attempts is None else max_fix_attempts
        self._slurm_jobs_submitted = 0

    def run(self, planner_output: dict) -> dict:
        try:
            validate_planner_output(planner_output)
        except PlannerSchemaValidationError as exc:
            raise CoderAgentError(f"Input doesn't match the Experiment Planner's output schema: {exc}") from exc

        self._slurm_jobs_submitted = 0
        plans = planner_output["experiment_plans"]
        expected_ids = [p["hypothesis_id"] for p in plans]
        ordered_plans = self._order_by_priority(plans, planner_output["priority_order"])

        network_available = self.network_check()
        gpu_available = self.gpu_check()
        logger.info(
            "Processing %d experiment plan(s); network_available=%s, gpu_available=%s",
            len(ordered_plans), network_available, gpu_available,
        )

        shared_dir, shared_files = self._setup_shared_infrastructure(planner_output, ordered_plans)

        experiments: List[dict] = []
        for plan in ordered_plans:
            experiments.append(
                self._process_plan(plan, shared_files, network_available=network_available, gpu_available=gpu_available)
            )

        result: dict = {
            "experiments": experiments,
            "shared_infrastructure_path": str(shared_dir),
            "source_hypothesis_ids": expected_ids,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": settings.llm_model,
        }

        try:
            validate_output(result, expected_hypothesis_ids=expected_ids)
        except SchemaValidationError as exc:
            debug_path = self._write_summary(result, suffix="_invalid")
            raise CoderAgentError(
                f"Assembled output failed schema validation: {exc}. "
                f"Raw (invalid) output written to {debug_path} for inspection. "
                f"Generated experiment code on disk under {self.experiments_dir} is unaffected."
            ) from exc

        summary_path = self._write_summary(result)
        logger.info("Wrote coder agent summary to %s", summary_path)
        return result

    # -- Ordering ---------------------------------------------------------------

    @staticmethod
    def _order_by_priority(plans: List[dict], priority_order: List[dict]) -> List[dict]:
        rank_by_id = {entry["hypothesis_id"]: entry["rank"] for entry in priority_order}
        # plans missing from priority_order (shouldn't happen given a validated
        # planner output) sort after every ranked plan, in their original order
        return sorted(plans, key=lambda p: rank_by_id.get(p["hypothesis_id"], len(plans) + 1))

    # -- Shared infrastructure -----------------------------------------------

    def _setup_shared_infrastructure(self, planner_output: dict, plans: List[dict]) -> tuple[Path, Dict[str, str]]:
        shared_dir = self.experiments_dir / "_shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        (self.experiments_dir / "__init__.py").touch()
        (shared_dir / "__init__.py").touch()

        shared_items = planner_output.get("shared_infrastructure") or []
        if not shared_items:
            return shared_dir, {}

        prompt = prompts.SHARED_INFRA_PROMPT.format(
            shared_items_block=json.dumps(shared_items, indent=2),
            plans_block=json.dumps(plans, indent=2),
        )
        response = self._call_json(prompt)
        files = response.get("files", {})
        self._write_files(shared_dir, files)
        logger.info("Wrote %d shared infrastructure file(s) to %s", len(files), shared_dir)
        return shared_dir, {name: content for name, content in files.items() if name.endswith(".py")}

    # -- Per-experiment processing --------------------------------------------

    def _process_plan(self, plan: dict, shared_files: Dict[str, str], network_available: bool, gpu_available: bool) -> dict:
        hypothesis_id = plan["hypothesis_id"]

        if not plan["feasible"]:
            logger.info("Skipping %s: marked infeasible by the Experiment Planner", hypothesis_id)
            return self._result(
                hypothesis_id,
                status="skipped",
                reason=f"Marked infeasible by the Experiment Planner: {plan['feasibility_notes']}",
                code_path=None,
            )

        experiment_dir = self.experiments_dir / hypothesis_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        generation = self._generate_experiment_files(plan, shared_files, network_available)
        return self._generate_run_and_diagnose(
            plan, shared_files, generation, experiment_dir, network_available, gpu_available
        )

    def _generate_run_and_diagnose(
        self,
        plan: dict,
        shared_files: Dict[str, str],
        generation: dict,
        experiment_dir: Path,
        network_available: bool,
        gpu_available: bool,
    ) -> dict:
        """Owns validate -> compile -> lint -> execute -> read-results for one
        plan, regenerating the code against the concrete error whenever a
        deterministic check fails, up to self.max_fix_attempts times. Every
        stop condition is a real check's own verdict — no model judgment
        decides whether the code is good enough."""
        hypothesis_id = plan["hypothesis_id"]
        fix_history: List[dict] = []
        outcome: dict = {}

        for attempt in range(self.max_fix_attempts + 1):
            outcome = self._attempt_once(plan, generation, experiment_dir, network_available, gpu_available)

            if fix_history:
                fix_history[-1]["resolved"] = self._cleared_previous_error(fix_history[-1]["error_source"], outcome)

            if "result" in outcome:
                return {**outcome["result"], "fix_attempts": len(fix_history), "fix_history": fix_history}

            if attempt == self.max_fix_attempts:
                break

            logger.info(
                "Fixing %s after %s failure (attempt %d/%d)",
                hypothesis_id, outcome["error_source"], attempt + 1, self.max_fix_attempts,
            )
            fix_history.append(
                {
                    "attempt": attempt + 1,
                    "error_source": outcome["error_source"],
                    "error_summary": _truncate(outcome["error_text"]),
                    "code_path": str(self._snapshot_attempt(experiment_dir, attempt + 1)),
                    "resolved": False,
                }
            )
            generation = self._regenerate_with_fix(
                plan, shared_files, generation, network_available, outcome["error_source"], outcome["error_text"]
            )

        attempted = f" after {len(fix_history)} fix attempt(s)" if fix_history else ""
        return {
            **self._result(
                hypothesis_id,
                status="code_generated_not_run",
                reason=f"{outcome['error_text']}{attempted}",
                code_path=str(experiment_dir),
                assumptions_made=generation.get("assumptions_made", []),
            ),
            "fix_attempts": len(fix_history),
            "fix_history": fix_history,
        }

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
        sections = generation.get("run_py_sections", {})
        assumptions_made = generation.get("assumptions_made", [])
        needs_gpu = bool(generation.get("needs_gpu", False))

        required_sections = ("load_data_function", "build_model_function", "run_experiment_function", "evaluate_function")
        missing_sections = [name for name in required_sections if not (sections.get(name) or "").strip()]
        if missing_sections:
            return {
                "error_source": "missing_sections",
                "error_text": f"Model response was missing required code section(s): {missing_sections} — run.py was not written or executed.",
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

        files = {
            "run.py": run_py,
            "README.md": generation.get("readme", ""),
            "requirements.txt": generation.get("requirements_txt", ""),
        }
        self._write_files(experiment_dir, files)

        compile_error = sandbox.compile_check([experiment_dir / "run.py"])
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

        complexity = plan["estimated_complexity"]
        requirements_path = experiment_dir / "requirements.txt"

        if complexity == "high" or (needs_gpu and not gpu_available):
            return self._handle_unrunnable_locally(
                plan, generation, run_py, experiment_dir, requirements_path, complexity
            )

        python_executable, env_error = sandbox.ensure_experiment_env(experiment_dir, requirements_path, network_available)
        if env_error:
            # Not routed through the fix loop: a missing package or an
            # unreachable index is an environment problem, and regenerating
            # the code can't resolve it.
            return {
                "result": self._result(
                    hypothesis_id,
                    status="code_generated_not_run",
                    reason=env_error,
                    code_path=str(experiment_dir),
                    assumptions_made=assumptions_made,
                )
            }

        timeout_seconds = sandbox.TIMEOUT_SECONDS_BY_COMPLEXITY[complexity]
        succeeded, message = sandbox.run_experiment(python_executable, experiment_dir / "run.py", experiment_dir, timeout_seconds)
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
        sbatch_path.write_text(sandbox.render_sbatch_template(hypothesis_id, requirements_path.exists()))

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
            return {"error_source": "self_review", "error_text": "Pre-submission review raised: " + "; ".join(concerns)}

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
            return leave_for_review(f"auto-submission failed ({submit_error}) — submit it yourself once resolved.")

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
        code_path: Optional[str],
        assumptions_made: Optional[List[str]] = None,
        results: Optional[dict] = None,
        slurm_job_id: Optional[str] = None,
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

    def _call_json(self, user_prompt: str) -> dict:
        try:
            return invoke_json(self.chat_model, prompts.SYSTEM_PROMPT, user_prompt)
        except LLMJSONError as exc:
            raise CoderAgentError(str(exc)) from exc

    @staticmethod
    def _shared_infra_block(shared_files: Dict[str, str]) -> str:
        if not shared_files:
            return "No shared infrastructure applies to this experiment — implement it standalone."
        blocks = "\n\n".join(f"--- experiments/_shared/{name} ---\n{content}" for name, content in shared_files.items())
        return f"Shared infrastructure already generated for this pipeline:\n\n{blocks}\n\n{prompts.SHARED_IMPORT_NOTE}"

    def _generate_experiment_files(self, plan: dict, shared_files: Dict[str, str], network_available: bool) -> dict:
        prompt = prompts.EXPERIMENT_CODEGEN_PROMPT.format(
            plan_block=json.dumps(plan, indent=2),
            shared_infra_block=self._shared_infra_block(shared_files),
            hypothesis_id=plan["hypothesis_id"],
            objective=plan["objective"],
            network_status="available" if network_available else "NOT available",
            network_note=(
                "Fetch real data as normal."
                if network_available
                else "Do not fetch remote data — generate/synthesize a small stand-in dataset instead, and say so clearly in assumptions_made and the README."
            ),
        )
        return self._call_json(prompt)

    def _regenerate_with_fix(
        self,
        plan: dict,
        shared_files: Dict[str, str],
        previous_generation: dict,
        network_available: bool,
        error_source: str,
        error_text: str,
    ) -> dict:
        prompt = prompts.EXPERIMENT_CODEGEN_FIX_PROMPT.format(
            hypothesis_id=plan["hypothesis_id"],
            plan_block=json.dumps(plan, indent=2),
            shared_infra_block=self._shared_infra_block(shared_files),
            previous_sections_block=json.dumps(previous_generation.get("run_py_sections", {}), indent=2),
            error_source=error_source,
            error_text=error_text,
            network_status="available" if network_available else "NOT available",
            network_note=(
                ""
                if network_available
                else " Do not fetch remote data — synthesize a small stand-in dataset instead, and say so in assumptions_made and the README."
            ),
        )
        return self._call_json(prompt)

    def _self_review(self, plan: dict, run_py: str) -> List[str]:
        """Reads the code back critically before it goes to a shared cluster.
        This is the one place model judgment is used, and only because the
        code cannot be executed here first — everywhere else a real check
        decides."""
        prompt = prompts.EXPERIMENT_SELF_REVIEW_PROMPT.format(
            hypothesis_id=plan["hypothesis_id"],
            plan_block=json.dumps(plan, indent=2),
            code_block=run_py,
        )
        response = self._call_json(prompt)
        if response.get("looks_correct"):
            return []
        return [str(concern) for concern in response.get("concerns", [])]

    # -- File / summary persistence -------------------------------------------

    @staticmethod
    def _write_files(directory: Path, files: Dict[str, str]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (directory / name).write_text(content)

    def _write_summary(self, result: dict, suffix: str = "") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"coder_agent_summary_{timestamp}{suffix}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return path


def run_coder_agent(planner_output: dict, output_dir: Optional[str | Path] = None) -> dict:
    """Module-level entry point — the stable call the pipeline (or a manual
    trigger) should use. Builds a default-configured CoderAgent and runs it
    once; use the CoderAgent class directly to reuse one model/config across
    multiple calls."""
    return CoderAgent(output_dir=output_dir).run(planner_output)
