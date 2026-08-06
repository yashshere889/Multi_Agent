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

from research_pipeline.agents.coder import prompts, sandbox
from research_pipeline.agents.coder.schema import SchemaValidationError, validate_output
from research_pipeline.agents.experiment_planner.schema import SchemaValidationError as PlannerSchemaValidationError
from research_pipeline.agents.experiment_planner.schema import validate_output as validate_planner_output
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import LLMJSONError, invoke_json

logger = logging.getLogger(__name__)


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
    ) -> None:
        # Reuses the pipeline's existing LLM client/config, same as the
        # Hypothesis and Experiment Planner agents, at a low temperature.
        self.chat_model = chat_model or get_chat_model(temperature=0.1)
        self.experiments_dir = Path(experiments_dir or settings.coder_experiments_dir)
        self.output_dir = Path(output_dir or settings.coder_output_dir)
        self.network_check = network_check or sandbox.has_network_access
        self.gpu_check = gpu_check or sandbox.has_gpu

    def run(self, planner_output: dict) -> dict:
        try:
            validate_planner_output(planner_output)
        except PlannerSchemaValidationError as exc:
            raise CoderAgentError(f"Input doesn't match the Experiment Planner's output schema: {exc}") from exc

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
            return {
                "hypothesis_id": hypothesis_id,
                "status": "skipped",
                "reason": f"Marked infeasible by the Experiment Planner: {plan['feasibility_notes']}",
                "code_path": None,
                "assumptions_made": [],
                "results": None,
            }

        experiment_dir = self.experiments_dir / hypothesis_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        generation = self._generate_experiment_files(plan, shared_files, network_available)
        sections = generation.get("run_py_sections", {})
        assumptions_made = generation.get("assumptions_made", [])
        needs_gpu = bool(generation.get("needs_gpu", False))

        required_sections = ("load_data_function", "build_model_function", "run_experiment_function", "evaluate_function")
        missing_sections = [name for name in required_sections if not (sections.get(name) or "").strip()]
        if missing_sections:
            return {
                "hypothesis_id": hypothesis_id,
                "status": "code_generated_not_run",
                "reason": f"Model response was missing required code section(s): {missing_sections} — run.py was not written or executed.",
                "code_path": str(experiment_dir),
                "assumptions_made": assumptions_made,
                "results": None,
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
                "hypothesis_id": hypothesis_id,
                "status": "code_generated_not_run",
                "reason": f"Generated code has a syntax error, not executed: {compile_error}",
                "code_path": str(experiment_dir),
                "assumptions_made": assumptions_made,
                "results": None,
            }

        complexity = plan["estimated_complexity"]
        requirements_path = experiment_dir / "requirements.txt"

        if complexity == "high" or (needs_gpu and not gpu_available):
            reason = (
                f"estimated_complexity is 'high'"
                if complexity == "high"
                else "experiment needs a GPU, none detected in this environment"
            )
            sbatch_path = experiment_dir / "run.sbatch"
            sbatch_path.write_text(sandbox.render_sbatch_template(hypothesis_id, requirements_path.exists()))
            return {
                "hypothesis_id": hypothesis_id,
                "status": "code_generated_not_run",
                "reason": f"{reason} — generated {sbatch_path.name} instead of running synchronously; review and submit it yourself with `sbatch run.sbatch`.",
                "code_path": str(experiment_dir),
                "assumptions_made": assumptions_made,
                "results": None,
            }

        python_executable, env_error = sandbox.ensure_experiment_env(experiment_dir, requirements_path, network_available)
        if env_error:
            return {
                "hypothesis_id": hypothesis_id,
                "status": "code_generated_not_run",
                "reason": env_error,
                "code_path": str(experiment_dir),
                "assumptions_made": assumptions_made,
                "results": None,
            }

        timeout_seconds = sandbox.TIMEOUT_SECONDS_BY_COMPLEXITY[complexity]
        succeeded, message = sandbox.run_experiment(python_executable, experiment_dir / "run.py", experiment_dir, timeout_seconds)
        if not succeeded:
            return {
                "hypothesis_id": hypothesis_id,
                "status": "code_generated_not_run",
                "reason": f"Execution failed: {message}",
                "code_path": str(experiment_dir),
                "assumptions_made": assumptions_made,
                "results": None,
            }

        results = self._read_results_json(experiment_dir)
        if results is None:
            return {
                "hypothesis_id": hypothesis_id,
                "status": "code_generated_not_run",
                "reason": "run.py exited successfully but did not produce a valid results.json",
                "code_path": str(experiment_dir),
                "assumptions_made": assumptions_made,
                "results": None,
            }

        return {
            "hypothesis_id": hypothesis_id,
            "status": "completed",
            "reason": "",
            "code_path": str(experiment_dir),
            "assumptions_made": assumptions_made,
            "results": results,
        }

    @staticmethod
    def _read_results_json(experiment_dir: Path) -> Optional[dict]:
        results_path = experiment_dir / "results.json"
        if not results_path.exists():
            return None
        try:
            data = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "metrics" not in data or "meets_success_criteria" not in data:
            return None
        data.setdefault("notes", "")
        return data

    # -- LLM calls -----------------------------------------------------------

    def _call_json(self, user_prompt: str) -> dict:
        try:
            return invoke_json(self.chat_model, prompts.SYSTEM_PROMPT, user_prompt)
        except LLMJSONError as exc:
            raise CoderAgentError(str(exc)) from exc

    def _generate_experiment_files(self, plan: dict, shared_files: Dict[str, str], network_available: bool) -> dict:
        if shared_files:
            blocks = "\n\n".join(f"--- experiments/_shared/{name} ---\n{content}" for name, content in shared_files.items())
            shared_infra_block = f"Shared infrastructure already generated for this pipeline:\n\n{blocks}\n\n{prompts.SHARED_IMPORT_NOTE}"
        else:
            shared_infra_block = "No shared infrastructure applies to this experiment — implement it standalone."

        prompt = prompts.EXPERIMENT_CODEGEN_PROMPT.format(
            plan_block=json.dumps(plan, indent=2),
            shared_infra_block=shared_infra_block,
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
