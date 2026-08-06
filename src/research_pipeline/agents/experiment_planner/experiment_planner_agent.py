"""Experiment Planner Agent.

Sits between the Hypothesis Agent and a future Coder Agent: takes the
Hypothesis Agent's output (3 hypotheses + supporting literature synthesis),
judges feasibility per hypothesis, and produces a fully implementation-ready
experiment plan for each — plus cross-cutting shared-infrastructure notes and
a priority order — all as structured JSON a Coder Agent can consume directly.

Input contract
--------------
`hypothesis_output: dict` — the exact output of
`research_pipeline.agents.hypothesis.run_hypothesis_agent` (or
`outputs/hypotheses_<timestamp>.json` loaded from disk). Validated against
`research_pipeline.agents.hypothesis.schema.HypothesisAgentOutput` on entry:
    {
      "literature_summary": str,
      "methods_overview": [{"method", "papers_using_it", "notes"}, ...],
      "gaps": [{"gap", "supporting_evidence", "notes"}, ...],
      "hypotheses": [ exactly 3 x {"id", "statement", "rationale",
                                     "related_gaps", "related_methods",
                                     "suggested_variables": {"independent", "dependent"}} ],
      "source_paper_ids": [...], "generated_at": ..., "model": ...,
    }
If this doesn't validate, ExperimentPlannerAgentError is raised immediately —
planning against a malformed hypothesis set is never attempted.

Planning assumptions
---------------------
Confirmed with the pipeline owner, baked into prompts.py:SYSTEM_PROMPT:
- Compute: a shared university HPC/SLURM GPU cluster — jobs of hours to a few
  days, single/small-multi-GPU scale, not large continuous training runs.
- Data: the literature's source papers are assumed minable for full
  methodological detail, not just title/abstract. This is a *forward-looking*
  assumption — the current Literature Agent only downloads PDFs and never
  parses them (see agents/literature/state.py's `Paper.local_path`), so a PDF
  text-extraction step is implicitly required before a Coder Agent could act
  on any plan step that relies on paper-level detail beyond title/abstract.

Output contract
----------------
A dict matching `agents.experiment_planner.schema.ExperimentPlannerOutput`:
    {
      "experiment_plans": [ one entry per input hypothesis, always — even
          flagged-infeasible hypotheses still get a full (simplified) plan;
          see schema.py:ExperimentPlan for the per-plan shape ],
      "shared_infrastructure": [str, ...],
      "priority_order": [ {"hypothesis_id", "rank", "justification"}, ... ],
      "source_hypothesis_ids": [str, ...],
      "generated_at": "<UTC ISO 8601>",
      "model": str,
    }
Validated (agents.experiment_planner.schema.validate_output) before being
returned — including that every input hypothesis id has a corresponding plan,
and that priority_order/experiment_plans reference exactly the same set of
ids. Also written to `<output_dir>/experiment_plan_<UTC timestamp>.json`. On a
schema validation failure, the raw (invalid) output is still written —
suffixed `_invalid` — for debugging, and ExperimentPlannerAgentError is raised.

Entry points
------------
    from research_pipeline.agents.experiment_planner import run_experiment_planner_agent
    result = run_experiment_planner_agent(hypothesis_output)  # dict from the Hypothesis Agent

Or, to reuse one configured model/output dir across multiple calls:
    agent = ExperimentPlannerAgent()
    result = agent.run(hypothesis_output)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.experiment_planner import prompts
from research_pipeline.agents.experiment_planner.schema import SchemaValidationError, validate_output
from research_pipeline.agents.hypothesis.schema import SchemaValidationError as HypothesisSchemaValidationError
from research_pipeline.agents.hypothesis.schema import validate_output as validate_hypothesis_output
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import LLMJSONError, invoke_json

logger = logging.getLogger(__name__)


class ExperimentPlannerAgentError(RuntimeError):
    """Raised when the agent can't produce schema-valid output, even after retries."""


class ExperimentPlannerAgent:
    def __init__(
        self,
        chat_model: Optional[BaseChatModel] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        # Reuses the pipeline's existing LLM client/config (research_pipeline.llm),
        # same as the Hypothesis Agent, at a low temperature suited to grounded,
        # implementation-precise planning rather than a separate client.
        self.chat_model = chat_model or get_chat_model(temperature=0.1)
        self.output_dir = Path(output_dir or settings.experiment_planner_output_dir)

    def run(self, hypothesis_output: dict) -> dict:
        try:
            validate_hypothesis_output(hypothesis_output)
        except HypothesisSchemaValidationError as exc:
            raise ExperimentPlannerAgentError(
                f"Input doesn't match the Hypothesis Agent's output schema: {exc}"
            ) from exc

        hypotheses = hypothesis_output["hypotheses"]
        expected_ids = [h["id"] for h in hypotheses]
        logger.info("Planning experiments for %d hypotheses", len(hypotheses))

        plans_by_id: Dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=len(hypotheses)) as pool:
            future_to_id = {
                pool.submit(self._plan_one, hypothesis, hypothesis_output): hypothesis["id"] for hypothesis in hypotheses
            }
            for future in as_completed(future_to_id):
                hypothesis_id = future_to_id[future]
                plan = future.result()
                plan["hypothesis_id"] = hypothesis_id  # never trust the model's echo over the source id
                plans_by_id[hypothesis_id] = plan

        # preserve the input's hypothesis order, not futures' completion order
        experiment_plans = [plans_by_id[hid] for hid in expected_ids]

        cross_cutting = self._plan_cross_cutting(experiment_plans)

        result: dict = {
            "experiment_plans": experiment_plans,
            "shared_infrastructure": cross_cutting.get("shared_infrastructure", []),
            "priority_order": cross_cutting.get("priority_order", []),
            "source_hypothesis_ids": expected_ids,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": settings.llm_model,
        }

        try:
            validate_output(result, expected_hypothesis_ids=expected_ids)
        except SchemaValidationError as exc:
            debug_path = self._write_output(result, suffix="_invalid")
            raise ExperimentPlannerAgentError(
                f"Assembled output failed schema validation: {exc}. "
                f"Raw (invalid) output written to {debug_path} for inspection."
            ) from exc

        output_path = self._write_output(result)
        logger.info("Wrote experiment planner output to %s", output_path)
        return result

    # -- LLM calls -----------------------------------------------------------

    def _call_json(self, user_prompt: str) -> dict:
        try:
            return invoke_json(self.chat_model, prompts.SYSTEM_PROMPT, user_prompt)
        except LLMJSONError as exc:
            raise ExperimentPlannerAgentError(str(exc)) from exc

    def _plan_one(self, hypothesis: dict, hypothesis_output: dict) -> dict:
        prompt = prompts.PLAN_PROMPT.format(
            hypothesis_block=json.dumps(hypothesis, indent=2),
            hypothesis_id=hypothesis["id"],
            literature_summary=hypothesis_output["literature_summary"],
            methods_overview_block=json.dumps(hypothesis_output["methods_overview"], indent=2),
            gaps_block=json.dumps(hypothesis_output["gaps"], indent=2),
        )
        return self._call_json(prompt)

    def _plan_cross_cutting(self, experiment_plans: List[dict]) -> dict:
        prompt = prompts.CROSS_CUTTING_PROMPT.format(n=len(experiment_plans), plans_block=json.dumps(experiment_plans, indent=2))
        return self._call_json(prompt)

    # -- Output persistence ----------------------------------------------------

    def _write_output(self, result: dict, suffix: str = "") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"experiment_plan_{timestamp}{suffix}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return path


def run_experiment_planner_agent(hypothesis_output: dict, output_dir: Optional[str | Path] = None) -> dict:
    """Module-level entry point — the stable call a Coder Agent (or the CLI)
    should use. Builds a default-configured ExperimentPlannerAgent and runs it
    once; use the ExperimentPlannerAgent class directly to reuse one
    model/config across multiple calls."""
    return ExperimentPlannerAgent(output_dir=output_dir).run(hypothesis_output)
