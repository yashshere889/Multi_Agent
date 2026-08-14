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

`hypothesis_ids: list[str] | None` (optional) narrows which of those hypotheses
are actually planned — the orchestrator uses it to plan only the hypothesis the
Hypothesis Agent ranked first (`selected_hypothesis_id`), so execution and
paper-writing effort goes to one hypothesis rather than three. It changes
*nothing* about the input contract: the full 3-hypothesis output is still
required and still validated in full, and an id that isn't in it is an error.
Omitted (the default), every hypothesis is planned, exactly as before.

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
      "experiment_plans": [ one entry per *planned* hypothesis, always — even
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

Internals
---------
`run()` executes a LangGraph StateGraph (see graph.py / state.py) rather than a
straight-line method, so every step — each LLM call and each deterministic step
— is its own traceable node, and the per-hypothesis planning fans out
concurrently via `Send` (replacing the previous ThreadPoolExecutor). That's
purely internal: the entry points above, their signatures, the returned dict,
and the raised ExperimentPlannerAgentError are all unchanged.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.experiment_planner import prompts
from research_pipeline.agents.experiment_planner.schema import (
    SchemaValidationError,
    clean_shared_infrastructure,
    priority_order_errors,
    validate_output,
)
from research_pipeline.agents.experiment_planner.state import ExperimentPlannerState
from research_pipeline.agents.hypothesis.schema import SchemaValidationError as HypothesisSchemaValidationError
from research_pipeline.agents.hypothesis.schema import validate_output as validate_hypothesis_output
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import LLMJSONError, invoke_json

logger = logging.getLogger(__name__)


def _coerce_priority_order(priority_order: list, expected_ids: List[str]) -> List[dict]:
    """Deterministically repairs a priority_order that still doesn't match
    expected_ids after a repair-prompt retry — the last line of defense so a
    single hallucinated/dropped hypothesis_id can never crash the whole
    orchestrator run (see _plan_cross_cutting). Keeps whichever entries the
    model got right (an id that's actually in expected_ids), in the model's
    own relative order, then appends any it omitted at the end with a generic
    justification, and renumbers ranks 1..n over the result. Never raises:
    which ids are valid is entirely mechanical here (exactly expected_ids), so
    this is Python deciding it rather than asking the model a third time."""
    expected_set = set(expected_ids)
    seen: set[str] = set()
    kept: List[dict] = []
    for entry in priority_order or []:
        if not isinstance(entry, dict):
            continue
        hid = entry.get("hypothesis_id")
        if hid in expected_set and hid not in seen:
            seen.add(hid)
            kept.append(entry)
    for hid in expected_ids:
        if hid not in seen:
            kept.append(
                {
                    "hypothesis_id": hid,
                    "justification": "Auto-assigned: the model's priority order didn't reference this hypothesis.",
                }
            )
    return [
        {"hypothesis_id": entry["hypothesis_id"], "rank": i + 1, "justification": entry.get("justification", "")}
        for i, entry in enumerate(kept)
    ]


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

    def run(self, hypothesis_output: dict, hypothesis_ids: Optional[List[str]] = None) -> dict:
        """Runs the agent's graph end to end. The graph is an internal detail, so
        callers (the CLI, the orchestrator, another agent) see only this
        signature, the returned dict, and ExperimentPlannerAgentError on failure.

        `hypothesis_ids` narrows *which* hypotheses get planned; the input is
        still validated in full either way. Omitting it plans every hypothesis,
        exactly as this agent always has.

        Node exceptions aren't swallowed by LangGraph, so an
        ExperimentPlannerAgentError raised inside a node propagates out of
        `.invoke` unchanged.
        """
        # Imported here rather than at module scope: graph.py needs this module's
        # ExperimentPlannerAgent for typing, so a top-level import would be circular.
        from research_pipeline.agents.experiment_planner.graph import build_experiment_planner_graph

        graph = build_experiment_planner_graph(self)
        final_state = graph.invoke(
            {"hypothesis_output": hypothesis_output, "hypothesis_ids": hypothesis_ids},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        return final_state["result"]

    # -- Graph nodes -----------------------------------------------------------
    # Thin adapters: each takes the graph state, delegates to the private helper
    # below that does the actual work, and returns only the keys it produces.

    def _node_validate_input(self, state: ExperimentPlannerState) -> dict:
        hypothesis_output = state["hypothesis_output"]
        try:
            validate_hypothesis_output(hypothesis_output)
        except HypothesisSchemaValidationError as exc:
            raise ExperimentPlannerAgentError(
                f"Input doesn't match the Hypothesis Agent's output schema: {exc}"
            ) from exc

        # The *full* hypothesis set is what gets validated, regardless of any
        # filter: this agent's input contract is still "a complete Hypothesis
        # Agent output". The filter below only narrows what's planned from it.
        hypotheses = hypothesis_output["hypotheses"]
        wanted = state.get("hypothesis_ids")
        if wanted is not None:
            available = [h["id"] for h in hypotheses]
            unknown = [hid for hid in wanted if hid not in available]
            if unknown:
                raise ExperimentPlannerAgentError(
                    f"Asked to plan hypothesis id(s) {unknown} that aren't in the input's hypotheses {available}"
                )
            hypotheses = [h for h in hypotheses if h["id"] in wanted]
            if not hypotheses:
                raise ExperimentPlannerAgentError("hypothesis_ids selected none of the input's hypotheses")

        expected_ids = [h["id"] for h in hypotheses]
        logger.info(
            "Planning experiments for %d of %d hypotheses%s",
            len(hypotheses), len(hypothesis_output["hypotheses"]),
            f" ({', '.join(expected_ids)})" if wanted is not None else "",
        )
        return {"hypotheses": hypotheses, "expected_ids": expected_ids}

    def _node_plan_one(self, state: ExperimentPlannerState) -> dict:
        """Runs once per hypothesis — the graph fans this out with one `Send` per
        hypothesis, each carrying its own `current_hypothesis`. The single-element
        list is what the state's operator.add reducer accumulates across branches."""
        hypothesis = state["current_hypothesis"]
        plan = self._plan_one(hypothesis, state["hypothesis_output"])
        plan["hypothesis_id"] = hypothesis["id"]  # never trust the model's echo over the source id
        return {"plans": [plan]}

    def _node_reorder_plans(self, state: ExperimentPlannerState) -> dict:
        # Send branches complete in whatever order they complete in, so the
        # input's hypothesis order is re-derived here rather than inherited.
        plans_by_id: Dict[str, dict] = {plan["hypothesis_id"]: plan for plan in state["plans"]}
        return {"experiment_plans": [plans_by_id[hid] for hid in state["expected_ids"]]}

    def _node_plan_cross_cutting(self, state: ExperimentPlannerState) -> dict:
        cross_cutting = self._plan_cross_cutting(state["experiment_plans"])
        return {
            "shared_infrastructure": cross_cutting.get("shared_infrastructure", []),
            "priority_order": cross_cutting.get("priority_order", []),
        }

    def _node_assemble_and_validate(self, state: ExperimentPlannerState) -> dict:
        expected_ids = state["expected_ids"]
        result: dict = {
            "experiment_plans": state["experiment_plans"],
            "shared_infrastructure": state["shared_infrastructure"],
            "priority_order": state["priority_order"],
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
        return {"result": result}

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
        """Asks the model for shared_infrastructure/priority_order across every
        plan, then makes sure priority_order actually stays inside the plan set
        it was shown — a 2026-08-12 production run crashed the whole
        orchestrator when the model ranked a hypothesis (H2) that had been
        filtered out before this call ever saw it, and schema.validate_output
        correctly rejected the mismatch but nothing downstream could recover
        from it. Two lines of defense before that ever reaches
        _node_assemble_and_validate: one repair-prompt retry (mirrors
        llm_json's JSON-parse repair, but for this semantic constraint), then a
        deterministic coercion if the model repeats the mistake — which
        hypothesis_ids are valid is entirely mechanical here (exactly the ids
        of the plans passed in), so Python decides it rather than asking the
        model a third time. This function therefore never raises on a
        priority_order problem; only a genuinely unparseable response (handled
        by _call_json/invoke_json already) still propagates.

        With exactly one plan — every orchestrated run, since the orchestrator
        always narrows to a single hypothesis before this agent ever runs —
        nothing can be "shared" by definition, so the LLM call is skipped
        entirely rather than trusted to notice that itself. A 2026-08-14
        production run (job 10229968, single-plan) got back
        "H1 and H2 both require the MultiHop-RAG evaluation harness" despite
        planning only H2 and having no H1 in this call at all: nearly a
        verbatim echo of CROSS_CUTTING_PROMPT's own worked example
        ("H1 and H3 both need the MultiHop-RAG eval harness"). That
        fabricated shared_infrastructure then misdirected
        _setup_shared_infrastructure into generating shared code for an
        unrelated task, which is how H2's actual experiment ended up
        importing a dependency (pandas) nothing had reason to declare."""
        expected_ids = [plan["hypothesis_id"] for plan in experiment_plans]
        if len(experiment_plans) == 1:
            return {
                "shared_infrastructure": [],
                "priority_order": [
                    {"hypothesis_id": expected_ids[0], "rank": 1, "justification": "Only plan in this run."}
                ],
            }
        prompt = prompts.CROSS_CUTTING_PROMPT.format(n=len(experiment_plans), plans_block=json.dumps(experiment_plans, indent=2))
        cross_cutting = self._call_json(prompt)
        errors = priority_order_errors(cross_cutting.get("priority_order") or [], expected_ids)

        if errors:
            logger.warning(
                "Cross-cutting priority_order didn't match the planned hypotheses (%s) — "
                "retrying once with a repair prompt",
                "; ".join(errors),
            )
            repair_prompt = prompts.CROSS_CUTTING_REPAIR_PROMPT.format(
                problem="; ".join(errors),
                n=len(expected_ids),
                expected_ids=json.dumps(expected_ids),
                previous_response=json.dumps(cross_cutting)[:4000],
            )
            cross_cutting = self._call_json(repair_prompt)
            errors = priority_order_errors(cross_cutting.get("priority_order") or [], expected_ids)

        if errors:
            logger.warning(
                "Cross-cutting priority_order still didn't match after a repair retry (%s) — "
                "deterministically coercing it instead of failing the run",
                "; ".join(errors),
            )
            cross_cutting["priority_order"] = _coerce_priority_order(
                cross_cutting.get("priority_order") or [], expected_ids
            )

        kept, dropped = clean_shared_infrastructure(cross_cutting.get("shared_infrastructure") or [], expected_ids)
        if dropped:
            logger.warning(
                "Dropped shared_infrastructure entr%s naming a hypothesis id outside this run's plans %s: %s",
                "y" if len(dropped) == 1 else "ies",
                expected_ids,
                dropped,
            )
        cross_cutting["shared_infrastructure"] = kept

        return cross_cutting

    # -- Output persistence ----------------------------------------------------

    def _write_output(self, result: dict, suffix: str = "") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"experiment_plan_{timestamp}{suffix}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return path


def run_experiment_planner_agent(
    hypothesis_output: dict,
    output_dir: Optional[str | Path] = None,
    hypothesis_ids: Optional[List[str]] = None,
) -> dict:
    """Module-level entry point — the stable call a Coder Agent (or the CLI)
    should use. Builds a default-configured ExperimentPlannerAgent and runs it
    once; use the ExperimentPlannerAgent class directly to reuse one
    model/config across multiple calls.

    `hypothesis_ids` is optional and defaults to planning all of them, so
    existing callers are unaffected."""
    return ExperimentPlannerAgent(output_dir=output_dir).run(hypothesis_output, hypothesis_ids=hypothesis_ids)
