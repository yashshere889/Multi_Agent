"""Hypothesis Agent.

Sits between the Literature (or Interdisciplinary Literature) Agent and the
Experiment Planner Agent: takes the papers found upstream, synthesizes the
literature, produces exactly 3 grounded, testable hypotheses, and ranks them
against each other so one can be taken forward — all as structured JSON, so
downstream agents can consume it programmatically without parsing prose.

Input contract
--------------
`papers: list[dict]`, each ideally shaped like the Literature Agent's `Paper`
(research_pipeline.agents.literature.state.Paper):
    {"title": str, "authors": list[str], "abstract": str, "year": int | None,
     "source": "arxiv" | "semantic_scholar", "arxiv_id" | "paper_id": str,
     "doi": str | None, ...}
This is also exactly the shape the Interdisciplinary Literature Agent's
`papers` key carries, so either agent can feed this one.

`interdisciplinary_context: list | None` (optional) — the Interdisciplinary
Literature Agent's `bridge_insights`. When given, it's shown to the model
alongside the literature synthesis as explicitly-labelled cross-field
inspiration (see prompts.INTERDISCIPLINARY_BLOCK) for both hypothesis
generation and ranking. When omitted, this agent behaves exactly as it did
before the parameter existed — standalone/CLI/disk-chained use is unaffected.

Only `title` and `abstract` are actually used today — the current Literature
Agent doesn't extract full text or per-section content (see
agents/literature/state.py). If a paper dict has a `full_text` (or `sections`)
field, this agent prefers it over the abstract automatically; nothing else
needs to change if the Literature Agent grows that field later.

Malformed or empty papers (missing dict, no title/abstract/full_text) are
logged and dropped rather than raising, as long as at least one usable paper
remains — see agents/hypothesis/papers.py:normalize_papers.

Output contract
----------------
A dict matching `agents.hypothesis.schema.HypothesisAgentOutput`:
    {
      "literature_summary": str,
      "methods_overview": [{"method", "papers_using_it", "notes"}, ...],
      "gaps": [{"gap", "supporting_evidence", "notes"}, ...],
      "hypotheses": [ exactly 3 x {"id", "statement", "rationale",
                                     "related_gaps", "related_methods",
                                     "suggested_variables": {"independent", "dependent"}} ],
      "ranking": [ one per hypothesis x {"hypothesis_id", "rank", "score",
                                          "justification"} ],
      "selected_hypothesis_id": str,    # the id ranked 1
      "source_paper_ids": [str, ...],   # papers actually analyzed, after normalization
      "generated_at": "<UTC ISO 8601>",
      "model": str,
    }
All 3 hypotheses are always generated and always returned — ranking is additive
and decides nothing on its own. `selected_hypothesis_id` is derived in Python
from whichever ranking entry holds rank 1, never read from a field the model
was asked to fill in separately, and the schema enforces that the ranks are a
permutation of 1..3 over exactly the generated ids. What (if anything) narrows
to the selected hypothesis is the caller's choice: the orchestrator plans only
the winner (see orchestrator/nodes.py:run_planner_node), while the CLI and any
direct caller still get everything.
Validated against that schema (agents.hypothesis.schema.validate_output) before
being returned. Also written to `<output_dir>/hypotheses_<UTC timestamp>.json`
so it can be inspected or reused without re-running the agent. On a schema
validation failure, the raw (invalid) output is still written — suffixed
`_invalid` — for debugging, and HypothesisAgentError is raised.

Entry points
------------
    from research_pipeline.agents.hypothesis import run_hypothesis_agent
    result = run_hypothesis_agent(papers)  # papers: list[dict]

Or, to reuse one configured model/output dir across multiple calls (e.g. from
an Experiment Planner Agent that runs this in a loop):
    agent = HypothesisAgent()
    result = agent.run(papers, research_question="...")

Internals
---------
`run()` executes a LangGraph StateGraph (see graph.py / state.py) rather than a
straight-line method, so every step — each LLM call and each deterministic step
— is its own traceable node, and the per-batch analysis fans out concurrently
via `Send`. That's purely internal: the entry points above, their signatures,
the returned dict, and the raised HypothesisAgentError are all unchanged.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.hypothesis import prompts
from research_pipeline.agents.hypothesis.papers import NormalizedPaper, chunk_papers, normalize_papers, paper_to_text
from research_pipeline.agents.hypothesis.schema import (
    HYPOTHESIS_FIELD_NAMES,
    SchemaValidationError,
    validate_output,
)
from research_pipeline.agents.hypothesis.state import HypothesisState
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import LLMJSONError, invoke_json, repair_keys

logger = logging.getLogger(__name__)

# Mirrors the "exactly 3" rule schema.validate_output enforces — used here only
# to decide whether a ranking call is worth making at all.
EXPECTED_HYPOTHESIS_COUNT = 3


class HypothesisAgentError(RuntimeError):
    """Raised when the agent can't produce schema-valid output, even after retries."""


class HypothesisAgent:
    def __init__(
        self,
        chat_model: Optional[BaseChatModel] = None,
        batch_max_chars: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        # Reuses the pipeline's existing LLM client/config (research_pipeline.llm) at
        # a lower temperature suited to grounded extraction/synthesis rather than
        # creating a separate client — see llm.py for how the endpoint is configured.
        self.chat_model = chat_model or get_chat_model(temperature=0.1)
        self.batch_max_chars = batch_max_chars or settings.hypothesis_batch_max_chars
        self.output_dir = Path(output_dir or settings.hypothesis_output_dir)

    def run(
        self,
        papers: List[dict],
        research_question: Optional[str] = None,
        interdisciplinary_context: Optional[List[dict]] = None,
    ) -> dict:
        """Runs the agent's graph end to end. The graph is an internal detail, so
        callers (the CLI, the orchestrator, another agent) see only this
        signature, the returned dict, and HypothesisAgentError on failure.

        `interdisciplinary_context` is the Interdisciplinary Literature Agent's
        `bridge_insights` when one ran upstream; omitting it leaves every prompt
        exactly as it was before that agent existed.

        Node exceptions aren't swallowed by LangGraph, so a HypothesisAgentError
        raised inside a node propagates out of `.invoke` unchanged.
        """
        # Imported here rather than at module scope: graph.py needs this module's
        # HypothesisAgent for typing, so a top-level import would be circular.
        from research_pipeline.agents.hypothesis.graph import build_hypothesis_graph

        graph = build_hypothesis_graph(self)
        final_state = graph.invoke(
            {
                "papers": papers,
                "research_question": research_question,
                "interdisciplinary_context": interdisciplinary_context or [],
            },
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        return final_state["result"]

    # -- Graph nodes -----------------------------------------------------------
    # Thin adapters: each takes the graph state, delegates to the private helper
    # below that does the actual work, and returns only the keys it produces.

    def _node_normalize_and_chunk(self, state: HypothesisState) -> dict:
        normalized = normalize_papers(state["papers"])
        if not normalized:
            raise HypothesisAgentError(
                "No usable papers were provided (empty list, or every paper was missing "
                "title/abstract/full_text)."
            )

        batches = chunk_papers(normalized, self.batch_max_chars)
        logger.info("Analyzing %d paper(s) across %d batch(es)", len(normalized), len(batches))
        return {
            "normalized": normalized,
            "all_paper_ids": [p["id"] for p in normalized],
            "batches": batches,
        }

    def _node_analyze_batch(self, state: HypothesisState) -> dict:
        """Runs once per batch — the graph fans this out with one `Send` per
        batch, each carrying its own `current_batch`. The single-element list is
        what the state's operator.add reducer accumulates across branches."""
        return {"partials": [self._analyze_batch(state["current_batch"])]}

    def _node_synthesize(self, state: HypothesisState) -> dict:
        synthesis = self._synthesize(state["partials"], state["all_paper_ids"], state.get("research_question"))
        return {
            "literature_summary": synthesis.get("literature_summary", ""),
            "methods_overview": synthesis.get("methods_overview", []),
            "gaps": synthesis.get("gaps", []),
        }

    def _node_generate_hypotheses(self, state: HypothesisState) -> dict:
        hypotheses_result = self._generate_hypotheses(
            state["literature_summary"],
            state["methods_overview"],
            state["gaps"],
            state.get("research_question"),
            state.get("interdisciplinary_context"),
        )
        # Repair mangled field names before anything downstream sees them. A
        # Barkla run (job 10334292) lost the whole question because the model
        # wrote `rationationale` on two of three hypotheses — the rationales
        # were complete, only the keys were doubled — and schema validation
        # rejected the assembled output three stages later. Deterministic
        # rename, no extra model call; anything it cannot decide is left for
        # validate_output to reject as before.
        return {
            "hypotheses": [
                repair_keys(hypothesis, HYPOTHESIS_FIELD_NAMES) if isinstance(hypothesis, dict) else hypothesis
                for hypothesis in hypotheses_result.get("hypotheses", [])
            ]
        }

    def _node_rank_hypotheses(self, state: HypothesisState) -> dict:
        """Scores the 3 hypotheses against each other and names the winner.

        The winner is *derived* from the ranking (the entry holding rank 1)
        rather than asked for separately, so the two can't disagree — and a
        ranking that isn't a clean permutation of 1..3 over the generated ids is
        rejected by schema.validate_output rather than papered over here.
        """
        hypotheses = state["hypotheses"]
        if len(hypotheses) != EXPECTED_HYPOTHESIS_COUNT:
            # Nothing worth an LLM call: assemble_and_validate is about to
            # reject this output for the count alone, and ranking a broken set
            # would only add noise to the error it reports.
            logger.warning(
                "Skipping ranking: expected %d hypotheses, got %d", EXPECTED_HYPOTHESIS_COUNT, len(hypotheses)
            )
            return {"ranking": [], "selected_hypothesis_id": ""}

        ranking = self._rank_hypotheses(
            hypotheses,
            state["literature_summary"],
            state["gaps"],
            state.get("research_question"),
            state.get("interdisciplinary_context"),
        ).get("ranking", [])

        selected = next(
            (entry.get("hypothesis_id", "") for entry in ranking if isinstance(entry, dict) and entry.get("rank") == 1),
            "",
        )
        logger.info("Ranked %d hypotheses; selected %s", len(ranking), selected or "(none — ranking was malformed)")
        return {"ranking": ranking, "selected_hypothesis_id": selected}

    def _node_assemble_and_validate(self, state: HypothesisState) -> dict:
        result: dict = {
            "literature_summary": state["literature_summary"],
            "methods_overview": state["methods_overview"],
            "gaps": state["gaps"],
            "hypotheses": state["hypotheses"],
            "ranking": state["ranking"],
            "selected_hypothesis_id": state["selected_hypothesis_id"],
            "source_paper_ids": state["all_paper_ids"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": settings.llm_model,
        }

        try:
            validate_output(result)
        except SchemaValidationError as exc:
            debug_path = self._write_output(result, suffix="_invalid")
            raise HypothesisAgentError(
                f"Assembled output failed schema validation: {exc}. "
                f"Raw (invalid) output written to {debug_path} for inspection."
            ) from exc

        output_path = self._write_output(result)
        logger.info("Wrote hypothesis agent output to %s", output_path)
        return {"result": result}

    # -- LLM calls -----------------------------------------------------------

    def _call_json(self, user_prompt: str) -> dict:
        try:
            return invoke_json(self.chat_model, prompts.SYSTEM_PROMPT, user_prompt)
        except LLMJSONError as exc:
            raise HypothesisAgentError(str(exc)) from exc

    def _analyze_batch(self, batch: List[NormalizedPaper]) -> dict:
        papers_block = "\n\n".join(paper_to_text(p) for p in batch)
        return self._call_json(prompts.BATCH_ANALYSIS_PROMPT.format(papers_block=papers_block))

    def _synthesize(self, partials: List[dict], all_paper_ids: List[str], research_question: Optional[str]) -> dict:
        prompt = prompts.SYNTHESIS_PROMPT.format(
            research_question_line=f"Original research question guiding this search: {research_question}\n\n" if research_question else "",
            partials_block=json.dumps(partials, indent=2),
            all_paper_ids=", ".join(all_paper_ids),
        )
        return self._call_json(prompt)

    def _generate_hypotheses(
        self,
        literature_summary: str,
        methods_overview: list,
        gaps: list,
        research_question: Optional[str],
        interdisciplinary_context: Optional[list] = None,
    ) -> dict:
        prompt = prompts.HYPOTHESIS_PROMPT.format(
            research_question_line=f"Original research question guiding this search: {research_question}\n\n" if research_question else "",
            literature_summary=literature_summary,
            methods_overview_block=json.dumps(methods_overview, indent=2),
            gaps_block=json.dumps(gaps, indent=2),
            interdisciplinary_block=self._interdisciplinary_block(interdisciplinary_context),
        )
        return self._call_json(prompt)

    def _rank_hypotheses(
        self,
        hypotheses: list,
        literature_summary: str,
        gaps: list,
        research_question: Optional[str],
        interdisciplinary_context: Optional[list] = None,
    ) -> dict:
        prompt = prompts.RANKING_PROMPT.format(
            research_question_line=f"Original research question guiding this search: {research_question}\n\n" if research_question else "",
            hypotheses_block=json.dumps(hypotheses, indent=2),
            literature_summary=literature_summary,
            gaps_block=json.dumps(gaps, indent=2),
            interdisciplinary_block=self._interdisciplinary_block(interdisciplinary_context),
        )
        return self._call_json(prompt)

    @staticmethod
    def _interdisciplinary_block(interdisciplinary_context: Optional[list]) -> str:
        """Empty string when no Interdisciplinary Literature Agent ran upstream,
        so the prompts are byte-identical to what they were before that agent
        existed."""
        if not interdisciplinary_context:
            return ""
        return prompts.INTERDISCIPLINARY_BLOCK.format(
            bridge_insights_block=json.dumps(interdisciplinary_context, indent=2)
        )

    # -- Output persistence ----------------------------------------------------

    def _write_output(self, result: dict, suffix: str = "") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"hypotheses_{timestamp}{suffix}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return path


def run_hypothesis_agent(
    papers: List[dict],
    research_question: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    interdisciplinary_context: Optional[List[dict]] = None,
) -> dict:
    """Module-level entry point — the stable call an Experiment Planner Agent
    (or the CLI) should use. Builds a default-configured HypothesisAgent and
    runs it once; use the HypothesisAgent class directly if you want to reuse
    one model/config across multiple calls.

    `interdisciplinary_context` is optional and defaults to nothing, so callers
    that don't run an Interdisciplinary Literature Agent are unaffected."""
    return HypothesisAgent(output_dir=output_dir).run(
        papers,
        research_question=research_question,
        interdisciplinary_context=interdisciplinary_context,
    )
