"""Hypothesis Agent.

Sits between the Literature Agent and a future Experiment Planner Agent: takes
the papers the Literature Agent found, synthesizes the literature, and
produces exactly 3 grounded, testable hypotheses — all as structured JSON, so
downstream agents can consume it programmatically without parsing prose.

Input contract
--------------
`papers: list[dict]`, each ideally shaped like the Literature Agent's `Paper`
(research_pipeline.agents.literature.state.Paper):
    {"title": str, "authors": list[str], "abstract": str, "year": int | None,
     "source": "arxiv" | "semantic_scholar", "arxiv_id" | "paper_id": str,
     "doi": str | None, ...}

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
      "source_paper_ids": [str, ...],   # papers actually analyzed, after normalization
      "generated_at": "<UTC ISO 8601>",
      "model": str,
    }
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
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.hypothesis import prompts
from research_pipeline.agents.hypothesis.papers import NormalizedPaper, chunk_papers, normalize_papers, paper_to_text
from research_pipeline.agents.hypothesis.schema import SchemaValidationError, validate_output
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import LLMJSONError, invoke_json

logger = logging.getLogger(__name__)


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

    def run(self, papers: List[dict], research_question: Optional[str] = None) -> dict:
        normalized = normalize_papers(papers)
        if not normalized:
            raise HypothesisAgentError(
                "No usable papers were provided (empty list, or every paper was missing "
                "title/abstract/full_text)."
            )

        all_paper_ids = [p["id"] for p in normalized]
        batches = chunk_papers(normalized, self.batch_max_chars)
        logger.info("Analyzing %d paper(s) across %d batch(es)", len(normalized), len(batches))

        partials = [self._analyze_batch(batch) for batch in batches]

        synthesis = self._synthesize(partials, all_paper_ids, research_question)
        literature_summary = synthesis.get("literature_summary", "")
        methods_overview = synthesis.get("methods_overview", [])
        gaps = synthesis.get("gaps", [])

        hypotheses_result = self._generate_hypotheses(literature_summary, methods_overview, gaps, research_question)
        hypotheses = hypotheses_result.get("hypotheses", [])

        result: dict = {
            "literature_summary": literature_summary,
            "methods_overview": methods_overview,
            "gaps": gaps,
            "hypotheses": hypotheses,
            "source_paper_ids": all_paper_ids,
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
        return result

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
        self, literature_summary: str, methods_overview: list, gaps: list, research_question: Optional[str]
    ) -> dict:
        prompt = prompts.HYPOTHESIS_PROMPT.format(
            research_question_line=f"Original research question guiding this search: {research_question}\n\n" if research_question else "",
            literature_summary=literature_summary,
            methods_overview_block=json.dumps(methods_overview, indent=2),
            gaps_block=json.dumps(gaps, indent=2),
        )
        return self._call_json(prompt)

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
) -> dict:
    """Module-level entry point — the stable call an Experiment Planner Agent
    (or the CLI) should use. Builds a default-configured HypothesisAgent and
    runs it once; use the HypothesisAgent class directly if you want to reuse
    one model/config across multiple calls."""
    return HypothesisAgent(output_dir=output_dir).run(papers, research_question=research_question)
