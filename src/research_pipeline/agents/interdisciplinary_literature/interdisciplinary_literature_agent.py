"""Interdisciplinary Literature Agent.

Sits between the Literature Agent and the Hypothesis Agent: takes the papers
the Literature Agent found in one field, works out which *adjacent* fields
could inform the same problem, searches those fields with the same arXiv /
Semantic Scholar / CORE clients the Literature Agent uses, and hands the
Hypothesis Agent a cross-pollinated paper pool plus explicit "bridge insights"
tying the cross-field work back to the core problem — all as structured JSON.

Input contract
--------------
`papers: list[dict]`, each ideally shaped like the Literature Agent's `Paper`
(research_pipeline.agents.literature.state.Paper) — exactly the same input the
Hypothesis Agent accepts today, so this agent drops into the chain without the
Literature Agent changing at all:
    {"title": str, "authors": list[str], "abstract": str, "year": int | None,
     "source": "arxiv" | "semantic_scholar" | "core", "arxiv_id" | "paper_id": str,
     "doi": str | None, ...}

Malformed or empty papers are logged and dropped rather than raising (see
agents/hypothesis/papers.py:normalize_papers, reused here so paper IDs match
what the Hypothesis Agent will derive from the same list), as long as at least
one usable paper remains.

Output contract
----------------
A dict matching `agents.interdisciplinary_literature.schema.InterdisciplinaryLiteratureOutput`:
    {
      "papers": [...],              # in-domain papers + cross_field_papers, deduped —
                                    #   ready to hand straight to run_hypothesis_agent
      "core_paper_ids": [str, ...], # the in-domain papers this agent started from
      "cross_field_papers": [...],  # subset of "papers", each carrying "discipline"
      "fields_explored": [{"field", "rationale", "queries"}, ...],
      "bridge_insights": [{"insight", "source_field",
                            "connection_to_core_problem", "supporting_paper_ids"}, ...],
      "research_question": str | None,
      "generated_at": "<UTC ISO 8601>",
      "model": str,
    }
Validated against that schema (…schema.validate_output) before being returned.
Also written to `<output_dir>/interdisciplinary_<UTC timestamp>.json` so it can
be inspected or reused without re-running the agent. On a schema validation
failure, the raw (invalid) output is still written — suffixed `_invalid` — for
debugging, and InterdisciplinaryLiteratureAgentError is raised.

Three things are deliberately *not* left to the model. Deduplication of the
cross-field results (against each other and against the in-domain papers) is
done in Python on the Literature Agent's own doi/normalized-title key, not by
asking the model which papers are the same. And `supporting_paper_ids` on every
bridge insight is filtered against the ids actually present in the merged pool,
so an id the model invented is dropped and logged rather than handed downstream
— the same rule the Writer Agent applies to citations. And which cross-field
papers survive the transferability screen is a threshold applied in Python
(agents/literature/relevance.py, shared with the Literature Agent's own screen):
the model returns one 0-5 score per paper and never sees the threshold, let
alone decides what to keep.

The in-domain papers are deliberately *not* re-screened — they arrive already
screened against the same research question, and re-filtering another agent's
validated output would make this agent silently lossy on its own input.

Entry points
------------
    from research_pipeline.agents.interdisciplinary_literature import (
        run_interdisciplinary_literature_agent,
    )
    result = run_interdisciplinary_literature_agent(papers, research_question="...")
    hyp = run_hypothesis_agent(result["papers"],
                               research_question=result["research_question"],
                               interdisciplinary_context=result["bridge_insights"])

Or, to reuse one configured model/output dir across multiple calls:
    agent = InterdisciplinaryLiteratureAgent()
    result = agent.run(papers, research_question="...")

Internals
---------
`run()` executes a LangGraph StateGraph (see graph.py / state.py) rather than a
straight-line method, so every step — each LLM call, each search, each
deterministic merge — is its own traceable node, and the per-field search fans
out concurrently via `Send`. That's purely internal: the entry points above,
their signatures, the returned dict, and the raised
InterdisciplinaryLiteratureAgentError are all unchanged by it.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.hypothesis.papers import normalize_papers
from research_pipeline.agents.interdisciplinary_literature import prompts
from research_pipeline.agents.interdisciplinary_literature.schema import SchemaValidationError, validate_output
from research_pipeline.agents.interdisciplinary_literature.state import InterdisciplinaryState
from research_pipeline.agents.literature.clients import search_arxiv, search_core, search_semantic_scholar

# Reused rather than re-implemented, exactly like _normalize_title below: "is
# this paper worth keeping?" gets one answer in this pipeline. Only the rubric
# differs here — cross-field papers are screened on whether their method could
# transfer, not on topical overlap, since being topically distant is the whole
# point of finding them.
from research_pipeline.agents.literature.relevance import (
    TRANSFER_RELEVANCE_CRITERION,
    apply_threshold,
    score_papers,
)

# The Literature Agent's own dedupe key, imported rather than re-implemented so
# "are these the same paper?" can only ever be answered one way in this
# pipeline. Copying it here would be exactly the kind of duplicate that drifts.
from research_pipeline.agents.literature.nodes import dedupe_key as _dedupe_key
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import LLMJSONError, invoke_json

logger = logging.getLogger(__name__)

# How much of each paper goes into a prompt digest. Abstracts are truncated and
# the paper count is capped so field identification stays one bounded call
# regardless of how many papers the Literature Agent found — this agent never
# needs the whole corpus in context, only enough of it to name adjacent fields.
DIGEST_MAX_PAPERS = 25
DIGEST_ABSTRACT_CHARS = 400


class InterdisciplinaryLiteratureAgentError(RuntimeError):
    """Raised when the agent can't produce schema-valid output, even after retries."""


def _paper_id(paper: dict, index: int) -> str:
    """The same id scheme agents/hypothesis/papers.py:normalize_paper uses, so
    ids in `core_paper_ids`/`supporting_paper_ids` line up with the ids the
    Hypothesis Agent will derive from the very same merged list."""
    return str(paper.get("arxiv_id") or paper.get("paper_id") or paper.get("doi") or f"paper_{index}")


def _digest(papers: List[dict], start_index: int = 0, limit: int = DIGEST_MAX_PAPERS) -> str:
    """Renders papers for a prompt, tagged with the id they carry at
    `start_index + i` in the merged pool — so an id the model cites back is
    directly resolvable against that pool."""
    lines = []
    for index, paper in enumerate(papers[:limit]):
        abstract = (paper.get("abstract") or "").strip().replace("\n", " ")
        if len(abstract) > DIGEST_ABSTRACT_CHARS:
            abstract = abstract[:DIGEST_ABSTRACT_CHARS] + "…"
        year = paper.get("year") if paper.get("year") is not None else "n.d."
        tag = f" [field: {paper['discipline']}]" if paper.get("discipline") else ""
        lines.append(
            f"[{_paper_id(paper, start_index + index)}] {paper.get('title') or '(untitled)'} ({year}){tag}\n"
            f"{abstract or '(no abstract available)'}"
        )
    return "\n\n".join(lines)


class InterdisciplinaryLiteratureAgent:
    def __init__(
        self,
        chat_model: Optional[BaseChatModel] = None,
        max_fields: Optional[int] = None,
        max_results_per_query: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
        search_arxiv_fn: Optional[Callable[[List[str], int], List[dict]]] = None,
        search_semantic_scholar_fn: Optional[Callable[[List[str], int], List[dict]]] = None,
        search_core_fn: Optional[Callable[[List[str], int], List[dict]]] = None,
    ) -> None:
        # Reuses the pipeline's existing LLM client/config (research_pipeline.llm)
        # at a lower temperature suited to grounded synthesis rather than
        # creating a separate client — see llm.py for how the endpoint is configured.
        self.chat_model = chat_model or get_chat_model(temperature=0.1)
        self.max_fields = max_fields or settings.interdisciplinary_max_fields
        self.max_results_per_query = max_results_per_query or settings.default_max_results_per_query
        self.output_dir = Path(output_dir or settings.interdisciplinary_output_dir)
        # Injectable for the same reason CoderAgent's network_check/gpu_check
        # are: they're the only three calls in this agent that touch the network,
        # so tests substitute them rather than reaching arXiv/CORE for real.
        self.search_arxiv = search_arxiv_fn or search_arxiv
        self.search_semantic_scholar = search_semantic_scholar_fn or search_semantic_scholar
        self.search_core = search_core_fn or search_core

    def run(self, papers: List[dict], research_question: Optional[str] = None) -> dict:
        """Runs the agent's graph end to end, returning the validated output
        dict documented in this module's docstring.

        Node exceptions aren't swallowed by LangGraph, so an
        InterdisciplinaryLiteratureAgentError raised inside a node propagates
        out of `.invoke` unchanged.
        """
        # Imported here rather than at module scope: graph.py needs this module's
        # InterdisciplinaryLiteratureAgent for typing, so a top-level import
        # would be circular.
        from research_pipeline.agents.interdisciplinary_literature.graph import (
            build_interdisciplinary_literature_graph,
        )

        graph = build_interdisciplinary_literature_graph(self)
        final_state = graph.invoke(
            {"papers": papers, "research_question": research_question},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        return final_state["result"]

    # -- Graph nodes -----------------------------------------------------------
    # Thin adapters: each takes the graph state, delegates to the private helper
    # below that does the actual work, and returns only the keys it produces.

    def _node_identify_adjacent_fields(self, state: InterdisciplinaryState) -> dict:
        core_papers = [p for p in (state["papers"] or []) if isinstance(p, dict)]
        # normalize_papers is the same drop-the-unusable pass the Hypothesis
        # Agent runs; used here only to decide whether there's anything to work
        # from, since the raw dicts are what get handed downstream.
        if not normalize_papers(core_papers):
            raise InterdisciplinaryLiteratureAgentError(
                "No usable papers were provided (empty list, or every paper was missing "
                "title/abstract/full_text)."
            )

        fields = self._identify_adjacent_fields(core_papers, state.get("research_question")).get("fields", [])
        fields = [f for f in fields if isinstance(f, dict) and f.get("field")][: self.max_fields]
        for field in fields:
            queries = [q for q in (field.get("queries") or []) if isinstance(q, str) and q.strip()]
            # A field with no usable query still gets searched — on its own
            # name — rather than silently contributing nothing.
            field["queries"] = queries or [field["field"]]
            field["rationale"] = str(field.get("rationale") or "")

        logger.info("Exploring %d adjacent field(s): %s", len(fields), [f["field"] for f in fields])
        return {
            "core_papers": core_papers,
            "core_paper_ids": [_paper_id(p, i) for i, p in enumerate(core_papers)],
            "fields_explored": fields,
        }

    def _node_search_field(self, state: InterdisciplinaryState) -> dict:
        """Runs once per adjacent field — the graph fans this out with one `Send`
        per field, each carrying its own `current_field`. The single-element list
        is what the state's operator.add reducer accumulates across branches."""
        field = state["current_field"]
        queries = field["queries"]
        found = list(self.search_arxiv(queries, self.max_results_per_query))
        found += list(self.search_semantic_scholar(queries, self.max_results_per_query))
        found += list(self.search_core(queries, self.max_results_per_query))
        for paper in found:
            paper["discipline"] = field["field"]
        logger.info("Field '%s': %d paper(s) found across %d query/queries", field["field"], len(found), len(queries))
        return {"field_results": [{"field": field["field"], "papers": found}]}

    def _node_merge_cross_field(self, state: InterdisciplinaryState) -> dict:
        """Deterministic: dedupes the cross-field results against each other and
        against the in-domain papers on the Literature Agent's own doi/
        normalized-title key. No model judgment involved."""
        core_papers = state["core_papers"]
        seen = {_dedupe_key(p) for p in core_papers if p.get("title")}

        cross_field: List[dict] = []
        for result in state.get("field_results", []) or []:
            for paper in result["papers"]:
                if not paper.get("title"):
                    continue
                key = _dedupe_key(paper)
                if key in seen:
                    continue
                seen.add(key)
                cross_field.append(paper)

        merged = list(core_papers) + cross_field
        logger.info(
            "Merged %d in-domain + %d unique cross-field paper(s) into a pool of %d",
            len(core_papers), len(cross_field), len(merged),
        )
        return {"merged_papers": merged, "cross_field_papers": cross_field}

    def _node_score_cross_field(self, state: InterdisciplinaryState) -> dict:
        """Screens the cross-field papers before they reach bridge synthesis or
        the merged pool.

        Until this node existed, every paper an adjacent-field query returned
        entered `papers` — and therefore became citable by the Writer — on the
        strength of a query the model generated from a rationale two hops from
        the research question. A query for an adjacent field's terminology
        routinely returns work with no bearing on the core problem at all.

        The in-domain papers are deliberately *not* re-screened here: they
        arrived from the Literature Agent, which has already screened them
        against the same research question, and re-filtering another agent's
        validated output would make this agent silently lossy on its input.

        `merged_papers` is rebuilt rather than filtered, because `_paper_id`
        derives an id from a paper's *position* in that pool — dropping a
        cross-field paper without renumbering would leave every id after it
        pointing at the wrong paper.
        """
        cross_field = state["cross_field_papers"]
        if not settings.enable_relevance_filter or not cross_field:
            return {}

        scores = score_papers(
            self.chat_model,
            self._scoring_objective(state),
            cross_field,
            criterion=TRANSFER_RELEVANCE_CRITERION,
            batch_max_chars=settings.relevance_batch_max_chars,
        )
        kept, dropped = apply_threshold(
            cross_field,
            scores,
            min_score=settings.interdisciplinary_relevance_min_score,
            # No rescue floor here, unlike the Literature Agent's screen: "no
            # adjacent field yielded anything transferable" is an answer this
            # agent already handles (_node_synthesize_bridges returns no
            # insights, and the merged pool falls back to the in-domain papers),
            # so forcing through the least-bad untransferable paper would defeat
            # the screen rather than protect the run.
            keep_min=0,
        )

        logger.info(
            "Transferability filter kept %d/%d cross-field paper(s) at a threshold of %d",
            len(kept), len(cross_field), settings.interdisciplinary_relevance_min_score,
        )
        for paper in dropped:
            logger.debug(
                "Dropped as untransferable (score %s): %s", paper.get("relevance_score"), paper.get("title")
            )
        return {"cross_field_papers": kept, "merged_papers": list(state["core_papers"]) + kept}

    def _scoring_objective(self, state: InterdisciplinaryState) -> str:
        """What the cross-field papers are screened against.

        `research_question` is optional on this agent's entry point, and with
        none supplied the in-domain titles are the only statement of the problem
        available — a far better objective than skipping the screen entirely,
        which would put the whole unfiltered cross-field pool back in play.
        """
        question = state.get("research_question")
        if question:
            return question
        titles = [p.get("title") for p in state["core_papers"][:10] if p.get("title")]
        return "The problem addressed by this body of work:\n- " + "\n- ".join(titles)

    def _node_synthesize_bridges(self, state: InterdisciplinaryState) -> dict:
        cross_field = state["cross_field_papers"]
        if not cross_field:
            # Nothing was found to bridge from, so there is nothing for the
            # model to ground an insight in — an empty list is the honest
            # answer, and asking anyway would invite an invented one.
            logger.warning("No cross-field papers found — returning no bridge insights")
            return {"bridge_insights": []}

        raw = self._synthesize_bridges(cross_field, state["core_papers"], state.get("research_question"))
        known_ids = {_paper_id(p, i) for i, p in enumerate(state["merged_papers"])}

        insights: List[dict] = []
        for insight in raw.get("bridge_insights", []) or []:
            if not isinstance(insight, dict) or not insight.get("insight"):
                continue
            cited = [str(pid) for pid in (insight.get("supporting_paper_ids") or [])]
            resolved = [pid for pid in cited if pid in known_ids]
            if len(resolved) != len(cited):
                logger.warning(
                    "Dropping %d unresolvable supporting_paper_id(s) from a bridge insight: %s",
                    len(cited) - len(resolved), sorted(set(cited) - known_ids),
                )
            insights.append(
                {
                    "insight": str(insight["insight"]),
                    "source_field": str(insight.get("source_field") or ""),
                    "connection_to_core_problem": str(insight.get("connection_to_core_problem") or ""),
                    "supporting_paper_ids": resolved,
                }
            )

        logger.info("Synthesized %d bridge insight(s)", len(insights))
        return {"bridge_insights": insights}

    def _node_assemble_and_validate(self, state: InterdisciplinaryState) -> dict:
        result: dict = {
            "papers": state["merged_papers"],
            "core_paper_ids": state["core_paper_ids"],
            "cross_field_papers": state["cross_field_papers"],
            "fields_explored": state["fields_explored"],
            "bridge_insights": state["bridge_insights"],
            "research_question": state.get("research_question"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": settings.llm_model,
        }

        try:
            validate_output(result)
        except SchemaValidationError as exc:
            debug_path = self._write_output(result, suffix="_invalid")
            raise InterdisciplinaryLiteratureAgentError(
                f"Assembled output failed schema validation: {exc}. "
                f"Raw (invalid) output written to {debug_path} for inspection."
            ) from exc

        output_path = self._write_output(result)
        logger.info("Wrote interdisciplinary literature agent output to %s", output_path)
        return {"result": result}

    # -- LLM calls -----------------------------------------------------------

    def _call_json(self, user_prompt: str) -> dict:
        try:
            return invoke_json(self.chat_model, prompts.SYSTEM_PROMPT, user_prompt)
        except LLMJSONError as exc:
            raise InterdisciplinaryLiteratureAgentError(str(exc)) from exc

    def _research_question_line(self, research_question: Optional[str]) -> str:
        return f"Original research question guiding this search: {research_question}\n\n" if research_question else ""

    def _identify_adjacent_fields(self, core_papers: List[dict], research_question: Optional[str]) -> dict:
        prompt = prompts.ADJACENT_FIELDS_PROMPT.format(
            research_question_line=self._research_question_line(research_question),
            papers_block=_digest(core_papers),
            max_fields=self.max_fields,
        )
        return self._call_json(prompt)

    def _synthesize_bridges(
        self, cross_field_papers: List[dict], core_papers: List[dict], research_question: Optional[str]
    ) -> dict:
        # Cross-field papers sit after the in-domain ones in the merged pool, so
        # they're digested at that offset — the ids the model cites are then the
        # ids everything downstream resolves against.
        prompt = prompts.BRIDGE_INSIGHTS_PROMPT.format(
            research_question_line=self._research_question_line(research_question),
            cross_field_block=_digest(cross_field_papers, start_index=len(core_papers)),
            core_block=_digest(core_papers),
        )
        return self._call_json(prompt)

    # -- Output persistence ----------------------------------------------------

    def _write_output(self, result: dict, suffix: str = "") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"interdisciplinary_{timestamp}{suffix}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return path


def run_interdisciplinary_literature_agent(
    papers: List[dict],
    research_question: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Module-level entry point — the stable call the Hypothesis Agent (or the
    CLI, or the orchestrator) should use. Builds a default-configured
    InterdisciplinaryLiteratureAgent and runs it once; use the
    InterdisciplinaryLiteratureAgent class directly if you want to reuse one
    model/config across multiple calls."""
    return InterdisciplinaryLiteratureAgent(output_dir=output_dir).run(papers, research_question=research_question)
